"""
Bar plot: serial CPU vs best CPU (MPI) vs best GPU (single precision), in
timesteps/s, for each benchmarked system, one panel per machine.

Usage: python plot_compare.py
Reads results_<machine>.csv next to this script, writes cpu_vs_gpu.png.
"""
import csv, os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MACHINES = [("md68", "RTX 4090 + i9-14900K"), ("md43", "GTX 1080 Ti + Xeon W-2135")]
SYSTEMS = [("planar_10k", "planar\n10k atoms"), ("planar_102k", "planar\n102k atoms"),
           ("polymer_solvent_229k", "polymer + solvent\n229k atoms")]
COLORS = {"serial": "#9e9e9e", "cpu": "#4c72b0", "gpu": "#dd8452"}


def load(machine):
    rows = list(csv.DictReader(open(os.path.join(HERE, f"results_{machine}.csv"))))
    out = {}
    for name, _ in SYSTEMS:
        sys_rows = [r for r in rows if r["system"] == name]
        cpu = [r for r in sys_rows if r["config"].startswith("CPU")]
        serial = next(r for r in cpu if int(r["mpi"]) == 1)
        best_cpu = max(cpu, key=lambda r: float(r["tps_mean"]))
        gpu = next(r for r in sys_rows if r["config"] == "GPU single")
        out[name] = [("serial CPU (1 core)", serial, "serial"),
                     (f"best CPU (MPI {best_cpu['mpi']})", best_cpu, "cpu"),
                     ("GPU, single precision", gpu, "gpu")]
    return out


fig, axes = plt.subplots(1, len(MACHINES), figsize=(12, 5), sharey=True)
width = 0.27
x = np.arange(len(SYSTEMS))

for ax, (machine, hw) in zip(axes, MACHINES):
    data = load(machine)
    for k in range(3):
        for i, (name, _) in enumerate(SYSTEMS):
            label, row, kind = data[name][k]
            tps = float(row["tps_mean"])
            serial = float(data[name][0][1]["tps_mean"])
            xpos = x[i] + (k - 1) * width
            ax.bar(xpos, tps, width * 0.92, color=COLORS[kind],
                   label=label.split(" (MPI")[0] if i == 0 else None)
            text = f"{tps:.0f}" if k == 0 else f"{tps:.0f}\n{tps / serial:.0f}x"
            if kind == "cpu":
                text = f"MPI {row['mpi']}\n" + text
            ax.text(xpos, tps * 1.08, text, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in SYSTEMS])
    ax.set_yscale("log")
    ax.set_ylim(3, 30000)
    ax.set_title(f"{machine}: {hw}")
    ax.grid(axis="y", which="major", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axes[0].set_ylabel("timesteps/s (log scale)")
axes[0].legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle("mesomem/dipole: serial CPU vs best CPU vs GPU (labels: steps/s, speedup over serial)")
fig.tight_layout()
out = os.path.join(HERE, "cpu_vs_gpu.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
