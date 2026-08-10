"""plot_bench.py — Unified benchmark plot script.

Reads results_{machine}.csv from each system folder, writes plots to
plots/{machine}/.

Usage:
    python plot_bench.py --machine md69
    python plot_bench.py --machine md69 --systems planar solvent
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# System name → subfolder containing results_{machine}.csv
SYSTEM_DIRS = {
    "planar":          "planar_benchmark",
    "solvent":         "solvent_benchmark",
    "polymer":         "polymer",
    "polymer_solvent": "polymer_solvent",
}

# Colour palette for OMP thread counts (GPU lines)
OMP_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink",
]

# Colour palette for MPI rank counts (CPU lines)
MPI_COLORS = [
    "tab:red", "tab:purple", "tab:brown", "tab:olive",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_gpu_col(df):
    if df["Mode"].dtype == object:
        pass  # Mode is already "GPU"/"CPU" string
    if "GPU" in df.columns and "Mode" not in df.columns:
        if df["GPU"].dtype == object:
            df["GPU"] = df["GPU"].astype(str).str.strip().str.lower() \
                                 .map({"true": True, "false": False})
        df["Mode"] = df["GPU"].map({True: "GPU", False: "CPU"})
    return df


def load_csv(base_dir, system, machine):
    folder = os.path.join(base_dir, SYSTEM_DIRS[system])
    path   = os.path.join(folder, f"results_{machine}.csv")
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    df = pd.read_csv(path)
    df = normalise_gpu_col(df)
    return df


def omp_color(omp_val, omp_vals_sorted):
    idx = list(omp_vals_sorted).index(omp_val)
    return OMP_COLORS[idx % len(OMP_COLORS)]


def mpi_color(mpi_val, mpi_vals_sorted):
    idx = list(mpi_vals_sorted).index(mpi_val)
    return MPI_COLORS[idx % len(MPI_COLORS)]


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — Absolute TPS vs N  (or bar chart for fixed-N)
# ---------------------------------------------------------------------------

def plot_tps(df, system, out_dir, machine):
    gpu = df[df["Mode"] == "GPU"]
    cpu = df[df["Mode"] == "CPU"]

    n_vals = sorted(df["N"].unique())
    fixed_n = len(n_vals) == 1

    fig, ax = plt.subplots(figsize=(9, 5))

    omp_vals = sorted(gpu["OMP"].unique()) if not gpu.empty else []
    mpi_vals = sorted(cpu["MPI"].unique()) if not cpu.empty else []

    if fixed_n:
        # Bar chart
        configs, tps_means, tps_stds, colors = [], [], [], []
        for omp in omp_vals:
            sub = gpu[gpu["OMP"] == omp]
            if not sub.empty:
                configs.append(f"GPU OMP={omp}")
                tps_means.append(sub["TPS_mean"].values[0])
                tps_stds.append(sub["TPS_std"].values[0])
                colors.append(omp_color(omp, omp_vals))
        for mpi in mpi_vals:
            sub = cpu[cpu["MPI"] == mpi]
            if not sub.empty:
                configs.append(f"CPU MPI={mpi}")
                tps_means.append(sub["TPS_mean"].values[0])
                tps_stds.append(sub["TPS_std"].values[0])
                colors.append(mpi_color(mpi, mpi_vals))
        x = np.arange(len(configs))
        ax.bar(x, tps_means, yerr=tps_stds, color=colors,
               capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Configuration")
    else:
        # Line plot, log-log
        for omp in omp_vals:
            sub = gpu[gpu["OMP"] == omp].sort_values("N")
            if sub.empty:
                continue
            c = omp_color(omp, omp_vals)
            ax.errorbar(sub["N"], sub["TPS_mean"], yerr=sub["TPS_std"],
                        marker="o", color=c, label=f"GPU OMP={omp}",
                        capsize=3)
        for i, mpi in enumerate(mpi_vals):
            sub = cpu[cpu["MPI"] == mpi].sort_values("N")
            if sub.empty:
                continue
            c = mpi_color(mpi, mpi_vals)
            ax.errorbar(sub["N"], sub["TPS_mean"], yerr=sub["TPS_std"],
                        marker="x", linestyle="--", color=c,
                        label=f"CPU MPI={mpi}", capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N atoms")
        ax.legend(fontsize=8)

    ax.set_ylabel("Timesteps / s")
    ax.set_title(f"{system} — Throughput ({machine})")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, f"{system}_tps.png")


# ---------------------------------------------------------------------------
# Plot 2 — Speedup vs N  (relative to MPI=1 CPU)
# ---------------------------------------------------------------------------

def plot_speedup(df, system, out_dir, machine):
    cpu = df[df["Mode"] == "CPU"]
    gpu = df[df["Mode"] == "GPU"]

    baseline = cpu[cpu["MPI"] == 1].set_index("N")["TPS_mean"]
    if baseline.empty:
        print(f"  [skip speedup] no MPI=1 CPU baseline for {system}")
        return

    n_vals   = sorted(df["N"].unique())
    fixed_n  = len(n_vals) == 1
    omp_vals = sorted(gpu["OMP"].unique()) if not gpu.empty else []
    mpi_vals = sorted(cpu["MPI"].unique()) if not cpu.empty else []

    fig, ax = plt.subplots(figsize=(9, 5))

    if fixed_n:
        configs, speedups, speedup_errs, colors = [], [], [], []
        n0 = n_vals[0]
        base_tps = baseline.get(n0, np.nan)
        base_std = cpu[(cpu["MPI"] == 1) & (cpu["N"] == n0)]["TPS_std"].values
        base_std = base_std[0] if len(base_std) else 0.0
        for omp in omp_vals:
            sub = gpu[gpu["OMP"] == omp]
            if not sub.empty:
                tps = sub["TPS_mean"].values[0]
                std = sub["TPS_std"].values[0]
                configs.append(f"GPU OMP={omp}")
                speedups.append(tps / base_tps)
                # error propagation: d(s/b)/s = sqrt((std/tps)^2 + (base_std/base_tps)^2) * speedup
                speedup_errs.append((tps / base_tps) *
                                    np.sqrt((std / tps) ** 2 + (base_std / base_tps) ** 2)
                                    if base_tps > 0 else 0.0)
                colors.append(omp_color(omp, omp_vals))
        for mpi in mpi_vals:
            sub = cpu[cpu["MPI"] == mpi]
            if not sub.empty:
                tps = sub["TPS_mean"].values[0]
                std = sub["TPS_std"].values[0]
                configs.append(f"CPU MPI={mpi}")
                speedups.append(tps / base_tps)
                speedup_errs.append((tps / base_tps) *
                                    np.sqrt((std / tps) ** 2 + (base_std / base_tps) ** 2)
                                    if base_tps > 0 else 0.0)
                colors.append(mpi_color(mpi, mpi_vals))
        x = np.arange(len(configs))
        ax.bar(x, speedups, yerr=speedup_errs, color=colors, capsize=4, alpha=0.85)
        ax.axhline(1.0, color="grey", linestyle=":", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Configuration")
    else:
        for omp in omp_vals:
            sub = gpu[gpu["OMP"] == omp].sort_values("N")
            sub = sub.set_index("N")
            speedup = sub["TPS_mean"].div(baseline).dropna()
            if speedup.empty:
                continue
            c = omp_color(omp, omp_vals)
            # error propagation for speedup = tps/base
            yerr = speedup * np.sqrt(
                (sub.loc[speedup.index, "TPS_std"] / sub.loc[speedup.index, "TPS_mean"]) ** 2 +
                (cpu[cpu["MPI"] == 1].set_index("N").loc[speedup.index, "TPS_std"] /
                 baseline.loc[speedup.index]) ** 2
            )
            ax.errorbar(speedup.index, speedup.values, yerr=yerr.values,
                        marker="o", color=c, label=f"GPU OMP={omp}", capsize=3)
        for mpi in mpi_vals:
            sub = cpu[cpu["MPI"] == mpi].sort_values("N")
            sub = sub.set_index("N")
            speedup = sub["TPS_mean"].div(baseline).dropna()
            if speedup.empty:
                continue
            c = mpi_color(mpi, mpi_vals)
            yerr = speedup * np.sqrt(
                (sub.loc[speedup.index, "TPS_std"] / sub.loc[speedup.index, "TPS_mean"]) ** 2 +
                (cpu[cpu["MPI"] == 1].set_index("N").loc[speedup.index, "TPS_std"] /
                 baseline.loc[speedup.index]) ** 2
            )
            ax.errorbar(speedup.index, speedup.values, yerr=yerr.values,
                        marker="x", linestyle="--", color=c,
                        label=f"CPU MPI={mpi}", capsize=3)
        ax.axhline(1.0, color="grey", linestyle=":", alpha=0.6, label="baseline")
        ax.set_xscale("log")
        ax.set_xlabel("N atoms")
        ax.legend(fontsize=8)

    ax.set_ylabel("Speedup (vs MPI=1 CPU)")
    ax.set_title(f"{system} — Speedup ({machine})")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, f"{system}_speedup.png")


# ---------------------------------------------------------------------------
# Plot 3 — GPU vs best-CPU ratio (summary across systems)
# ---------------------------------------------------------------------------

SYSTEM_COLORS = {
    "planar":          "tab:blue",
    "solvent":         "tab:orange",
    "polymer":         "tab:green",
    "polymer_solvent": "tab:purple",
}


def plot_gpu_vs_cpu(all_data, out_dir, machine):
    """Horizontal bar: best GPU TPS / best CPU TPS for each system (largest N only)."""
    rows = []   # list of (system, n, ratio)

    for system, df in all_data.items():
        gpu = df[df["Mode"] == "GPU"]
        cpu = df[df["Mode"] == "CPU"]
        if gpu.empty or cpu.empty:
            continue
        n_max = df["N"].max()
        gpu_n = gpu[gpu["N"] == n_max]["TPS_mean"]
        cpu_n = cpu[cpu["N"] == n_max]["TPS_mean"]
        if gpu_n.empty or cpu_n.empty:
            continue
        rows.append((system, n_max, gpu_n.max() / cpu_n.max()))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 1.0)))

    for y_pos, (system, n, ratio) in enumerate(rows):
        color = SYSTEM_COLORS.get(system, "tab:grey")
        ax.barh(y_pos, ratio, color=color, alpha=0.80, label=system)
        ax.text(ratio + 0.05, y_pos, f"×{ratio:.1f}", va="center", fontsize=9)

    ax.axvline(1.0, color="grey", linestyle=":", alpha=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{s}  (N={n:,})" for s, n, _ in rows], fontsize=9)
    ax.set_xlabel("Best GPU TPS / Best CPU TPS")
    ax.set_title(f"GPU vs Best CPU — largest N per system ({machine})")
    ax.grid(True, axis="x", ls="-", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, f"gpu_vs_cpu_ratio.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine",  required=True)
    parser.add_argument("--systems",  nargs="+",
                        default=list(SYSTEM_DIRS.keys()),
                        choices=list(SYSTEM_DIRS.keys()))
    parser.add_argument("--base_dir", default=".",
                        help="Repo root (where system subdirectories live)")
    args = parser.parse_args()

    out_dir = os.path.join(args.base_dir, "plots", args.machine)
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}
    for system in args.systems:
        print(f"\n=== {system} ===")
        df = load_csv(args.base_dir, system, args.machine)
        if df is None:
            continue
        all_data[system] = df
        plot_tps(df, system, out_dir, args.machine)
        plot_speedup(df, system, out_dir, args.machine)

    # Summary across systems
    if all_data:
        plot_gpu_vs_cpu(all_data, out_dir, args.machine)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
