import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

df = pd.read_csv('kokkos_sweep_lmp_kokkos_89_dev_20260520_2120.csv')  # edit filename

# One figure per N value.
# Lines  = (neigh, comm) combinations, coloured consistently across figures.
# Sort   = solid (host) vs dashed (device) — shows whether it matters.
# x-axis = OMP thread count.

# Fixed colour per (neigh, comm) combo
combo_colors = {
    ("half", "host"):   "tab:blue",
    ("half", "device"): "tab:orange",
    ("full", "host"):   "tab:green",
    ("full", "device"): "tab:red",
}
sort_styles = {"host": "-", "device": "--"}
sort_markers = {"host": "o", "device": "s"}

omp_vals = sorted(df["OMP"].unique())

for n_val, df_n in sorted(df.groupby("N")):
    fig, ax = plt.subplots(figsize=(9, 5))

    for (neigh, comm), df_combo in df_n.groupby(["neigh", "comm"]):
        color = combo_colors.get((neigh, comm), "black")
        label_base = f"neigh={neigh}, comm={comm}"

        for sort_val, df_sort in df_combo.groupby("sort"):
            df_sort = df_sort.sort_values("OMP")
            ax.plot(df_sort["OMP"], df_sort["TPS"],
                    color=color,
                    linestyle=sort_styles[sort_val],
                    marker=sort_markers[sort_val],
                    label=f"{label_base}, sort={sort_val}")

    ax.set_title(f"Kokkos option sweep — N={n_val} atoms")
    ax.set_xlabel("OMP threads")
    ax.set_ylabel("Timesteps/s")
    ax.set_xticks(omp_vals)
    ax.grid(True, ls="-", alpha=0.3)

    # Build a two-section legend: combos by colour, sort by linestyle
    combo_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="-",
                      label=f"neigh={n}, comm={co}")
        for (n, co), c in combo_colors.items()
    ]
    sort_handles = [
        mlines.Line2D([], [], color="grey", marker=sort_markers[s],
                      linestyle=sort_styles[s], label=f"sort={s}")
        for s in ["host", "device"]
    ]
    ax.legend(handles=combo_handles + sort_handles,
              title="Configuration", fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(f"kokkos_sweep_N{n_val}.png", dpi=200, transparent=True)
    plt.show()
    plt.close()
