"""compare_bench.py — Compare two benchmark runs (e.g. old vs new LAMMPS build).

Reads results_{old}.csv and results_{new}.csv from each system folder,
matches rows on (Mode, N, MPI, OMP), prints a table with the throughput
ratio new/old, and writes one bar plot per system to
plots/compare_{old}_vs_{new}/.

A ratio > 1 means the new build is faster.

Usage:
    python compare_bench.py --old md69 --new md69_p13Nov2026
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from plot_bench import SYSTEM_DIRS, load_csv, save

KEYS = ["Mode", "N", "MPI", "OMP"]


def compare_system(base_dir, system, old, new):
    df_old = load_csv(base_dir, system, old)
    df_new = load_csv(base_dir, system, new)
    if df_old is None or df_new is None:
        return None

    cols = KEYS + ["TPS_mean", "TPS_std"]
    df = pd.merge(df_old[cols], df_new[cols], on=KEYS, suffixes=("_old", "_new"))
    if df.empty:
        print("  no matching configurations")
        return None

    df["ratio"] = df["TPS_mean_new"] / df["TPS_mean_old"]
    df = df.sort_values(["Mode", "N", "MPI", "OMP"], ascending=[False, True, True, True])
    return df.reset_index(drop=True)


def print_table(df):
    print(f"  {'Mode':4} {'N':>8} {'MPI':>4} {'OMP':>4} {'TPS_old':>10} {'TPS_new':>10} {'new/old':>8}")
    for _, r in df.iterrows():
        print(f"  {r['Mode']:4} {r['N']:>8d} {r['MPI']:>4d} {r['OMP']:>4d} "
              f"{r['TPS_mean_old']:>10.1f} {r['TPS_mean_new']:>10.1f} {r['ratio']:>8.3f}")


def plot_ratio(df, system, out_dir, old, new):
    labels = [f"{r['Mode']} N={r['N']}\nMPI={r['MPI']} OMP={r['OMP']}" for _, r in df.iterrows()]
    colors = ["tab:green" if m == "GPU" else "tab:gray" for m in df["Mode"]]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df)), 4))
    ax.bar(range(len(df)), df["ratio"], color=colors)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(f"TPS ratio  {new} / {old}")
    ax.set_title(f"{system} — new vs old build (green: GPU, gray: CPU)")
    save(fig, out_dir, f"{system}_ratio.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="machine label of the baseline CSVs")
    parser.add_argument("--new", required=True, help="machine label of the new CSVs")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEM_DIRS.keys()),
                        choices=list(SYSTEM_DIRS.keys()))
    parser.add_argument("--base_dir", default=".")
    args = parser.parse_args()

    out_dir = os.path.join(args.base_dir, "plots", f"compare_{args.old}_vs_{args.new}")
    os.makedirs(out_dir, exist_ok=True)

    for system in args.systems:
        print(f"\n=== {system} ===")
        df = compare_system(args.base_dir, system, args.old, args.new)
        if df is None:
            continue
        print_table(df)
        plot_ratio(df, system, out_dir, args.old, args.new)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
