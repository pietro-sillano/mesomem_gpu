"""
Compare GPU Kokkos precision (double / mixed / single) against the
CPU-only plain MPI runs (1, 2, 4, ... ranks up to --cores).

Needs three builds made by compile_local.sh, named <prefix>, <prefix>-mixed
and <prefix>-single (default prefix: _build/local):
  _build/local          (KOKKOS_PREC=double, also used for the CPU runs)
  _build/local-mixed    (KOKKOS_PREC=mixed)
  _build/local-single   (KOKKOS_PREC=single)

Usage: python3 compare.py [--machine NAME] [--build_prefix _build/local]
                          [--cores 8] [--replicas 2]
Writes results_<machine>.csv next to this script.
"""
import argparse, csv, os, re, socket, subprocess, time

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(ROOT, "benchmarks")

BUILD_SUFFIX = {"gpu-double": "", "gpu-mixed": "-mixed", "gpu-single": "-single"}

# name, folder, input script, extra -v variables, GPU steps, CPU steps
SYSTEMS = [
    ("planar_10k",  "planar_benchmark", "planar.lmp", ["-v", "N", "10000"],  2000, 500),
    ("planar_102k", "planar_benchmark", "planar.lmp", ["-v", "N", "102400"], 1000, 100),
    ("polymer_solvent_229k", "polymer_solvent", "polymer_solvent.lmp",
     # path relative to the system folder: the script reads ../../${data_file}
     ["-v", "data_file", "input/combined_N35280_poly64000_r45.5_d0.10.data"], 1000, 50),
]


def configs(cores):
    """(label, build, mpi ranks, omp threads, extra lmp args, is_gpu)"""
    out = []
    for prec in ("double", "mixed", "single"):
        out.append((f"GPU {prec}", f"gpu-{prec}", 1, 1,
                    ["-k", "on", "g", "1", "t", "1", "-sf", "kk",
                     "-pk", "kokkos", "newton", "on", "neigh", "half", "comm", "device"], True))
    ranks, np_ = [], 1
    while np_ <= cores:
        ranks.append(np_)
        np_ *= 2
    if ranks[-1] != cores:
        ranks.append(cores)
    for np_ in ranks:
        out.append((f"CPU MPI {np_}", "gpu-double", np_, 1, [], False))
    return out


def parse_log(path):
    """timesteps/s and the last potential energy printed by thermo"""
    tps, pe, cols = None, None, None
    with open(path) as f:
        for line in f:
            if "Performance:" in line and "timesteps/s" in line:
                tps = float(re.search(r"([\d.]+) timesteps/s", line).group(1))
            words = line.split()
            if words[:1] == ["Step"]:
                cols = words
            elif cols and len(words) == len(cols) and words[0].isdigit():
                pe = float(words[cols.index("PotEng")])
            elif line.startswith("Loop time"):
                cols = None
    return tps, pe


def run_one(env_sh, mpi, omp, extra, system, steps, run_dir):
    name, folder, script, sysvars = system[0], system[1], system[2], system[3]
    os.makedirs(run_dir, exist_ok=True)
    lmp = ["lmp", "-in", os.path.join(BENCH, folder, script),
           "-v", "t_run", str(steps)] + sysvars + extra
    cmd = (f"source {env_sh} && exec mpirun -np {mpi} --map-by socket:PE={omp} --bind-to core "
           + " ".join(f"'{a}'" for a in lmp))
    env = dict(os.environ, OMP_NUM_THREADS=str(omp), OMP_PROC_BIND="spread", OMP_PLACES="cores")
    t0 = time.time()
    res = subprocess.run(["bash", "-c", cmd], cwd=run_dir, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"      FAILED: {res.stderr.strip().splitlines()[-1:]}")
        return None, None
    tps, pe = parse_log(os.path.join(run_dir, "log.lammps"))
    print(f"      {tps} steps/s, PE={pe}, {time.time()-t0:.1f}s wall")
    return tps, pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default=socket.gethostname().split(".")[0])
    ap.add_argument("--build_prefix", default="_build/local",
                    help="build folder of the double build, relative to the repo root")
    ap.add_argument("--cores", type=int, default=8, help="max CPU cores for CPU runs")
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--systems", nargs="+", default=[s[0] for s in SYSTEMS],
                    help="subset of systems to run")
    args = ap.parse_args()

    env_files = {b: os.path.join(ROOT, args.build_prefix + sfx, "env.sh")
                 for b, sfx in BUILD_SUFFIX.items()}
    for build, path in env_files.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing build {build}: {path} (run compile_local.sh)")

    rows = []
    for system in [s for s in SYSTEMS if s[0] in args.systems]:
        name, folder, _, _, steps_gpu, steps_cpu = system
        print(f"== {name} ==")
        for label, build, mpi, omp, extra, is_gpu in configs(args.cores):
            steps = steps_gpu if is_gpu else steps_cpu
            print(f"  -- {label} ({steps} steps)")
            tps_list, pe = [], None
            for rep in range(args.replicas):
                tag = label.replace(" ", "_")
                run_dir = os.path.join(BENCH, folder, "bench_runs", f"prec_{args.machine}_{name}_{tag}_rep{rep}")
                tps, pe = run_one(env_files[build], mpi, omp, extra, system, steps, run_dir)
                if tps is None:
                    break
                tps_list.append(tps)
            if not tps_list:
                continue
            mean = sum(tps_list) / len(tps_list)
            std = (sum((t - mean) ** 2 for t in tps_list) / len(tps_list)) ** 0.5
            rows.append({"system": name, "config": label, "build": build, "mpi": mpi,
                         "omp": omp, "steps": steps, "tps_mean": round(mean, 3),
                         "tps_std": round(std, 3), "pe_last": pe, "machine": args.machine})

    out = os.path.join(HERE, f"results_{args.machine}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
