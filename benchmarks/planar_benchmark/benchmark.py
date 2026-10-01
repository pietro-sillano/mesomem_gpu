"""benchmark.py — Planar membrane (dipole_sphere) benchmark.

GPU sweep (no mpirun):
    python benchmark.py --machine md69 --test_gpu --omp_list 1 4 8 \\
        --steps_gpu 1000 --lmp_bin lmp_kokkos_89_dev --hwthread

CPU MPI scaling:
    python benchmark.py --machine md69 --steps_cpu 500 \\
        --mpi_list 1 4 8 16 --hwthread
"""

import subprocess, os, argparse, time, re, platform
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_hw_info():
    cpu = "Unknown"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu = re.sub(".*model name.*:", "", line, 1).strip()
                        break
    except Exception:
        pass
    gpu = {"name": "None", "cap": "N/A"}
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits",
            shell=True).decode().strip().split(",")
        gpu = {"name": out[0].strip(), "cap": out[1].strip()}
    except Exception:
        pass
    return cpu, gpu


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_performance(log_path):
    tps = 0.0
    if not os.path.exists(log_path):
        return 0.0
    with open(log_path) as f:
        for line in f:
            if "Performance:" in line and "timesteps/s" in line:
                for p in line.split(","):
                    if "timesteps/s" in p:
                        try:
                            tps = float(p.strip().split()[0])
                        except Exception:
                            pass
    return tps


# ---------------------------------------------------------------------------
# Replica runner
# ---------------------------------------------------------------------------

def run_replicas(cmd, base_dir, env, n_replicas):
    tps_vals, wall_vals = [], []
    for rep in range(n_replicas):
        sim_dir = f"{base_dir}_rep{rep}"
        os.makedirs(sim_dir, exist_ok=True)
        print(f"    replica {rep+1}/{n_replicas}  dir={sim_dir}")
        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=sim_dir, check=True, env=env)
        except Exception as e:
            print(f"    ERROR: {e}")
            return None
        wall_vals.append(time.time() - t0)
        tps_vals.append(parse_performance(os.path.join(sim_dir, "log.lammps")))
    return {
        "TPS_mean":  np.mean(tps_vals),  "TPS_std":  np.std(tps_vals),
        "Wall_mean": np.mean(wall_vals), "Wall_std": np.std(wall_vals),
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_gpu(args, n, omp, lmp_input, hw_info):
    cpu_model, gpu_info = hw_info
    newton = "on" if args.neigh == "half" else "off"
    base_dir = os.path.join("bench_runs", f"N{n}_GPU_O{omp}")

    cmd  = [args.lmp_bin, "-in", lmp_input]
    cmd += ["-v", "N", str(n), "-v", "t_run", str(args.steps_gpu)]
    cmd += ["-v", "atomstyle", args.atomstyle]
    cmd += ["-k", "on", "g", "1", "t", str(omp), "-sf", "kk"]
    cmd += ["-pk", "kokkos", "newton", newton, "neigh", args.neigh, "comm", "device"]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    env["OMP_PROC_BIND"]   = "spread"
    env["OMP_PLACES"]      = "cores"

    print(f"--> GPU  N={n} OMP={omp}: {' '.join(cmd)}")
    rep = run_replicas(cmd, base_dir, env, args.replicas)
    if rep is None:
        return None
    return {"N": n, "Mode": "GPU", "MPI": 1, "OMP": omp, "Neigh": args.neigh,
            **rep, "CPU_Info": cpu_model, "GPU_Name": gpu_info["name"],
            "GPU_Cap": gpu_info["cap"], "LMP_Binary": os.path.basename(args.lmp_bin),
            "Machine": args.machine}


def run_cpu(args, n, mpi, lmp_input, hw_info):
    cpu_model, gpu_info = hw_info
    base_dir = os.path.join("bench_runs", f"N{n}_CPU_M{mpi}")

    cmd  = ["mpirun", "-np", str(mpi), "--map-by", "socket:PE=1"]
    if args.hwthread:
        cmd += ["--use-hwthread-cpus"]
    cmd += [args.lmp_bin, "-in", lmp_input]
    cmd += ["-v", "N", str(n), "-v", "t_run", str(args.steps_cpu)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"]   = "spread"
    env["OMP_PLACES"]      = "cores"

    print(f"--> CPU  N={n} MPI={mpi}: {' '.join(cmd)}")
    rep = run_replicas(cmd, base_dir, env, args.replicas)
    if rep is None:
        return None
    return {"N": n, "Mode": "CPU", "MPI": mpi, "OMP": 1, "Neigh": "none",
            **rep, "CPU_Info": cpu_model, "GPU_Name": gpu_info["name"],
            "GPU_Cap": gpu_info["cap"], "LMP_Binary": os.path.basename(args.lmp_bin),
            "Machine": args.machine}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine",   required=True,
                        help="Machine name — used in the CSV filename")
    # lmp_kokkos is the portable default; locally use lmp_kokkos_89_dev
    parser.add_argument("--lmp_bin",   default="lmp_kokkos")
    parser.add_argument("--steps_gpu", type=int, default=1000)
    parser.add_argument("--steps_cpu", type=int, default=500)
    parser.add_argument("--omp_list",  type=int, nargs="+", default=[1, 4, 8],
                        help="OMP thread counts for GPU test")
    parser.add_argument("--mpi_list",  type=int, nargs="+", default=[1, 4, 8, 16, 16],
                        help="MPI rank counts for CPU test")
    parser.add_argument("--n_list",    type=int, nargs="+",
                        default=[2500, 10000, 102400],
                        help="System sizes (must match files in input/)")
    parser.add_argument("--atomstyle", default="hybrid angle sphere dipole",
                        help="atom style for the GPU test (dipole_sphere_angle enables device sorting)")
    parser.add_argument("--replicas",  type=int, default=3)
    parser.add_argument("--test_gpu",  action="store_true")
    parser.add_argument("--hwthread",  action="store_true")
    parser.add_argument("--neigh",     default="half")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    lmp_input = os.path.abspath("planar.lmp")
    hw_info   = get_hw_info()
    results   = []

    # GPU test
    if args.test_gpu:
        for n in args.n_list:
            data = os.path.join("input", f"lattice_d_0.85_N_{n}_ds")
            if not os.path.exists(data):
                print(f"  [SKIP] data file not found: {data}")
                continue
            for omp in args.omp_list:
                row = run_gpu(args, n, omp, lmp_input, hw_info)
                if row:
                    results.append(row)

    # CPU test (always — needed as speedup baseline)
    for n in args.n_list:
        data = os.path.join("input", f"lattice_d_0.85_N_{n}_ds")
        if not os.path.exists(data):
            print(f"  [SKIP] data file not found: {data}")
            continue
        for mpi in args.mpi_list:
            row = run_cpu(args, n, mpi, lmp_input, hw_info)
            if row:
                results.append(row)

    df    = pd.DataFrame(results)
    fname = f"results_{args.machine}.csv"
    df.to_csv(fname, index=False)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
