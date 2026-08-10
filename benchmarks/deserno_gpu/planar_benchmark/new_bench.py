import subprocess
import os
import argparse
import time
import csv
import pandas as pd
import platform
import re
import matplotlib.pyplot as plt
import numpy as np


def get_hw_info():
    """Detects CPU and GPU specifications."""
    cpu = "Unknown"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu = re.sub(".*model name.*:", "", line, 1).strip()
                        break
    except: pass

    gpu = {"name": "None", "cap": "N/A"}
    try:
        cmd = "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode().strip().split(',')
        gpu = {"name": out[0].strip(), "cap": out[1].strip()}
    except: pass

    return cpu, gpu

def parse_performance(log_path):
    """Extracts timesteps/s from log.lammps."""
    tps = 0.0
    if not os.path.exists(log_path): return 0.0
    with open(log_path, 'r') as f:
        for line in f:
            if "Performance:" in line and "timesteps/s" in line:
                parts = line.split(',')
                for p in parts:
                    if "timesteps/s" in p:
                        try: tps = float(p.strip().split()[0])
                        except: pass
    return tps

def run_bench(args, n_atoms, mpi, omp, use_gpu, hw_info):
    cpu_model, gpu_info = hw_info
    timestamp = time.strftime("%H%M%S")
    bin_tag = os.path.basename(args.lmp_bin)
    sim_id = f"N{n_atoms}_M{mpi}_O{omp}_G{int(use_gpu)}_{bin_tag}_{timestamp}"
    sim_dir = os.path.join("bench_runs", sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    lipid_file = os.path.join(script_dir, "lipid.txt")

    # --map-by socket:PE=<omp> allocates <omp> CPUs per rank and binds accordingly.
    # This is the modern replacement for the deprecated --cpus-per-proc + --bind-to pair.
    # With --use-hwthread-cpus the PEs are hwthreads; without, they are physical cores.
    cmd = ["mpirun", "-np", str(mpi), "--map-by", f"socket:PE={omp}"]
    if args.hwthread:
        cmd += ["--use-hwthread-cpus"]

    cmd += [args.lmp_bin, "-in", os.path.join(script_dir, "planar.lmp")]
    cmd += ["-v", "N", str(n_atoms), "-v", "t_run", str(args.steps)]
    cmd += ["-v", "lipid_file", lipid_file]

    # Let OMP threads spread across the allocated cores without migration.
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "cores"

    notes = []
    if use_gpu:
        cmd += ["-k", "on", "g", "1", "t", str(omp), "-sf", "kk"]
        cmd += ["-pk", "kokkos", "newton", args.newton, "neigh", args.neigh, "comm", "device"]
        notes.append("Kokkos_GPU")
    else:
        if omp > 1:
            cmd += ["-k", "on", "t", str(omp), "-sf", "kk"]
            notes.append("Kokkos_OMP")

    print(f"--> Running: {' '.join(cmd)}")

    start = time.time()
    try:
        subprocess.run(cmd, cwd=sim_dir, check=True, env=env)
        wall = time.time() - start
        tps = parse_performance(os.path.join(sim_dir, "log.lammps"))

        return {
            "N": n_atoms, "MPI": mpi, "OMP": omp, "GPU": use_gpu,
            "TPS": tps, "Wall_Time": wall, "CPU_Info": cpu_model,
            "GPU_Name": gpu_info['name'], "GPU_Cap": gpu_info['cap'],
            "LMP_Binary": bin_tag,
            "Notes": ", ".join(notes)
        }
    except Exception as e:
        print(f"Error in {sim_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp_bin", default="lmp_kokkos_61",
                        help="LAMMPS binary (e.g. lmp_kokkos_61 or lmp_kokkos_89)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n_list", type=int, nargs='+',
                        default=[600, 2400, 9600, 38400, 153600, 614400],
                        help="Total atom counts (must be divisible by 6). "
                             "These correspond to square bilayers of L=10,20,40,80,160,320.")
    parser.add_argument("--mpi_list", type=int, nargs='+', default=[1])
    parser.add_argument("--omp_list", type=int, nargs='+', default=[1])
    parser.add_argument("--test_gpu", action='store_true', help="Run GPU benchmark")
    parser.add_argument("--hwthread", action='store_true', help="Enable --use-hwthread-cpus")
    # Kokkos package flags
    parser.add_argument("--newton", default="on")
    parser.add_argument("--neigh", default="half")

    args = parser.parse_args()
    hw_info = get_hw_info()
    results = []

    gpu_options = [True] if args.test_gpu else [False]

    for n in args.n_list:
        for mpi in args.mpi_list:
            for omp in args.omp_list:
                for use_gpu in gpu_options:
                    res = run_bench(args, n, mpi, omp, use_gpu, hw_info)
                    if res: results.append(res)

    df = pd.DataFrame(results)
    bin_tag = os.path.basename(args.lmp_bin)
    fname = f"bench_deserno_{bin_tag}_{time.strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=False)
    print(f"Results saved to {fname}")

if __name__ == "__main__":
    main()


    # --- Example commands ---

    # Smoke test (CPU, N=600, 100 steps):
    # python new_bench.py --steps 100 --n_list 600 --mpi_list 1 --omp_list 1

    # GPU sweep on CC6.1 machine:
    # python new_bench.py --steps 1000 --n_list 600 2400 9600 38400 153600 614400 --mpi_list 1 --omp_list 4 --lmp_bin lmp_kokkos_61 --test_gpu --hwthread

    # GPU sweep on CC8.9 machine:
    # python new_bench.py --steps 1000 --n_list 600 2400 9600 38400 153600 614400 --mpi_list 1 --omp_list 4 --lmp_bin lmp_kokkos_89_dev --test_gpu --hwthread

    # MPI scaling (CPU, no GPU):
    # python new_bench.py --steps 100 --n_list 600 2400 9600 38400 153600 614400 --mpi_list 1 4 8 12 --omp_list 1 --hwthread --lmp_bin lmp_kokkos_61
