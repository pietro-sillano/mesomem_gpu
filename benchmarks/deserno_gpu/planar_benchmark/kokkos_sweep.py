import subprocess
import os
import argparse
import time
import itertools
import pandas as pd
import platform
import re


def get_hw_info():
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
    tps = 0.0
    if not os.path.exists(log_path): return 0.0
    with open(log_path, 'r') as f:
        for line in f:
            if "Performance:" in line and "timesteps/s" in line:
                for p in line.split(','):
                    if "timesteps/s" in p:
                        try: tps = float(p.strip().split()[0])
                        except: pass
    return tps

def run_one(args, n_atoms, omp, neigh, newton, comm, sort, hw_info):
    cpu_model, gpu_info = hw_info
    timestamp = time.strftime("%H%M%S")
    bin_tag = os.path.basename(args.lmp_bin)
    combo_tag = f"neigh{neigh}_newton{newton}_comm{comm}_sort{sort}_O{omp}"
    sim_id = f"N{n_atoms}_{combo_tag}_{bin_tag}_{timestamp}"
    sim_dir = os.path.join("kokkos_sweep_runs", sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    lipid_file = os.path.join(script_dir, "lipid.txt")

    cmd = ["mpirun", "-np", "1", "--map-by", f"socket:PE={omp}"]
    if args.hwthread:
        cmd += ["--use-hwthread-cpus"]

    cmd += [args.lmp_bin, "-in", os.path.join(script_dir, "planar.lmp")]
    cmd += ["-v", "N", str(n_atoms), "-v", "t_run", str(args.steps)]
    cmd += ["-v", "lipid_file", lipid_file]
    cmd += ["-k", "on", "g", "1", "t", str(omp), "-sf", "kk"]
    cmd += ["-pk", "kokkos",
            "newton", newton,
            "neigh",  neigh,
            "comm",   comm,
            "sort",   sort]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "cores"

    print(f"--> {combo_tag}  N={n_atoms}")

    start = time.time()
    try:
        subprocess.run(cmd, cwd=sim_dir, check=True, env=env)
        wall = time.time() - start
        tps = parse_performance(os.path.join(sim_dir, "log.lammps"))
        return {
            "N": n_atoms, "OMP": omp,
            "neigh": neigh, "newton": newton, "comm": comm, "sort": sort,
            "TPS": tps, "Wall_Time": wall,
            "CPU_Info": cpu_model, "GPU_Name": gpu_info["name"],
            "GPU_Cap": gpu_info["cap"], "LMP_Binary": bin_tag,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def kokkos_combinations():
    """Valid Kokkos GPU parameter combinations.
    neigh=full requires newton=off; neigh=half uses newton=on.
    """
    for neigh, newton in [("half", "on"), ("full", "off")]:
        for comm in ["host", "device"]:
            for sort in ["host", "device"]:
                yield neigh, newton, comm, sort


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Kokkos GPU package options for the Deserno planar benchmark.")
    parser.add_argument("--lmp_bin", default="lmp_kokkos_89")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n_list", type=int, nargs='+', default=[9600, 38400],
                        help="System sizes to test (total atoms, divisible by 6).")
    parser.add_argument("--omp_list", type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument("--hwthread", action='store_true')

    args = parser.parse_args()
    hw_info = get_hw_info()
    results = []

    combos = list(kokkos_combinations())
    total = len(args.n_list) * len(args.omp_list) * len(combos)
    print(f"Running {total} configurations "
          f"({len(combos)} Kokkos combos × {len(args.omp_list)} OMP × {len(args.n_list)} N)\n")

    for n in args.n_list:
        for omp in args.omp_list:
            for neigh, newton, comm, sort in combos:
                res = run_one(args, n, omp, neigh, newton, comm, sort, hw_info)
                if res:
                    results.append(res)

    df = pd.DataFrame(results)
    bin_tag = os.path.basename(args.lmp_bin)
    fname = f"kokkos_sweep_{bin_tag}_{time.strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=False)
    print(f"\nResults saved to {fname}")

    # Quick summary: best TPS per N
    print("\n--- Best configurations per N ---")
    for n, grp in df.groupby("N"):
        best = grp.loc[grp["TPS"].idxmax()]
        print(f"  N={n:>7}  TPS={best['TPS']:.1f}  "
              f"neigh={best['neigh']} newton={best['newton']} "
              f"comm={best['comm']} sort={best['sort']} OMP={best['OMP']}")


if __name__ == "__main__":
    main()


    # Quick run (2 sizes, default OMP sweep):
    # python kokkos_sweep.py --steps 500 --n_list 9600 38400 --lmp_bin lmp_kokkos_89 --hwthread

    # Full sweep including large N:
    # python kokkos_sweep.py --steps 1000 --n_list 9600 38400 153600 --omp_list 1 4 8 --lmp_bin lmp_kokkos_89_dev --hwthread
