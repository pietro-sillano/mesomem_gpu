"""benchmark.py — Vesicle + ring-polymer + solvent benchmark.

GPU sweep (no mpirun):
    python benchmark.py --machine md69 --test_gpu --omp_list 1 4 8 \\
        --steps_gpu 1000 --lmp_bin lmp_kokkos_89_dev --hwthread

CPU MPI scaling:
    python benchmark.py --machine md69 --steps_cpu 500 \\
        --mpi_list 1 4 8 --hwthread
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
# Solvent geometry
# ---------------------------------------------------------------------------

def solvent_geometry(n_mem, density, info_path):
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"Vesicle info file not found: {info_path}\n"
            f"Run: python spherical_vesicle_ds.py --path input --N {n_mem}")
    meta = {}
    with open(info_path) as f:
        for line in f:
            k, v = line.strip().split()
            meta[k] = float(v)
    radius   = meta["radius"]
    box_half = meta["box_half"]

    V_box = (2 * box_half) ** 3
    V_in  = (4.0 / 3.0) * 3.141592653589793 * radius ** 3
    V_out = V_box - V_in
    N_in  = int(density * V_in)
    N_out = int(density * V_out)
    return radius, box_half, N_in, N_out


# ---------------------------------------------------------------------------
# Prep stage
# ---------------------------------------------------------------------------

def prepared_data_path(n_mem, n_pol, radius, density):
    return os.path.join("input",
                        f"combined_N{n_mem}_poly{n_pol}_r{radius:.1f}_d{density:.2f}.data")


def ensure_prepared_data(combined_file, n_mem, n_pol, radius, N_in, N_out,
                         density, lmp_bin, prep_mpi=6, force=False):
    out_file = prepared_data_path(n_mem, n_pol, radius, density)
    if os.path.exists(out_file) and not force:
        print(f"   [reuse] {out_file}")
        return out_file
    print(f"   [prepare] {out_file}  (mpirun -np {prep_mpi})")
    prep_dir = os.path.join("bench_runs", "_prep",
                            f"N{n_mem}_poly{n_pol}_d{density:.2f}")
    os.makedirs(prep_dir, exist_ok=True)
    cmd = [
        "mpirun", "-np", str(prep_mpi), "--map-by", "socket:PE=1",
        "--use-hwthread-cpus",
        lmp_bin,
        "-in", os.path.abspath("prepare_polymer_solvent.lmp"),
        "-v", "combined_file", os.path.abspath(combined_file),
        "-v", "N_in",          str(N_in),
        "-v", "N_out",         str(N_out),
        "-v", "radius",        f"{radius:.6f}",
        "-v", "out_file",      os.path.abspath(out_file),
    ]
    print(f"--> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=prep_dir, check=True, env=os.environ.copy())
    if not os.path.exists(out_file):
        raise RuntimeError(f"prepare_polymer_solvent.lmp did not produce {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Replica runner
# ---------------------------------------------------------------------------

def run_replicas(cmd, base_dir, env, n_replicas):
    tps_vals, wall_vals = [], []
    for rep in range(n_replicas):
        sim_dir = f"{base_dir}_rep{rep}"
        os.makedirs(sim_dir, exist_ok=True)
        print(f"    replica {rep+1}/{n_replicas}")
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

def run_gpu(args, n_total, data_file, omp, hw_info):
    cpu_model, gpu_info = hw_info
    newton   = "on" if args.neigh == "half" else "off"
    base_dir = os.path.join("bench_runs", f"N{n_total}_GPU_O{omp}")

    cmd  = [args.lmp_bin, "-in", os.path.abspath("polymer_solvent.lmp")]
    cmd += ["-v", "t_run", str(args.steps_gpu), "-v", "data_file", data_file]
    cmd += ["-k", "on", "g", "1", "t", str(omp), "-sf", "kk"]
    cmd += ["-pk", "kokkos", "newton", newton, "neigh", args.neigh, "comm", "device"]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    env["OMP_PROC_BIND"]   = "spread"
    env["OMP_PLACES"]      = "cores"

    print(f"--> GPU  N={n_total} OMP={omp}: {' '.join(cmd)}")
    rep = run_replicas(cmd, base_dir, env, args.replicas)
    if rep is None:
        return None
    return {"N": n_total, "Mode": "GPU", "MPI": 1, "OMP": omp, "Neigh": args.neigh,
            **rep, "CPU_Info": cpu_model, "GPU_Name": gpu_info["name"],
            "GPU_Cap": gpu_info["cap"], "LMP_Binary": os.path.basename(args.lmp_bin),
            "Machine": args.machine}


def run_cpu(args, n_total, data_file, mpi, hw_info):
    cpu_model, gpu_info = hw_info
    base_dir = os.path.join("bench_runs", f"N{n_total}_CPU_M{mpi}")

    cmd  = ["mpirun", "-np", str(mpi), "--map-by", "socket:PE=1"]
    if args.hwthread:
        cmd += ["--use-hwthread-cpus"]
    cmd += [args.lmp_bin, "-in", os.path.abspath("polymer_solvent.lmp")]
    cmd += ["-v", "t_run", str(args.steps_cpu), "-v", "data_file", data_file]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"]   = "spread"
    env["OMP_PLACES"]      = "cores"

    print(f"--> CPU  N={n_total} MPI={mpi}: {' '.join(cmd)}")
    rep = run_replicas(cmd, base_dir, env, args.replicas)
    if rep is None:
        return None
    return {"N": n_total, "Mode": "CPU", "MPI": mpi, "OMP": 1, "Neigh": "none",
            **rep, "CPU_Info": cpu_model, "GPU_Name": gpu_info["name"],
            "GPU_Cap": gpu_info["cap"], "LMP_Binary": os.path.basename(args.lmp_bin),
            "Machine": args.machine}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine",         required=True)
    parser.add_argument("--lmp_bin",         default="lmp_kokkos")
    parser.add_argument("--steps_gpu",       type=int, default=1000)
    parser.add_argument("--steps_cpu",       type=int, default=500)
    parser.add_argument("--omp_list",        type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--mpi_list",        type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--data_file",       default="input/combined_N35280_poly64000.data")
    parser.add_argument("--solvent_density", type=float, default=0.1)
    parser.add_argument("--prep_mpi",        type=int, default=6)
    parser.add_argument("--force_prep",      action="store_true")
    parser.add_argument("--replicas",        type=int, default=3)
    parser.add_argument("--test_gpu",        action="store_true")
    parser.add_argument("--hwthread",        action="store_true")
    parser.add_argument("--neigh",           default="half")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    combined_file = args.data_file
    if not os.path.exists(combined_file):
        raise FileNotFoundError(f"Combined data file not found: {combined_file}")

    m = re.search(r"combined_N(\d+)_poly(\d+)", combined_file)
    if not m:
        raise ValueError(f"Cannot parse N_mem/N_pol from: {combined_file}")
    n_mem = int(m.group(1))
    n_pol = int(m.group(2))

    info_path = os.path.join("input", f"vesicle_ds_N{n_mem}_d0.80.info")
    radius, box_half, N_in, N_out = solvent_geometry(n_mem, args.solvent_density, info_path)
    n_total = n_mem + n_pol + N_in + N_out
    print(f"\nVesicle r={radius:.2f}, box=±{box_half:.2f}, "
          f"N_in={N_in}, N_out={N_out}")
    print(f"Total: {n_mem} membrane + {n_pol} polymer + {N_in+N_out} solvent = {n_total}")

    data_file = ensure_prepared_data(
        combined_file, n_mem, n_pol, radius, N_in, N_out,
        args.solvent_density, args.lmp_bin,
        prep_mpi=args.prep_mpi, force=args.force_prep)

    hw_info = get_hw_info()
    results = []

    if args.test_gpu:
        for omp in args.omp_list:
            row = run_gpu(args, n_total, data_file, omp, hw_info)
            if row:
                results.append(row)

    for mpi in args.mpi_list:
        row = run_cpu(args, n_total, data_file, mpi, hw_info)
        if row:
            results.append(row)

    df    = pd.DataFrame(results)
    fname = f"results_{args.machine}.csv"
    df.to_csv(fname, index=False)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
