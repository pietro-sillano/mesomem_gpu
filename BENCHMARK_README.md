# Benchmark Suite

Performance benchmarks for LAMMPS membrane / polymer / solvent simulations using the Kokkos GPU backend.

---

## Systems

| Key | Folder | Model | N atoms |
|-----|--------|-------|---------|
| `planar` | `planar_benchmark_89/` | Planar dipole-sphere membrane | variable (400–1M) |
| `solvent` | `solvent_benchmark_89_uvm_off/` | Spherical vesicle + LJ solvent | variable (~10k–250k) |
| `polymer` | `polymer/` | Spherical vesicle + ring-polymer | fixed (~99k) |
| `polymer_solvent` | `polymer_solvent/` | Spherical vesicle + ring-polymer + LJ solvent | fixed (~200k+) |

Each system folder contains:
- `benchmark.py` — the benchmark driver
- `results_{machine}.csv` — output (created on first run)
- `bench_runs/` — per-run LAMMPS working directories (created at runtime)

---

## Quick start

```bash
# GPU + CPU sweep, 3 replicas each, save to results_md69.csv
cd planar_benchmark_89
python benchmark.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --test_gpu --omp_list 1 4 8 --mpi_list 1 4 8 12 \
    --steps_gpu 5000 --steps_cpu 500 --hwthread

# Generate all plots into plots/md69/
cd ..
python plot_bench.py --machine md69
```

---

## `benchmark.py` — common interface

Every system exposes the same set of arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--machine` | *(required)* | Label written into the CSV filename (`results_{machine}.csv`) |
| `--lmp_bin` | `lmp_kokkos` | LAMMPS binary. Locally use `lmp_kokkos_89_dev` or `lmp_kokkos_61` |
| `--steps_gpu` | 1000 | MD steps for each GPU replica |
| `--steps_cpu` | 500 | MD steps for each CPU replica |
| `--omp_list` | `1 4 8` | OpenMP thread counts for the **GPU test** |
| `--mpi_list` | `1 4 8` | MPI rank counts for the **CPU test** |
| `--replicas` | 3 | Runs per configuration; CSV stores mean ± std |
| `--test_gpu` | off | Enable the GPU test |
| `--hwthread` | off | Pass `--use-hwthread-cpus` to mpirun |
| `--neigh` | `half` | Kokkos neighbour list (`half` or `full`); `newton` is derived automatically |

System-specific arguments (e.g. `--n_list`, `--solvent_density`, `--data_file`) are described below.

### Two test modes

**GPU test** (`--test_gpu`): runs the binary directly — no `mpirun`, one GPU, OMP thread sweep.
```
lmp_bin -in input.lmp -v t_run N ... -k on g 1 t {omp} -sf kk -pk kokkos ... comm device
```

**CPU test** (always runs — it is the speedup baseline): pure MPI, OMP=1.
```
mpirun -np {mpi} --map-by socket:PE=1 lmp_bin -in input.lmp -v t_run N ...
```

OMP environment variables set for every run:
```
OMP_NUM_THREADS={omp}
OMP_PROC_BIND=spread
OMP_PLACES=cores
```

### CSV schema

```
N, Mode, MPI, OMP, Neigh,
TPS_mean, TPS_std, Wall_mean, Wall_std,
GPU_Name, GPU_Cap, CPU_Info, LMP_Binary, Machine
```

---

## System-specific notes

### `planar_benchmark_89/`

Pre-generated lattice data files are required in `input/`:

```
input/lattice_d_0.85_N_{N}_ds      # for each N in --n_list
```

The script skips any N for which the file is missing.

```bash
python benchmark.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --test_gpu --n_list 2500 10000 102400 501264 \
    --omp_list 1 4 8 --mpi_list 1 4 8 12 --hwthread
```

### `solvent_benchmark_89_uvm_off/`

`--n_list` specifies **membrane** atom counts. Total atom count (membrane + solvent) is computed automatically from the vesicle geometry and `--solvent_density`.

Requires base vesicle files in `input/`:
```
input/vesicle_ds_N{N}_d0.85.data    # bare vesicle, one per N in --n_list
```

A CPU **prep stage** (add solvent + minimise) runs automatically the first time and caches the result in `input/`. Re-run prep with `--force_prep`.

```bash
python benchmark.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --test_gpu --n_list 2420 5120 9680 20480 50000 \
    --omp_list 1 4 8 --mpi_list 1 4 8 --solvent_density 0.4 \
    --prep_mpi 8 --hwthread
```

### `polymer/`

Fixed system — no N sweep. Default data file: `input/combined_N35280_poly64000.data` (~99k atoms).

```bash
python benchmark.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --test_gpu --omp_list 1 4 8 --mpi_list 1 4 8 --hwthread
```

### `polymer_solvent/`

Fixed system with solvent. A CPU prep stage runs automatically (same caching logic as the solvent system). Default data file: `input/combined_N35280_poly64000.data`.

```bash
python benchmark.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --test_gpu --omp_list 1 4 8 --mpi_list 1 4 8 \
    --solvent_density 0.4 --prep_mpi 8 --hwthread
```

---

## `plot_bench.py` — unified plot script

Reads `results_{machine}.csv` from each system folder and writes plots to `plots/{machine}/`.

```bash
python plot_bench.py --machine md69
```



## LAMMPS compilation

```
cmake ../cmake -D BUILD_MPI=yes -D BUILD_OMP=yes -D PKG_KOKKOS=yes  -D Kokkos_ARCH_ADA89=yes -D Kokkos_ENABLE_CUDA=yes -D Kokkos_ENABLE_OPENMP=yes -D Kokkos_ENABLE_SERIAL=yes -D PKG_MOLECULE=yes -D PKG_DIPOLE=yes -D LAMMPS_MACHINE=kokkos_89_dev -D PKG_EXTRA-PAIR=yes -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 -D Kokkos_ENABLE_CUDA_UVM=OFF
make -j16
make install
```


