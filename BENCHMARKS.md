# Benchmarks

Performance benchmarks for the MesoMem membrane / polymer / solvent
systems using the Kokkos GPU backend. All benchmark material lives under
[`benchmarks/`](benchmarks/). See [`BUILDING.md`](BUILDING.md) to get a
`lmp` binary first.

## Systems

| Key | Folder | Model | N atoms |
|-----|--------|-------|---------|
| `planar` | `benchmarks/planar_benchmark/` | Planar dipole-sphere membrane | variable (400–1M) |
| `solvent` | `benchmarks/solvent_benchmark/` | Spherical vesicle + LJ solvent | variable (~2k–50k membrane) |
| `polymer` | `benchmarks/polymer/` | Spherical vesicle + ring-polymer | fixed (~99k) |
| `polymer_solvent` | `benchmarks/polymer_solvent/` | Spherical vesicle + ring-polymer + LJ solvent | fixed (~200k+) |
| `deserno` | `benchmarks/deserno_gpu/` | Cooke/Deserno lipid bilayer + torus, generated in-script (no data files) | variable |

Each system folder contains:
- `benchmark.py` — the benchmark driver
- `results_{machine}.csv` — output (created on first run)
- `bench_runs/` — per-run LAMMPS working directories (gitignored, created at runtime)

## Quick start

```bash
cd benchmarks/planar_benchmark

# GPU + CPU sweep, 3 replicas each, save to results_md69.csv
python benchmark.py --machine md69 --lmp_bin lmp \
    --test_gpu --omp_list 1 4 8 --mpi_list 1 4 8 12 \
    --steps_gpu 5000 --steps_cpu 500 --hwthread

# Generate all plots into benchmarks/plots/md69/
cd ..
python plot_bench.py --machine md69
```

`--lmp_bin` should point at (or resolve via PATH to) the `lmp` binary
produced by `compile_local.sh`/`compile_hpc.sh` — `source
_build/local/env.sh` first.

## `benchmark.py` — common interface

Every system exposes the same set of arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--machine` | *(required)* | Label written into the CSV filename (`results_{machine}.csv`) |
| `--lmp_bin` | `lmp_kokkos` | LAMMPS binary name/path |
| `--steps_gpu` | 1000 | MD steps for each GPU replica |
| `--steps_cpu` | 500 | MD steps for each CPU replica |
| `--omp_list` | `1 4 8` | OpenMP thread counts for the **GPU test** |
| `--mpi_list` | `1 4 8` | MPI rank counts for the **CPU test** |
| `--replicas` | 3 | Runs per configuration; CSV stores mean ± std |
| `--test_gpu` | off | Enable the GPU test |
| `--hwthread` | off | Pass `--use-hwthread-cpus` to mpirun |
| `--neigh` | `half` | Kokkos neighbour list (`half` or `full`); `newton` is derived automatically |

System-specific arguments (e.g. `--n_list`, `--solvent_density`,
`--data_file`) are described below.

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

## System-specific notes

### `planar_benchmark/`

Pre-generated lattice data files are required in `input/`:
```
input/lattice_d_0.85_N_{N}_ds      # for each N in --n_list
```
The script skips any N for which the file is missing.

```bash
python benchmark.py --machine md69 --lmp_bin lmp \
    --test_gpu --n_list 2500 10000 102400 501264 \
    --omp_list 1 4 8 --mpi_list 1 4 8 12 --hwthread
```

`processors * * 1` confines the domain decomposition to the xy-plane —
important for a 2D membrane in a 3D box (splitting along z would put most
atoms on one rank).

### `solvent_benchmark/`

`--n_list` specifies **membrane** atom counts. Total atom count (membrane
+ solvent) is computed automatically from the vesicle geometry and
`--solvent_density`.

Requires base vesicle files in `input/`:
```
input/vesicle_ds_N{N}_d0.85.data    # bare vesicle, one per N in --n_list
```

A CPU **prep stage** (add solvent + minimise) runs automatically the first
time and caches the result in `input/`. Re-run prep with `--force_prep`.
Ensure `comm device` is active — solvent dramatically increases halo atom
counts at the membrane–solvent interface.

```bash
python benchmark.py --machine md69 --lmp_bin lmp \
    --test_gpu --n_list 2420 5120 9680 20480 50000 \
    --omp_list 1 4 8 --mpi_list 1 4 8 --solvent_density 0.4 \
    --prep_mpi 8 --hwthread
```

### `polymer/`

Fixed system — no N sweep. Default data file: `input/combined_N35280_poly64000.data` (~99k atoms).
Ring polymers use `bond_style fene` + `angle_style harmonic`, both ported
to Kokkos — no extra flags needed beyond the standard GPU command.

```bash
python benchmark.py --machine md69 --lmp_bin lmp \
    --test_gpu --omp_list 1 4 8 --mpi_list 1 4 8 --hwthread
```

Vesicle/polymer data prep:
```bash
python spherical_vesicle_ds.py --path ./input --N 35000 --target_dist 0.8 --box_factor 1.2
python combine_vesicle_polymer.py --vesicle ./input/vesicle_ds_N35280_d0.80.data \
    --polymer input/poly_relaxed_64000.data --out ./input/combined_N35280_poly64000.data
```

### `polymer_solvent/`

The most expensive system — membrane + polymer + solvent. A CPU prep
stage runs automatically (same caching logic as `solvent_benchmark/`).
Default data file: `input/combined_N35280_poly64000.data`.

```bash
python benchmark.py --machine md69 --lmp_bin lmp \
    --test_gpu --omp_list 1 4 8 --mpi_list 1 4 8 \
    --solvent_density 0.4 --prep_mpi 8 --hwthread
```

### `deserno_gpu/`

Cooke/Deserno lipid bilayer benchmark: generates the bilayer directly in
LAMMPS (no pre-generated data files) and supports a Kokkos option sweep.

```bash
cd benchmarks/deserno_gpu/planar_benchmark

python new_bench.py --machine md69 --lmp_bin lmp \
    --test_gpu --n_list 600 2400 9600 38400 153600 614400 \
    --omp_list 1 4 8 --mpi_list 1 4 8 --hwthread

# Kokkos option sweep (neigh x comm x sort x OMP)
python kokkos_sweep.py --machine md69 --lmp_bin lmp \
    --n_list 9600 38400 --omp_list 1 4 8 --steps 1000 --hwthread
python plot_kokkos_sweep.py   # edit CSV filename at top of file
```

The `torus_benchmark/` folder is a standalone bent-membrane check:
```bash
mpirun -np 1 --bind-to core --use-hwthread-cpus lmp \
    -i torus_edges.lmp -v R 5.0 -v h 20.0 -v kt 18 \
    -k on g 1 t 4 -sf kk -pk kokkos newton off neigh full comm host
```

## `plot_bench.py` — unified plot script

Reads `results_{machine}.csv` from each system folder and writes plots to
`benchmarks/plots/{machine}/`.

```bash
cd benchmarks
python plot_bench.py --machine md69
```

## Running simulations — the canonical GPU command

```bash
lmp -in simulation.lmp \
  -k on g 1 t 1 \
  -sf kk \
  -pk kokkos newton on neigh half comm device
```

| Flag | Value | Reason |
|------|-------|--------|
| `-k on g 1` | 1 GPU | Use one GPU per MPI rank (standard for single-node) |
| `-k on … t 1` | OMP threads | See "OpenMP threads" below |
| `-sf kk` | Kokkos suffix | Automatically maps all styles to their `kk` variants |
| `newton on` | Newton's 3rd law | Reduces force evaluations by ~2x; works with `neigh half` |
| `neigh half` | Half neighbour list | Combined with `newton on` this is always the best option |
| `comm device` | On-device halo | Avoids PCIe round-trips; works with both `hybrid angle sphere dipole` and `dipole_sphere_angle` |

### Half vs full neighbour list

Always use `neigh half` + `newton on`. It evaluates each pair interaction
once and accumulates forces on both atoms, halving the number of pair
evaluations. `neigh full` + `newton off` computes each pair twice — it can
be faster on some GPU architectures where the extra bandwidth is cheaper
than the atomic update, but for our systems `half` wins consistently.

### Neighbour list rebuild frequency

The LAMMPS default (`delay 0 every 1 check yes`) rebuilds every step if
any atom has moved more than `skin/2`. For membrane simulations at
moderate temperature you can safely use:

```lammps
neighbor        1.0 bin
neigh_modify    delay 5 every 1 check yes
```

A skin of 1.0 (LJ units, on top of the pair cutoff) gives a comfortable
buffer for typical membrane dynamics.

> TODO: test the binsize and neighbour list settings further (not yet
> systematically swept).

### OpenMP threads

With `comm device`, nearly all work stays on the GPU and the CPU is mostly
idle. OMP threads are used for Kokkos host-side fallbacks (rarely hit) and
for thread-level parallelism in a handful of non-ported styles.

**Recommendation: use `t 1` (one OMP thread).** Multiple threads add
scheduling overhead without measurable benefit once `comm device` is
active. Set the environment variables regardless:

```bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

### MPI parallelism

MPI is useful for CPU-only runs (pure MPI scaling):

```bash
mpirun -np 8 --map-by socket:PE=1 --use-hwthread-cpus \
    lmp -in simulation.lmp -v t_run 500
```

`--map-by socket:PE=1` pins each MPI rank to one physical core and
prevents OMP threads from fighting over the same core.

### `CUDA_UVM=OFF`

Unified Virtual Memory lets the driver lazily migrate pages between host
and device — convenient for systems that don't fit in GPU memory, but
slow. `compile_local.sh`/`compile_hpc.sh` build with `Kokkos_ENABLE_CUDA_UVM=OFF`,
so Kokkos manages explicit device allocations for significantly better
memory throughput. Always use `UVM=OFF` for production runs.

## Kokkos sweep findings (Deserno benchmark)

From `benchmarks/deserno_gpu/planar_benchmark/`:

```
Best configurations per N:
  N=   9600  TPS=1539.1  neigh=full newton=off comm=device sort=device OMP=1
  N=  38400  TPS=706.4   neigh=half newton=on  comm=device sort=host   OMP=4
  N= 153600  TPS=283.4   neigh=half newton=on  comm=device sort=host   OMP=4
```

`comm=device` and `sort=device` matter the most — moving everything to
device is the biggest win. Device sorting requires the custom
`dipole_sphere_angle` atom style (`-var atomstyle dipole_sphere_angle`);
with the default `hybrid angle sphere dipole` style LAMMPS falls back to
host sorting. `neigh full` + `newton off` can win for smaller systems.

These sweeps were measured with the old `mesomem` pair style and the
custom atom style, before the switch to `mesomem/dipole`.

## Timing and profiling

### Accurate GPU timings

By default CUDA kernels are asynchronous — LAMMPS may report misleadingly
low times because the CPU timer fires before the GPU finishes. For an
accurate per-step breakdown:

```bash
export CUDA_LAUNCH_BLOCKING=1
```

Add to the input script:
```lammps
timer full sync
```
This forces a CPU-GPU sync at each timer boundary. Use only for profiling
(adds ~5-10% overhead), not production.

### Reading the LAMMPS timing output

| Section | What to look for |
|---------|-----------------|
| `Pair` | Should dominate; expected ~60-80% for membrane systems |
| `Comm` | With `comm device` this should be < 10%; if > 20% suspect a missing `comm device` |
| `Neigh` | Occasional spike is normal (every ~5 steps); sustained high value -> increase skin or `binsize` |
| `Other` | Includes I/O and thermo; keep dump frequency low during benchmarks |

## Visualizing with OVITO

Dump trajectories are automatically recognized. For the polymer/solvent
data files, pick the LAMMPS "hybrid angle sphere dipole" atom style when
prompted (same columns as the custom `dipole_sphere_angle` style, which
OVITO does not know).

## Kokkos resources

- LAMMPS Kokkos documentation: <https://docs.lammps.org/Speed_kokkos.html>
- Kokkos performance paper (OSTI): <https://www.osti.gov/servlets/purl/2588325>
