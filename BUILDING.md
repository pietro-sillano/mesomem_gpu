# Building

Two standalone scripts — each builds LAMMPS with Kokkos (CUDA + OpenMP),
MPI, and the Python API against the custom MesoMem sources in [`cpp_files/`](cpp_files/).
See [`README.md`](README.md) for what those sources are and why a pinned
commit of the pietro-sillano/lammps fork is required.

```bash
# Local machine (auto-detects your GPU architecture via nvidia-smi)
./compile_local.sh

# HPC cluster (module-based toolchain; run inside an interactive/batch
# GPU job, e.g. on Snellius:
#   srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=16 --time=01:00:00 --pty bash
./compile_hpc.sh
```

`compile_hpc.sh` has the module names and GPU architecture (default:
`AMPERE80` for Snellius A100 nodes) set as plain variables near the top of
the file — edit them if you're building on a different cluster.

## Build settings

Both scripts have these plain variables near the top of the file
(`compile_local.sh` also accepts them from the environment, e.g.
`KOKKOS_PREC=mixed ./compile_local.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `KOKKOS_PREC` | `double` | Kokkos precision: `double`, `mixed` (float math, double accumulation) or `single` |
| `CUSTOM_ATOM_STYLE` | `no` | `yes` also builds the optional `dipole_sphere_angle(/kk)` atom style (only needed for Kokkos device sorting, which gives no speedup, see [`BENCHMARKS.md`](BENCHMARKS.md#host-vs-device-sorting)) |
| `JOBS` (local only) | `8` | parallel compile jobs |

Kokkos is always built with the CUDA backend (GPU) plus the OpenMP backend
(host threads, `-k on g 1 t N`). The default local build goes to
`_build/local/`; a non-default `KOKKOS_PREC` goes to its own folder
`_build/local-<prec>/` (e.g. `_build/local-mixed/`), so several builds can
coexist.

## Build + benchmark in one go

[`build_and_bench.sh`](build_and_bench.sh) runs `compile_local.sh` with the
default settings for the GPU of the machine it runs on, then the planar,
solvent, polymer and polymer_solvent benchmarks, writing
`benchmarks/*/results_<hostname>.csv`:

```bash
./build_and_bench.sh
```

If `_build/<local|hpc>/lammps-src` already exists from a previous run and
is at the pinned commit, re-running either script skips the LAMMPS fetch and just does an
incremental rebuild (the custom `cpp_files/` sources are always re-copied
first, so local edits are picked up). Pass `--force` to wipe it and fetch
+ build from scratch:

```bash
./compile_local.sh --force
./compile_hpc.sh --force
```

## What each script does

1. Checks prerequisites (`cmake`, `mpicc`, `python3`, and `nvcc` for GPU
   builds) and picks a Kokkos GPU architecture (auto-detected locally via
   `nvidia-smi`, hardcoded for HPC).
2. Fetches the pinned commit of the pietro-sillano/lammps fork (which
   contains `pair_style mesomem/dipole`) with a shallow `git fetch`;
   an existing source tree at a different commit is refetched. It also
   patches Kokkos' `nvcc_wrapper` in that tree: as shipped, it does not
   strip the quotes from CMake's `@objects1.rsp` response file and linking
   `liblammps.so` fails with "cannot specify '-o' with '-c' ... with
   multiple files".
3. Copies the Kokkos pair style from `cpp_files/` into `src/KOKKOS/` (plus
   the custom atom style into `src/DIPOLE/` and `src/KOKKOS/` if
   `CUSTOM_ATOM_STYLE=yes`).
4. Configures with CMake (`PKG_KOKKOS` + CUDA and OpenMP backends,
   `KOKKOS_PREC`, `PKG_DIPOLE`, `PKG_MOLECULE`, `PKG_EXTRA-PAIR`,
   `PKG_PYTHON`, MPI, OpenMP, shared libs) and builds + installs.
5. Creates a Python virtualenv and installs the LAMMPS Python bindings
   into it via `python/install.py`.
6. Writes `env.sh` (PATH and LD_LIBRARY_PATH, including the Python library
   folder, which `liblammps.so` needs when `python3` comes from a
   conda/micromamba environment) and runs a smoke test that loads
   `atom_style hybrid angle sphere dipole` and `pair_style mesomem/dipole`
   through the Python API, checks that `mesomem/dipole/kk` (and with
   `CUSTOM_ATOM_STYLE=yes` also `dipole_sphere_angle(/kk)`) was built, and
   prints `OK: ...` on success.

## Output layout

Each script is self-contained under `_build/local/` or `_build/hpc/`
(gitignored — nothing outside that folder is touched):

- `_build/<local|hpc>/lammps-src/` — fetched LAMMPS source (+ MesoMem files copied in)
- `_build/<local|hpc>/install/` — installed `lmp` binary + `liblammps.so`
- `_build/<local|hpc>/venv/` — Python virtualenv with the `lammps` Python module installed
- `_build/<local|hpc>/env.sh` — source this to put `lmp` and `liblammps.so` on your PATH/LD_LIBRARY_PATH

## Using a finished build

```bash
source _build/local/env.sh
source _build/local/venv/bin/activate
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half comm device -in your_script.lmp
```

or from Python:

```python
from lammps import lammps
lmp = lammps()
lmp.command('atom_style hybrid angle sphere dipole')
lmp.command('pair_style mesomem/dipole 2.5')
```

See [`BENCHMARKS.md`](BENCHMARKS.md) for the canonical GPU run command and
performance tuning notes once you have a build.

## Quick GPU smoke run on HPC

[`run_gpu_small.sh`](run_gpu_small.sh) runs a short polymer+solvent GPU job
against an existing `compile_hpc.sh` build — useful to sanity-check a GPU
allocation before launching a full benchmark sweep:

```bash
srun --partition=gpu_a100 --ntasks=1 --gpus-per-node=1 \
     --cpus-per-task=8 --time=00:05:00 --pty ./run_gpu_small.sh
```

It sources `_build/hpc/env.sh` (loading the same modules `compile_hpc.sh`
used) and runs `benchmarks/polymer_solvent/polymer_solvent.lmp` for a
small number of steps (`T_RUN`, default 200) on one GPU, writing to a
timestamped run directory under `benchmarks/polymer_solvent/bench_runs/`.
Requires `./compile_hpc.sh` to have been run first.

## Batch GPU job

[`slurm_job.sh`](slurm_job.sh) is a `sbatch`-submittable version of the
same run — 1 MPI task, 4 OMP threads, 1 GPU:

```bash
sbatch slurm_job.sh
```
