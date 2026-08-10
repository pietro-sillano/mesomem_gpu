# Building

Two standalone scripts, no arguments needed — each builds LAMMPS with
Kokkos/CUDA, MPI, and the Python API against the custom MesoMem sources in
[`cpp_files/`](cpp_files/). See [`README.md`](README.md) for what those
sources are and why a pinned LAMMPS commit is required.

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

## What each script does

1. Checks prerequisites (`cmake`, `mpicc`, `python3`, `nvcc`) and picks a
   Kokkos GPU architecture (auto-detected locally via `nvidia-smi`,
   hardcoded for HPC).
2. Fetches the pinned LAMMPS commit with a shallow `git fetch`.
3. Copies the custom MesoMem source files from `cpp_files/` into the right
   package subfolders (`src/`, `src/DIPOLE/`, `src/KOKKOS/`).
4. Configures with CMake (`PKG_KOKKOS` + CUDA, `PKG_DIPOLE`,
   `PKG_MOLECULE`, `PKG_EXTRA-PAIR`, `PKG_PYTHON`, MPI, shared libs) and
   builds + installs.
5. Creates a Python virtualenv and installs the LAMMPS Python bindings
   into it via `python/install.py`.
6. Runs a smoke test that loads `atom_style mesomem` and
   `pair_style mesomem` through the Python API and prints
   `OK: ...` on success.

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
lmp.command('atom_style mesomem')
lmp.command('pair_style mesomem 2.5')
```

See [`BENCHMARKS.md`](BENCHMARKS.md) for the canonical GPU run command and
performance tuning notes once you have a build.
