# MesoMem GPU

GPU-accelerated LAMMPS (Kokkos/CUDA) simulations of dipole-sphere membrane
models ("MesoMem"): planar membranes, vesicles, and vesicle + polymer +
solvent systems. See [`Mesomem_gpu.md`](Mesomem_gpu.md) for the physics/
performance best-practice guide and [`BENCHMARK_README.md`](BENCHMARK_README.md)
for the benchmark suite.

## What's essential for the GPU implementation

The GPU (Kokkos) build adds three custom LAMMPS components on top of stock
LAMMPS. All of their source lives in [`cpp_files/`](cpp_files/) — this is
the folder that matters if you're modifying the physics or porting it to a
new LAMMPS version:

| File(s) | What it is | LAMMPS package/location |
|---|---|---|
| `pair_membrane_sillano_v2.{cpp,h}` | CPU reference pair style (`pair_style membrane_sillanov2`) | `src/` (core) |
| `pair_membrane_sillano_v2_kokkos.{cpp,h}` | GPU pair style (`pair_style membrane_sillanov2/kk`) | `src/KOKKOS/` |
| `atom_vec_dipole_sphere_angle.{cpp,h}` | CPU atom style (`atom_style dipole_sphere_angle`) — merges charge, dipole, sphere radius/mass, bonds, angles into one contiguous layout | `src/DIPOLE/` |
| `atom_vec_dipole_sphere_angle_kokkos.{cpp,h}` | GPU atom style (`atom_style dipole_sphere_angle/kk`) | `src/KOKKOS/` |
| `fix_langevin_kokkos.cpp` | Patched Kokkos Langevin thermostat (modifies the stock LAMMPS file) | `src/KOKKOS/` |

Why a custom atom style: standard LAMMPS `hybrid` atom styles disable
`comm device` (on-GPU halo exchange) because the data layout isn't
guaranteed contiguous. `dipole_sphere_angle` stores everything in one
Kokkos view so the whole simulation — including communication — can stay
on the GPU. See [`Mesomem_gpu.md`](Mesomem_gpu.md) section 1 for details.

Required upstream LAMMPS packages: `KOKKOS` (GPU), `DIPOLE`, `MOLECULE`
(bond `fene`, angle `harmonic`/`cosine` used by the polymer/vesicle
systems), `EXTRA-PAIR` (`pair_style cosine/squared`, used for the LJ
solvent), `PYTHON` (Python API), plus MPI.

This repo does **not** ship a full LAMMPS source tree — the compile
scripts fetch a pinned upstream LAMMPS commit (git SHA hardcoded in the
script) and drop the files above into it, so the checked-in content stays
small and always builds against the exact LAMMPS version the MesoMem
KOKKOS code was written for.

**Why a pinned commit, not `lammps-stable`:** the KOKKOS atom-style API
(`AtomVecKokkos::sync`/`modified`/`sync_pinned`) changed after the last
stable LAMMPS release, so building against `lammps-stable.tar.gz` fails.
The scripts `git fetch --depth 1` a specific known-good `develop`-branch
commit instead.

## Building

Two standalone scripts, no arguments needed — each builds LAMMPS with
Kokkos/CUDA, MPI, and the Python API:

```bash
# Local machine (auto-detects your GPU architecture via nvidia-smi)
./compile_local.sh

# HPC cluster (module-based toolchain; run inside an interactive/batch
# GPU job, e.g. `srun --partition=gpu --gpus=1 --pty bash` on Snellius)
./compile_hpc.sh
```

`compile_hpc.sh` has the module names and GPU architecture (default:
`AMPERE80` for Snellius A100 nodes) set as plain variables near the top of
the file — edit them if you're building on a different cluster.

Each script is self-contained under `_build/local/` or `_build/hpc/`:

- `_build/<local|hpc>/lammps-src/` — fetched LAMMPS source (+ MesoMem files copied in)
- `_build/<local|hpc>/install/` — installed `lmp` binary + `liblammps.so`
- `_build/<local|hpc>/venv/` — Python virtualenv with the `lammps` Python module installed
- `_build/<local|hpc>/env.sh` — source this to put `lmp` and `liblammps.so` on your PATH/LD_LIBRARY_PATH

Both scripts finish with a smoke test that loads `atom_style
dipole_sphere_angle` and `pair_style membrane_sillanov2` through the
Python API and prints `OK: ...` on success.

To use a finished build later:

```bash
source _build/local/env.sh
source _build/local/venv/bin/activate
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half comm device -in your_script.lmp
```

## Repository layout

- `cpp_files/` — custom MesoMem C++ sources (see table above)
- `compile_local.sh`, `compile_hpc.sh` — build scripts (these are what you run)
- `Mesomem_gpu.md` — physics + Kokkos performance best-practice guide
- `BENCHMARK_README.md` — benchmark suite usage
- `planar_benchmark/`, `polymer/`, `solvent_benchmark/`, `polymer_solvent/`,
  `deserno_gpu/` — per-system input scripts and `benchmark.py` sweep drivers
- `diagnostics/` — small standalone correctness checks (angle/sphere/dsa styles)
- `plots/`, `plot_bench.py` — benchmark plotting
- `lammps/` — a local, already-patched LAMMPS checkout used for day-to-day
  development on this machine (not tracked in git — see `.gitignore`;
  reproduce it anywhere with `compile_local.sh`)
