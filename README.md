# MesoMem GPU

GPU-accelerated LAMMPS (Kokkos/CUDA) simulations of dipole-sphere membrane
models ("MesoMem"): planar membranes, vesicles, and vesicle + polymer +
solvent systems.

- [`BUILDING.md`](BUILDING.md) — how to compile LAMMPS with this package (local / HPC)
- [`BENCHMARKS.md`](BENCHMARKS.md) — the benchmark suite and GPU performance tuning
- this file — what MesoMem is, and exactly what it changes in LAMMPS

## What MesoMem adds to LAMMPS

The membrane model needs one custom atom style and one custom pair style,
each with a CPU and a GPU (Kokkos) implementation. All of their source
lives in [`cpp_files/`](cpp_files/) — this is the folder that matters if
you're modifying the physics or porting it to a new LAMMPS version:

| File(s) | What it is | LAMMPS package/location |
|---|---|---|
| `pair_mesomem.{cpp,h}` | CPU reference pair style (`pair_style mesomem`) | `src/` (core) |
| `pair_mesomem_kokkos.{cpp,h}` | GPU pair style (`pair_style mesomem/kk`) | `src/KOKKOS/` |
| `atom_vec_dipole_sphere_angle.{cpp,h}` | CPU atom style (`atom_style dipole_sphere_angle`) — merges charge, dipole, sphere radius/mass, bonds, angles into one contiguous layout | `src/DIPOLE/` |
| `atom_vec_dipole_sphere_angle_kokkos.{cpp,h}` | GPU atom style (`atom_style dipole_sphere_angle/kk`) | `src/KOKKOS/` |
| `fix_langevin_kokkos.cpp` | Patched Kokkos Langevin thermostat (overwrites the stock LAMMPS file) | `src/KOKKOS/` |

`compile_local.sh`/`compile_hpc.sh` (see [`BUILDING.md`](BUILDING.md))
fetch a vanilla LAMMPS source tree and copy these files into the package
subfolders above, following LAMMPS's own layout convention so CMake's
per-package glob picks them up automatically.

### Why a custom atom style

Standard LAMMPS `hybrid` atom styles disable `comm device` (on-GPU halo
exchange) because the data layout isn't guaranteed contiguous across the
combined properties. `dipole_sphere_angle` stores charge, dipole, sphere
radius/mass, bonds, and angles in one contiguous Kokkos view, enabling:

- `comm device` (halo exchange entirely on GPU)
- Kokkos device-side neighbour sorting
- no forced host<->device transfers between force and communication steps

#### Data file column order

The `Atoms` section for `dipole_sphere_angle` uses a layout different from
plain `dipole_sphere`:

```
id  type  x  y  z  molecule  diameter  density  q  mux  muy  muz
```

`data_atom_post` converts `diameter -> radius` and `(density, diameter) ->
rmass` at read time, so the data file itself stores physical diameter and
density.

### Required upstream LAMMPS packages

`KOKKOS` (GPU), `DIPOLE`, `MOLECULE` (bond `fene`, angle
`harmonic`/`cosine`, used by the polymer/vesicle systems), `EXTRA-PAIR`
(`pair_style cosine/squared`, used for the LJ solvent), `PYTHON` (Python
API), plus MPI.

### Why a pinned LAMMPS commit, not `lammps-stable`

This repo does **not** ship a full LAMMPS source tree — the build scripts
fetch a pinned upstream LAMMPS commit (git SHA hardcoded in the scripts)
instead of the `lammps-stable` release. The Kokkos atom-style API
(`AtomVecKokkos::sync`/`modified`/`sync_pinned`) changed after the last
stable LAMMPS release, so building `atom_vec_dipole_sphere_angle_kokkos`
against `lammps-stable.tar.gz` fails to compile. The scripts `git fetch
--depth 1` a specific known-good `develop`-branch commit instead, so the
checked-in content stays small while always building against the exact
LAMMPS version this KOKKOS code was written for.

## Repository layout

- `cpp_files/` — custom MesoMem C++ sources (see table above)
- `compile_local.sh`, `compile_hpc.sh` — build scripts, see [`BUILDING.md`](BUILDING.md)
- `benchmarks/` — per-system input scripts, `benchmark.py` sweep drivers, and plots; see [`BENCHMARKS.md`](BENCHMARKS.md)
- `diagnostics/` — small standalone correctness checks (angle/sphere/dsa atom and pair styles, CPU vs GPU)
- `lammps/` — a local, already-patched LAMMPS checkout used for day-to-day
  development on this machine (not tracked in git — see `.gitignore`;
  reproduce it anywhere with `compile_local.sh`)
