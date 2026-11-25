# MesoMem GPU

GPU-accelerated LAMMPS (Kokkos/CUDA) simulations of dipole-sphere membrane
models ("MesoMem"): planar membranes, vesicles, and vesicle + polymer +
solvent systems.

- [`BUILDING.md`](BUILDING.md) — how to compile LAMMPS with this package (local / HPC)
- [`BENCHMARKS.md`](BENCHMARKS.md) — the benchmark suite and GPU performance tuning
- this file — what MesoMem is, and exactly what it changes in LAMMPS

## What MesoMem adds to LAMMPS

The membrane model needs a pair style and a custom atom style, each with
a CPU and a GPU (Kokkos) implementation.

**Pair style — upstream.** The MesoMem potential is part of upstream
LAMMPS as `pair_style mesomem/dipole` (DIPOLE package), with official
KOKKOS (`mesomem/dipole/kk`) and OPENMP (`mesomem/dipole/omp`) variants,
starting with the `patch_13Nov2026` release (docs:
`doc/src/pair_mesomem_dipole.rst`). This repo no longer carries its own
copy. Syntax (no global cutoff; one cutoff per type pair):

```
pair_style mesomem/dipole
pair_coeff 1 1 sigma eps k_tilt k_splay r_c w_c zeta c0
```

The old local `pair_mesomem` (removed, see tag `pre-mesomem-dipole`) had
the opposite sign of the splay torque, rounded zeta to an integer on the
GPU, and had no guard for zero dipoles; upstream fixes all three.

**Atom style — custom.** Its source lives in [`cpp_files/`](cpp_files/):

| File(s) | What it is | LAMMPS package/location |
|---|---|---|
| `atom_vec_dipole_sphere_angle.{cpp,h}` | CPU atom style (`atom_style dipole_sphere_angle`) — merges charge, dipole, sphere radius/mass, bonds, angles into one contiguous layout | `src/DIPOLE/` |
| `atom_vec_dipole_sphere_angle_kokkos.{cpp,h}` | GPU atom style (`atom_style dipole_sphere_angle/kk`) | `src/KOKKOS/` |

The Kokkos Langevin `omega yes` fix that used to be patched in here
(`omega_thermostat_kokkos()`) is also upstream now.

`compile_local.sh`/`compile_hpc.sh` (see [`BUILDING.md`](BUILDING.md))
fetch a vanilla LAMMPS source tree and copy the atom style files into the
package subfolders above, following LAMMPS's own layout convention so CMake's
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
`git fetch --depth 1` a pinned upstream LAMMPS commit (SHA hardcoded in
the scripts): the **`patch_13Nov2026`** feature release
(`68b557e3a2`). It is the first release containing `mesomem/dipole` and
its KOKKOS port; `stable_30Sep2026` does not have it. The custom KOKKOS
atom style also tracks the `AtomVecKokkos` API, which changes between
releases, so it is only guaranteed to compile against the pinned commit.

## Repository layout

- `cpp_files/` — custom atom style C++ sources (see table above)
- `compile_local.sh`, `compile_hpc.sh` — build scripts, see [`BUILDING.md`](BUILDING.md).
  Local builds go to `_build/local_sm<XY>/`, one per GPU type (md6*: `sm89`,
  md4*: `sm61`); `source env_local.sh` loads the right one on any workstation
- `benchmarks/` — per-system input scripts, `benchmark.py` sweep drivers, and plots; see [`BENCHMARKS.md`](BENCHMARKS.md)
- `diagnostics/` — small standalone correctness checks (angle/sphere/dsa atom and pair styles, CPU vs GPU).
  `run_mesomem_cpu_gpu.sh` on patch_13Nov2026 (RTX 4090): `mesomem/dipole`
  vs `mesomem/dipole/kk` per-atom forces agree to 3e-14 and torques to 7e-14
  (relative to the largest value)
- `lammps/` — a local, already-patched LAMMPS checkout used for day-to-day
  development on this machine (not tracked in git — see `.gitignore`;
  reproduce it anywhere with `compile_local.sh`)
