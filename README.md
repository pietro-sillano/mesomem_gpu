# MesoMem GPU

GPU-accelerated LAMMPS (Kokkos/CUDA) simulations of dipole-sphere membrane
models ("MesoMem"): planar membranes, vesicles, and vesicle + polymer +
solvent systems.

- [`BUILDING.md`](BUILDING.md) — how to compile LAMMPS with this package (local / HPC)
- [`BENCHMARKS.md`](BENCHMARKS.md) — the benchmark suite and GPU performance tuning
- this file — what MesoMem is, and exactly what it changes in LAMMPS

## What MesoMem adds to LAMMPS

The CPU pair style `pair_style mesomem/dipole` (DIPOLE package,
`src/DIPOLE/pair_mesomem_dipole.{cpp,h}`, documented in
`doc/src/pair_mesomem_dipole.rst`) is being contributed to LAMMPS itself and
lives in the [pietro-sillano/lammps](https://github.com/pietro-sillano/lammps)
fork, which the build scripts check out. This repository adds the GPU
(Kokkos) version of that pair style plus an optional custom atom style. All
of their source lives in [`cpp_files/`](cpp_files/) -- this is the folder
that matters if you're modifying the physics or porting it to a new LAMMPS
version:

| File(s) | What it is | LAMMPS package/location |
|---|---|---|
| `pair_mesomem_dipole_kokkos.{cpp,h}` | GPU pair style (`pair_style mesomem/dipole/kk`), a Kokkos port of `pair_mesomem_dipole.cpp` with identical physics | `src/KOKKOS/` |
| `atom_vec_dipole_sphere_angle.{cpp,h}` | Optional CPU atom style (`atom_style dipole_sphere_angle`) -- merges charge, dipole, sphere radius/mass, bonds, angles into one atom style | `src/DIPOLE/` |
| `atom_vec_dipole_sphere_angle_kokkos.{cpp,h}` | Optional GPU atom style (`atom_style dipole_sphere_angle/kk`) | `src/KOKKOS/` |

`compile_local.sh`/`compile_hpc.sh` (see [`BUILDING.md`](BUILDING.md))
fetch the pinned LAMMPS fork and copy these files into the package
subfolders above, following LAMMPS's own layout convention so CMake's
per-package glob picks them up automatically.

Any change to the physics must be made in both `pair_mesomem_dipole.cpp`
(fork) and `pair_mesomem_dipole_kokkos.cpp` (here); the Kokkos kernel
follows the CPU `compute()` step by step (same section labels A-G).

### Atom style: hybrid (default) or custom

The pair style needs per-atom dipole (orientation), torque, and finite-size
sphere data; the polymer/vesicle systems also need bonds and angles. Two
atom styles provide this, with the **same** `Atoms` column layout, so every
data file works with both:

- `atom_style hybrid angle sphere dipole` -- **default** in all input
  scripts. Recent LAMMPS Kokkos handles hybrid atom styles on the GPU,
  including `comm device` (halo exchange entirely on the GPU). Only Kokkos
  atom *sorting* is not yet supported on the device for hybrid styles:
  LAMMPS prints a warning and sorts on the host instead.
- `atom_style dipole_sphere_angle` -- custom single (non-hybrid) atom style
  from `cpp_files/`. It additionally enables Kokkos device-side sorting.
  Select it in any benchmark script with `-var atomstyle dipole_sphere_angle`.

Earlier LAMMPS versions also forced `comm host` for hybrid atom styles,
which is why the custom style was originally required; that is no longer
the case, and the stock Kokkos `fix langevin` now thermostats rotations
(`omega yes`) on the GPU, so the patched `fix_langevin_kokkos.cpp` that
used to live here has been removed.

#### Data file column order

The `Atoms` section of both `hybrid angle sphere dipole` (in exactly this
sub-style order) and `dipole_sphere_angle` is:

```
id  type  x  y  z  molecule  diameter  density  q  mux  muy  muz
```

For both atom styles the Atoms section reads the physical diameter and
density; LAMMPS converts them to radius and per-atom mass at read time.

### Required upstream LAMMPS packages

`KOKKOS` (GPU), `DIPOLE`, `MOLECULE` (bond `fene`, angle
`harmonic`/`cosine`, used by the polymer/vesicle systems), `EXTRA-PAIR`
(`pair_style cosine/squared`, used for the LJ solvent), `PYTHON` (Python
API), plus MPI.

### Why a pinned LAMMPS commit, not `lammps-stable`

This repo does **not** ship a full LAMMPS source tree -- the build scripts
`git fetch --depth 1` a pinned commit (git SHA hardcoded in the scripts) of
the [pietro-sillano/lammps](https://github.com/pietro-sillano/lammps) fork's
`develop` branch. That commit contains the reviewed `pair_style
mesomem/dipole` and the current Kokkos API the files in `cpp_files/` are
written against (the `lammps-stable` release predates both). Once
`mesomem/dipole` is merged into official LAMMPS, the pin can move to an
upstream `lammps/lammps` commit.

### Changes relative to the old `pair_style mesomem`

`mesomem/dipole` is the renamed, reviewed version of the former `mesomem`
style; `pair_coeff` takes the same 8 coefficients. Compared with the old
`pair_mesomem*.cpp` files it fixes the sign of the radial force coming
from the C0 term of the tilt energy and the sign of the splay torque (both
now equal the exact derivatives of the energy), skips the tilt/splay terms
for particles with a zero dipole, and accepts non-integer `zeta` on the GPU
too. The benchmark CSVs and plots in `benchmarks/` were produced with the
old `mesomem` style.

## Repository layout

- `cpp_files/` — custom MesoMem C++ sources (see table above)
- `compile_local.sh`, `compile_hpc.sh` — build scripts, see [`BUILDING.md`](BUILDING.md)
- `benchmarks/` — per-system input scripts, `benchmark.py` sweep drivers, and plots; see [`BENCHMARKS.md`](BENCHMARKS.md)
- `diagnostics/` — small standalone correctness checks (angle/sphere/dsa atom and pair styles, CPU vs GPU)
- `lammps/` — a local, already-patched LAMMPS checkout used for day-to-day
  development on this machine (not tracked in git — see `.gitignore`;
  reproduce it anywhere with `compile_local.sh`)
