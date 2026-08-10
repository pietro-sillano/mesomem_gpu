
# MesoMem GPU — Best-Practice Guide

## TODO
- [ ] test the binsize and neighbour lists settings

---

## Benchmark systems

We consider the following systems in our benchmark suite:

| System | Description |
|--------|-------------|
| Planar membrane | Flat dipole-sphere bilayer, periodic |
| Vesicle | Closed spherical membrane |
| Vesicle + solvent | Vesicle surrounded by explicit LJ solvent |
| Vesicle + polymer | Vesicle with encapsulated ring-polymer chains |
| Vesicle + polymer + solvent | Full system: membrane + polymer + solvent |

---

## 1. Custom atom style — why it matters

Standard LAMMPS offers `atom_style sphere`, `dipole`, and `hybrid` combinations. Hybrid styles disable several Kokkos optimisations (notably `comm device`) because data layout cannot be guaranteed contiguous on the device.

For all mesomem systems we use a **custom monolithic atom style** that merges the three required properties:

```
atom_style dipole_sphere_angle
```

This style stores charge `q`, dipole vector `mu`, sphere radius, mass, bonds and angles in a single contiguous Kokkos view, enabling:
- `comm device` (halo exchange entirely on GPU)
- Kokkos device-side neighbour sorting
- No forced host↔device transfers between force and communication steps

> **You need to compile LAMMPS with the custom `atom_vec_dipole_sphere_angle` files.** See Section 2 for build instructions.

### Data file column order

The Atoms section for `dipole_sphere_angle` uses a **different layout** than plain `dipole_sphere`:

```
id  type  x  y  z  molecule  diameter  density  q  mux  muy  muz
```

`data_atom_post` converts `diameter → radius` and `(density, diameter) → rmass` at read time, so you write physical diameter and density in the file.

---

## 2. Building LAMMPS

A full build guide is in [`installation.md`](installation.md). The essential CMake flags are:

```bash
cmake ../cmake \
  -D PKG_KOKKOS=yes \
  -D Kokkos_ENABLE_CUDA=yes \
  -D Kokkos_ARCH_AMPERE89=yes \      # match your GPU: VOLTA70, TURING75, AMPERE80, …
  -D CMAKE_CXX_COMPILER=nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA_UVM=OFF \    # see note below
  -D PKG_DIPOLE=yes \
  -D PKG_MOLECULE=yes \
  -D BUILD_MPI=yes \
  -D BUILD_OMP=yes
```

**`CUDA_UVM=OFF` is important.** Unified Virtual Memory lets the driver lazily migrate pages between host and device, which is convenient for bigger systems that dont fit onto the GPU memory but slow. With UVM off, Kokkos manages explicit device allocations and achieves significantly better memory throughput. Always use `UVM=OFF` for production runs.

---

## 3. Running on GPU — the canonical command

```bash
lmp_kokkos \
  -in simulation.lmp \
  -k on g 1 t 1 \
  -sf kk \
  -pk kokkos newton on neigh half comm device
```

| Flag | Value | Reason |
|------|-------|--------|
| `-k on g 1` | 1 GPU | Use one GPU per MPI rank (standard for single-node) |
| `-k on … t 1` | OMP threads | See Section 4 |
| `-sf kk` | Kokkos suffix | Automatically maps all styles to their `kk` variants |
| `newton on` | Newton's 3rd law | Reduces force evaluations by ~2×; works with `neigh half` |
| `neigh half` | Half neighbour list | Combined with `newton on` this is always the best option |
| `comm device` | On-device halo | Avoids PCIe round-trips; requires monolithic atom style (no hybrid) |

---

## 4. OpenMP threads

With `comm device`, nearly all work stays on the GPU and the CPU is mostly idle. OMP threads are used for Kokkos host-side fallbacks (rarely hit) and for thread-level parallelism in a handful of non-ported styles.

**Recommendation: use `t 1` (one OMP thread).**

Multiple threads add scheduling overhead without measurable benefit once `comm device` is active. Set the environment variables regardless:

```bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

---

## 5. Neighbour list tuning

### Half vs full list

Always use `neigh half` + `newton on`. It evaluates each pair interaction once and accumulates forces on both atoms, halving the number of pair evaluations. `neigh full` + `newton off` computes each pair twice — it can be faster on some GPU architectures where the extra bandwidth is cheaper than the atomic update, but for our systems `half` wins consistently.

### Neighbour list rebuild frequency

The LAMMPS default (`delay 0 every 1 check yes`) rebuilds every step if any atom has moved more than `skin/2`. For membrane simulations at moderate temperature you can safely use:

```lammps
neighbor        1.0 bin
neigh_modify    delay 5 every 1 check yes
```

A skin of 1.0 (in LJ units, on top of the pair cutoff) gives a comfortable buffer for typical membrane dynamics.

---

## 6. MPI parallelism 

MPI is useful for CPU-only runs (pure MPI scaling). For CPU benchmarks:

```bash
mpirun -np 8 --map-by socket:PE=1 --use-hwthread-cpus \
    lmp_kokkos -in simulation.lmp -v t_run 500
```

`--map-by socket:PE=1` pins each MPI rank to one physical core and prevents OMP threads from fighting over the same core.

---

## 7. Per-system recommendations

### Planar membrane

```lammps
atom_style      dipole_sphere_angle
neighbor        1.0 bin
neigh_modify    delay 5 every 1 check yes binsize 5.0
processors      * * 1          # confine decomposition to xy plane
boundary        p p p
```

- `processors * * 1`: forces the domain decomposition to stay in-plane. In a 3D box with a 2D membrane, splitting along z would put most atoms in one rank.
- Large N (> 500k): the lattice generator (`planar_lattice_ds.py`) can produce the required data file; runtime is dominated by the pair style.

### Vesicle (closed membrane)

```lammps
atom_style      dipole_sphere_angle
neighbor        1.0 bin
neigh_modify    delay 5 every 1 check yes
```

No special processor decomposition needed. The default 3D decomposition works well.

### Vesicle + solvent

The LJ solvent greatly increases the number of atoms and the fraction of time spent in the neighbour list. Ensure `comm device` is active — solvent dramatically increases halo atom counts.

A CPU *prep stage* (add solvent particles + short minimisation) is run once and cached. The benchmark scripts handle this automatically. See `solvent_benchmark/benchmark.py`.

### Vesicle + polymer

Ring polymers use `bond_style fene` and `angle_style harmonic`, both ported to Kokkos. No extra flags required beyond the standard GPU command. The combined data file (`combined_N*_poly*.data`) must be prepared separately.

### Vesicle + polymer + solvent

The most expensive system. The prep stage runs automatically via `polymer_solvent/benchmark.py`. The same Kokkos flags apply; `comm device` is especially important here given the large halo at the membrane–solvent interface.

---

## 8. Timing and profiling

### Accurate GPU timings

By default CUDA kernels are asynchronous — LAMMPS may report misleadingly low times because the CPU timer fires before the GPU finishes. For accurate per-step breakdown:

```bash
export CUDA_LAUNCH_BLOCKING=1
```

Add to the input script:

```lammps
timer full sync
```

This forces a CPU–GPU synchronisation at each timer boundary. Use only for profiling, not production (adds ~5–10% overhead).

### Reading the LAMMPS timing output

Focus on these sections of the final timing table:

| Section | What to look for |
|---------|-----------------|
| `Pair` | Should dominate; expected ~60–80% for membrane systems |
| `Comm` | With `comm device` this should be < 10%; if > 20% suspect a hybrid style or missing `comm device` |
| `Neigh` | Occasional spike is normal (every ~5 steps); sustained high value → increase skin or `binsize` |
| `Other` | Includes I/O and thermo; keep dump frequency low during benchmarks |

### Benchmark scripts

Each system folder contains a `benchmark.py` that sweeps N, OMP, and MPI counts, stores results in `results_{machine}.csv`, and supports replica averaging. See [`BENCHMARK_README.md`](BENCHMARK_README.md) for usage.

---

## 9. Kokkos resources

- LAMMPS Kokkos documentation: <https://docs.lammps.org/Speed_kokkos.html>
- Kokkos performance paper (OSTI): <https://www.osti.gov/servlets/purl/2588325>
