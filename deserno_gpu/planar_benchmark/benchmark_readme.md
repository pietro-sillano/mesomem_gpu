## Deserno / Cooke bilayer benchmark

A separate benchmark for the Cooke lipid bilayer model lives in `deserno_gpu/planar_benchmark/`. It generates the bilayer directly in LAMMPS (no pre-generated data files) and supports a Kokkos option sweep.

```bash
cd deserno_gpu/planar_benchmark

# Standard benchmark (same interface as above)
python new_bench.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --test_gpu --n_list 600 2400 9600 38400 153600 614400 \
    --omp_list 1 4 8 --mpi_list 1 4 8 --hwthread

# Kokkos option sweep (neigh × comm × sort × OMP)
python kokkos_sweep.py --machine md69 --lmp_bin lmp_kokkos_89_dev \
    --n_list 9600 38400 --omp_list 1 4 8 --steps 1000 --hwthread

# Plot sweep results
python plot_kokkos_sweep.py   # edit CSV filename at top of file
```

---