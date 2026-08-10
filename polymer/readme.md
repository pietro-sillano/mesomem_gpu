# Commands

* Generating a spherical vesicle

```bash
python spherical_vesicle_ds.py --path ./input --N 35000 --target_dist 0.8 --box_factor 1.2
```

* Combining with relaxed polymer

```bash
python combine_vesicle_polymer.py --vesicle ./input/vesicle_ds_N35280_d0.80.data --polymer input/poly_relaxed_64000.data  --out ./input/combined_N35280_poly64000.data
```

* Run benchmark for GPU
```bash
python polymer_bench.py --steps 10000 --omp_list 1 4 16 --test_gpu --hwthread --neigh full --lmp_bin lmp_kokkos_89_dev --data_file ./input/combined_N35280_poly64000.data
```

* Run benchmark for CPU
```bash
python polymer_bench.py --steps 250 --mpi_list 1 4 8 16 --omp_list 1 --hwthread --neigh full --lmp_bin lmp_kokkos_89_dev --data_file ./input/combined_N35280_poly64000.data
```


# How to visualzie with Ovito
Lammps datafile you need to choose:

Dump trajectories are automatically recognized.


