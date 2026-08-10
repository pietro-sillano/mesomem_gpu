# Commands


```
mpirun -np 1 --bind-to core --use-hwthread-cpus lmp_kokkos_61 -i torus_edges.lmp -v R 5.0 -v h 20.0 -v kt 18 -k on g 1 t 4 -sf kk -pk kokkos newton off neigh full comm host
```

