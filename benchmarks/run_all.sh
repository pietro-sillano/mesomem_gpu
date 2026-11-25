#!/bin/bash
# run_all.sh -- full benchmark sweep (all four MesoMem systems) + plots for one
# hardware type. Results go to results_<type>.csv in each system folder and
# plots to plots/<type>/.
#
# Usage (after "micromamba activate gpu_env" and "source ../env_local.sh"):
#   ./run_all.sh md6x      # md6* workstations: i9-14900K + RTX 4090
#   ./run_all.sh md4x      # md4* workstations: Xeon W-2135 + GTX 1080 Ti
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

M=${1:?usage: ./run_all.sh <md6x|md4x>}
# MPI ranks for the CPU runs, up to the number of hardware threads
case "$M" in
  md6x) MPI="1 4 8 16" ;;
  md4x) MPI="1 2 4 6 12" ;;
  *) echo "unknown hardware type '$M', add its MPI list to run_all.sh"; exit 1 ;;
esac
MPI_MAX=${MPI##* }
COMMON="--machine $M --test_gpu --steps_gpu 5000 --steps_cpu 500 --hwthread"

echo "== $M: $(command -v lmp)"
(cd planar_benchmark && python benchmark.py $COMMON \
   --n_list 2500 10000 102400 --omp_list 1 4 8 --mpi_list $MPI)
(cd solvent_benchmark && python benchmark.py $COMMON \
   --n_list 2420 5120 9680 20480 50000 --solvent_density 0.1 --omp_list 1 4 8 --mpi_list $MPI)
(cd polymer && python benchmark.py $COMMON --omp_list 4 --mpi_list $MPI)
(cd polymer_solvent && python benchmark.py $COMMON \
   --solvent_density 0.1 --omp_list 4 --mpi_list $MPI_MAX)

python plot_bench.py --machine "$M"
