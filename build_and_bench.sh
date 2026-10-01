#!/usr/bin/env bash
#
# build_and_bench.sh -- build LAMMPS with compile_local.sh (default settings:
# CUDA + OpenMP, double precision, hybrid atom style with host sorting) for the
# GPU of the machine it runs on, then run the planar, solvent, polymer and
# polymer_solvent benchmarks.
#
# Usage: ./build_and_bench.sh
#
# Results are labelled with the machine's short hostname:
#   benchmarks/*/results_<hostname>.csv
# Re-running skips the LAMMPS fetch and rebuilds incrementally.

set -euo pipefail

MACHINE="$(hostname -s)"
PY=/home/pietro/micromamba/envs/gpu_env/bin/python   # needs numpy + pandas
MPI_LIST="1 4"     # CPU reference runs (MPI ranks)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/benchmarks"
LOG="$SCRIPT_DIR/build_${MACHINE}.log"
export PATH=/usr/local/cuda/bin:$PATH

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
echo "== building (log: $LOG) =="
"$SCRIPT_DIR/compile_local.sh" > "$LOG" 2>&1 || { echo "ERROR: build failed, see $LOG"; exit 1; }
grep "^OK:" "$LOG"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_build/local/env.sh"
echo "== using $(command -v lmp) =="
if ldd "$(command -v lmp)" | grep "not found"; then
  echo "ERROR: lmp has unresolved shared libraries"; exit 1
fi

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
echo "== planar =="
cd "$BENCH/planar_benchmark"
$PY benchmark.py --machine "$MACHINE" --lmp_bin lmp --test_gpu --omp_list 1 --mpi_list $MPI_LIST \
    --n_list 2500 10000 102400 --steps_gpu 2000 --steps_cpu 500 --replicas 3 > "bench_${MACHINE}.log" 2>&1

echo "== solvent =="
cd "$BENCH/solvent_benchmark"
$PY benchmark.py --machine "$MACHINE" --lmp_bin lmp --test_gpu --omp_list 1 --mpi_list 1 \
    --n_list 2420 9680 20480 --steps_gpu 2000 --steps_cpu 100 --replicas 3 > "bench_${MACHINE}.log" 2>&1

echo "== polymer =="
cd "$BENCH/polymer"
$PY benchmark.py --machine "$MACHINE" --lmp_bin lmp --test_gpu --omp_list 1 --mpi_list 1 \
    --steps_gpu 1000 --steps_cpu 20 --replicas 3 > "bench_${MACHINE}.log" 2>&1

echo "== polymer_solvent =="
cd "$BENCH/polymer_solvent"
$PY benchmark.py --machine "$MACHINE" --lmp_bin lmp --test_gpu --omp_list 1 --mpi_list 1 \
    --steps_gpu 1000 --steps_cpu 20 --replicas 3 > "bench_${MACHINE}.log" 2>&1

echo "== done: results in $BENCH/*/results_${MACHINE}.csv =="
