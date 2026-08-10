#!/usr/bin/env bash
#
# run_gpu_small.sh -- short GPU smoke run of the polymer+solvent system
#
# Loads the modules/build produced by compile_hpc.sh and runs a small
# number of steps of benchmarks/polymer_solvent/polymer_solvent.lmp on
# one GPU. Meant to be launched directly as the srun command, e.g. on
# Snellius:
#
#   srun --partition=gpu_a100 --ntasks=1 --gpus-per-node=1 \
#        --cpus-per-task=8 --time=00:05:00 --pty ./run_gpu_small.sh
#
# stdout/stderr are left unredirected so they show up directly in the
# srun --pty terminal; a copy is also kept in the run directory's
# log.lammps (LAMMPS) file for later inspection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/_build/hpc/env.sh"
SYSTEM_DIR="$SCRIPT_DIR/benchmarks/polymer_solvent"
DATA_FILE="input/combined_N35280_poly64000_r45.5_d0.10.data"
T_RUN="${T_RUN:-200}"

[[ -f "$ENV_FILE" ]] || {
  echo "ERROR: $ENV_FILE not found. Run ./compile_hpc.sh first." >&2
  exit 1
}
[[ -f "$SYSTEM_DIR/$DATA_FILE" ]] || {
  echo "ERROR: $SYSTEM_DIR/$DATA_FILE not found." >&2
  exit 1
}

echo "-- loading modules and the compile_hpc.sh build --"
# shellcheck disable=SC1090
source "$ENV_FILE"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

RUN_DIR="$SYSTEM_DIR/bench_runs/gpu_small_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "-- running polymer_solvent.lmp for $T_RUN steps on 1 GPU --"
echo "   run dir: $RUN_DIR"

lmp -in "$SYSTEM_DIR/polymer_solvent.lmp" \
    -v t_run "$T_RUN" \
    -v data_file "$DATA_FILE" \
    -k on g 1 t 1 \
    -sf kk \
    -pk kokkos newton on neigh half comm device
