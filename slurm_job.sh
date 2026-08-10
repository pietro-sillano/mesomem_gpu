#!/bin/bash
#Set job requirements
#SBATCH --partition=gpu_a100
#SBATCH -n 1                    # number of MPI tasks
#SBATCH --cpus-per-task 4       # threads for mpi task
#SBATCH --gpus-per-node 1
#SBATCH -t 00:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.sillano@tudelft.nl

set -euo pipefail

#Loading modules (must match the toolchain compile_hpc.sh built with)
module load 2025
module load foss/2025b
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.9.1

# sbatch copies this script into /var/spool/slurm/.../job.../ and runs it
# from there, so ${BASH_SOURCE[0]} does NOT point at the repo. Use
# SLURM_SUBMIT_DIR (set by Slurm to the directory sbatch was run from).
SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set, this script must be run via sbatch}"
SYSTEM_DIR="$SCRIPT_DIR/benchmarks/polymer_solvent"

[[ -f "$SCRIPT_DIR/_build/hpc/env.sh" ]] || {
  echo "ERROR: $SCRIPT_DIR/_build/hpc/env.sh not found. Run ./compile_hpc.sh first." >&2
  exit 1
}
source "$SCRIPT_DIR/_build/hpc/env.sh"

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# polymer_solvent.lmp does "read_data ../../${v_data}", so run from a
# directory two levels below polymer_solvent/ (matches benchmark.py's
# bench_runs/<name>/ layout) with data_file relative to polymer_solvent/.
RUN_DIR="$SYSTEM_DIR/bench_runs/slurm_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

srun lmp -in "$SYSTEM_DIR/polymer_solvent.lmp" \
    -v t_run 2000 \
    -v data_file input/combined_N35280_poly64000_r45.5_d0.10.data \
    -k on g 1 t 4 \
    -sf kk \
    -pk kokkos newton on neigh half comm device
