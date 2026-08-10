#!/bin/bash
#Set job requirements
#SBATCH --partition=gpu_a100
#SBATCH -n 1                    # number of MPI tasks
#SBATCH --cpus-per-task 4       # threads for mpi task
#SBATCH --gpus-per-node 1
#SBATCH -t 01:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.sillano@tudelft.nl

#Loading modules (must match the toolchain compile_hpc.sh built with)
module load 2025
module load foss/2025b
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.9.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/benchmarks/polymer_solvent"

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
