#!/bin/bash
# Compare pair_style mesomem/dipole (CPU) with mesomem/dipole/kk (GPU):
# runs mesomem_cpu_gpu.lmp once per mode and prints the largest per-atom
# force/torque difference relative to the largest CPU value.
#
# Usage (from this folder, after "source ../env_local.sh"):
#   ./run_mesomem_cpu_gpu.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LMP=${LMP:-lmp}
command -v "$LMP" >/dev/null || { echo "ERROR: $LMP not found, run 'source env_local.sh' first"; exit 1; }
GPU_FLAGS="-k on g 1 -sf kk -pk kokkos newton on neigh half comm device"

for mode in cpu gpu; do
  dir="run_mesomem_${mode}"
  mkdir -p "$dir"
  flags=""
  [ "$mode" = "gpu" ] && flags="$GPU_FLAGS"
  echo "=== $mode: $LMP -in ../mesomem_cpu_gpu.lmp $flags"
  (cd "$dir" && $LMP -in ../mesomem_cpu_gpu.lmp $flags > log.out 2>&1)
  grep -A1 "^ *Step" "$dir/log.out" | tail -1
done

python3 - <<'EOF'
import numpy as np
cpu = np.loadtxt("run_mesomem_cpu/forces.dump", skiprows=9)
gpu = np.loadtxt("run_mesomem_gpu/forces.dump", skiprows=9)
assert (cpu[:, 0] == gpu[:, 0]).all(), "atom ids differ"
for name, cols in (("force", slice(1, 4)), ("torque", slice(4, 7))):
    ref = np.abs(cpu[:, cols]).max()
    diff = np.abs(cpu[:, cols] - gpu[:, cols]).max()
    print(f"{name:6s}: max |CPU| = {ref:.6g}, max |CPU-GPU| = {diff:.3g}, relative = {diff/ref:.3g}")
EOF
