#!/bin/bash
# ab_old_new.sh -- back-to-back GPU timing of the baseline binary (old pair
# style membrane_sillanov2, LAMMPS 5ea3b58, produced results_md6x_old.csv) and
# the current build (mesomem/dipole/kk) on the planar benchmark. The inputs are
# identical except for the pair_style line. Both binaries see the same GPU
# state, so this separates a real slowdown from a GPU busy with another job
# (the processes on the GPU are listed first).
#
# Usage (after "source ../../env_local.sh"):
#   ./ab_old_new.sh                # N=2500, 2000 steps
#   N=10000 STEPS=5000 ./ab_old_new.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OLD_LMP=${OLD_LMP:-$HOME/.local/bin/lmp_kokkos_89_dev}
NEW_LMP=${NEW_LMP:-lmp}
N=${N:-2500}
STEPS=${STEPS:-2000}
KK="-k on g 1 t 1 -sf kk -pk kokkos newton on neigh half comm device"

echo "== GPU load and processes (should be ~0% and empty) =="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "old: $OLD_LMP"
echo "new: $(command -v "$NEW_LMP")"

# run dirs two levels below this folder, like benchmark.py (read_data ../../input/...)
mkdir -p bench_runs/ab_old bench_runs/ab_new
sed 's#^pair_style .*#pair_style      membrane_sillanov2 ${v_rcut}#' planar.lmp > bench_runs/ab_old/planar.lmp
cp planar.lmp bench_runs/ab_new/planar.lmp

echo "== N=$N, $STEPS steps, timesteps/s =="
for rep in 1 2; do
  for v in old new; do
    bin=$OLD_LMP
    [ "$v" = "new" ] && bin=$NEW_LMP
    dir="bench_runs/ab_$v"
    (cd "$dir" && OMP_NUM_THREADS=1 "$bin" -in planar.lmp -v N "$N" -v t_run "$STEPS" $KK \
       -log "log_$rep.lammps" > "out_$rep.txt" 2>&1) || { echo "$v failed:"; tail -3 "$dir/out_$rep.txt"; exit 1; }
    tps=$(grep "^Performance" "$dir/log_$rep.lammps" | sed 's/.*, \([0-9.]*\) timesteps\/s.*/\1/')
    echo "  $v  rep $rep: $tps"
  done
done
