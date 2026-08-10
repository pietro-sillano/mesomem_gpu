#!/bin/bash
LMP=${LMP:-lmp_kokkos_89_dev}
GPU_FLAGS="-k on g 1 -sf kk -pk kokkos newton off neigh full comm device"

for style in angle sphere; do
  for mode in cpu gpu; do
    dir="run_${style}_${mode}"
    mkdir -p "$dir"
    if [ "$mode" = "cpu" ]; then
      cmd="$LMP -in ../lj_${style}.lmp"
    else
      cmd="$LMP -in ../lj_${style}.lmp $GPU_FLAGS"
    fi
    echo "=== $style / $mode ==="
    echo "    $cmd"
    (cd "$dir" && eval $cmd 2>&1 | tail -20)
    echo ""
  done
done
