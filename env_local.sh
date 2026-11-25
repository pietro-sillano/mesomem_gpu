# env_local.sh -- load the compile_local.sh build that matches this machine's GPU
#
# Usage (from any directory):
#   source /martini/pietro/mesomem_gpu/env_local.sh
#
# Picks ./_build/local_sm<XY>/ from the GPU compute capability, so the same
# command works on every workstation (md6*: sm89, md4*: sm61). Source it
# AFTER activating a micromamba env, so this lmp comes first on PATH.

_mm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_mm_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
_mm_env="$_mm_root/_build/local_sm${_mm_cap/./}/env.sh"

if [[ -f "$_mm_env" ]]; then
  source "$_mm_env"
  echo "loaded $_mm_env"
  echo "lmp -> $(command -v lmp)"
else
  echo "ERROR: no build for compute capability $_mm_cap ($_mm_env missing), run ./compile_local.sh"
fi
unset _mm_root _mm_cap _mm_env
