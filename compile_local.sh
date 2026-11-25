#!/usr/bin/env bash
#
# compile_local.sh -- build LAMMPS + MesoMem GPU package on a local machine
#
# Fetches a pinned LAMMPS commit, drops in the custom MesoMem source files
# from cpp_files/, and builds LAMMPS with KOKKOS (GPU/CUDA), MPI, and the
# Python API. The GPU architecture is auto-detected.
#
# If ./_build/local_<host>/lammps-src already exists at the pinned commit,
# the LAMMPS fetch is skipped and re-running this script just does an
# incremental rebuild (custom source files are always re-copied first, so
# local edits to cpp_files/ are picked up). Pass --force to wipe it and
# fetch + build from scratch.
#
# Usage: ./compile_local.sh [--force]
#
# Everything ends up under ./_build/local_<host>/ (source, cmake build dir,
# install, python venv) so nothing outside that folder is touched.

set -euo pipefail

FORCE="no"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE="yes"; shift ;;
    -h|--help) echo "Usage: $0 [--force]"; exit 0 ;;
    *) echo "Unknown option: $1"; echo "Usage: $0 [--force]"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAMMPS_GIT_URL="https://github.com/lammps/lammps.git"
# Pinned LAMMPS commit: the "patch_13Nov2026" feature release. It is the
# first release that ships pair_style mesomem/dipole (DIPOLE) together with
# its KOKKOS port (mesomem/dipole/kk), and the custom KOKKOS atom style in
# cpp_files/ is known to compile against it. stable_30Sep2026 does NOT
# contain mesomem/dipole.
LAMMPS_REF="68b557e3a2148ea9b31dda5b55800215fb8be677"

# one build per host: /martini is shared between workstations with
# different GPUs (e.g. md43: GTX 1080 Ti, md69: RTX 4090)
BUILD_ROOT="$SCRIPT_DIR/_build/local_$(hostname -s)"
SRC_DIR="$BUILD_ROOT/lammps-src"
INSTALL_PREFIX="$BUILD_ROOT/install"
VENV_DIR="$BUILD_ROOT/venv"
ENV_FILE="$BUILD_ROOT/env.sh"
JOBS="$(nproc)"

echo "== MesoMem GPU build (local) =="
echo "  work dir: $BUILD_ROOT"
echo "  force:    $FORCE"
echo

# --------------------------------------------------------------------------
# Step 0: check prerequisites and detect the GPU architecture
# --------------------------------------------------------------------------
command -v cmake   >/dev/null 2>&1 || { echo "ERROR: cmake not found"; exit 1; }
command -v mpicc   >/dev/null 2>&1 || { echo "ERROR: MPI compiler (mpicc) not found"; exit 1; }
# always build against the system python, even if a micromamba/conda env is
# active: liblammps.so links to libpython, and switching interpreters between
# runs breaks CMake's cached Python lookup
PYTHON_BIN="/usr/bin/python3"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: $PYTHON_BIN not found"; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found, no GPU detected"; exit 1; }
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
case "$COMPUTE_CAP" in
  6.0) GPU_ARCH=PASCAL60 ;;
  6.1) GPU_ARCH=PASCAL61 ;;
  7.0) GPU_ARCH=VOLTA70 ;;
  7.5) GPU_ARCH=TURING75 ;;
  8.0) GPU_ARCH=AMPERE80 ;;
  8.6) GPU_ARCH=AMPERE86 ;;
  8.9) GPU_ARCH=ADA89 ;;
  9.0) GPU_ARCH=HOPPER90 ;;
  *) echo "ERROR: unrecognized GPU compute capability '$COMPUTE_CAP', edit compile_local.sh to add it"; exit 1 ;;
esac
echo "  GPU:    compute capability $COMPUTE_CAP -> Kokkos_ARCH_$GPU_ARCH"
echo "  python: $PYTHON_BIN"

# nvcc/CUDA is commonly installed outside PATH
if ! command -v nvcc >/dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found (checked PATH and /usr/local/cuda/bin)"; exit 1; }
echo

# --------------------------------------------------------------------------
# Step 1: fetch the pinned LAMMPS source tree (skipped if already present)
# --------------------------------------------------------------------------
# an existing checkout is only reused if it is at the pinned commit, so
# bumping LAMMPS_REF above automatically triggers a fresh fetch
CURRENT_REF="$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ "$FORCE" == "no" && "$CURRENT_REF" == "$LAMMPS_REF" ]]; then
  echo "-- reusing existing LAMMPS source at $SRC_DIR (pass --force to refetch from scratch) --"
else
  echo "-- fetching LAMMPS ($LAMMPS_REF) --"
  rm -rf "$SRC_DIR"
  mkdir -p "$SRC_DIR"
  git -C "$SRC_DIR" init -q
  git -C "$SRC_DIR" remote add origin "$LAMMPS_GIT_URL"
  git -C "$SRC_DIR" fetch --depth 1 origin "$LAMMPS_REF"
  git -C "$SRC_DIR" checkout -q FETCH_HEAD
fi

# --------------------------------------------------------------------------
# Step 2: drop in the custom MesoMem source files
#
# Placement follows LAMMPS's package layout: files belong in the src/
# subfolder of the package that owns them so CMake's per-package glob
# picks them up.
# --------------------------------------------------------------------------
echo "-- installing MesoMem custom source files --"
cp "$SCRIPT_DIR"/cpp_files/atom_vec_dipole_sphere_angle.cpp \
   "$SCRIPT_DIR"/cpp_files/atom_vec_dipole_sphere_angle.h \
   "$SRC_DIR/src/DIPOLE/"

cp "$SCRIPT_DIR"/cpp_files/atom_vec_dipole_sphere_angle_kokkos.cpp \
   "$SCRIPT_DIR"/cpp_files/atom_vec_dipole_sphere_angle_kokkos.h \
   "$SRC_DIR/src/KOKKOS/"

# --------------------------------------------------------------------------
# Step 3: configure, build, install
# --------------------------------------------------------------------------
echo "-- configuring with CMake --"
mkdir -p "$SRC_DIR/build"
cd "$SRC_DIR/build"

cmake \
  -D BUILD_MPI=yes \
  -D BUILD_SHARED_LIBS=yes \
  -D CMAKE_BUILD_TYPE=Release \
  -D PKG_PYTHON=yes \
  -D PKG_DIPOLE=yes \
  -D PKG_MOLECULE=yes \
  -D PKG_EXTRA-PAIR=yes \
  -D PKG_KOKKOS=yes \
  -D Kokkos_ENABLE_CUDA=yes \
  -D "Kokkos_ARCH_${GPU_ARCH}=yes" \
  -D Kokkos_ENABLE_CUDA_UVM=OFF \
  -D CMAKE_CXX_COMPILER="$SRC_DIR/lib/kokkos/bin/nvcc_wrapper" \
  -U "Python_*" -U "_Python_*" \
  -D Python_EXECUTABLE="$PYTHON_BIN" \
  -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  "$SRC_DIR/cmake"

echo "-- building (this can take a while) --"
cmake --build . -j "$JOBS"
cmake --build . --target install

# CMAKE_INSTALL_LIBDIR (lib vs lib64) is distro-dependent, so find the
# installed shared library instead of hardcoding a path.
LIBLAMMPS="$(find "$INSTALL_PREFIX" -name 'liblammps.so*' -print -quit)"
[[ -n "$LIBLAMMPS" ]] || { echo "ERROR: liblammps.so not found under $INSTALL_PREFIX after install"; exit 1; }
LIB_DIR="$(dirname "$LIBLAMMPS")"

# --------------------------------------------------------------------------
# Step 4: Python API - build a venv and install the lammps python bindings
# --------------------------------------------------------------------------
echo "-- setting up Python virtual environment --"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null

echo "-- installing LAMMPS python bindings --"
cd "$SRC_DIR/python"
python install.py \
  -p "$SRC_DIR/python/lammps" \
  -l "$LIBLAMMPS" \
  -v "$SRC_DIR/src/version.h" \
  -f

deactivate

# --------------------------------------------------------------------------
# Step 5: write env.sh so this build can be reloaded later
# --------------------------------------------------------------------------
{
  echo "# source this file to use the local MesoMem GPU LAMMPS build"
  echo "export PATH=\"$INSTALL_PREFIX/bin:\${PATH:-}\""
  echo "export LD_LIBRARY_PATH=\"$LIB_DIR:\${LD_LIBRARY_PATH:-}\""
} > "$ENV_FILE"
echo "-- wrote $ENV_FILE --"

# --------------------------------------------------------------------------
# Step 6: smoke test - confirm the atom/pair styles load via Python
# --------------------------------------------------------------------------
echo "-- running smoke test --"
# shellcheck disable=SC1090
source "$ENV_FILE"
source "$VENV_DIR/bin/activate"

python3 -c "
from lammps import lammps
lmp = lammps()
lmp.command('units lj')
lmp.command('atom_style dipole_sphere_angle')
lmp.command('pair_style mesomem/dipole')
print('OK: dipole_sphere_angle atom_style + mesomem/dipole pair_style loaded through the Python API')
"

deactivate

echo
echo "== local build complete =="
echo "  lmp binary: $INSTALL_PREFIX/bin/lmp"
echo "  to use this build later:  source $ENV_FILE && source $VENV_DIR/bin/activate"
