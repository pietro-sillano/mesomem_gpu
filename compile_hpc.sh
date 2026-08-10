#!/usr/bin/env bash
#
# compile_hpc.sh -- build LAMMPS + MesoMem GPU package on an HPC cluster
#
# Fetches a pinned LAMMPS commit, drops in the custom MesoMem source files
# from cpp_files/, and builds LAMMPS with KOKKOS (GPU/CUDA), MPI, and the
# Python API, using the cluster's module system.
#
# Run this from an interactive GPU job, e.g. on Snellius:
#   srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=16 --time=01:00:00 --pty bash
#   ./compile_hpc.sh
#
# The module names/versions and GPU_ARCH below are set for Snellius
# (A100 GPUs). Edit them if you're building on a different cluster.
#
# Everything ends up under ./_build/hpc/ (source, cmake build dir,
# install, python venv) so nothing outside that folder is touched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAMMPS_GIT_URL="https://github.com/lammps/lammps.git"
# Pinned LAMMPS commit (develop branch, "30 Mar 2026" snapshot) that the
# custom KOKKOS files in cpp_files/ are known to compile against. The
# stable/*.tar.gz release is NOT compatible: AtomVecKokkos's sync/modified
# API changed (uint64_t masks, sync_pinned) after the last stable tag.
LAMMPS_REF="5ea3b58ad8d72ddc1b50c102033578181d34bbbd"

# GPU architecture on the cluster's GPU partition (Snellius: A100 -> AMPERE80)
GPU_ARCH="AMPERE80"

BUILD_ROOT="$SCRIPT_DIR/_build/hpc"
SRC_DIR="$BUILD_ROOT/lammps-src"
INSTALL_PREFIX="$BUILD_ROOT/install"
VENV_DIR="$BUILD_ROOT/venv"
ENV_FILE="$BUILD_ROOT/env.sh"
JOBS="$(nproc)"

echo "== MesoMem GPU build (hpc) =="
echo "  work dir: $BUILD_ROOT"
echo "  GPU arch: $GPU_ARCH"
echo

# --------------------------------------------------------------------------
# Step 0: load the cluster toolchain
# --------------------------------------------------------------------------
command -v module >/dev/null 2>&1 || {
  echo "ERROR: 'module' command not found. Run this from an HPC login/compute"
  echo "       shell (e.g. via 'srun --pty bash'), not a plain local shell."
  exit 1
}

echo "-- loading modules --"
module purge
module load 2025
module load foss/2025b
module load Python/3.13.5-GCCcore-14.3.0
# Adjust the CUDA module name/version to whatever is available on the
# cluster's GPU partition (check with 'module avail CUDA').
module load CUDA/12.9.1

PYTHON_BIN="$(command -v python3)"
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found after loading modules"; exit 1; }
echo "  python: $PYTHON_BIN"
echo

# --------------------------------------------------------------------------
# Step 1: fetch the pinned LAMMPS source tree
# --------------------------------------------------------------------------
echo "-- fetching LAMMPS ($LAMMPS_REF) --"
rm -rf "$SRC_DIR"
mkdir -p "$SRC_DIR"
git -C "$SRC_DIR" init -q
git -C "$SRC_DIR" remote add origin "$LAMMPS_GIT_URL"
git -C "$SRC_DIR" fetch --depth 1 origin "$LAMMPS_REF"
git -C "$SRC_DIR" checkout -q FETCH_HEAD

# --------------------------------------------------------------------------
# Step 2: drop in the custom MesoMem source files
#
# Placement follows LAMMPS's package layout: files belong in the src/
# subfolder of the package that owns them so CMake's per-package glob
# picks them up.
# --------------------------------------------------------------------------
echo "-- installing MesoMem custom source files --"
cp "$SCRIPT_DIR"/cpp_files/pair_mesomem.cpp \
   "$SCRIPT_DIR"/cpp_files/pair_mesomem.h \
   "$SRC_DIR/src/"

cp "$SCRIPT_DIR"/cpp_files/atom_vec_mesomem.cpp \
   "$SCRIPT_DIR"/cpp_files/atom_vec_mesomem.h \
   "$SRC_DIR/src/DIPOLE/"

cp "$SCRIPT_DIR"/cpp_files/atom_vec_mesomem_kokkos.cpp \
   "$SCRIPT_DIR"/cpp_files/atom_vec_mesomem_kokkos.h \
   "$SCRIPT_DIR"/cpp_files/pair_mesomem_kokkos.cpp \
   "$SCRIPT_DIR"/cpp_files/pair_mesomem_kokkos.h \
   "$SCRIPT_DIR"/cpp_files/fix_langevin_kokkos.cpp \
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
# Step 5: write env.sh so batch jobs can load this build later
# --------------------------------------------------------------------------
{
  echo "# source this file to use the hpc MesoMem GPU LAMMPS build"
  echo "module purge"
  echo "module load 2025"
  echo "module load foss/2025b"
  echo "module load Python/3.13.5-GCCcore-14.3.0"
  echo "module load CUDA/12.9.1"
  echo "export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
  echo "export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\""
} > "$ENV_FILE"
echo "-- wrote $ENV_FILE --"

# --------------------------------------------------------------------------
# Step 6: smoke test - confirm the mesomem atom/pair styles load via Python
# --------------------------------------------------------------------------
echo "-- running smoke test --"
# shellcheck disable=SC1090
source "$ENV_FILE"
source "$VENV_DIR/bin/activate"

python3 -c "
from lammps import lammps
lmp = lammps()
lmp.command('units lj')
lmp.command('atom_style mesomem')
lmp.command('pair_style mesomem 2.5')
print('OK: mesomem atom_style + pair_style loaded through the Python API')
"

deactivate

echo
echo "== hpc build complete =="
echo "  lmp binary: $INSTALL_PREFIX/bin/lmp"
echo "  to use this build later:  source $ENV_FILE && source $VENV_DIR/bin/activate"
