

to start the interactive shell 
`srun --partition=genoa --ntasks=1 --cpus-per-task=8 --time=00:30:00 --pty bash`

```
module purge
module load 2025
module load foss/2025b
module load Python/3.13.5-GCCcore-14.3.0
cd ~/Projects/MesoMemLive/
git clone https://gitlab.tudelft.nl/idema-group/mesomem.git wget --no-check-certificate https://download.lammps.org/tars/lammps-stable.tar.gz tar -xvf lammps-stable.tar.gz
cp mesomem/cpp_files/*.cpp mesomem/cpp_files/*.h lammps-22Jul2025/src/.
cd ~/Projects/MesoMemLive/lammps-22Jul2025/
mkdir build && cd build
cmake -D BUILD_MPI=yes \
     -D BUILD_SHARED_LIBS=yes \
     -D PKG_PYTHON=yes \
     -D PKG_BROWNIAN=yes \
     -D PKG_EXTRA-PAIR=yes \
     -D PKG_MOLECULE=yes \
     -D PKG_DIPOLE=yes \
     -D Python_EXECUTABLE=$(which python3) \
     -D CMAKE_INSTALL_PREFIX=$HOME/Projects/MesoMemLive/lammps-install \
     ../cmake
cmake --build . -j 8 cmake --build . --target install
export LD_LIBRARY_PATH=$HOME/Projects/MesoMemLive/lammps-install/lib64:$LD_LIBRARY_PATH
cat > $HOME/Projects/MesoMemLive/env.sh << 'EOF'
module purge
module load 2025
module load foss/2025b
module load Python/3.13.5-GCCcore-14.3.0
export PATH=$HOME/Projects/MesoMemLive/lammps-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/Projects/MesoMemLive/lammps-install/lib64:$LD_LIBRARY_PATH
EOF
source $HOME/Projects/MesoMemLive/env.sh
source $HOME/Projects/MesoMemLive/venv/bin/activate
pip install --upgrade pip
cd $HOME/Projects/MesoMemLive/lammps-22Jul2025/python
python install.py \
-p $HOME/Projects/MesoMemLive/lammps-22Jul2025/python/lammps \
-l $HOME/Projects/MesoMemLive/lammps-install/lib64/liblammps.so \
-v $HOME/Projects/MesoMemLive/lammps-22Jul2025/src/version.h \
-f

Verify
source $HOME/Projects/MesoMemLive/venv/bin/activate
python -c "
from lammps import lammps
lmp = lammps()
lmp.command('pair_style mesomem 2.5')
print('OK: mesomem pair style loaded through Python bindings')
"
```