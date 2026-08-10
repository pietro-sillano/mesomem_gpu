"""
Generate a planar hexagonal lattice data file for atom_style dipole_sphere_angle
(monolithic, non-hybrid).

Per atom_vec_dipole_sphere_angle.cpp the Atoms section column order is:
    id type x y z molecule diameter density q mux muy muz

The data_atom_post hook converts diameter -> radius and (density,diameter) ->
rmass internally, so the file stores diameter and density as written.
molecule is set to 0 (unused for a pure-membrane system with no bonds/angles).

Usage:
    python planar_lattice_ds.py --path . --N 320 --d 0.85
"""

import os
import sys
import argparse
import numpy as np


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, required=True)
    p.add_argument("--N", type=int, default=50,
                   help="particles per square edge (total = N*N then trimmed to square)")
    p.add_argument("--d", type=float, default=0.85,
                   help="interparticle distance")
    p.add_argument("--diameter", type=float, default=1.0)
    p.add_argument("--density", type=float, default=1.0)
    return p.parse_args()


def generate_hex_grid(nx, ny, min_diam):
    xs, ys = [], []
    for row in range(ny):
        y = row * min_diam * np.sqrt(3) / 2.0
        x_off = min_diam / 2.0 if row % 2 == 1 else 0.0
        for col in range(nx):
            xs.append(x_off + col * min_diam)
            ys.append(y)
    return np.array([xs, ys]).T


def write_data(outfile, x, y, z, mux, muy, muz, types,
               Lx, Ly, Lz, diameter, density, q):
    n = len(x)
    with open(outfile, "w") as f:
        f.write("LAMMPS data file for atom_style dipole_sphere_angle\n\n")
        f.write(f"{n} atoms\n")
        f.write(f"{int(types.max())} atom types\n\n")
        f.write(f"{-Lx/2:.6f} {Lx/2:.6f} xlo xhi\n")
        f.write(f"{-Ly/2:.6f} {Ly/2:.6f} ylo yhi\n")
        f.write(f"{-Lz/2:.6f} {Lz/2:.6f} zlo zhi\n\n")
        # Atoms section: id type x y z molecule diameter density q mux muy muz
        f.write("Atoms # dipole_sphere_angle\n\n")
        for i in range(n):
            f.write(
                f"{i+1} {int(types[i])} "
                f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} "
                f"0 "                                   # molecule id (unused)
                f"{diameter:.6f} {density:.6f} "
                f"{q:.6f} "
                f"{mux[i]:.6f} {muy[i]:.6f} {muz[i]:.6f}\n"
            )
    print(f"Wrote {n} atoms to {outfile}")


def main():
    args = parse_arguments()

    centers = generate_hex_grid(args.N, args.N, args.d)

    # trim to a square footprint (drop columns past ymax of staggered grid)
    ymax = centers[:, 1].max()
    centers = centers[centers[:, 0] < ymax]

    edge = args.d
    Lx = centers[:, 0].max() - centers[:, 0].min() + edge
    Ly = centers[:, 1].max() - centers[:, 1].min() + edge
    Lz = Lx

    x = centers[:, 0] - Lx / 2.0
    y = centers[:, 1] - Ly / 2.0
    z = np.zeros_like(x)

    types = np.ones(x.shape[0], dtype=np.int32)
    mux = np.zeros_like(x)
    muy = np.zeros_like(x)
    muz = np.ones_like(x)

    n_atoms = x.shape[0]
    # Match the original naming convention: filename uses N*N (pre-trim),
    # so existing planar.lmp invocations with -v N <N*N> still resolve.
    label_n = args.N * args.N
    outfile = os.path.join(
        args.path, f"lattice_d_{args.d:.2f}_N_{label_n}_ds")
    write_data(outfile, x, y, z, mux, muy, muz, types,
               Lx, Ly, Lz, args.diameter, args.density, q=1.0)

    print(f"Box: {Lx:.3f} x {Ly:.3f} x {Lz:.3f}")
    print(f"Run: lmp_kokkos -in planar.lmp -v N {n_atoms} -v t_run 500 "
          f"-k on g 1 t 1 -sf kk -pk kokkos newton off neigh full comm device")


if __name__ == "__main__":
    main()
