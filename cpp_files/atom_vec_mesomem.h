/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* Custom atom style combining dipole_sphere (membrane) with molecular
   angle topology (ring polymer).  Data file column order:
     atom-ID  type  q  diameter  density  x  y  z  mux  muy  muz  mol-ID
   All dipole/sphere per-atom fields are present for every atom; polymer
   atoms get q=0, diameter=1, density=1, mu=0.  Membrane atoms get mol-ID=0.
*/

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(mesomem,AtomVecMesomem);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_MESOMEM_H
#define LMP_ATOM_VEC_MESOMEM_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecMesomem : virtual public AtomVec {
 public:
  AtomVecMesomem(class LAMMPS *);
  ~AtomVecMesomem() override;

  void grow_pointers() override;
  void create_atom_post(int) override;
  void data_atom_post(int) override;
  void pack_data_pre(int) override;
  void pack_data_post(int) override;
  void pack_restart_pre(int) override;
  void pack_restart_post(int) override;
  void unpack_restart_init(int) override;
  void read_data_general_to_restricted(int, int) override;
  void write_data_restricted_to_general() override;
  void write_data_restore_restricted() override;

 protected:
  // dipole_sphere per-atom fields
  double **mu;
  double **mu_hold;
  double *radius, *rmass;
  double **omega;
  double radius_one, rmass_one;

  // molecular (angle) topology fields
  int *num_bond, *num_angle;
  int **bond_type, **angle_type;
  int **nspecial;

  int any_bond_negative, any_angle_negative;
  int bond_per_atom, angle_per_atom;
  int *bond_negative, *angle_negative;
};

}    // namespace LAMMPS_NS

#endif
#endif
