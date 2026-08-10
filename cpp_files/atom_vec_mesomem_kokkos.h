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

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(mesomem/kk,AtomVecMesomemKokkos);
AtomStyle(mesomem/kk/device,AtomVecMesomemKokkos);
AtomStyle(mesomem/kk/host,AtomVecMesomemKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_MESOMEM_KOKKOS_H
#define LMP_ATOM_VEC_MESOMEM_KOKKOS_H

#include "atom_vec_mesomem.h"
#include "atom_vec_kokkos.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecMesomemKokkos : public AtomVecKokkos,
                                        public AtomVecMesomem {
 public:
  AtomVecMesomemKokkos(class LAMMPS *);
  void init() override;

  void grow(int) override;
  void grow_pointers() override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;
  void sync(ExecutionSpace space, uint64_t mask) override;
  void modified(ExecutionSpace space, uint64_t mask) override;
  void sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag = 0) override;

 protected:
  // Raw CPU-side pointers (used by non-Kokkos base class methods)
  double *q;
  double **torque;
  tagint *molecule;
  tagint **bond_atom;
  tagint **angle_atom1, **angle_atom2, **angle_atom3;
  tagint **special;
  // d_x, d_v, d_f, d_q, d_mu, d_radius, d_rmass, d_omega, d_torque,
  // d_tag, d_type, d_mask, d_image (and h_ counterparts) are inherited
  // from AtomVecKokkos — do not re-declare here.
};

}    // namespace LAMMPS_NS

#endif
#endif
