// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* Kokkos atom style combining dipole_sphere (membrane) with angle
   molecular topology (ring polymer) in one native style.
   Enables GPU comm via the field-registry-based pack/unpack in AtomVecKokkos.

   This file intentionally contains NO pack/unpack overrides — those are
   handled entirely by the base class AtomVecKokkos using the field registry
   that AtomVecMesomem sets up in its constructor.
*/

#include "atom_vec_mesomem_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecMesomemKokkos::AtomVecMesomemKokkos(LAMMPS *lmp)
    : AtomVec(lmp), AtomVecKokkos(lmp), AtomVecMesomem(lmp),
      q(nullptr), torque(nullptr), molecule(nullptr),
      bond_atom(nullptr), angle_atom1(nullptr), angle_atom2(nullptr),
      angle_atom3(nullptr), special(nullptr)
{
}

/* ----------------------------------------------------------------------
   process field strings to initialize data structs for all other methods
------------------------------------------------------------------------- */

void AtomVecMesomemKokkos::init()
{
  AtomVecMesomem::init();

  set_atom_masks();
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecMesomemKokkos::grow(int n)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA, nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atomKK->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR, "Per-processor system is too big");

  atomKK->sync(Device, ALL_MASK);
  atomKK->modified(Device, ALL_MASK);

  // base fields
  memoryKK->grow_kokkos(atomKK->k_tag,    atomKK->tag,    nmax, "atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type,   atomKK->type,   nmax, "atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask,   atomKK->mask,   nmax, "atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image,  atomKK->image,  nmax, "atom:image");
  memoryKK->grow_kokkos(atomKK->k_x,      atomKK->x,      nmax, "atom:x");
  memoryKK->grow_kokkos(atomKK->k_v,      atomKK->v,      nmax, "atom:v");
  memoryKK->grow_kokkos(atomKK->k_f,      atomKK->f,      nmax, "atom:f");

  // sphere fields
  memoryKK->grow_kokkos(atomKK->k_radius, atomKK->radius, nmax, "atom:radius");
  memoryKK->grow_kokkos(atomKK->k_rmass,  atomKK->rmass,  nmax, "atom:rmass");
  memoryKK->grow_kokkos(atomKK->k_omega,  atomKK->omega,  nmax, "atom:omega");
  memoryKK->grow_kokkos(atomKK->k_torque, atomKK->torque, nmax, "atom:torque");

  // dipole fields
  memoryKK->grow_kokkos(atomKK->k_q,      atomKK->q,      nmax, "atom:q");
  memoryKK->grow_kokkos(atomKK->k_mu,     atomKK->mu,     nmax, "atom:mu");

  // molecular/angle fields
  memoryKK->grow_kokkos(atomKK->k_molecule,   atomKK->molecule,   nmax, "atom:molecule");
  memoryKK->grow_kokkos(atomKK->k_nspecial,   atomKK->nspecial,   nmax, 3, "atom:nspecial");
  memoryKK->grow_kokkos(atomKK->k_special,    atomKK->special,    nmax, atomKK->maxspecial,
                        "atom:special");
  memoryKK->grow_kokkos(atomKK->k_num_bond,   atomKK->num_bond,   nmax, "atom:num_bond");
  memoryKK->grow_kokkos(atomKK->k_bond_type,  atomKK->bond_type,  nmax, atomKK->bond_per_atom,
                        "atom:bond_type");
  memoryKK->grow_kokkos(atomKK->k_bond_atom,  atomKK->bond_atom,  nmax, atomKK->bond_per_atom,
                        "atom:bond_atom");
  memoryKK->grow_kokkos(atomKK->k_num_angle,  atomKK->num_angle,  nmax, "atom:num_angle");
  memoryKK->grow_kokkos(atomKK->k_angle_type, atomKK->angle_type, nmax, atomKK->angle_per_atom,
                        "atom:angle_type");
  memoryKK->grow_kokkos(atomKK->k_angle_atom1, atomKK->angle_atom1, nmax, atomKK->angle_per_atom,
                        "atom:angle_atom1");
  memoryKK->grow_kokkos(atomKK->k_angle_atom2, atomKK->angle_atom2, nmax, atomKK->angle_per_atom,
                        "atom:angle_atom2");
  memoryKK->grow_kokkos(atomKK->k_angle_atom3, atomKK->angle_atom3, nmax, atomKK->angle_per_atom,
                        "atom:angle_atom3");

  grow_pointers();
  atomKK->sync(Host, ALL_MASK);

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecMesomemKokkos::grow_pointers()
{
  // base fields
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.view_device();
  h_tag = atomKK->k_tag.view_host();

  type = atomKK->type;
  d_type = atomKK->k_type.view_device();
  h_type = atomKK->k_type.view_host();
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.view_device();
  h_mask = atomKK->k_mask.view_host();
  image = atomKK->image;
  d_image = atomKK->k_image.view_device();
  h_image = atomKK->k_image.view_host();

  x = atomKK->x;
  d_x = atomKK->k_x.view_device();
  h_x = atomKK->k_x.view_hostkk();
  v = atomKK->v;
  d_v = atomKK->k_v.view_device();
  h_v = atomKK->k_v.view_hostkk();
  f = atomKK->f;
  d_f = atomKK->k_f.view_device();
  h_f = atomKK->k_f.view_hostkk();

  // sphere fields
  radius = atomKK->radius;
  d_radius = atomKK->k_radius.view_device();
  h_radius = atomKK->k_radius.view_hostkk();
  rmass = atomKK->rmass;
  d_rmass = atomKK->k_rmass.view_device();
  h_rmass = atomKK->k_rmass.view_hostkk();
  omega = atomKK->omega;
  d_omega = atomKK->k_omega.view_device();
  h_omega = atomKK->k_omega.view_hostkk();
  torque = atomKK->torque;
  d_torque = atomKK->k_torque.view_device();
  h_torque = atomKK->k_torque.view_hostkk();

  // dipole fields
  q = atomKK->q;
  d_q = atomKK->k_q.view_device();
  h_q = atomKK->k_q.view_hostkk();
  mu = atomKK->mu;
  d_mu = atomKK->k_mu.view_device();
  h_mu = atomKK->k_mu.view_hostkk();

  // molecular/angle fields
  molecule = atomKK->molecule;
  d_molecule = atomKK->k_molecule.view_device();
  h_molecule = atomKK->k_molecule.view_host();

  nspecial = atomKK->nspecial;
  d_nspecial = atomKK->k_nspecial.view_device();
  h_nspecial = atomKK->k_nspecial.view_hostkk();
  special = atomKK->special;
  d_special = atomKK->k_special.view_device();
  h_special = atomKK->k_special.view_hostkk();

  num_bond = atomKK->num_bond;
  d_num_bond = atomKK->k_num_bond.view_device();
  h_num_bond = atomKK->k_num_bond.view_host();
  bond_type = atomKK->bond_type;
  d_bond_type = atomKK->k_bond_type.view_device();
  h_bond_type = atomKK->k_bond_type.view_hostkk();
  bond_atom = atomKK->bond_atom;
  d_bond_atom = atomKK->k_bond_atom.view_device();
  h_bond_atom = atomKK->k_bond_atom.view_hostkk();

  num_angle = atomKK->num_angle;
  d_num_angle = atomKK->k_num_angle.view_device();
  h_num_angle = atomKK->k_num_angle.view_host();
  angle_type = atomKK->angle_type;
  d_angle_type = atomKK->k_angle_type.view_device();
  h_angle_type = atomKK->k_angle_type.view_hostkk();
  angle_atom1 = atomKK->angle_atom1;
  d_angle_atom1 = atomKK->k_angle_atom1.view_device();
  h_angle_atom1 = atomKK->k_angle_atom1.view_hostkk();
  angle_atom2 = atomKK->angle_atom2;
  d_angle_atom2 = atomKK->k_angle_atom2.view_device();
  h_angle_atom2 = atomKK->k_angle_atom2.view_hostkk();
  angle_atom3 = atomKK->angle_atom3;
  d_angle_atom3 = atomKK->k_angle_atom3.view_device();
  h_angle_atom3 = atomKK->k_angle_atom3.view_hostkk();
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecMesomemKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, ALL_MASK & ~F_MASK & ~TORQUE_MASK);

  Sorter.sort(LMPDeviceType(), d_tag);
  Sorter.sort(LMPDeviceType(), d_type);
  Sorter.sort(LMPDeviceType(), d_mask);
  Sorter.sort(LMPDeviceType(), d_image);
  Sorter.sort(LMPDeviceType(), d_x);
  Sorter.sort(LMPDeviceType(), d_v);

  // sphere
  Sorter.sort(LMPDeviceType(), d_radius);
  Sorter.sort(LMPDeviceType(), d_rmass);
  Sorter.sort(LMPDeviceType(), d_omega);

  // dipole
  Sorter.sort(LMPDeviceType(), d_q);
  Sorter.sort(LMPDeviceType(), d_mu);

  // molecular/angle
  Sorter.sort(LMPDeviceType(), d_molecule);
  Sorter.sort(LMPDeviceType(), d_num_bond);
  Sorter.sort(LMPDeviceType(), d_bond_type);
  Sorter.sort(LMPDeviceType(), d_bond_atom);
  Sorter.sort(LMPDeviceType(), d_nspecial);
  Sorter.sort(LMPDeviceType(), d_special);
  Sorter.sort(LMPDeviceType(), d_num_angle);
  Sorter.sort(LMPDeviceType(), d_angle_type);
  Sorter.sort(LMPDeviceType(), d_angle_atom1);
  Sorter.sort(LMPDeviceType(), d_angle_atom2);
  Sorter.sort(LMPDeviceType(), d_angle_atom3);

  atomKK->modified(Device, ALL_MASK & ~F_MASK & ~TORQUE_MASK);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomemKokkos::sync(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (mask & X_MASK)      atomKK->k_x.sync_device();
    if (mask & V_MASK)      atomKK->k_v.sync_device();
    if (mask & F_MASK)      atomKK->k_f.sync_device();
    if (mask & TAG_MASK)    atomKK->k_tag.sync_device();
    if (mask & TYPE_MASK)   atomKK->k_type.sync_device();
    if (mask & MASK_MASK)   atomKK->k_mask.sync_device();
    if (mask & IMAGE_MASK)  atomKK->k_image.sync_device();
    if (mask & RADIUS_MASK) atomKK->k_radius.sync_device();
    if (mask & RMASS_MASK)  atomKK->k_rmass.sync_device();
    if (mask & OMEGA_MASK)  atomKK->k_omega.sync_device();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync_device();
    if (mask & Q_MASK)      atomKK->k_q.sync_device();
    if (mask & MU_MASK)     atomKK->k_mu.sync_device();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync_device();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.sync_device();
      atomKK->k_special.sync_device();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.sync_device();
      atomKK->k_bond_type.sync_device();
      atomKK->k_bond_atom.sync_device();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.sync_device();
      atomKK->k_angle_type.sync_device();
      atomKK->k_angle_atom1.sync_device();
      atomKK->k_angle_atom2.sync_device();
      atomKK->k_angle_atom3.sync_device();
    }
  } else if (space == Host) {
    if (mask & X_MASK)      atomKK->k_x.sync_host();
    if (mask & V_MASK)      atomKK->k_v.sync_host();
    if (mask & F_MASK)      atomKK->k_f.sync_host();
    if (mask & TAG_MASK)    atomKK->k_tag.sync_host();
    if (mask & TYPE_MASK)   atomKK->k_type.sync_host();
    if (mask & MASK_MASK)   atomKK->k_mask.sync_host();
    if (mask & IMAGE_MASK)  atomKK->k_image.sync_host();
    if (mask & RADIUS_MASK) atomKK->k_radius.sync_host();
    if (mask & RMASS_MASK)  atomKK->k_rmass.sync_host();
    if (mask & OMEGA_MASK)  atomKK->k_omega.sync_host();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync_host();
    if (mask & Q_MASK)      atomKK->k_q.sync_host();
    if (mask & MU_MASK)     atomKK->k_mu.sync_host();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync_host();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.sync_host();
      atomKK->k_special.sync_host();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.sync_host();
      atomKK->k_bond_type.sync_host();
      atomKK->k_bond_atom.sync_host();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.sync_host();
      atomKK->k_angle_type.sync_host();
      atomKK->k_angle_atom1.sync_host();
      atomKK->k_angle_atom2.sync_host();
      atomKK->k_angle_atom3.sync_host();
    }
  } else if (space == HostKK) {
    if (mask & X_MASK)      atomKK->k_x.sync_hostkk();
    if (mask & V_MASK)      atomKK->k_v.sync_hostkk();
    if (mask & F_MASK)      atomKK->k_f.sync_hostkk();
    if (mask & TAG_MASK)    atomKK->k_tag.sync_host();
    if (mask & TYPE_MASK)   atomKK->k_type.sync_host();
    if (mask & MASK_MASK)   atomKK->k_mask.sync_host();
    if (mask & IMAGE_MASK)  atomKK->k_image.sync_host();
    if (mask & RADIUS_MASK) atomKK->k_radius.sync_hostkk();
    if (mask & RMASS_MASK)  atomKK->k_rmass.sync_hostkk();
    if (mask & OMEGA_MASK)  atomKK->k_omega.sync_hostkk();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync_hostkk();
    if (mask & Q_MASK)      atomKK->k_q.sync_hostkk();
    if (mask & MU_MASK)     atomKK->k_mu.sync_hostkk();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync_host();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.sync_hostkk();
      atomKK->k_special.sync_hostkk();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.sync_host();
      atomKK->k_bond_type.sync_hostkk();
      atomKK->k_bond_atom.sync_hostkk();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.sync_host();
      atomKK->k_angle_type.sync_hostkk();
      atomKK->k_angle_atom1.sync_hostkk();
      atomKK->k_angle_atom2.sync_hostkk();
      atomKK->k_angle_atom3.sync_hostkk();
    }
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomemKokkos::modified(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (mask & X_MASK)      atomKK->k_x.modify_device();
    if (mask & V_MASK)      atomKK->k_v.modify_device();
    if (mask & F_MASK)      atomKK->k_f.modify_device();
    if (mask & TAG_MASK)    atomKK->k_tag.modify_device();
    if (mask & TYPE_MASK)   atomKK->k_type.modify_device();
    if (mask & MASK_MASK)   atomKK->k_mask.modify_device();
    if (mask & IMAGE_MASK)  atomKK->k_image.modify_device();
    if (mask & RADIUS_MASK) atomKK->k_radius.modify_device();
    if (mask & RMASS_MASK)  atomKK->k_rmass.modify_device();
    if (mask & OMEGA_MASK)  atomKK->k_omega.modify_device();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify_device();
    if (mask & Q_MASK)      atomKK->k_q.modify_device();
    if (mask & MU_MASK)     atomKK->k_mu.modify_device();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify_device();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.modify_device();
      atomKK->k_special.modify_device();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.modify_device();
      atomKK->k_bond_type.modify_device();
      atomKK->k_bond_atom.modify_device();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.modify_device();
      atomKK->k_angle_type.modify_device();
      atomKK->k_angle_atom1.modify_device();
      atomKK->k_angle_atom2.modify_device();
      atomKK->k_angle_atom3.modify_device();
    }
  } else if (space == Host) {
    if (mask & X_MASK)      atomKK->k_x.modify_host();
    if (mask & V_MASK)      atomKK->k_v.modify_host();
    if (mask & F_MASK)      atomKK->k_f.modify_host();
    if (mask & TAG_MASK)    atomKK->k_tag.modify_host();
    if (mask & TYPE_MASK)   atomKK->k_type.modify_host();
    if (mask & MASK_MASK)   atomKK->k_mask.modify_host();
    if (mask & IMAGE_MASK)  atomKK->k_image.modify_host();
    if (mask & RADIUS_MASK) atomKK->k_radius.modify_host();
    if (mask & RMASS_MASK)  atomKK->k_rmass.modify_host();
    if (mask & OMEGA_MASK)  atomKK->k_omega.modify_host();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify_host();
    if (mask & Q_MASK)      atomKK->k_q.modify_host();
    if (mask & MU_MASK)     atomKK->k_mu.modify_host();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify_host();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.modify_host();
      atomKK->k_special.modify_host();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.modify_host();
      atomKK->k_bond_type.modify_host();
      atomKK->k_bond_atom.modify_host();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.modify_host();
      atomKK->k_angle_type.modify_host();
      atomKK->k_angle_atom1.modify_host();
      atomKK->k_angle_atom2.modify_host();
      atomKK->k_angle_atom3.modify_host();
    }
  } else if (space == HostKK) {
    if (mask & X_MASK)      atomKK->k_x.modify_hostkk();
    if (mask & V_MASK)      atomKK->k_v.modify_hostkk();
    if (mask & F_MASK)      atomKK->k_f.modify_hostkk();
    if (mask & TAG_MASK)    atomKK->k_tag.modify_host();
    if (mask & TYPE_MASK)   atomKK->k_type.modify_host();
    if (mask & MASK_MASK)   atomKK->k_mask.modify_host();
    if (mask & IMAGE_MASK)  atomKK->k_image.modify_host();
    if (mask & RADIUS_MASK) atomKK->k_radius.modify_hostkk();
    if (mask & RMASS_MASK)  atomKK->k_rmass.modify_hostkk();
    if (mask & OMEGA_MASK)  atomKK->k_omega.modify_hostkk();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify_hostkk();
    if (mask & Q_MASK)      atomKK->k_q.modify_hostkk();
    if (mask & MU_MASK)     atomKK->k_mu.modify_hostkk();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify_host();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.modify_hostkk();
      atomKK->k_special.modify_hostkk();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.modify_host();
      atomKK->k_bond_type.modify_hostkk();
      atomKK->k_bond_atom.modify_hostkk();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.modify_host();
      atomKK->k_angle_type.modify_hostkk();
      atomKK->k_angle_atom1.modify_hostkk();
      atomKK->k_angle_atom2.modify_hostkk();
      atomKK->k_angle_atom3.modify_hostkk();
    }
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomemKokkos::sync_pinned(ExecutionSpace space, uint64_t mask,
                                                  int async_flag)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3_lr>(atomKK->k_x,space,async_flag);
    if ((mask & V_MASK) && atomKK->k_v.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_v,space,async_flag);
    if ((mask & F_MASK) && atomKK->k_f.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_f,space,async_flag);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync_device())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space,async_flag);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync_device())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_type,space,async_flag);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync_device())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_mask,space,async_flag);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync_device())
      perform_pinned_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space,async_flag);
    if ((mask & RADIUS_MASK) && atomKK->k_radius.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_radius,space,async_flag);
    if ((mask & RMASS_MASK) && atomKK->k_rmass.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_rmass,space,async_flag);
    if ((mask & OMEGA_MASK) && atomKK->k_omega.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_omega,space,async_flag);
    if ((mask & TORQUE_MASK) && atomKK->k_torque.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_torque,space,async_flag);
    if ((mask & Q_MASK) && atomKK->k_q.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_q,space,async_flag);
    if ((mask & MU_MASK) && atomKK->k_mu.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_4>(atomKK->k_mu,space,async_flag);
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync_device())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space,async_flag);
    if (mask & SPECIAL_MASK) {
      if (atomKK->k_nspecial.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_int_2d>(atomKK->k_nspecial,space,async_flag);
      if (atomKK->k_special.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_special,space,async_flag);
    }
    if (mask & BOND_MASK) {
      if (atomKK->k_num_bond.need_sync_device())
        perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_num_bond,space,async_flag);
      if (atomKK->k_bond_type.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_int_2d>(atomKK->k_bond_type,space,async_flag);
      if (atomKK->k_bond_atom.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_bond_atom,space,async_flag);
    }
    if (mask & ANGLE_MASK) {
      if (atomKK->k_num_angle.need_sync_device())
        perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_num_angle,space,async_flag);
      if (atomKK->k_angle_type.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_int_2d>(atomKK->k_angle_type,space,async_flag);
      if (atomKK->k_angle_atom1.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_angle_atom1,space,async_flag);
      if (atomKK->k_angle_atom2.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_angle_atom2,space,async_flag);
      if (atomKK->k_angle_atom3.need_sync_device())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_angle_atom3,space,async_flag);
    }
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3_lr>(atomKK->k_x,space,async_flag);
    if ((mask & V_MASK) && atomKK->k_v.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_v,space,async_flag);
    if ((mask & F_MASK) && atomKK->k_f.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_f,space,async_flag);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync_host())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space,async_flag);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync_host())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_type,space,async_flag);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync_host())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_mask,space,async_flag);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync_host())
      perform_pinned_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space,async_flag);
    if ((mask & RADIUS_MASK) && atomKK->k_radius.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_radius,space,async_flag);
    if ((mask & RMASS_MASK) && atomKK->k_rmass.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_rmass,space,async_flag);
    if ((mask & OMEGA_MASK) && atomKK->k_omega.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_omega,space,async_flag);
    if ((mask & TORQUE_MASK) && atomKK->k_torque.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_torque,space,async_flag);
    if ((mask & Q_MASK) && atomKK->k_q.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_q,space,async_flag);
    if ((mask & MU_MASK) && atomKK->k_mu.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_4>(atomKK->k_mu,space,async_flag);
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync_host())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space,async_flag);
    if (mask & SPECIAL_MASK) {
      if (atomKK->k_nspecial.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_int_2d>(atomKK->k_nspecial,space,async_flag);
      if (atomKK->k_special.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_special,space,async_flag);
    }
    if (mask & BOND_MASK) {
      if (atomKK->k_num_bond.need_sync_host())
        perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_num_bond,space,async_flag);
      if (atomKK->k_bond_type.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_int_2d>(atomKK->k_bond_type,space,async_flag);
      if (atomKK->k_bond_atom.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_bond_atom,space,async_flag);
    }
    if (mask & ANGLE_MASK) {
      if (atomKK->k_num_angle.need_sync_host())
        perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_num_angle,space,async_flag);
      if (atomKK->k_angle_type.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_int_2d>(atomKK->k_angle_type,space,async_flag);
      if (atomKK->k_angle_atom1.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_angle_atom1,space,async_flag);
      if (atomKK->k_angle_atom2.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_angle_atom2,space,async_flag);
      if (atomKK->k_angle_atom3.need_sync_host())
        perform_pinned_copy_transform<DAT::ttransform_tagint_2d>(atomKK->k_angle_atom3,space,async_flag);
    }
  }
}
