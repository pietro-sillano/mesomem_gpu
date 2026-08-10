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

#include "atom_vec_mesomem.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "math_const.h"
#include "memory.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

AtomVecMesomem::AtomVecMesomem(LAMMPS *lmp) : AtomVec(lmp)
{
  mass_type = PER_ATOM;
  molecular = Atom::MOLECULAR;
  bonds_allow = angles_allow = 1;

  atom->q_flag = atom->mu_flag = 1;
  atom->radius_flag = atom->rmass_flag = 1;
  atom->omega_flag = atom->torque_flag = 1;
  atom->molecule_flag = 1;

  mu_hold = nullptr;

  fields_grow = {"q", "mu", "radius", "rmass", "omega", "torque",
                 "molecule", "num_bond", "bond_type", "bond_atom",
                 "num_angle", "angle_type", "angle_atom1", "angle_atom2",
                 "angle_atom3", "nspecial", "special"};
  fields_copy = {"q", "mu", "radius", "rmass", "omega",
                 "molecule", "num_bond", "bond_type", "bond_atom",
                 "num_angle", "angle_type", "angle_atom1", "angle_atom2",
                 "angle_atom3", "nspecial", "special"};
  fields_comm       = {"mu3"};
  fields_comm_vel   = {"mu3", "omega"};
  fields_reverse    = {"torque"};
  fields_border     = {"q", "mu", "radius", "rmass", "molecule"};
  fields_border_vel = {"q", "mu", "radius", "rmass", "omega", "molecule"};
  fields_exchange   = {"q", "mu", "radius", "rmass", "omega",
                       "molecule", "num_bond", "bond_type", "bond_atom",
                       "num_angle", "angle_type", "angle_atom1", "angle_atom2",
                       "angle_atom3", "nspecial", "special"};
  fields_restart    = {"q", "mu", "radius", "rmass", "omega",
                       "molecule", "num_bond", "bond_type", "bond_atom",
                       "num_angle", "angle_type", "angle_atom1", "angle_atom2",
                       "angle_atom3"};
  fields_create     = {"q", "mu", "radius", "rmass", "omega",
                       "molecule", "num_bond", "num_angle", "nspecial"};
  // Data file column order (OVITO angle+sphere+dipole convention):
  //   atom-ID type x y z mol-ID diameter density q mux muy muz
  fields_data_atom  = {"id", "type", "x", "molecule", "radius", "rmass", "q", "mu3"};
  fields_data_vel   = {"id", "v", "omega"};

  setup_fields();

  bond_per_atom = angle_per_atom = 0;
  bond_negative = angle_negative = nullptr;
}

/* ---------------------------------------------------------------------- */

AtomVecMesomem::~AtomVecMesomem()
{
  delete[] bond_negative;
  delete[] angle_negative;
  if (mu_hold) memory->destroy(mu_hold);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::grow_pointers()
{
  // dipole_sphere raw pointers
  mu     = atom->mu;
  radius = atom->radius;
  rmass  = atom->rmass;
  omega  = atom->omega;

  // molecular raw pointers (used in restart pre/post)
  num_bond   = atom->num_bond;
  bond_type  = atom->bond_type;
  num_angle  = atom->num_angle;
  angle_type = atom->angle_type;
  nspecial   = atom->nspecial;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::create_atom_post(int ilocal)
{
  radius[ilocal] = 0.5;
  rmass[ilocal]  = 4.0 * MY_PI / 3.0 * 0.5 * 0.5 * 0.5;
}

/* ----------------------------------------------------------------------
   data_atom_post: dipole_sphere diameter/density conversion + mu magnitude
                   + zero molecular topology
------------------------------------------------------------------------- */

void AtomVecMesomem::data_atom_post(int ilocal)
{
  // sphere: input is diameter stored in radius, input is density stored in rmass
  radius_one = 0.5 * atom->radius[ilocal];
  radius[ilocal] = radius_one;
  if (radius_one > 0.0)
    rmass[ilocal] *= 4.0 * MY_PI / 3.0 * radius_one * radius_one * radius_one;

  if (rmass[ilocal] <= 0.0)
    error->one(FLERR, "Invalid density in Atoms section of data file");

  omega[ilocal][0] = omega[ilocal][1] = omega[ilocal][2] = 0.0;

  // dipole: compute scalar magnitude from vector components
  double *mu_one = mu[ilocal];
  mu_one[3] = sqrt(mu_one[0]*mu_one[0] + mu_one[1]*mu_one[1] + mu_one[2]*mu_one[2]);

  // molecular: zero topology (bonds/angles are added later from Bonds/Angles sections)
  num_bond[ilocal]    = 0;
  num_angle[ilocal]   = 0;
  nspecial[ilocal][0] = nspecial[ilocal][1] = nspecial[ilocal][2] = 0;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::pack_data_pre(int ilocal)
{
  radius_one = radius[ilocal];
  rmass_one  = rmass[ilocal];

  radius[ilocal] *= 2.0;
  if (radius_one != 0.0)
    rmass[ilocal] = rmass_one / (4.0 * MY_PI / 3.0 * radius_one * radius_one * radius_one);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::pack_data_post(int ilocal)
{
  radius[ilocal] = radius_one;
  rmass[ilocal]  = rmass_one;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::pack_restart_pre(int ilocal)
{
  if (bond_per_atom < atom->bond_per_atom) {
    delete[] bond_negative;
    bond_per_atom = atom->bond_per_atom;
    bond_negative = new int[bond_per_atom];
  }
  if (angle_per_atom < atom->angle_per_atom) {
    delete[] angle_negative;
    angle_per_atom = atom->angle_per_atom;
    angle_negative = new int[angle_per_atom];
  }

  any_bond_negative = 0;
  for (int m = 0; m < num_bond[ilocal]; m++) {
    if (bond_type[ilocal][m] < 0) {
      bond_negative[m] = 1;
      bond_type[ilocal][m] = -bond_type[ilocal][m];
      any_bond_negative = 1;
    } else
      bond_negative[m] = 0;
  }

  any_angle_negative = 0;
  for (int m = 0; m < num_angle[ilocal]; m++) {
    if (angle_type[ilocal][m] < 0) {
      angle_negative[m] = 1;
      angle_type[ilocal][m] = -angle_type[ilocal][m];
      any_angle_negative = 1;
    } else
      angle_negative[m] = 0;
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::pack_restart_post(int ilocal)
{
  if (any_bond_negative)
    for (int m = 0; m < num_bond[ilocal]; m++)
      if (bond_negative[m]) bond_type[ilocal][m] = -bond_type[ilocal][m];

  if (any_angle_negative)
    for (int m = 0; m < num_angle[ilocal]; m++)
      if (angle_negative[m]) angle_type[ilocal][m] = -angle_type[ilocal][m];
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::unpack_restart_init(int ilocal)
{
  nspecial[ilocal][0] = nspecial[ilocal][1] = nspecial[ilocal][2] = 0;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::read_data_general_to_restricted(int nlocal_previous, int nlocal)
{
  AtomVec::read_data_general_to_restricted(nlocal_previous, nlocal);

  for (int i = nlocal_previous; i < nlocal; i++)
    domain->general_to_restricted_vector(mu[i]);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::write_data_restricted_to_general()
{
  AtomVec::write_data_restricted_to_general();

  int nlocal = atom->nlocal;
  memory->create(mu_hold, nlocal, 3, "atomvec:mu_hold");
  for (int i = 0; i < nlocal; i++) {
    memcpy(&mu_hold[i][0], &mu[i][0], 3*sizeof(double));
    domain->restricted_to_general_vector(mu[i]);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMesomem::write_data_restore_restricted()
{
  AtomVec::write_data_restore_restricted();

  if (!mu_hold) return;

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    memcpy(&mu[i][0], &mu_hold[i][0], 3*sizeof(double));
  memory->destroy(mu_hold);
  mu_hold = nullptr;
}
