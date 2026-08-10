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

// Contributing author: Pietro Sillano (TU Delft), 2026
// Kokkos port of pair_membrane_sillano_v2
// Structurally modeled on pair_lj_cut_dipole_cut_kokkos.



#define INCLUDE_RADIAL

#include "pair_membrane_sillano_v2_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMembraneSillanov2Kokkos<DeviceType>::PairMembraneSillanov2Kokkos(LAMMPS *lmp) :
    PairMembraneSillanov2(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read  = X_MASK | F_MASK | TORQUE_MASK | TYPE_MASK | MU_MASK |
                   ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMembraneSillanov2Kokkos<DeviceType>::~PairMembraneSillanov2Kokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->destroy_kokkos(k_cutsq, cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMembraneSillanov2Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space, F_MASK | TORQUE_MASK);

  x      = atomKK->k_x.view<DeviceType>();
  f      = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  mu     = atomKK->k_mu.view<DeviceType>();
  type   = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall   = atom->nlocal + atom->nghost;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  newton_pair = force->newton_pair;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh  = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist     = k_list->d_ilist;
  int inum    = list->inum;

  copymode = 1;

  EV_FLOAT ev;

#define MEMBRANE_DISPATCH(STACK)                                                         \
  if (evflag) {                                                                          \
    if (neighflag == HALF) {                                                             \
      if (newton_pair) Kokkos::parallel_reduce(                                          \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALF,1,1,STACK>>(0,inum),    \
          *this, ev);                                                                    \
      else             Kokkos::parallel_reduce(                                          \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALF,0,1,STACK>>(0,inum),    \
          *this, ev);                                                                    \
    } else if (neighflag == HALFTHREAD) {                                                \
      if (newton_pair) Kokkos::parallel_reduce(                                          \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALFTHREAD,1,1,STACK>>(0,inum),\
          *this, ev);                                                                    \
      else             Kokkos::parallel_reduce(                                          \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALFTHREAD,0,1,STACK>>(0,inum),\
          *this, ev);                                                                    \
    } else {                                                                             \
      if (newton_pair) Kokkos::parallel_reduce(                                          \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<FULL,1,1,STACK>>(0,inum),    \
          *this, ev);                                                                    \
      else             Kokkos::parallel_reduce(                                          \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<FULL,0,1,STACK>>(0,inum),    \
          *this, ev);                                                                    \
    }                                                                                    \
  } else {                                                                               \
    if (neighflag == HALF) {                                                             \
      if (newton_pair) Kokkos::parallel_for(                                             \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALF,1,0,STACK>>(0,inum),    \
          *this);                                                                        \
      else             Kokkos::parallel_for(                                             \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALF,0,0,STACK>>(0,inum),    \
          *this);                                                                        \
    } else if (neighflag == HALFTHREAD) {                                                \
      if (newton_pair) Kokkos::parallel_for(                                             \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALFTHREAD,1,0,STACK>>(0,inum),\
          *this);                                                                        \
      else             Kokkos::parallel_for(                                             \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<HALFTHREAD,0,0,STACK>>(0,inum),\
          *this);                                                                        \
    } else {                                                                             \
      if (newton_pair) Kokkos::parallel_for(                                             \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<FULL,1,0,STACK>>(0,inum),    \
          *this);                                                                        \
      else             Kokkos::parallel_for(                                             \
          Kokkos::RangePolicy<DeviceType, TagPairMembraneV2<FULL,0,0,STACK>>(0,inum),    \
          *this);                                                                        \
    }                                                                                    \
  }

  if (atom->ntypes > MAX_TYPES_STACKPARAMS) {
    MEMBRANE_DISPATCH(false)
  } else {
    MEMBRANE_DISPATCH(true)
  }

#undef MEMBRANE_DISPATCH

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0]; virial[1] += ev.v[1]; virial[2] += ev.v[2];
    virial[3] += ev.v[3]; virial[4] += ev.v[4]; virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }
  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

/* ----------------------------------------------------------------------
   main per-atom kernel: i loops over its own neighbor list
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void PairMembraneSillanov2Kokkos<DeviceType>::operator()(
    TagPairMembraneV2<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
    const int ii, EV_FLOAT &ev) const
{
  // atomic views for NEIGHFLAG != FULL
  Kokkos::View<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>> a_f = f;
  Kokkos::View<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>> a_torque = torque;

  const int i = d_ilist[ii];
  const KK_FLOAT xtmp = x(i,0);
  const KK_FLOAT ytmp = x(i,1);
  const KK_FLOAT ztmp = x(i,2);
  const int itype = type(i);
  const int jnum = d_numneigh[i];

  // cache orientation of i
  const KK_FLOAT inv_mag_i = 1.0 / mu(i,3);
  const KK_FLOAT nix = mu(i,0) * inv_mag_i;
  const KK_FLOAT niy = mu(i,1) * inv_mag_i;
  const KK_FLOAT niz = mu(i,2) * inv_mag_i;

  // accumulators for atom i (non-atomic; flushed to a_f / a_torque at end)
  KK_FLOAT fxi = 0.0, fyi = 0.0, fzi = 0.0;
  KK_FLOAT txi = 0.0, tyi = 0.0, tzi = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    const KK_FLOAT factor_lj = special_lj[sbmask(j)];
    j &= NEIGHMASK;
    const int jtype = type(j);

    const KK_FLOAT delx = xtmp - x(j,0);
    const KK_FLOAT dely = ytmp - x(j,1);
    const KK_FLOAT delz = ztmp - x(j,2);
    const KK_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

    const KK_FLOAT cutsq_ij = STACKPARAMS ? m_cutsq[itype][jtype] : d_cutsq(itype,jtype);
    if (rsq >= cutsq_ij) continue;

    const params_membrane &p = STACKPARAMS ? m_params[itype][jtype] : params(itype,jtype);

    const KK_FLOAT r     = sqrt(rsq);
    const KK_FLOAT inv_r = 1.0 / r;
    const KK_FLOAT rhatx = delx * inv_r;
    const KK_FLOAT rhaty = dely * inv_r;
    const KK_FLOAT rhatz = delz * inv_r;

    // ----- 1. Isotropic LJ/cos branch -----
    const KK_FLOAT eps_val   = p.eps;
    const KK_FLOAT sigma_val = p.sigma;
    const KK_FLOAT rmin      = sigma_val;
    KK_FLOAT Ulj    = 0.0;
    KK_FLOAT eps_lj = 0.0;

    if (r < rmin) {
      const KK_FLOAT t  = sigma_val * inv_r;
      const KK_FLOAT t2 = t * t;
      const KK_FLOAT t4 = t2 * t2;
      Ulj    = eps_val * (t4 - 2.0 * t2);
      eps_lj = 4.0 * eps_val * inv_r * (t4 - t2);
    } else {
      const KK_FLOAT rcut  = sqrt(cutsq_ij);
      const int     zt_i  = p.zeta_int;           // integer zeta
      const KK_FLOAT denom = 1.0 / (rcut - rmin);
      const KK_FLOAT g     = MY_PI * 0.5 * (r - rmin) * denom;
      const KK_FLOAT cos_t = cos(g);
      const KK_FLOAT sin_t = sin(g);

      // integer-exponent specialization of pow(cos_t, 2*zeta - 1)
      const int n = 2 * zt_i - 1;
      KK_FLOAT cos_pow = 1.0;
      for (int k = 0; k < n; ++k) cos_pow *= cos_t;
      const KK_FLOAT cos_2zt = cos_pow * cos_t;

      Ulj = -eps_val * cos_2zt;
      const KK_FLOAT dU_dg = eps_val * (2.0 * (KK_FLOAT)zt_i) * cos_pow * sin_t;
      const KK_FLOAT dg_dr = MY_PI * 0.5 * denom;
      eps_lj = -dU_dg * dg_dr;
    }

    KK_FLOAT fx = eps_lj * rhatx;
    KK_FLOAT fy = eps_lj * rhaty;
    KK_FLOAT fz = eps_lj * rhatz;
    KK_FLOAT tx = 0.0, ty = 0.0, tz = 0.0;
    KK_FLOAT txj = 0.0, tyj = 0.0, tzj = 0.0;

    KK_FLOAT U_ang_sum = 0.0;
    KK_FLOAT w = 0.0;

    // ----- 2. Anisotropic (tilt / splay) branch -----
    const KK_FLOAT wr = p.wc;
    if (r < wr) {
      // weight w(r) = exp( r^2 / ( rga^2 * ((r/wr)^4 - 1) ) )
      const KK_FLOAT rga    = 0.5 * wr;
      const KK_FLOAT r_wr   = r / wr;
      const KK_FLOAT r_wr_2 = r_wr * r_wr;
      const KK_FLOAT r_wr_4 = r_wr_2 * r_wr_2;
      const KK_FLOAT denom_w = r_wr_4 - 1.0;   // negative for r < wr

      const KK_FLOAT rga_sq = rga * rga;
      if (denom_w < -1e-14) {
        const KK_FLOAT val_exp = (r * r) / (rga_sq * denom_w);
        w = exp(val_exp);
      }

      // orientation of j
      const KK_FLOAT inv_mag_j = 1.0 / mu(j,3);
      const KK_FLOAT njx = mu(j,0) * inv_mag_j;
      const KK_FLOAT njy = mu(j,1) * inv_mag_j;
      const KK_FLOAT njz = mu(j,2) * inv_mag_j;

      const KK_FLOAT nirhat = nix*rhatx + niy*rhaty + niz*rhatz;
      const KK_FLOAT njrhat = njx*rhatx + njy*rhaty + njz*rhatz;
      const KK_FLOAT ninj   = nix*njx   + niy*njy   + niz*njz;

      // rhat x ni
      const KK_FLOAT rxnix = rhaty*niz - rhatz*niy;
      const KK_FLOAT rxniy = rhatz*nix - rhatx*niz;
      const KK_FLOAT rxniz = rhatx*niy - rhaty*nix;
      // rhat x nj
      const KK_FLOAT rxnjx = rhaty*njz - rhatz*njy;
      const KK_FLOAT rxnjy = rhatz*njx - rhatx*njz;
      const KK_FLOAT rxnjz = rhatx*njy - rhaty*njx;

      const KK_FLOAT c0_ij  = p.c0;
      const KK_FLOAT sin_a2 = 0.5 * r * c0_ij;

      const KK_FLOAT kt = p.ktilt;
      const KK_FLOAT diff_i = nirhat + sin_a2;
      const KK_FLOAT diff_j = njrhat - sin_a2;

      const KK_FLOAT Utilt = 0.5 * kt * (diff_i*diff_i + diff_j*diff_j);

      // u = n - (n.r)r
      const KK_FLOAT uix = nix - nirhat*rhatx;
      const KK_FLOAT uiy = niy - nirhat*rhaty;
      const KK_FLOAT uiz = niz - nirhat*rhatz;
      const KK_FLOAT ujx = njx - njrhat*rhatx;
      const KK_FLOAT ujy = njy - njrhat*rhaty;
      const KK_FLOAT ujz = njz - njrhat*rhatz;

      const KK_FLOAT ft_pref = -kt * inv_r;
      const KK_FLOAT tilt_fx = ft_pref * (diff_i*uix + diff_j*ujx);
      const KK_FLOAT tilt_fy = ft_pref * (diff_i*uiy + diff_j*ujy);
      const KK_FLOAT tilt_fz = ft_pref * (diff_i*uiz + diff_j*ujz);

      fx += tilt_fx * w;
      fy += tilt_fy * w;
      fz += tilt_fz * w;

      // splay
      const KK_FLOAT ks = p.ksplay;
      const KK_FLOAT ninjx = niy*njz - niz*njy;
      const KK_FLOAT ninjy = niz*njx - nix*njz;
      const KK_FLOAT ninjz = nix*njy - niy*njx;

      const KK_FLOAT splay_arg = ninj - 1.0 + 2.0 * sin_a2 * sin_a2;
      const KK_FLOAT Usplay    = 0.5 * ks * splay_arg * splay_arg;

      U_ang_sum = Utilt + Usplay;

#ifdef INCLUDE_RADIAL
      if (w > 0.0) {
        // Radial correction from d/dr of weight w(r)
        const KK_FLOAT rad_num  = 2.0 * w * (r_wr_4 + 1.0) * r;
        const KK_FLOAT rad_den  = rga_sq * denom_w * denom_w;
        const KK_FLOAT f_rad    = U_ang_sum * (rad_num / rad_den);
        fx += f_rad * rhatx;
        fy += f_rad * rhaty;
        fz += f_rad * rhatz;

        // radial part from tilt (derivative of sin_a2 = 0.5*r*c0 wrt r)
        const KK_FLOAT f_rad_tilt = -0.5 * kt * (diff_i * c0_ij + diff_j * c0_ij);
        fx += w * f_rad_tilt * rhatx;
        fy += w * f_rad_tilt * rhaty;
        fz += w * f_rad_tilt * rhatz;

        // radial part from splay
        const KK_FLOAT f_rad_splay = -ks * splay_arg * (c0_ij * c0_ij * r);
        fx += w * f_rad_splay * rhatx;
        fy += w * f_rad_splay * rhaty;
        fz += w * f_rad_splay * rhatz;
      }
#endif

      // torques
      const KK_FLOAT splay_pref = ks * splay_arg;
      tx  += w * (kt * diff_i * rxnix + splay_pref * ninjx);
      ty  += w * (kt * diff_i * rxniy + splay_pref * ninjy);
      tz  += w * (kt * diff_i * rxniz + splay_pref * ninjz);

      txj += w * (kt * diff_j * rxnjx - splay_pref * ninjx);
      tyj += w * (kt * diff_j * rxnjy - splay_pref * ninjy);
      tzj += w * (kt * diff_j * rxnjz - splay_pref * ninjz);
    }

    // ----- FINAL per-pair: apply factor_lj and accumulate -----
    fx *= factor_lj;  fy *= factor_lj;  fz *= factor_lj;
    tx *= factor_lj;  ty *= factor_lj;  tz *= factor_lj;
    txj *= factor_lj; tyj *= factor_lj; tzj *= factor_lj;

    fxi += fx; fyi += fy; fzi += fz;
    txi += tx; tyi += ty; tzi += tz;

    if ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) &&
        (NEWTON_PAIR || j < nlocal)) {
      a_f(j,0) -= fx;
      a_f(j,1) -= fy;
      a_f(j,2) -= fz;
      a_torque(j,0) += txj;
      a_torque(j,1) += tyj;
      a_torque(j,2) += tzj;
    }

    if (EVFLAG) {
      KK_FLOAT evdwl = factor_lj * (Ulj + w * U_ang_sum);
      if (eflag_global) {
        const KK_FLOAT half_or_full =
            ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) &&
             (NEWTON_PAIR || j < nlocal)) ? 1.0 : 0.5;
        ev.evdwl += half_or_full * evdwl;
      }
      if (eflag_atom || vflag_either)
        ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, evdwl, fx, fy, fz,
                                            delx, dely, delz);
    }
  }

  a_f(i,0) += fxi;
  a_f(i,1) += fyi;
  a_f(i,2) += fzi;
  a_torque(i,0) += txi;
  a_torque(i,1) += tyi;
  a_torque(i,2) += tzi;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void PairMembraneSillanov2Kokkos<DeviceType>::operator()(
    TagPairMembraneV2<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
    const int ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>(
      TagPairMembraneV2<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairMembraneSillanov2Kokkos<DeviceType>::ev_tally_xyz(
    EV_FLOAT &ev, int i, int j, const KK_FLOAT &epair,
    KK_FLOAT fx, KK_FLOAT fy, KK_FLOAT fz,
    KK_FLOAT delx, KK_FLOAT dely, KK_FLOAT delz) const
{
  Kokkos::View<KK_ACC_FLOAT*, typename DAT::ttransform_kkacc_1d::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>> v_eatom =
      d_eatom;
  Kokkos::View<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>> v_vatom =
      d_vatom;

  if (eflag_atom) {
    const KK_ACC_FLOAT epairhalf = 0.5 * epair;
    if (NEIGHFLAG == FULL || newton_pair || i < nlocal) v_eatom[i] += epairhalf;
    if (NEIGHFLAG != FULL && (newton_pair || j < nlocal)) v_eatom[j] += epairhalf;
  }

  if (vflag_either) {
    const KK_FLOAT v0 = delx*fx;
    const KK_FLOAT v1 = dely*fy;
    const KK_FLOAT v2 = delz*fz;
    const KK_FLOAT v3 = delx*fy;
    const KK_FLOAT v4 = delx*fz;
    const KK_FLOAT v5 = dely*fz;

    if (vflag_global) {
      if (NEIGHFLAG != FULL) {
        if (NEWTON_PAIR) {
          ev.v[0] += v0; ev.v[1] += v1; ev.v[2] += v2;
          ev.v[3] += v3; ev.v[4] += v4; ev.v[5] += v5;
        } else {
          if (i < nlocal) {
            ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
            ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
          }
          if (j < nlocal) {
            ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
            ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
          }
        }
      } else {
        ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
      }
    }

    if (vflag_atom) {
      if (NEIGHFLAG == FULL || NEWTON_PAIR || i < nlocal) {
        v_vatom(i,0) += 0.5*v0; v_vatom(i,1) += 0.5*v1; v_vatom(i,2) += 0.5*v2;
        v_vatom(i,3) += 0.5*v3; v_vatom(i,4) += 0.5*v4; v_vatom(i,5) += 0.5*v5;
      }
      if (NEIGHFLAG != FULL && (NEWTON_PAIR || j < nlocal)) {
        v_vatom(j,0) += 0.5*v0; v_vatom(j,1) += 0.5*v1; v_vatom(j,2) += 0.5*v2;
        v_vatom(j,3) += 0.5*v3; v_vatom(j,4) += 0.5*v4; v_vatom(j,5) += 0.5*v5;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMembraneSillanov2Kokkos<DeviceType>::allocate()
{
  PairMembraneSillanov2::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq, cutsq, n+1, n+1, "pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  k_params = Kokkos::DualView<params_membrane**, Kokkos::LayoutRight, DeviceType>(
      "PairMembraneSillanov2::params", n+1, n+1);
  params = k_params.template view<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMembraneSillanov2Kokkos<DeviceType>::init_style()
{
  PairMembraneSillanov2::init_style();

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa) error->all(FLERR, "Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS
  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PairMembraneSillanov2Kokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairMembraneSillanov2::init_one(i, j);

  // zeta must be a positive integer for the device kernel; warn & round.
  double zeta_ij = zeta[i][j];
  int    zeta_int = (int) std::round(zeta_ij);
  if (std::fabs(zeta_ij - (double)zeta_int) > 1e-12) {
    error->warning(FLERR,
        "membrane_sillanov2/kk assumes integer zeta; rounding input to nearest int");
  }
  if (zeta_int < 1) {
    error->all(FLERR, "membrane_sillanov2/kk requires zeta >= 1");
  }

  k_params.view_host()(i,j).cutsq    = cutone * cutone;
  k_params.view_host()(i,j).sigma    = sigma[i][j];
  k_params.view_host()(i,j).eps      = eps[i][j];
  k_params.view_host()(i,j).ktilt    = ktilt[i][j];
  k_params.view_host()(i,j).ksplay   = ksplay[i][j];
  k_params.view_host()(i,j).wc       = weight_rcut[i][j];
  k_params.view_host()(i,j).c0       = c0[i][j];
  k_params.view_host()(i,j).zeta_int = zeta_int;
  k_params.view_host()(j,i) = k_params.view_host()(i,j);

  if (i < MAX_TYPES_STACKPARAMS+1 && j < MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.view_host()(i,j);
    m_cutsq[i][j]  = m_cutsq[j][i]  = cutone * cutone;
  }

  k_cutsq.view_host()(i,j) = k_cutsq.view_host()(j,i) = cutone * cutone;
  k_cutsq.modify_host();
  k_params.modify_host();

  return cutone;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairMembraneSillanov2Kokkos<DeviceType>::sbmask(const int &j) const
{
  return j >> SBBITS & 3;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairMembraneSillanov2Kokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMembraneSillanov2Kokkos<LMPHostType>;
#endif
}
