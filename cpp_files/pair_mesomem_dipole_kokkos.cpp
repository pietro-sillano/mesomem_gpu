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

// Contributing author: Pietro Sillano, 2026

#include "pair_mesomem_dipole_kokkos.h"

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
using MathConst::MY_PI2;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMesomemDipoleKokkos<DeviceType>::PairMesomemDipoleKokkos(LAMMPS *lmp) : PairMesomemDipole(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TORQUE_MASK | TYPE_MASK | MU_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMesomemDipoleKokkos<DeviceType>::~PairMesomemDipoleKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMesomemDipoleKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | TORQUE_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  mu = atomKK->k_mu.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  for (int k = 0; k < 4; k++) special_lj[k] = static_cast<KK_FLOAT>(force->special_lj[k]);
  newton_pair = force->newton_pair;

  // get the neighbor list and neighbors used in operator()

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  int inum = list->inum;

  // loop over neighbors of my atoms

  copymode = 1;

  EV_FLOAT ev;

  // compute kernel NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS

#define MESOMEM_DIPOLE_LAUNCH(NEIGH,STACK)                                                          \
  if (evflag) {                                                                                    \
    if (newton_pair)                                                                               \
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,                                      \
                              TagPairMesomemDipole<NEIGH,1,1,STACK>>(0,inum),*this,ev);           \
    else                                                                                           \
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,                                      \
                              TagPairMesomemDipole<NEIGH,0,1,STACK>>(0,inum),*this,ev);           \
  } else {                                                                                         \
    if (newton_pair)                                                                               \
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,                                         \
                           TagPairMesomemDipole<NEIGH,1,0,STACK>>(0,inum),*this);                 \
    else                                                                                           \
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,                                         \
                           TagPairMesomemDipole<NEIGH,0,0,STACK>>(0,inum),*this);                 \
  }

  if (atom->ntypes > MAX_TYPES_STACKPARAMS) {
    if (neighflag == HALF) { MESOMEM_DIPOLE_LAUNCH(HALF,false) }
    else if (neighflag == HALFTHREAD) { MESOMEM_DIPOLE_LAUNCH(HALFTHREAD,false) }
    else { MESOMEM_DIPOLE_LAUNCH(FULL,false) }
  } else {
    if (neighflag == HALF) { MESOMEM_DIPOLE_LAUNCH(HALF,true) }
    else if (neighflag == HALFTHREAD) { MESOMEM_DIPOLE_LAUNCH(HALFTHREAD,true) }
    else { MESOMEM_DIPOLE_LAUNCH(FULL,true) }
  }

#undef MESOMEM_DIPOLE_LAUNCH

  if (eflag_global) eng_vdwl += static_cast<double>(ev.evdwl);
  if (vflag_global) {
    virial[0] += static_cast<double>(ev.v[0]);
    virial[1] += static_cast<double>(ev.v[1]);
    virial[2] += static_cast<double>(ev.v[2]);
    virial[3] += static_cast<double>(ev.v[3]);
    virial[4] += static_cast<double>(ev.v[4]);
    virial[5] += static_cast<double>(ev.v[5]);
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
   per-atom kernel, same physics as PairMesomemDipole::compute()
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void PairMesomemDipoleKokkos<DeviceType>::operator()(TagPairMesomemDipole<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
                                                     const int ii, EV_FLOAT &ev) const
{
  // The f and torque arrays are atomic for Half/Thread neighbor style
  Kokkos::View<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;
  Kokkos::View<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_torque = torque;

  const KK_FLOAT zero = static_cast<KK_FLOAT>(0.0);
  const KK_FLOAT one = static_cast<KK_FLOAT>(1.0);
  const KK_FLOAT half = static_cast<KK_FLOAT>(0.5);
  const KK_FLOAT two = static_cast<KK_FLOAT>(2.0);

  const int i = d_ilist[ii];
  const KK_FLOAT xtmp = x(i,0);
  const KK_FLOAT ytmp = x(i,1);
  const KK_FLOAT ztmp = x(i,2);
  const int itype = type(i);
  const int jnum = d_numneigh[i];

  const KK_FLOAT mui = mu(i,3);
  KK_FLOAT nix = zero, niy = zero, niz = zero;
  if (mui > zero) {
    const KK_FLOAT inv_mag_i = one / mui;
    nix = mu(i,0) * inv_mag_i;
    niy = mu(i,1) * inv_mag_i;
    niz = mu(i,2) * inv_mag_i;
  }

  KK_ACC_FLOAT fx_i = 0.0;
  KK_ACC_FLOAT fy_i = 0.0;
  KK_ACC_FLOAT fz_i = 0.0;
  KK_ACC_FLOAT torquex_i = 0.0;
  KK_ACC_FLOAT torquey_i = 0.0;
  KK_ACC_FLOAT torquez_i = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i,jj);
    const KK_FLOAT factor_lj = special_lj[sbmask(j)];
    j &= NEIGHMASK;
    const int jtype = type(j);

    const KK_FLOAT delx = xtmp - x(j,0);
    const KK_FLOAT dely = ytmp - x(j,1);
    const KK_FLOAT delz = ztmp - x(j,2);
    const KK_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    const KK_FLOAT cutsq_ij = STACKPARAMS ? m_cutsq[itype][jtype] : d_cutsq(itype,jtype);
    if (rsq >= cutsq_ij) continue;

    const params_mesomem_dipole &p = STACKPARAMS ? m_params[itype][jtype] : params(itype,jtype);

    const KK_FLOAT r = Kokkos::sqrt(rsq);
    const KK_FLOAT inv_r = one / r;
    const KK_FLOAT rhatx = delx * inv_r;
    const KK_FLOAT rhaty = dely * inv_r;
    const KK_FLOAT rhatz = delz * inv_r;

    // --- 1. Isotropic (LJ/Cos) Force ---

    const KK_FLOAT eps_val = p.eps;
    const KK_FLOAT rmin = p.sigma;
    KK_FLOAT Ulj, eps_lj;

    if (r < rmin) {
      const KK_FLOAT t = rmin * inv_r;
      const KK_FLOAT t2 = t * t;
      const KK_FLOAT t4 = t2 * t2;
      Ulj = eps_val * (t4 - two * t2);
      eps_lj = static_cast<KK_FLOAT>(4.0) * eps_val * inv_r * (t4 - t2);
    } else {
      const KK_FLOAT denom = one / (p.cut - rmin);
      const KK_FLOAT g = static_cast<KK_FLOAT>(MY_PI2) * (r - rmin) * denom;
      const KK_FLOAT cos_t = Kokkos::cos(g);
      const KK_FLOAT sin_t = Kokkos::sin(g);

      // cos_t^(2*zeta-1), with a multiplication loop for integer zeta
      KK_FLOAT cos_pow;
      if (p.zeta_int > 0) {
        cos_pow = one;
        for (int k = 0; k < 2 * p.zeta_int - 1; k++) cos_pow *= cos_t;
      } else {
        cos_pow = Kokkos::pow(cos_t, two * p.zeta - one);
      }
      const KK_FLOAT cos_2zt = cos_pow * cos_t;

      Ulj = -eps_val * cos_2zt;

      // dU/dg * dg/dr
      const KK_FLOAT dU_dg = eps_val * (two * p.zeta) * cos_pow * sin_t;
      const KK_FLOAT dg_dr = static_cast<KK_FLOAT>(MY_PI2) * denom;
      eps_lj = -dU_dg * dg_dr;
    }

    KK_FLOAT fx = eps_lj * rhatx;
    KK_FLOAT fy = eps_lj * rhaty;
    KK_FLOAT fz = eps_lj * rhatz;
    KK_FLOAT tx = zero, ty = zero, tz = zero;
    KK_FLOAT tx_j = zero, ty_j = zero, tz_j = zero;
    KK_FLOAT evdwl = Ulj;

    // --- 2. Anisotropic (Tilt/Splay) Force ---

    const KK_FLOAT wr = p.wc;
    const KK_FLOAT muj = mu(j,3);

    if (r < wr && mui > zero && muj > zero) {

      // --- A. Weight Calculation ---
      const KK_FLOAT rga = half * wr;
      const KK_FLOAT r_wr = r / wr;

      // D = (r/wc)^4
      const KK_FLOAT r_wr_2 = r_wr * r_wr;
      const KK_FLOAT r_wr_4 = r_wr_2 * r_wr_2;
      const KK_FLOAT denom_w = r_wr_4 - one;    // always negative for r < wr

      // w is mathematically 0.0 close to the singularity at r = wr
      KK_FLOAT w = zero;
      const KK_FLOAT rga_sq = rga * rga;
      if (denom_w < static_cast<KK_FLOAT>(-1e-14)) w = Kokkos::exp(rsq / (rga_sq * denom_w));

      // --- B. Vector Normalization ---
      const KK_FLOAT inv_mag_j = one / muj;
      const KK_FLOAT njx = mu(j,0) * inv_mag_j;
      const KK_FLOAT njy = mu(j,1) * inv_mag_j;
      const KK_FLOAT njz = mu(j,2) * inv_mag_j;

      // --- C. Dot Products ---
      const KK_FLOAT nirhat = nix*rhatx + niy*rhaty + niz*rhatz;
      const KK_FLOAT njrhat = njx*rhatx + njy*rhaty + njz*rhatz;
      const KK_FLOAT ninj = nix*njx + niy*njy + niz*njz;

      // --- D. Tilt Calculation ---
      // rhat x ni and rhat x nj
      const KK_FLOAT rh_x_nix = rhaty*niz - rhatz*niy;
      const KK_FLOAT rh_x_niy = rhatz*nix - rhatx*niz;
      const KK_FLOAT rh_x_niz = rhatx*niy - rhaty*nix;
      const KK_FLOAT rh_x_njx = rhaty*njz - rhatz*njy;
      const KK_FLOAT rh_x_njy = rhatz*njx - rhatx*njz;
      const KK_FLOAT rh_x_njz = rhatx*njy - rhaty*njx;

      const KK_FLOAT c0 = p.c0;
      const KK_FLOAT sin_a2 = half * r * c0;
      const KK_FLOAT kt = p.ktilt;

      const KK_FLOAT diff_i = nirhat + sin_a2;
      const KK_FLOAT diff_j = njrhat - sin_a2;

      const KK_FLOAT Utilt = half * kt * (diff_i*diff_i + diff_j*diff_j);

      // Vector u = n - (n.r)r
      const KK_FLOAT uix = nix - nirhat*rhatx;
      const KK_FLOAT uiy = niy - nirhat*rhaty;
      const KK_FLOAT uiz = niz - nirhat*rhatz;
      const KK_FLOAT ujx = njx - njrhat*rhatx;
      const KK_FLOAT ujy = njy - njrhat*rhaty;
      const KK_FLOAT ujz = njz - njrhat*rhatz;

      // Tilt force (angular part only), scaled by weight
      const KK_FLOAT ft_pref = -kt * inv_r * w;
      fx += ft_pref * (diff_i*uix + diff_j*ujx);
      fy += ft_pref * (diff_i*uiy + diff_j*ujy);
      fz += ft_pref * (diff_i*uiz + diff_j*ujz);

      // --- E. Splay Calculation ---
      const KK_FLOAT ks = p.ksplay;
      const KK_FLOAT ni_x_njx = niy*njz - niz*njy;
      const KK_FLOAT ni_x_njy = niz*njx - nix*njz;
      const KK_FLOAT ni_x_njz = nix*njy - niy*njx;

      const KK_FLOAT splay_arg = ninj - one + two * sin_a2 * sin_a2;
      const KK_FLOAT Usplay = half * ks * splay_arg * splay_arg;

      // --- F. Radial Contribution deriving from - (Utilt + Usplay) * (dw/dr) ---
      const KK_FLOAT U_ang_sum = Utilt + Usplay;

      if (w > zero) {
        // Factor = w * [ 2 * (D+1) ] / [ rga^2 * (D-1)^2 ]
        const KK_FLOAT rad_numerator = two * w * (r_wr_4 + one) * r;
        const KK_FLOAT rad_denominator = rga_sq * denom_w * denom_w;
        KK_FLOAT f_rad = U_ang_sum * (rad_numerator / rad_denominator);

        // radial parts coming from the r dependence of sin_a2 in tilt and splay
        f_rad += w * half * kt * c0 * (diff_j - diff_i);
        f_rad -= w * ks * splay_arg * (c0 * c0 * r);

        fx += f_rad * rhatx;
        fy += f_rad * rhaty;
        fz += f_rad * rhatz;
      }

      // --- G. Torques ---
      const KK_FLOAT splay_pref = ks * splay_arg;

      tx = w * (kt * diff_i * rh_x_nix - splay_pref * ni_x_njx);
      ty = w * (kt * diff_i * rh_x_niy - splay_pref * ni_x_njy);
      tz = w * (kt * diff_i * rh_x_niz - splay_pref * ni_x_njz);

      tx_j = w * (kt * diff_j * rh_x_njx + splay_pref * ni_x_njx);
      ty_j = w * (kt * diff_j * rh_x_njy + splay_pref * ni_x_njy);
      tz_j = w * (kt * diff_j * rh_x_njz + splay_pref * ni_x_njz);

      evdwl += w * U_ang_sum;
    }

    // --- FINAL APPLY ---

    fx *= factor_lj;
    fy *= factor_lj;
    fz *= factor_lj;

    fx_i += static_cast<KK_ACC_FLOAT>(fx);
    fy_i += static_cast<KK_ACC_FLOAT>(fy);
    fz_i += static_cast<KK_ACC_FLOAT>(fz);
    torquex_i += static_cast<KK_ACC_FLOAT>(factor_lj*tx);
    torquey_i += static_cast<KK_ACC_FLOAT>(factor_lj*ty);
    torquez_i += static_cast<KK_ACC_FLOAT>(factor_lj*tz);

    if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < nlocal)) {
      a_f(j,0) -= static_cast<KK_ACC_FLOAT>(fx);
      a_f(j,1) -= static_cast<KK_ACC_FLOAT>(fy);
      a_f(j,2) -= static_cast<KK_ACC_FLOAT>(fz);
      a_torque(j,0) += static_cast<KK_ACC_FLOAT>(factor_lj*tx_j);
      a_torque(j,1) += static_cast<KK_ACC_FLOAT>(factor_lj*ty_j);
      a_torque(j,2) += static_cast<KK_ACC_FLOAT>(factor_lj*tz_j);
    }

    if (EVFLAG) {
      evdwl *= factor_lj;
      if (eflag_global)
        ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<nlocal)))?static_cast<KK_ACC_FLOAT>(1.0):static_cast<KK_ACC_FLOAT>(0.5))*static_cast<KK_ACC_FLOAT>(evdwl);
      if (eflag_atom || vflag_either)
        ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, evdwl, fx, fy, fz, delx, dely, delz);
    }
  }

  a_f(i,0) += fx_i;
  a_f(i,1) += fy_i;
  a_f(i,2) += fz_i;
  a_torque(i,0) += torquex_i;
  a_torque(i,1) += torquey_i;
  a_torque(i,2) += torquez_i;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void PairMesomemDipoleKokkos<DeviceType>::operator()(TagPairMesomemDipole<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
                                                     const int ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>(TagPairMesomemDipole<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairMesomemDipoleKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, int i, int j, const KK_FLOAT &epair,
                                                       KK_FLOAT fx, KK_FLOAT fy, KK_FLOAT fz,
                                                       KK_FLOAT delx, KK_FLOAT dely, KK_FLOAT delz) const
{
  Kokkos::View<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = d_eatom;
  Kokkos::View<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = d_vatom;

  const KK_ACC_FLOAT half = static_cast<KK_ACC_FLOAT>(0.5);

  if (eflag_atom) {
    const KK_ACC_FLOAT epairhalf = half * static_cast<KK_ACC_FLOAT>(epair);
    if (NEIGHFLAG == FULL || newton_pair || i < nlocal) v_eatom[i] += epairhalf;
    if (NEIGHFLAG != FULL && (newton_pair || j < nlocal)) v_eatom[j] += epairhalf;
  }

  if (vflag_either) {
    KK_ACC_FLOAT v[6];
    v[0] = static_cast<KK_ACC_FLOAT>(delx*fx);
    v[1] = static_cast<KK_ACC_FLOAT>(dely*fy);
    v[2] = static_cast<KK_ACC_FLOAT>(delz*fz);
    v[3] = static_cast<KK_ACC_FLOAT>(delx*fy);
    v[4] = static_cast<KK_ACC_FLOAT>(delx*fz);
    v[5] = static_cast<KK_ACC_FLOAT>(dely*fz);

    if (vflag_global) {
      if (NEIGHFLAG != FULL) {
        if (NEWTON_PAIR) { // neigh half, newton on
          for (int k = 0; k < 6; k++) ev.v[k] += v[k];
        } else { // neigh half, newton off
          if (i < nlocal)
            for (int k = 0; k < 6; k++) ev.v[k] += half*v[k];
          if (j < nlocal)
            for (int k = 0; k < 6; k++) ev.v[k] += half*v[k];
        }
      } else { // neigh full
        for (int k = 0; k < 6; k++) ev.v[k] += half*v[k];
      }
    }

    if (vflag_atom) {
      if (NEIGHFLAG == FULL || NEWTON_PAIR || i < nlocal)
        for (int k = 0; k < 6; k++) v_vatom(i,k) += half*v[k];
      if (NEIGHFLAG != FULL && (NEWTON_PAIR || j < nlocal))
        for (int k = 0; k < 6; k++) v_vatom(j,k) += half*v[k];
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMesomemDipoleKokkos<DeviceType>::allocate()
{
  PairMesomemDipole::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  k_params = Kokkos::DualView<params_mesomem_dipole**,Kokkos::LayoutRight,DeviceType>("PairMesomemDipole::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairMesomemDipoleKokkos<DeviceType>::init_style()
{
  PairMesomemDipole::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairMesomemDipoleKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairMesomemDipole::init_one(i,j);

  // use the multiplication loop in the kernel when zeta is a (small) positive integer

  int zeta_int = -1;
  const double zeta_ij = zeta[i][j];
  if ((zeta_ij >= 1.0) && (zeta_ij <= 64.0) && (zeta_ij == std::round(zeta_ij)))
    zeta_int = static_cast<int>(zeta_ij);

  k_params.view_host()(i,j).cut = static_cast<KK_FLOAT>(cut[i][j]);
  k_params.view_host()(i,j).sigma = static_cast<KK_FLOAT>(sigma[i][j]);
  k_params.view_host()(i,j).eps = static_cast<KK_FLOAT>(eps[i][j]);
  k_params.view_host()(i,j).ktilt = static_cast<KK_FLOAT>(ktilt[i][j]);
  k_params.view_host()(i,j).ksplay = static_cast<KK_FLOAT>(ksplay[i][j]);
  k_params.view_host()(i,j).wc = static_cast<KK_FLOAT>(weight_rcut[i][j]);
  k_params.view_host()(i,j).zeta = static_cast<KK_FLOAT>(zeta_ij);
  k_params.view_host()(i,j).c0 = static_cast<KK_FLOAT>(c0[i][j]);
  k_params.view_host()(i,j).zeta_int = zeta_int;

  k_params.view_host()(j,i) = k_params.view_host()(i,j);
  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.view_host()(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = static_cast<KK_FLOAT>(cutone*cutone);
  }

  k_cutsq.view_host()(i,j) = k_cutsq.view_host()(j,i) = cutone*cutone;
  k_cutsq.modify_host();
  k_params.modify_host();

  return cutone;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairMesomemDipoleKokkos<DeviceType>::sbmask(const int& j) const
{
  return j >> SBBITS & 3;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairMesomemDipoleKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMesomemDipoleKokkos<LMPHostType>;
#endif
}
