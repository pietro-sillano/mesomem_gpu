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

// Contributing author: Pietro Sillano, 2026

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mesomem/dipole/kk,PairMesomemDipoleKokkos<LMPDeviceType>);
PairStyle(mesomem/dipole/kk/device,PairMesomemDipoleKokkos<LMPDeviceType>);
PairStyle(mesomem/dipole/kk/host,PairMesomemDipoleKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_MESOMEM_DIPOLE_KOKKOS_H
#define LMP_PAIR_MESOMEM_DIPOLE_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_mesomem_dipole.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

struct params_mesomem_dipole {
  KOKKOS_INLINE_FUNCTION
  params_mesomem_dipole() :
    cut(0), sigma(0), eps(0), ktilt(0), ksplay(0), wc(0), zeta(0), c0(0), zeta_int(-1) {}
  KOKKOS_INLINE_FUNCTION
  params_mesomem_dipole(int /*i*/) :
    cut(0), sigma(0), eps(0), ktilt(0), ksplay(0), wc(0), zeta(0), c0(0), zeta_int(-1) {}

  KK_FLOAT cut, sigma, eps, ktilt, ksplay, wc, zeta, c0;
  int zeta_int;    // zeta as integer if it is one (fast path), -1 otherwise
};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
struct TagPairMesomemDipole {};

template<class DeviceType>
class PairMesomemDipoleKokkos : public PairMesomemDipole {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMesomemDipoleKokkos(class LAMMPS *);
  ~PairMesomemDipoleKokkos() override;

  void compute(int, int) override;

  void init_style() override;
  double init_one(int, int) override;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMesomemDipole<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
                  const int, EV_FLOAT &ev) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMesomemDipole<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
                  const int) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, int i, int j, const KK_FLOAT &epair,
                    KK_FLOAT fx, KK_FLOAT fy, KK_FLOAT fz,
                    KK_FLOAT delx, KK_FLOAT dely, KK_FLOAT delz) const;

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:
  Kokkos::DualView<params_mesomem_dipole**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_mesomem_dipole**,
    Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  // hardwired to space for MAX_TYPES_STACKPARAMS atom types
  params_mesomem_dipole m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  KK_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_3 torque;
  typename AT::t_int_1d_randomread type;
  typename AT::t_kkfloat_1d_4_randomread mu;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  DAT::ttransform_kkfloat_2d k_cutsq;
  typename AT::t_kkfloat_2d d_cutsq;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  KK_FLOAT special_lj[4];

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  void allocate() override;
  friend void pair_virial_fdotr_compute<PairMesomemDipoleKokkos>(PairMesomemDipoleKokkos*);
};

}

#endif
#endif
