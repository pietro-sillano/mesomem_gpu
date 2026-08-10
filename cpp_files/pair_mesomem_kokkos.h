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

// Contributing author: Pietro Sillano (TU Delft), 2026
// Kokkos port of pair_mesomem

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mesomem/kk,        PairMesomemKokkos<LMPDeviceType>);
PairStyle(mesomem/kk/device, PairMesomemKokkos<LMPDeviceType>);
PairStyle(mesomem/kk/host,   PairMesomemKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_MESOMEM_KOKKOS_H
#define LMP_PAIR_MESOMEM_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_mesomem.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

struct params_mesomem {
  KOKKOS_INLINE_FUNCTION
  params_mesomem()
      : cutsq(0.0), sigma(0.0), eps(0.0), ktilt(0.0), ksplay(0.0),
        wc(0.0), c0(0.0), zeta_int(0) {}
  KOKKOS_INLINE_FUNCTION
  params_mesomem(int /*dummy*/)
      : cutsq(0.0), sigma(0.0), eps(0.0), ktilt(0.0), ksplay(0.0),
        wc(0.0), c0(0.0), zeta_int(0) {}

  KK_FLOAT cutsq;
  KK_FLOAT sigma;
  KK_FLOAT eps;
  KK_FLOAT ktilt;
  KK_FLOAT ksplay;
  KK_FLOAT wc;         // weight_rcut
  KK_FLOAT c0;         // spontaneous curvature
  int     zeta_int;   // zeta rounded to nearest integer (hot-loop uses integer pow)
};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
struct TagPairMesomem {};

template<class DeviceType>
class PairMesomemKokkos : public PairMesomem {
 public:
  enum { EnabledNeighFlags = FULL | HALFTHREAD | HALF };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMesomemKokkos(class LAMMPS *);
  ~PairMesomemKokkos() override;

  void compute(int, int) override;

  void init_style() override;
  double init_one(int, int) override;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMesomem<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
                  const int, EV_FLOAT &ev) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, bool STACKPARAMS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMesomem<NEIGHFLAG,NEWTON_PAIR,EVFLAG,STACKPARAMS>,
                  const int) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, int i, int j, const KK_FLOAT &epair,
                    KK_FLOAT fx, KK_FLOAT fy, KK_FLOAT fz,
                    KK_FLOAT delx, KK_FLOAT dely, KK_FLOAT delz) const;

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:
  Kokkos::DualView<params_mesomem**, Kokkos::LayoutRight, DeviceType> k_params;
  typename Kokkos::DualView<params_mesomem**,
      Kokkos::LayoutRight, DeviceType>::t_dev_const_um params;
  // stack-resident params for small ntypes (fast path)
  params_mesomem m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
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

  int neighflag, newton_pair;
  int nlocal, nall, eflag, vflag;

  double special_lj[4];

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  void allocate() override;
  friend void pair_virial_fdotr_compute<PairMesomemKokkos>(
      PairMesomemKokkos*);
};

}

#endif
#endif
