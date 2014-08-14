/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(sph/rho,ComputeSphFluidRho)

#else

#ifndef LMP_COMPUTE_SPH_FLUID_RHO_H
#define LMP_COMPUTE_SPH_FLUID_RHO_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSphFluidRho : public Compute {
 public:
  ComputeSphFluidRho(class LAMMPS *, int, char **);
  ~ComputeSphFluidRho();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  double *rhoVector;
};

}

#endif
#endif
