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

ComputeStyle(sph/fluid/stress,ComputeSpheFluidStress)

#else

#ifndef LMP_COMPUTE_SPH_FLUID_STRESS_H
#define LMP_COMPUTE_SPH_FLUID_STRESS_H

#include "compute.h"
#include <Eigen/Eigen>
using namespace Eigen;

namespace LAMMPS_NS {

class ComputeSpheFluidStress : public Compute {
 public:
  ComputeSpheFluidStress(class LAMMPS *, int, char **);
  ~ComputeSpheFluidStress();
  void init();
  void compute_peratom();
  double memory_usage();
  Matrix3d Deviator(Matrix3d);

 private:
  int nmax;
  double **stresstensorVector;
};

}

#endif
#endif
