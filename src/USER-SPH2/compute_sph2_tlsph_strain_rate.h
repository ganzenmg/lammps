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

ComputeStyle(sph2/tlsph_strain_rate,ComputeSph2TLSPHStrainRate)

#else

#ifndef LMP_COMPUTE_SPH2_TLSPH_STRAIN_RATE_H
#define LMP_COMPUTE_SPH2_TLSPH_STRAIN_RATE_H

#include "compute.h"
#include <Eigen/Eigen>
using namespace Eigen;

namespace LAMMPS_NS {

class ComputeSph2TLSPHStrainRate : public Compute {
 public:
  ComputeSph2TLSPHStrainRate(class LAMMPS *, int, char **);
  ~ComputeSph2TLSPHStrainRate();
  void init();
  void compute_peratom();
  double memory_usage();
  Matrix3d Deviator(Matrix3d);

 private:
  int nmax;
  double **strain_rate_array;
};

}

#endif
#endif
