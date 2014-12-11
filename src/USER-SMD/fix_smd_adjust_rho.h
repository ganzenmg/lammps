/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */


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

#ifdef FIX_CLASS

FixStyle(smd/adjust_rho,FixSphFluidAdjustRho)

#else

#ifndef LMP_FIX_SPH2_ADJUST_RHO_H
#define LMP_FIX_SPH2_ADJUST_RHO_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSphFluidAdjustRho : public Fix {
 public:
  FixSphFluidAdjustRho(class LAMMPS *, int, char **);
  int setmask();
  virtual void init();
  virtual void final_integrate();

 private:
  class NeighList *list;
 protected:
  int mass_require;
  double rho_target;
  int NN_target;

  class Pair *pair;
};

}

#endif
#endif
