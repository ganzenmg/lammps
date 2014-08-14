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

ComputeStyle(pdgcg/damage,ComputePDGCGDamage)

#else

#ifndef LMP_COMPUTE_PDGCG_DAMAGE_ATOM_H
#define LMP_COMPUTE_PDGCG_DAMAGE_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputePDGCGDamage : public Compute {
 public:
  ComputePDGCGDamage(class LAMMPS *, int, char **);
  ~ComputePDGCGDamage();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  double *damage;
  int ifix_peri;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: More than one compute damage/atom

It is not efficient to use compute ke/atom more than once.

E: Compute damage/atom requires peridynamic potential

Damage is a Peridynamic-specific metric.  It requires you
to be running a Peridynamics simulation.

*/
