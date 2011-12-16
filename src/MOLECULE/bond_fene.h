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

#ifdef BOND_CLASS

BondStyle(fene,BondFENE)

#else

#ifndef LMP_BOND_FENE_H
#define LMP_BOND_FENE_H

#include "stdio.h"
#include "bond.h"

namespace LAMMPS_NS {

class BondFENE : public Bond {
 public:
  BondFENE(class LAMMPS *);
  virtual ~BondFENE();
  virtual void compute(int, int);
  void coeff(int, char **);
  void init_style();
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  double single(int, double, int, int);

 protected:
  double TWO_1_3;
  double *k,*r0,*epsilon,*sigma;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

W: FENE bond too long: %ld %d %d %g

UNDOCUMENTED

E: Bad FENE bond

Two atoms in a FENE bond have become so far apart that the bond cannot
be computed.

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

W: Use special bonds = 0,1,1 with bond style fene

Most FENE models need this setting for the special_bonds command.

W: FENE bond too long: %ld %g

UNDOCUMENTED

*/
