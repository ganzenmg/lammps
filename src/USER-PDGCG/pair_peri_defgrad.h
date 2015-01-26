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

#ifdef PAIR_CLASS

PairStyle(peri/defgrad,PairPeriDefgrad)

#else

//ss

#ifndef LMP_PAIR_PERI_DEFGRAD_H
#define LMP_PAIR_PERI_DEFGRAD_H

#include "pair.h"
#include <Eigen/Eigen>
using namespace Eigen;

namespace LAMMPS_NS {

class PairPeriDefgrad : public Pair {
 public:
  PairPeriDefgrad(class LAMMPS *);
  virtual ~PairPeriDefgrad();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();
  void init_list(int, class NeighList *);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *) {}
  void read_restart_settings(FILE *) {}
  virtual double memory_usage();

  void kernel_and_derivative(double h, double r, double &wf, double &wfd);

 protected:
  int ifix_peri;
  double **bulkmodulus;
  double **syield, **smax, **alpha, **G0;
  double cutoff_global;

  void allocate();

  int nmax;
  Matrix2d *F, *K, *PK1, *PK1_as;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Pair style peri requires atom style peri

UNDOCUMENTED

E: Pair peri requires an atom map, see atom_modify

Even for atomic systems, an atom map is required to find Peridynamic
bonds.  Use the atom_modify command to define one.

E: Pair peri requires a lattice be defined

Use the lattice command for this purpose.

E: Pair peri lattice is not identical in x, y, and z

The lattice defined by the lattice command must be cubic.

E: Fix peri neigh does not exist

Somehow a fix that the pair style defines has been deleted.

*/
