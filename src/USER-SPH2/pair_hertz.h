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

PairStyle(tlsph/ipc_hertz,PairHertz)

#else

#ifndef LMP_TLSPH_HERTZ_H
#define LMP_TLSPH_HERTZ_H

#include "pair.h"

namespace LAMMPS_NS {

class PairHertz : public Pair {
 public:
  PairHertz(class LAMMPS *);
  virtual ~PairHertz();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();
  void init_list(int, class NeighList *);
  virtual double memory_usage();

 protected:
  double **bulkmodulus;
  double **kn;

  double *onerad_dynamic,*onerad_frozen;
  double *maxrad_dynamic,*maxrad_frozen;

  void allocate();
};

}

#endif
#endif

