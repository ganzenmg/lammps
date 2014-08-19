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

#include "string.h"
#include "compute_sph2_damage.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSph2Damage::ComputeSph2Damage(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph2/damage command");
  if (atom->rho_flag != 1) error->all(FLERR,"compute sph2/damage command requires atom_style with damage (e.g. sph2)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  damage_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSph2Damage::~ComputeSph2Damage()
{
  memory->sfree(damage_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSph2Damage::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"sph2/damage") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute sph2/damage");
}

/* ---------------------------------------------------------------------- */

void ComputeSph2Damage::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow rhoVector array if necessary

  if (atom->nlocal > nmax) {
    memory->sfree(damage_vector);
    nmax = atom->nmax;
    damage_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:damage_vector");
    vector_atom = damage_vector;
  }

  double *damage = atom->damage;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              damage_vector[i] = damage[i];
      }
      else {
              damage_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSph2Damage::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
