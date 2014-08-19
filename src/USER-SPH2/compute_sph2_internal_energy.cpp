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
#include "compute_sph2_internal_energy.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSph2InternalEnergy::ComputeSph2InternalEnergy(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph2/internal_energy command");
  if (atom->rho_flag != 1) error->all(FLERR,"compute sph2/internal_energy command requires atom_style with internal_energy (e.g. sph2)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  internal_energy_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSph2InternalEnergy::~ComputeSph2InternalEnergy()
{
  memory->sfree(internal_energy_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSph2InternalEnergy::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"sph2/internal_energy") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute sph2/internal_energy");
}

/* ---------------------------------------------------------------------- */

void ComputeSph2InternalEnergy::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow rhoVector array if necessary

  if (atom->nlocal > nmax) {
    memory->sfree(internal_energy_vector);
    nmax = atom->nmax;
    internal_energy_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:internal_energy_vector");
    vector_atom = internal_energy_vector;
  }

  double *e = atom->e;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              internal_energy_vector[i] = e[i];
      }
      else {
              internal_energy_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSph2InternalEnergy::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
