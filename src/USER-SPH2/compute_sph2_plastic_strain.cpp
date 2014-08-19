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
#include "compute_sph2_plastic_strain.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSph2PlasticStrain::ComputeSph2PlasticStrain(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph2/plastic_strain command");
  if (atom->rho_flag != 1) error->all(FLERR,"compute sph2/plastic_strain command requires atom_style with plastic_strain (e.g. sph2)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  plastic_strain_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSph2PlasticStrain::~ComputeSph2PlasticStrain()
{
  memory->sfree(plastic_strain_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSph2PlasticStrain::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"sph2/plastic_strain") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute sph2/plastic_strain");
}

/* ---------------------------------------------------------------------- */

void ComputeSph2PlasticStrain::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow rhoVector array if necessary

  if (atom->nlocal > nmax) {
    memory->sfree(plastic_strain_vector);
    nmax = atom->nmax;
    plastic_strain_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:plastic_strain_vector");
    vector_atom = plastic_strain_vector;
  }

  double *plastic_strain = atom->eff_plastic_strain;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              plastic_strain_vector[i] = plastic_strain[i];
      }
      else {
              plastic_strain_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSph2PlasticStrain::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
