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
#include "compute_tlsph_contact_radius.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTlsphContactRadius::ComputeTlsphContactRadius(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Number of arguments for compute tlsph/contact_radius command != 3");
  if (atom->contact_radius_flag != 1) error->all(FLERR,"compute tlsph/contact_radius command requires tlsph atom_style");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  contact_radius_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTlsphContactRadius::~ComputeTlsphContactRadius()
{
  memory->sfree(contact_radius_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphContactRadius::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"tlsph/contact_radius") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute tlsph/contact_radius");
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphContactRadius::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow evector array if necessary

  if (atom->nlocal > nmax) {
    memory->sfree(contact_radius_vector);
    nmax = atom->nmax;
    contact_radius_vector = (double *) memory->smalloc(nmax*sizeof(double),"evector/atom:evector");
    vector_atom = contact_radius_vector;
  }

  double *contact_radius = atom->contact_radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              contact_radius_vector[i] = contact_radius[i];
      }
      else {
              contact_radius_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeTlsphContactRadius::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
