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

/* ----------------------------------------------------------------------
   Contributing author: Mike Parks (SNL)
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_pdgcg_damage.h"
#include "fix_peri_neigh_gcg.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePDGCGDamage::ComputePDGCGDamage(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute damage/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  damage = NULL;
}

/* ---------------------------------------------------------------------- */

ComputePDGCGDamage::~ComputePDGCGDamage()
{
  memory->destroy(damage);
}

/* ---------------------------------------------------------------------- */

void ComputePDGCGDamage::init()
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"pdgcg/damage") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute pdgcg/damage");

  // find associated PERI_NEIGH fix that must exist

  ifix_peri = -1;
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"PERI_NEIGH_GCG") == 0) ifix_peri = i;
  if (ifix_peri == -1)
    error->all(FLERR,"Compute damage/atom requires peridynamic potential");
}

/* ---------------------------------------------------------------------- */

void ComputePDGCGDamage::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow damage array if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(damage);
    nmax = atom->nmax;
    memory->create(damage,nmax,"damage/atom:damage");
    vector_atom = damage;
  }

  // compute damage for each atom in group

  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double *vfrac = atom->vfrac;
  double *vinter = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->vinter;
  tagint **partner = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->partner;
  int *npartner = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->npartner;
  int i,j,jj,jnum;

  double damage_temp;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      jnum = npartner[i];
      damage_temp = 0.0;
      for (jj = 0; jj < jnum; jj++) {
        if (partner[i][jj] == 0) continue;
        
        // look up local index of this partner particle
        // skip if particle is "lost"
        
        j = atom->map(partner[i][jj]);
        if (j < 0) continue;
        
        damage_temp += vfrac[j];
      }

      if (vinter[i] != 0.0) damage[i] = 1.0 - damage_temp/vinter[i];
      else damage[i] = 0.0;

    } else damage[i] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputePDGCGDamage::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
