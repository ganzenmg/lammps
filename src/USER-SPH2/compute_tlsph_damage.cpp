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
#include "compute_tlsph_damage.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTlsphDamage::ComputeTlsphDamage(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute tlsph/eff_plastic_strain command");

    peratom_flag = 1;
    size_peratom_cols = 0;

    nmax = 0;
    damage_output = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTlsphDamage::~ComputeTlsphDamage() {
    memory->destroy(damage_output);
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphDamage::init() {
    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "tlsph/damage") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute tlsph/damage");
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphDamage::compute_peratom() {
    invoked_peratom = update->ntimestep;

    if (atom->nlocal > nmax) {
        memory->destroy(damage_output);
        nmax = atom->nmax;
        memory->create(damage_output, nmax, "tlsph/damage_output");
        vector_atom = damage_output;
    }

    double *damage = atom->damage;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            damage_output[i] = damage[i];
        } else {
            damage_output[i] = 0.0;
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeTlsphDamage::memory_usage() {
    double bytes = nmax * sizeof(double);
    return bytes;
}
