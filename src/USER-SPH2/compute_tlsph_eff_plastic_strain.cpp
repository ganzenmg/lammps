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
#include "compute_tlsph_eff_plastic_strain.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTlsphEffPlasticStrain::ComputeTlsphEffPlasticStrain(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute tlsph/eff_plastic_strain command");

    peratom_flag = 1;
    size_peratom_cols = 0;

    nmax = 0;
    epl = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTlsphEffPlasticStrain::~ComputeTlsphEffPlasticStrain() {
    memory->destroy(epl);
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphEffPlasticStrain::init() {
    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "tlsph/eff_plastic_strain") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute tlsph/eff_plastic_strain");
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphEffPlasticStrain::compute_peratom() {
    invoked_peratom = update->ntimestep;

    if (atom->nlocal > nmax) {
        memory->destroy(epl);
        nmax = atom->nmax;
        memory->create(epl, nmax, "tlsph/eff_plastic_strain:epl");
        vector_atom = epl;
    }

    double *eff_plastic_strain = atom->eff_plastic_strain;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            epl[i] = eff_plastic_strain[i];
        } else {
            epl[i] = 0.0;
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeTlsphEffPlasticStrain::memory_usage() {
    double bytes = nmax * sizeof(double);
    return bytes;
}
