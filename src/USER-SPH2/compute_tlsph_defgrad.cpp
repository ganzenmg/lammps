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
#include "compute_tlsph_defgrad.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <iostream>
#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include <Eigen/Eigen>
using namespace Eigen;
using namespace std;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTlsphDefgrad::ComputeTlsphDefgrad(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute TLSPH/defgrad command");

    peratom_flag = 1;
    size_peratom_cols = 10;

    nmax = 0;
    defgradVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTlsphDefgrad::~ComputeTlsphDefgrad() {
    memory->sfree(defgradVector);
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphDefgrad::init() {

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "defgrad/atom") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute defgrad/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphDefgrad::compute_peratom() {
    invoked_peratom = update->ntimestep;

    // grow vector array if necessary

    if (atom->nlocal > nmax) {
        memory->destroy(defgradVector);
        nmax = atom->nmax;
        memory->create(defgradVector, nmax, size_peratom_cols, "defgradVector");
        array_atom = defgradVector;
    }

    // copy data to output array
    int itmp = 0;
    double *detF = (double *) force->pair->extract("detF_ptr", itmp);
    if (detF == NULL) {
        error->all(FLERR, "compute TLSPH/defgrad failed to access detF array");
    }

    Matrix3d *F = (Matrix3d *) force->pair->extract("F_ptr", itmp);
    if (F == NULL) {
        error->all(FLERR, "compute defgrad/atom failed to access F array");
    }

    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            defgradVector[i][0] = F[i](0, 0);
            defgradVector[i][1] = F[i](0, 1);
            defgradVector[i][2] = F[i](0, 2);
            defgradVector[i][3] = F[i](1, 0);
            defgradVector[i][4] = F[i](1, 1);
            defgradVector[i][5] = F[i](1, 2);
            defgradVector[i][6] = F[i](2, 0);
            defgradVector[i][7] = F[i](2, 1);
            defgradVector[i][8] = F[i](2, 2);
            defgradVector[i][9] = detF[i];
        } else {
            for (int j = 0; j < size_peratom_cols; j++) {
                defgradVector[i][j] = 0.0;
            }
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeTlsphDefgrad::memory_usage() {
    double bytes = size_peratom_cols * nmax * sizeof(double);
    return bytes;
}
