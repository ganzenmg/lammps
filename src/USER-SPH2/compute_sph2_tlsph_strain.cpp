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
#include "compute_sph2_tlsph_strain.h"
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

ComputeSph2TLSPHstrain::ComputeSph2TLSPHstrain(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute sph2/tlsph_strain command");

    peratom_flag = 1;
    size_peratom_cols = 6;

    nmax = 0;
    strainVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSph2TLSPHstrain::~ComputeSph2TLSPHstrain() {
    memory->sfree(strainVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSph2TLSPHstrain::init() {

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "sph2/tlsph_strain") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute sph2/tlsph_strain");
}

/* ---------------------------------------------------------------------- */

void ComputeSph2TLSPHstrain::compute_peratom() {
    invoked_peratom = update->ntimestep;

    // grow vector array if necessary

    if (atom->nlocal > nmax) {
        memory->destroy(strainVector);
        nmax = atom->nmax;
        memory->create(strainVector, nmax, size_peratom_cols, "strainVector");
        array_atom = strainVector;
    }

    // copy data to output array
    int itmp = 0;
    Matrix3d *F = (Matrix3d *) force->pair->extract("sph2/tlsph/F_ptr", itmp);
    if (F == NULL) {
        error->all(FLERR, "compute sph2/tlsph_strain failed to access F array");
    }

    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    Matrix3d E, eye;
    eye.setIdentity();

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
        	E = 0.5 * (F[i] * F[i].transpose() - eye); // Green-Lagrange strain
            strainVector[i][0] = E(0, 0);
            strainVector[i][1] = E(1, 1);
            strainVector[i][2] = E(2, 2);
            strainVector[i][3] = E(0, 1);
            strainVector[i][4] = E(0, 2);
            strainVector[i][5] = E(1, 2);
        } else {
            for (int j = 0; j < size_peratom_cols; j++) {
                strainVector[i][j] = 0.0;
            }
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSph2TLSPHstrain::memory_usage() {
    double bytes = size_peratom_cols * nmax * sizeof(double);
    return bytes;
}
