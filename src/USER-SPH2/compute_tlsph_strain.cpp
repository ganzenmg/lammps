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
#include "compute_tlsph_strain.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <Eigen/Eigen>
using namespace Eigen;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeStrainTensor::ComputeStrainTensor(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute straintensor command");

    peratom_flag = 1;
    size_peratom_cols = 6;

    nmax = 0;
    straintensorVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeStrainTensor::~ComputeStrainTensor() {
    memory->sfree(straintensorVector);
}

/* ---------------------------------------------------------------------- */

void ComputeStrainTensor::init() {

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "straintensor") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute straintensor");
}

/* ---------------------------------------------------------------------- */

void ComputeStrainTensor::compute_peratom() {
    invoked_peratom = update->ntimestep;

    // grow vector array if necessary

    if (atom->nlocal > nmax) {
        memory->destroy(straintensorVector);
        nmax = atom->nmax;
        memory->create(straintensorVector, nmax, size_peratom_cols, "defgradVector");
        array_atom = straintensorVector;
    }

    // copy data to output array
    int itmp = 0;

    Matrix3d *F = (Matrix3d *) force->pair->extract("F_ptr", itmp);
    if (F == NULL) {
        error->all(FLERR, "compute straintensor failed to access F array");
    }

    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    Matrix3d E, eye;
    eye.setIdentity();
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            E = 0.5 * (F[i] * F[i].transpose() - eye);
            straintensorVector[i][0] = E(0, 0);
            straintensorVector[i][1] = E(1, 1);
            straintensorVector[i][2] = E(2, 2);
            straintensorVector[i][3] = E(0, 1); // xy
            straintensorVector[i][4] = E(0, 2); // xz
            straintensorVector[i][5] = E(1, 2); // yz


        } else {
            for (int j = 0; j < size_peratom_cols; j++) {
                straintensorVector[i][j] = 0.0;
            }
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeStrainTensor::memory_usage() {
    double bytes = size_peratom_cols * nmax * sizeof(double);
    return bytes;
}
