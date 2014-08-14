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
#include "compute_tlsph_stress.h"
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

ComputeTlsphStress::ComputeTlsphStress(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute tlsph/stress command");

	peratom_flag = 1;
	size_peratom_cols = 6;

	nmax = 0;
	stresstensorVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTlsphStress::~ComputeTlsphStress() {
	memory->sfree(stresstensorVector);
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphStress::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "tlsph/stress") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute tlsph/stress");
}

/* ---------------------------------------------------------------------- */

void ComputeTlsphStress::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow vector array if necessary

	if (atom->nlocal > nmax) {
		memory->destroy(stresstensorVector);
		nmax = atom->nmax;
		memory->create(stresstensorVector, nmax, size_peratom_cols, "stresstensorVector");
		array_atom = stresstensorVector;
	}

	// copy data to output array
	int itmp = 0;

	Matrix3d *T = (Matrix3d *) force->pair->extract("CauchyStress_ptr", itmp);
	if (T == NULL) {
		error->all(FLERR, "compute tlsph/stress failed to access Cauchy Stress array");
	}

	int nlocal = atom->nlocal;

	for (int i = 0; i < nlocal; i++) {
		stresstensorVector[i][0] = T[i](0, 0); // xx
		stresstensorVector[i][1] = T[i](1, 1); // yy
		stresstensorVector[i][2] = T[i](2, 2); // zz
		stresstensorVector[i][3] = T[i](0, 1); // xy
		stresstensorVector[i][4] = T[i](0, 2); // xz
		stresstensorVector[i][5] = T[i](1, 2); // yz
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeTlsphStress::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}
