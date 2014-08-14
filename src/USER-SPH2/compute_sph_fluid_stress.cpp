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
#include "compute_sph_fluid_stress.h"
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

ComputeSpheFluidStress::ComputeSpheFluidStress(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute sph/fluid/stress command");

	peratom_flag = 1;
	size_peratom_cols = 9;

	nmax = 0;
	stresstensorVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSpheFluidStress::~ComputeSpheFluidStress() {
	memory->sfree(stresstensorVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSpheFluidStress::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "sph/fluid/stress") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute tlsph/stress");
}

/* ---------------------------------------------------------------------- */

void ComputeSpheFluidStress::compute_peratom() {
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

	Matrix3d *T = (Matrix3d *) force->pair->extract("fluid_stressTensor_ptr", itmp);
	if (T == NULL) {
		error->all(FLERR, "compute sph/fluid/stress failed to access stress array");
	}

	int nlocal = atom->nlocal;
	Matrix3d Td;

	for (int i = 0; i < nlocal; i++) {

		Td = Deviator(T[i]);

		stresstensorVector[i][0] = T[i](0, 0); // xx
		stresstensorVector[i][1] = T[i](1, 1); // yy
		stresstensorVector[i][2] = T[i](2, 2); // zz
		stresstensorVector[i][3] = T[i](0, 1); // xy
		stresstensorVector[i][4] = T[i](0, 2); // xz
		stresstensorVector[i][5] = T[i](1, 2); // yz

		stresstensorVector[i][6] = Td(0,1); // xy
		stresstensorVector[i][7] = Td(0,2); // xz
		stresstensorVector[i][8] = Td(1,2); // yz
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSpheFluidStress::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}


/*
 * deviator of a tensor
 */
Matrix3d ComputeSpheFluidStress::Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}
