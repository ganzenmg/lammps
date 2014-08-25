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

#include "stdio.h"
#include "string.h"
#include "fix_sph_adjust_rho.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <Eigen/Eigen>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSphFluidAdjustRho::FixSphFluidAdjustRho(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	if ((atom->e_flag != 1) || (atom->rho_flag != 1))
		error->all(FLERR, "fix sph_fluid command requires atom_style with both energy and density");

	if (narg != 5)
		error->all(FLERR, "Illegal number of arguments for fix sph_adjust_rho command");

	rho_target = atof(arg[3]);
	NN_target = atof(arg[4]);

	time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixSphFluidAdjustRho::setmask() {
	int mask = 0;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSphFluidAdjustRho::init() {
}

/* ---------------------------------------------------------------------- */

void FixSphFluidAdjustRho::final_integrate() {

	// update v, rho, and e of atoms in group

	double *rho = atom->rho;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;
	double scale;
	double *rmass = atom->rmass;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;

	int itmp = 0;
	int *numNeighs = (int *) force->pair->extract("sph2/ulsph/numNeighs_ptr", itmp);
	if (numNeighs == NULL) {
		error->all(FLERR, "compute sph2/ulsph_num_neighs failed to access numNeighs array");
	}

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {

			scale = rho[i] / rho_target;
			rmass[i] /= sqrt(scale);
			//vfrac[i] *= sqrt(scale);

			//printf("old radius: %f ", radius[i]);
			//if (NN_target > 0) {
			//	scale = 12.0 / numNeighs[i];
			//	radius[i] *= sqrt(scale);
			//}
			//printf(" new radius: %f\n", radius[i]);

		}
	}
}

