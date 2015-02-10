/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

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
#include "fix_smd_move_triangulated_surface.h"
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
#include "domain.h"
#include <Eigen/Eigen>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSMDMoveTriSurf::FixSMDMoveTriSurf(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	if ((atom->e_flag != 1) || (atom->rho_flag != 1))
		error->all(FLERR, "fix fix smd/move_tri_surf command requires atom_style with both energy and density");

	if (narg < 3)
		error->all(FLERR, "Illegal number of arguments for fix fix smd/move_tri_surf command");

	int iarg = 3;

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix fix smd/move_tri_surf is active for group: %s \n", arg[1]);
	}
	while (true) {

		if (iarg >= narg) {
			break;
		}

		if (strcmp(arg[iarg], "*LINEAR") == 0) {
			linearFlag = true;
			if (comm->me == 0) {
				printf("... will move surface in a linear fashion\n");
			}

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following *LINEAR");
			}
			vx = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following *LINEAR");
			}
			vy = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected three floats following *LINEAR");
			}
			vz = force->numeric(FLERR, arg[iarg]);

		} else if (strcmp(arg[iarg], "rotate") == 0) {
			rotateFlag = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following adjust_radius");
			}

			wx = force->numeric(FLERR, arg[iarg]);

		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for fix smd/move_tri_surf: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;

	}

	if (comm->me == 0) {
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}

	time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixSMDMoveTriSurf::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDMoveTriSurf::init() {
	dtv = update->dt;
}

/* ----------------------------------------------------------------------
 ------------------------------------------------------------------------- */

void FixSMDMoveTriSurf::initial_integrate(int vflag) {
	double **x = atom->x;
	double **v = atom->v;
	double **vest = atom->vest;
	double **tlsph_fold = atom->tlsph_fold;
	int *mol = atom->molecule;

	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	int i;

	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {

			v[i][0] = vx;
			v[i][1] = vy;
			v[i][2] = vz;

			vest[i][0] = vx;
			vest[i][1] = vy;
			vest[i][2] = vz;

			x[i][0] += dtv * vx;
			x[i][1] += dtv * vy;
			x[i][2] += dtv * vz;

			/*
			 * if this is a triangle, move the vertices as well
			 */

			if (mol[i] == 65535) {
				tlsph_fold[i][0] += dtv * vx;
				tlsph_fold[i][1] += dtv * vy;
				tlsph_fold[i][2] += dtv * vz;

				tlsph_fold[i][3] += dtv * vx;
				tlsph_fold[i][4] += dtv * vy;
				tlsph_fold[i][5] += dtv * vz;

				tlsph_fold[i][6] += dtv * vx;
				tlsph_fold[i][7] += dtv * vy;
				tlsph_fold[i][8] += dtv * vz;
			}

		}
	}
}

/* ---------------------------------------------------------------------- */

void FixSMDMoveTriSurf::reset_dt() {
	dtv = update->dt;
}
