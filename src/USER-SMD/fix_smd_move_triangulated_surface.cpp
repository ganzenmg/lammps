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

	rotateFlag = linearFlag = false;

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

		} else if (strcmp(arg[iarg], "*ROTATE") == 0) {
			rotateFlag = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			origin(0) = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			origin(1) = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			origin(2) = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			rotation_axis(0) = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			rotation_axis(1) = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			rotation_axis(2) = force->numeric(FLERR, arg[iarg]);

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected 7 floats following *ROTATE: origin, rotation axis, and angular velocity");
			}
			angular_velocity = force->numeric(FLERR, arg[iarg]);

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

	// set comm sizes needed by this fix
	comm_forward = 12;

	atom->add_callback(0);

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
	double **x0 = atom->x0;
	double **v = atom->v;
	double **vest = atom->vest;
	double **tlsph_fold = atom->tlsph_fold;
	int *mol = atom->molecule;

	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	int i;

	Vector3d v1, v2, v3, n, center, R, vel;

	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	if (linearFlag) { // translate particles
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

				if (mol[i] >= 65535) {
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

	if (rotateFlag) { // rotate particles
		Vector3d xnew, R_new, x_correct;

		for (i = 0; i < nlocal; i++) {
			if (mask[i] & groupbit) {

				center << x[i][0], x[i][1], x[i][2];
				R = center - origin;
				vel = angular_velocity * R.cross(rotation_axis);

				// new incorrect position at new time
				xnew = center + dtv * vel;

				// line from origin to new incorrect position
				R_new = xnew - origin;

				// new correct position
				x_correct = origin + R.norm() * R_new / R_new.norm();

				vel = (x_correct - center) / dtv;

//				printf("x_old: %f %f %f\n", center(0), center(1), center(2));
//				printf("x_new: %f %f %f\n", xnew(0), xnew(1), xnew(2));
//				printf("R: %f %f %f\n", R(0), R(1), R(2));
//				printf("R_new: %f %f %f\n", R_new(0), R_new(1), R_new(2));
//				printf("x_cor: %f %f %f\n\n", x_correct(0), x_correct(1), x_correct(2));
				//printf("rot vel: %f %f %f\n", vel(0), vel(1), vel(2));

				v[i][0] = vel(0);
				v[i][1] = vel(1);
				v[i][2] = vel(2);

				vest[i][0] = vel(0);
				vest[i][1] = vel(1);
				vest[i][2] = vel(2);

//				x[i][0] += dtv * vel(0);
//				x[i][1] += dtv * vel(1);
//				x[i][2] += dtv * vel(2);
				x[i][0] = x_correct(0);
				x[i][1] = x_correct(1);
				x[i][2] = x_correct(2);

				/*
				 * if this is a triangle, rotate the vertices as well
				 */

				if (mol[i] >= 65535) {

					v1 << tlsph_fold[i][0], tlsph_fold[i][1], tlsph_fold[i][2];
					R = v1 - origin;
					vel = angular_velocity * R.cross(rotation_axis);
					xnew = v1 + dtv * vel;
					R_new = xnew - origin;
					x_correct = origin + R.norm() * R_new / R_new.norm();
					tlsph_fold[i][0] = x_correct(0);
					tlsph_fold[i][1] = x_correct(1);
					tlsph_fold[i][2] = x_correct(2);
					v1 = x_correct;

					v2 << tlsph_fold[i][3], tlsph_fold[i][4], tlsph_fold[i][5];
					R = v2 - origin;
					vel = angular_velocity * R.cross(rotation_axis);
					xnew = v2 + dtv * vel;
					R_new = xnew - origin;
					x_correct = origin + R.norm() * R_new / R_new.norm();
					tlsph_fold[i][3] = x_correct(0);
					tlsph_fold[i][4] = x_correct(1);
					tlsph_fold[i][5] = x_correct(2);
					v2 = x_correct;

					v3 << tlsph_fold[i][6], tlsph_fold[i][7], tlsph_fold[i][8];
					R = v3 - origin;
					vel = angular_velocity * R.cross(rotation_axis);
					xnew = v3 + dtv * vel;
					R_new = xnew - origin;
					x_correct = origin + R.norm() * R_new / R_new.norm();
					tlsph_fold[i][6] = x_correct(0);
					tlsph_fold[i][7] = x_correct(1);
					tlsph_fold[i][8] = x_correct(2);
					v3 = x_correct;

					// recalculate triangle normal
					n = (v2 - v1).cross(v2 - v3);
					x0[i][0] = n(0);
					x0[i][1] = n(1);
					x0[i][2] = n(2);

				}

			}
		}
	}

	// we changed tlsph_fold, x0. perform communication to ghosts
	comm->forward_comm_fix(this);

}

/* ---------------------------------------------------------------------- */

void FixSMDMoveTriSurf::reset_dt() {
	dtv = update->dt;
}

/* ---------------------------------------------------------------------- */

int FixSMDMoveTriSurf::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	double **x0 = atom->x0;
	double **tlsph_fold = atom->tlsph_fold;

	//printf("in FixSMDIntegrateTlsph::pack_forward_comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = x0[j][0];
		buf[m++] = x0[j][1];
		buf[m++] = x0[j][2];

		buf[m++] = tlsph_fold[j][0];
		buf[m++] = tlsph_fold[j][1];
		buf[m++] = tlsph_fold[j][2];
		buf[m++] = tlsph_fold[j][3];
		buf[m++] = tlsph_fold[j][4];
		buf[m++] = tlsph_fold[j][5];
		buf[m++] = tlsph_fold[j][6];
		buf[m++] = tlsph_fold[j][7];
		buf[m++] = tlsph_fold[j][8];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void FixSMDMoveTriSurf::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	double **x0 = atom->x0;
	double **tlsph_fold = atom->tlsph_fold;

	//printf("in FixSMDMoveTriSurf::unpack_forward_comm\n");
	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		x0[i][0] = buf[m++];
		x0[i][1] = buf[m++];
		x0[i][2] = buf[m++];

		tlsph_fold[i][0] = buf[m++];
		tlsph_fold[i][1] = buf[m++];
		tlsph_fold[i][2] = buf[m++];
		tlsph_fold[i][3] = buf[m++];
		tlsph_fold[i][4] = buf[m++];
		tlsph_fold[i][5] = buf[m++];
		tlsph_fold[i][6] = buf[m++];
		tlsph_fold[i][7] = buf[m++];
		tlsph_fold[i][8] = buf[m++];
	}
}
