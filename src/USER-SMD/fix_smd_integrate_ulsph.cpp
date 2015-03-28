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
#include "fix_smd_integrate_ulsph.h"
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

FixSMDIntegrateUlsph::FixSMDIntegrateUlsph(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	if ((atom->e_flag != 1) || (atom->rho_flag != 1))
		error->all(FLERR, "fix smd/integrate_ulsph command requires atom_style with both energy and density");

	if (narg < 3)
		error->all(FLERR, "Illegal number of arguments for fix smd/integrate_ulsph command");

	adjust_radius_flag = false;
	xsphFlag = false;
	vlimit = -1.0;
	int iarg = 3;

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix smd/integrate_ulsph is active for group: %s \n", arg[1]);
	}
	while (true) {

		if (iarg >= narg) {
			break;
		}

		if (strcmp(arg[iarg], "xsph") == 0) {
			xsphFlag = true;
			if (comm->me == 0) {
				printf("... will use XSPH time integration\n");
			}
		} else if (strcmp(arg[iarg], "adjust_radius") == 0) {
			adjust_radius_flag = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following adjust_radius");
			}

			adjust_radius_factor = force->numeric(FLERR, arg[iarg]);
			if (comm->me == 0) {
				printf("... will adjust smoothing length dynamically with factor %g\n", adjust_radius_factor);
			}
		} else if (strcmp(arg[iarg], "limit_velocity") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following limit_velocity");
			}
			vlimit = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... will limit velocities to <= %g\n", vlimit);
			}
		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for smd/integrate_ulsph: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;

	}

	if (comm->me == 0) {
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}

	// set comm sizes needed by this fix
	comm_forward = 3;
	atom->add_callback(0);

	time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixSMDIntegrateUlsph::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateUlsph::init() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixSMDIntegrateUlsph::initial_integrate(int vflag) {
	// update v and x and rho and e of atoms in group

	double **x = atom->x;
	double **x0 = atom->x0;
	double **v = atom->v;
	double **f = atom->f;
	double **vest = atom->vest;
	double *rho = atom->rho;
	double *drho = atom->drho;
	double *e = atom->e;
	double *de = atom->de;
	double *rmass = atom->rmass;

	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	int i, itmp;
	double dtfm, vsq, scale;
	double vxsph_x, vxsph_y, vxsph_z;

	/*
	 * update the reference configuration if needed
	 */

	int *updateFlag_ptr = (int *) force->pair->extract("smd/ulsph/updateFlag_ptr", itmp);
	if (updateFlag_ptr == NULL) {
		error->one(FLERR,
				"fix smd/integrate_ulsph failed to access updateFlag pointer. Check if a pair style exist which calculates this quantity.");
	}

	// sum all update flag across processors
	int updateFlag = 0;
	MPI_Allreduce(updateFlag_ptr, &updateFlag, 1, MPI_INT, MPI_MAX, world);

	if (updateFlag > 0) {
		if (comm->me == 0) {
			printf("updating ref config at step: %ld\n", update->ntimestep);
		}
	}

	/*
	 * get smoothed velocities from ULSPH pair style
	 */

	Vector3d *smoothVel = (Vector3d *) force->pair->extract("smd/ulsph/smoothVel_ptr", itmp);

	if (xsphFlag) {
		if (smoothVel == NULL) {
			error->one(FLERR, "fix smd/integrate_ulsph failed to access smoothVel array");
		}
	}

	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			dtfm = dtf / rmass[i];

			//e[i] += dtf * de[i]; // half-step update of particle internal energy
			//rho[i] += dtf * drho[i]; // ... and density

			v[i][0] += dtfm * f[i][0];
			v[i][1] += dtfm * f[i][1];
			v[i][2] += dtfm * f[i][2];

			if (vlimit > 0.0) {
				vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
				if (vsq > vlimitsq) {
					scale = sqrt(vlimitsq / vsq);
					v[i][0] *= scale;
					v[i][1] *= scale;
					v[i][2] *= scale;

					vest[i][0] = v[i][0];
					vest[i][1] = v[i][1];
					vest[i][2] = v[i][2];
				}
			}

			/*
			 * store current coords in x0
			 */

			if (updateFlag > 0) {
				x0[i][0] = x[i][0];
				x0[i][1] = x[i][1];
				x0[i][2] = x[i][2];
			}

			if (xsphFlag) {

				// construct XSPH velocity
				vxsph_x = v[i][0] + 0.5 * smoothVel[i](0);
				vxsph_y = v[i][1] + 0.5 * smoothVel[i](1);
				vxsph_z = v[i][2] + 0.5 * smoothVel[i](2);

				vest[i][0] = vxsph_x + dtfm * f[i][0];
				vest[i][1] = vxsph_y + dtfm * f[i][1];
				vest[i][2] = vxsph_z + dtfm * f[i][2];

				x[i][0] += dtv * vxsph_x;
				x[i][1] += dtv * vxsph_y;
				x[i][2] += dtv * vxsph_z;

			} else {

				// extrapolate velocity from half- to full-step
				vest[i][0] = v[i][0] + dtfm * f[i][0];
				vest[i][1] = v[i][1] + dtfm * f[i][1];
				vest[i][2] = v[i][2] + dtfm * f[i][2];

				x[i][0] += dtv * v[i][0];
				x[i][1] += dtv * v[i][1];
				x[i][2] += dtv * v[i][2];
			}
		}
	}

// update of reference config has changed x0
// communicate these quantities now to ghosts
	comm->forward_comm_fix(this);
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateUlsph::final_integrate() {

// update v, rho, and e of atoms in group

	double **v = atom->v;
	double **f = atom->f;
	double *e = atom->e;
	double *de = atom->de;
	double *rho = atom->rho;
	double *drho = atom->drho;
	double *radius = atom->radius;
	double *contact_radius = atom->contact_radius;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;
	double dtfm, vsq, scale;
	double *rmass = atom->rmass;

	/*
	 * get current number of SPH neighbors from ULSPH pair style
	 */

	int itmp;
	int *nn = (int *) force->pair->extract("smd/ulsph/numNeighs_ptr", itmp);
	if (nn == NULL) {
		error->one(FLERR, "fix smd/integrate_ulsph failed to accesss num_neighs array");
	}

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {

			dtfm = dtf / rmass[i];
			v[i][0] += dtfm * f[i][0];
			v[i][1] += dtfm * f[i][1];
			v[i][2] += dtfm * f[i][2];

			if (vlimit > 0.0) {
				vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
				if (vsq > vlimitsq) {
					scale = sqrt(vlimitsq / vsq);
					v[i][0] *= scale;
					v[i][1] *= scale;
					v[i][2] *= scale;
				}
			}

			e[i] += dtf * de[i];
			rho[i] += dtf * drho[i];

			if (adjust_radius_flag) {
				radius[i] = adjust_radius_factor * pow(rmass[i] / rho[i], 1. / domain->dimension); // Monaghan approach for setting the radius
//				double dh = - (adjust_radius_factor/domain->dimension) *
//						pow(rmass[i] / rho[i], 1. / domain->dimension) * (1.0 / rho[i]) * drho[i];
//				radius[i] += dh * dtv;
			}

			if (nn[i] < 20) {
				radius[i] *= 1.01;
			} //else if (nn[i] > 35) {
			//	radius[i] /= 1.01;
			//}

			//radius[i] = MAX(radius[i], 1.5*contact_radius[i]);
//			radius[i] = MIN(radius[i], 4.0*contact_radius[i]);

		}
	}
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateUlsph::reset_dt() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

int FixSMDIntegrateUlsph::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	double **x0 = atom->x0;

//printf("in FixSMDIntegrateTlsph::pack_forward_comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = x0[j][0];
		buf[m++] = x0[j][1];
		buf[m++] = x0[j][2];
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateUlsph::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	double **x0 = atom->x0;

//printf("in FixSMDIntegrateTlsph::unpack_forward_comm\n");
	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		x0[i][0] = buf[m++];
		x0[i][1] = buf[m++];
		x0[i][2] = buf[m++];
	}
}
