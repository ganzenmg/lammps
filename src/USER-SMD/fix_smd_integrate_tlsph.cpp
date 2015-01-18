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
#include "stdlib.h"
#include "string.h"
#include "fix_smd_integrate_tlsph.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "pair.h"
#include "neigh_list.h"
#include <Eigen/Eigen>
#include "domain.h"
#include "neighbor.h"
#include "comm.h"
#include "modify.h"
#include "stdio.h"
#include <iostream>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixSMDIntegrateTlsph::FixSMDIntegrateTlsph(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg < 3) {
		printf("narg=%d\n", narg);
		error->all(FLERR, "Illegal fix smd/integrate_tlsph command");
	}

	adjust_radius_flag = false;
	reinitReferenceConfigurationFlag = false;
	updateReferenceConfigurationFlag = false;
	xsphFlag = false;
	vlimit = -1.0;
	int iarg = 3;

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix smd/integrate_tlsph is active for group: %s \n", arg[1]);
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
		} else if (strcmp(arg[iarg], "update") == 0) {
			updateReferenceConfigurationFlag = true;
			if (comm->me == 0) {
				printf("... will update the reference configuration\n");
			}
		} else if (strcmp(arg[iarg], "reinit") == 0) {
			reinitReferenceConfigurationFlag = true;
			if (comm->me == 0) {
				printf("... will reinitialize reference configuration with current configuration\n");
			}
		} else if (strcmp(arg[iarg], "adjust_radius") == 0) {
			adjust_radius_flag = true;

			if (comm->me == 0) {
				printf("... will evolve smoothing length dynamically based on deformation\n");
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
			sprintf(msg, "Illegal keyword for smd/integrate_tlsph: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;
	}

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
	}

	time_integrate = 1;
	nRefConfigUpdates = 0;

	// set comm sizes needed by this fix
	comm_forward = 5;

	atom->add_callback(0);

}

/* ---------------------------------------------------------------------- */

int FixSMDIntegrateTlsph::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::init() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;

	/*
	 * need this to initialize x0 coordinates if not properly set in data file
	 */
	if (reinitReferenceConfigurationFlag == true) {
		double **x = atom->x;
		double **x0 = atom->x0;
		int *mask = atom->mask;
		int nlocal = atom->nlocal;

		for (int i = 0; i < nlocal; i++) {

			if (mask[i] & groupbit) {

				// re-set x0 coordinates
				x0[i][0] = x[i][0];
				x0[i][1] = x[i][1];
				x0[i][2] = x[i][2];

			}
		}
	}
}

/* ----------------------------------------------------------------------
 ------------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::initial_integrate(int vflag) {
	double dtfm, vsq, scale;

	// update v and x of atoms in group

	double **x = atom->x;
	double **v = atom->v;
	double **vest = atom->vest;
	double **f = atom->f;
	double *rmass = atom->rmass;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	int itmp;
	double vxsph_x, vxsph_y, vxsph_z;
	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	/*
	 * update the reference configuration if needed
	 */

	if (updateReferenceConfigurationFlag) {
		int *updateFlag_ptr = (int *) force->pair->extract("smd/tlsph/updateFlag_ptr", itmp);
		if (updateFlag_ptr == NULL) {
			error->one(FLERR,
					"fix smd/integrate_tlsph failed to access updateFlag pointer. Check if a pair style exist which calculates this quantity.");
		}

		// sum all update flag across processors
		int updateFlag;
		MPI_Allreduce(updateFlag_ptr, &updateFlag, 1, MPI_INT, MPI_MAX, world);

		if (updateFlag > 0) {
			if ((neighbor->ago == 0) && update->ntimestep > update->firststep + 1) {
				if (comm->me == 0) {
					printf("updating ref config at step: %ld\n", update->ntimestep);
				}

				FixSMDIntegrateTlsph::updateReferenceConfiguration();
			}
		}
	}

	Vector3d *smoothVelDifference = (Vector3d *) force->pair->extract("smd/tlsph/smoothVel_ptr", itmp);

	if (xsphFlag) {
		if (smoothVelDifference == NULL) {
			error->one(FLERR,
					"fix smd/integrate_tlsph failed to access smoothVel array. Check if a pair style exist which calculates this quantity.");
		}
	}

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			dtfm = dtf / rmass[i];

			// 1st part of Velocity_Verlet: push velocties 1/2 time increment ahead
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

			if (xsphFlag) {

				// construct XSPH velocity
				vxsph_x = v[i][0] + 0.5 * smoothVelDifference[i](0);
				vxsph_y = v[i][1] + 0.5 * smoothVelDifference[i](1);
				vxsph_z = v[i][2] + 0.5 * smoothVelDifference[i](2);

				vest[i][0] = v[i][0] + dtfm * f[i][0];
				vest[i][1] = v[i][1] + dtfm * f[i][1];
				vest[i][2] = v[i][2] + dtfm * f[i][2];

				x[i][0] += dtv * vxsph_x;
				x[i][1] += dtv * vxsph_y;
				x[i][2] += dtv * vxsph_z;
			} else {

				// extrapolate velocity from half- to full-step
				vest[i][0] = v[i][0] + dtfm * f[i][0];
				vest[i][1] = v[i][1] + dtfm * f[i][1];
				vest[i][2] = v[i][2] + dtfm * f[i][2];

				x[i][0] += dtv * v[i][0]; // 2nd part of Velocity-Verlet: push positions one full time increment ahead
				x[i][1] += dtv * v[i][1];
				x[i][2] += dtv * v[i][2];
			}
		}
	}

}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::final_integrate() {
	double dtfm, vsq, scale;

// update v of atoms in group

	double **v = atom->v;
	double **f = atom->f;
	double *e = atom->e;
	double *de = atom->de;
	double *rmass = atom->rmass;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;
	int i;

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			dtfm = dtf / rmass[i];

			v[i][0] += dtfm * f[i][0]; // 3rd part of Velocity-Verlet: push velocities another half time increment ahead
			v[i][1] += dtfm * f[i][1]; // both positions and velocities are now defined at full time-steps.
			v[i][2] += dtfm * f[i][2];

			// limit velocity
			if (vlimit > 0.0) {
				vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
				if (vsq > vlimitsq) {
					scale = sqrt(vlimitsq / vsq);
					v[i][0] *= scale;
					v[i][1] *= scale;
					v[i][2] *= scale;
				}
			}

			e[i] += dtv * de[i];

		}
	}
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::reset_dt() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::updateReferenceConfiguration() {
	double **defgrad0 = atom->tlsph_fold;
	double *radius = atom->radius;
	//double *contact_radius = atom->contact_radius;
	double **x = atom->x;
	double **x0 = atom->x0;
	double *vfrac = atom->vfrac;
	int nlocal = atom->nlocal;
	int i, itmp;
	double J;
	Matrix3d Ftotal;
	Matrix3d C, eye;
	int *mask = atom->mask;
	if (igroup == atom->firstgroup) {
		nlocal = atom->nfirst;
	}

	eye.setIdentity();

	Matrix3d F0;
	// access current deformation gradient
	// copy data to output array
	itmp = 0;

	Matrix3d *Fincr = (Matrix3d *) force->pair->extract("smd/tlsph/Fincr_ptr", itmp);
	if (Fincr == NULL) {
		error->all(FLERR, "FixSMDIntegrateTlsph::updateReferenceConfiguration() failed to access Fincr array");
	}

	int *numNeighsRefConfig = (int *) force->pair->extract("smd/tlsph/numNeighsRefConfig_ptr", itmp);
	if (numNeighsRefConfig == NULL) {
		error->all(FLERR, "FixSMDIntegrateTlsph::updateReferenceConfiguration() failed to access numNeighsRefConfig array");
	}

	nRefConfigUpdates++;

	for (i = 0; i < nlocal; i++) {

		if (mask[i] & groupbit) {

				// need determinant of old deformation gradient associated with reference configuration
				F0(0, 0) = defgrad0[i][0];
				F0(0, 1) = defgrad0[i][1];
				F0(0, 2) = defgrad0[i][2];
				F0(1, 0) = defgrad0[i][3];
				F0(1, 1) = defgrad0[i][4];
				F0(1, 2) = defgrad0[i][5];
				F0(2, 0) = defgrad0[i][6];
				F0(2, 1) = defgrad0[i][7];
				F0(2, 2) = defgrad0[i][8];

				// re-set x0 coordinates
				x0[i][0] = x[i][0];
				x0[i][1] = x[i][1];
				x0[i][2] = x[i][2];

				// compute current total deformation gradient
				Ftotal = F0 * Fincr[i];	// this is the total deformation gradient: reference deformation times incremental deformation

				// store the current deformation gradient the reference deformation gradient
				defgrad0[i][0] = Ftotal(0, 0);
				defgrad0[i][1] = Ftotal(0, 1);
				defgrad0[i][2] = Ftotal(0, 2);
				defgrad0[i][3] = Ftotal(1, 0);
				defgrad0[i][4] = Ftotal(1, 1);
				defgrad0[i][5] = Ftotal(1, 2);
				defgrad0[i][6] = Ftotal(2, 0);
				defgrad0[i][7] = Ftotal(2, 1);
				defgrad0[i][8] = Ftotal(2, 2);

				/*
				 * Adjust particle volume as the reference configuration is changed.
				 * We safeguard against excessive deformations by limiting the adjustment range
				 * to the intervale J \in [0.9..1.1]
				 */
				J = Fincr[i].determinant();
				J = MAX(J, 0.9);
				J = MIN(J, 1.1);
				vfrac[i] *= J;

				if (adjust_radius_flag) {
					radius[i] *= pow(J, 1. / domain->dimension);
				}

				//do not allow radius to grow excessively
				//radius[i] = MIN(radius[i], 10.0 * contact_radius[i]); // this only makes sense for well defined contact radii with compact regular initial node spacings
				//radius[i] = MAX(radius[i], 2.0 * contact_radius[i]);

		}

	}

	// update of reference config could have changed x0, vfrac, radius
	// communicate these quantities now to ghosts: x0, vfrac, radius
	comm->forward_comm_fix(this);
}

/* ---------------------------------------------------------------------- */

int FixSMDIntegrateTlsph::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;

//printf("in FixSMDIntegrateTlsph::pack_forward_comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = x0[j][0];
		buf[m++] = x0[j][1];
		buf[m++] = x0[j][2];

		buf[m++] = vfrac[j];
		buf[m++] = radius[j];
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;

//printf("in FixSMDIntegrateTlsph::unpack_forward_comm\n");
	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		x0[i][0] = buf[m++];
		x0[i][1] = buf[m++];
		x0[i][2] = buf[m++];

		vfrac[i] = buf[m++];
		radius[i] = buf[m++];
	}
}
