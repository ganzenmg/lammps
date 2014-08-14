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
#include "fix_tlsph_integrate.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "pair.h"
#include <Eigen/Eigen>
#include "domain.h"
#include "neighbor.h"
#include "comm.h"
#include "modify.h"
#include "pair_tlsph.h"

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixTlsphIntegrate::FixTlsphIntegrate(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg < 4) {
		printf("narg=%d\n", narg);
		error->all(FLERR, "Illegal fix tlsph/integrate command");
	}

	vlimit = atof(arg[3]);
	if (vlimit > 0.0) {
		if (comm->me == 0) {
			error->message(FLERR,
					"*** fix tlsph/integrate will cap velocities ***");
		}
	}

	updateReferenceConfigurationFlag = false;
	xsphFlag = false;

	for (int iarg = 4; iarg < narg; iarg++) {
		if (strcmp(arg[iarg], "update") == 0) {
			updateReferenceConfigurationFlag = true;
			if (comm->me == 0) {
				error->message(FLERR,
						"*** fix tlsph/integrate will update the reference configuration every with every neighborlist build ***");
			}
		}

		if (strcmp(arg[iarg], "xsph") == 0) {
			xsphFlag = true;
			if (comm->me == 0) {
				error->message(FLERR,
						"*** fix tlsph/integrate will use XSPH time integration ***");
			}
		}
	}

	time_integrate = 1;
	nRefConfigUpdates = 0;

	// set comm sizes needed by this fix
	comm_forward = 5;

}

/* ---------------------------------------------------------------------- */

int FixTlsphIntegrate::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixTlsphIntegrate::init() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;

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

/* ----------------------------------------------------------------------
 ------------------------------------------------------------------------- */

void FixTlsphIntegrate::initial_integrate(int vflag) {
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
	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	/*
	 * update the reference configuration if needed
	 */
//	if (updateReferenceConfigurationFlag) {
//		if ((neighbor->ago == 0) && update->ntimestep > update->firststep + 1) {
//			if (comm->me == 0) {
//				printf("updating ref config at step: %ld\n", update->ntimestep);
//			}
//			FixTlsphIntegrate::updateReferenceConfiguration();
//		}
//	}
	if (updateReferenceConfigurationFlag) {
		int *updateFlag_ptr = (int *) force->pair->extract("updateFlag_ptr",
				itmp);
		if (updateFlag_ptr == NULL) {
			error->one(FLERR,
					"fix tlsph/integrate failed to access updateFlag pointer. Check if a pair style exist which calculates this quantity.");
		}

		// sum all update flag across processors
		int updateFlag;
		MPI_Allreduce(updateFlag_ptr, &updateFlag, 1, MPI_INT, MPI_MAX, world);

		if (updateFlag > 0) {
			if ((neighbor->ago == 0)
					&& update->ntimestep > update->firststep + 1) {
				if (comm->me == 0) {
					printf("updating ref config at step: %ld\n",
							update->ntimestep);
				}

				FixTlsphIntegrate::updateReferenceConfiguration();
			}
		}
	}

	Vector3d *smoothVel = (Vector3d *) force->pair->extract("smoothVel_ptr",
			itmp);

	if (xsphFlag) {
		if (smoothVel == NULL) {
			error->one(FLERR,
					"fix tlsph/integrate failed to access smoothVel array. Check if a pair style exist which calculates this quantity.");
		}
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

			// extrapolate velocity from half- to full-step
			vest[i][0] = v[i][0] + dtfm * f[i][0];
			vest[i][1] = v[i][1] + dtfm * f[i][1];
			vest[i][2] = v[i][2] + dtfm * f[i][2];

			if (xsphFlag) {
				x[i][0] += dtv * (v[i][0] + 0.5 * smoothVel[i](0));
				x[i][1] += dtv * (v[i][1] + 0.5 * smoothVel[i](1));
				x[i][2] += dtv * (v[i][2] + 0.5 * smoothVel[i](2));
			} else {
				x[i][0] += dtv * v[i][0];
				x[i][1] += dtv * v[i][1];
				x[i][2] += dtv * v[i][2];
			}
		}
	}

}

/* ---------------------------------------------------------------------- */

void FixTlsphIntegrate::final_integrate() {
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
			v[i][0] += dtfm * f[i][0];
			v[i][1] += dtfm * f[i][1];
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

void FixTlsphIntegrate::reset_dt() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;
}

/* ---------------------------------------------------------------------- */

void FixTlsphIntegrate::updateReferenceConfiguration() {
	double **defgrad0 = atom->tlsph_fold;
	double *radius = atom->radius;
	double *contact_radius = atom->contact_radius;
	double **x = atom->x;
	double **x0 = atom->x0;
	double *vfrac = atom->vfrac;
	int *mol = atom->molecule;
	int nlocal = atom->nlocal;
	int i, itmp;
	double J, J0;
	int *mask = atom->mask;
	if (igroup == atom->firstgroup) {
		nlocal = atom->nfirst;
	}

	Matrix3d F0;
	// access current deformation gradient
	// copy data to output array
	itmp = 0;
	Matrix3d *F = (Matrix3d *) force->pair->extract("F_ptr", itmp);
	if (F == NULL) {
		error->all(FLERR,
				"FixTlsphIntegrate::updateReferenceConfiguration() failed to access F array");
	}

	Matrix3d *Fincr = (Matrix3d *) force->pair->extract("Fincr_ptr", itmp);
	if (Fincr == NULL) {
		error->all(FLERR,
				"FixTlsphIntegrate::updateReferenceConfiguration() failed to access Fincr array");
	}

	int *numNeighsRefConfig = (int *) force->pair->extract(
			"numNeighsRefConfig_ptr", itmp);
	if (numNeighsRefConfig == NULL) {
		error->all(FLERR,
				"FixTlsphIntegrate::updateReferenceConfiguration() failed to access numNeighsRefConfig array");
	}

	nRefConfigUpdates++;

	for (i = 0; i < nlocal; i++) {

		/*
		 * only update coordinates if plastic strain exceeds a certain threshold
		 */
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
			J0 = F0.determinant();

			// re-set x0 coordinates
			x0[i][0] = x[i][0];
			x0[i][1] = x[i][1];
			x0[i][2] = x[i][2];

			// make the current deformation gradient the reference deformation gradient
			defgrad0[i][0] = F[i](0, 0);
			defgrad0[i][1] = F[i](0, 1);
			defgrad0[i][2] = F[i](0, 2);
			defgrad0[i][3] = F[i](1, 0);
			defgrad0[i][4] = F[i](1, 1);
			defgrad0[i][5] = F[i](1, 2);
			defgrad0[i][6] = F[i](2, 0);
			defgrad0[i][7] = F[i](2, 1);
			defgrad0[i][8] = F[i](2, 2);

			// adjust particle volumes
			J = F[i].determinant();
			vfrac[i] *= J / J0;

			//if ((J > 2.0) || (J < 0.1)) {
			//	printf("in update reference config: mol=%d, J is %f\n", mol[i], J);
			//}

//			double radiusfactor = MAX(Fincr[i](0, 0), Fincr[i](1, 1));
//			radiusfactor = MAX(radiusfactor, Fincr[i](2, 2));
//
//			if (radiusfactor > 1.0) {
//				radius[i] = radiusfactor * 2.01 * 0.5;
//			}

			if (numNeighsRefConfig[i] < 20) {
				radius[i] *= 1.2;
			} else if (numNeighsRefConfig[i] > 50) {
				radius[i] *= 0.9;
			}

			// do not allow radius to grow excessively
			radius[i] = MIN(radius[i], 20.0 * contact_radius[i]);

		}

	}

	// update of reference config could have changed x0, vfrac, radius
	// communicate these quantities now to ghosts: x0, vfrac, radius
	comm->forward_comm_fix(this);
}

/* ---------------------------------------------------------------------- */

int FixTlsphIntegrate::pack_comm(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
	int i, j, m;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;

	//printf("in pack_comm neigh gcgc\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = x0[j][0];
		buf[m++] = x0[j][1];
		buf[m++] = x0[j][2];

		buf[m++] = vfrac[j];
		buf[m++] = radius[j];
	}
	return 5;
}

/* ---------------------------------------------------------------------- */

void FixTlsphIntegrate::unpack_comm(int n, int first, double *buf) {
	int i, m, last;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;

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
