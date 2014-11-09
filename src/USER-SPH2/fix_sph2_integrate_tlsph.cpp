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
#include "fix_sph2_integrate_tlsph.h"
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

FixSph2IntegrateTlsph::FixSph2IntegrateTlsph(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg < 5) {
		printf("narg=%d\n", narg);
		error->all(FLERR, "Illegal fix tlsph/integrate command");
	}

	vlimit = atof(arg[3]);
	if (vlimit > 0.0) {
		if (comm->me == 0) {
			error->message(FLERR, "*** fix tlsph/integrate will cap velocities ***");
		}
	}

	smoothPeriod = force->inumeric(FLERR, arg[4]);
	updateReferenceConfigurationFlag = false;
	xsphFlag = false;

	for (int iarg = 5; iarg < narg; iarg++) {
		if (strcmp(arg[iarg], "update") == 0) {
			updateReferenceConfigurationFlag = true;
			if (comm->me == 0) {
				error->message(FLERR, "*** fix tlsph/integrate will update the reference configuration ***");
			}
		}

		if (strcmp(arg[iarg], "xsph") == 0) {
			xsphFlag = true;
			if (comm->me == 0) {
				error->message(FLERR, "*** fix tlsph/integrate will use XSPH time integration ***");
			}
		}
	}

	time_integrate = 1;
	nRefConfigUpdates = 0;

	// set comm sizes needed by this fix
	comm_forward = 5;

	atom->add_callback(0);

}

/* ---------------------------------------------------------------------- */

int FixSph2IntegrateTlsph::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSph2IntegrateTlsph::init() {
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

void FixSph2IntegrateTlsph::initial_integrate(int vflag) {
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
	 * perform smoothing of stress field
	 */

	if (smoothPeriod > 0) {
		if (update->ntimestep % smoothPeriod == 0) {
			printf("smoothing\n");
			smooth_fields();
		}
	}

	/*
	 * update the reference configuration if needed
	 */

	if (updateReferenceConfigurationFlag) {
		int *updateFlag_ptr = (int *) force->pair->extract("sph2/tlsph/updateFlag_ptr", itmp);
		if (updateFlag_ptr == NULL) {
			error->one(FLERR,
					"fix tlsph/integrate failed to access updateFlag pointer. Check if a pair style exist which calculates this quantity.");
		}

		// sum all update flag across processors
		int updateFlag;
		MPI_Allreduce(updateFlag_ptr, &updateFlag, 1, MPI_INT, MPI_MAX, world);

		if (updateFlag > 0) {
			if ((neighbor->ago == 0) && update->ntimestep > update->firststep + 1) {
				if (comm->me == 0) {
					printf("updating ref config at step: %ld\n", update->ntimestep);
				}

				FixSph2IntegrateTlsph::updateReferenceConfiguration();
			}
		}
	}

	Vector3d *smoothVel = (Vector3d *) force->pair->extract("sph2/tlsph/smoothVel_ptr", itmp);

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
				vest[i][0] -= 0.5 * smoothVel[i](0); // extrapolated velocities need to be smoothed as well
				vest[i][1] -= 0.5 * smoothVel[i](1);
				vest[i][2] -= 0.5 * smoothVel[i](2);

				x[i][0] += dtv * (v[i][0] - 0.5 * smoothVel[i](0));
				x[i][1] += dtv * (v[i][1] - 0.5 * smoothVel[i](1));
				x[i][2] += dtv * (v[i][2] - 0.5 * smoothVel[i](2));
			} else {
				x[i][0] += dtv * v[i][0];
				x[i][1] += dtv * v[i][1];
				x[i][2] += dtv * v[i][2];
			}
		}
	}

}

/* ---------------------------------------------------------------------- */

void FixSph2IntegrateTlsph::final_integrate() {
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

void FixSph2IntegrateTlsph::reset_dt() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;
}

/* ---------------------------------------------------------------------- */

void FixSph2IntegrateTlsph::updateReferenceConfiguration() {
	double **defgrad0 = atom->tlsph_fold;
	double *radius = atom->radius;
	double *contact_radius = atom->contact_radius;
	double **x = atom->x;
	double **x0 = atom->x0;
	double *vfrac = atom->vfrac;
	int *mol = atom->molecule;
	int nlocal = atom->nlocal;
	int i, itmp;
	double J, J0, vol0;
	Matrix3d Ftotal;
	int *mask = atom->mask;
	if (igroup == atom->firstgroup) {
		nlocal = atom->nfirst;
	}

	Matrix3d F0;
	// access current deformation gradient
	// copy data to output array
	itmp = 0;

	Matrix3d *Fincr = (Matrix3d *) force->pair->extract("sph2/tlsph/Fincr_ptr", itmp);
	if (Fincr == NULL) {
		error->all(FLERR, "FixSph2IntegrateTlsph::updateReferenceConfiguration() failed to access Fincr array");
	}

	int *numNeighsRefConfig = (int *) force->pair->extract("sph2/tlsph/numNeighsRefConfig_ptr", itmp);
	if (numNeighsRefConfig == NULL) {
		error->all(FLERR, "FixSph2IntegrateTlsph::updateReferenceConfiguration() failed to access numNeighsRefConfig array");
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

			// compute current total deformation gradient
			Ftotal = F0 * Fincr[i]; // this is the total deformation gradient: reference deformation times incremental deformation

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

			// adjust particle volumes
			J = Fincr[i].determinant();
			vfrac[i] *= J;

			if (numNeighsRefConfig[i] < 25) {
				radius[i] *= 1.1;
			} else if (numNeighsRefConfig[i] > 80) {
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

int FixSph2IntegrateTlsph::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;

//printf("in FixSph2IntegrateTlsph::pack_forward_comm\n");
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

void FixSph2IntegrateTlsph::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double **x0 = atom->x0;

//printf("in FixSph2IntegrateTlsph::unpack_forward_comm\n");
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

void FixSph2IntegrateTlsph::smooth_fields() {
	int i, j, ii, jj, m, n, inum, jnum;
	int *ilist, *jlist, *numneigh, **firstneigh;
	int *touch, **firsttouch;
	double *shear, *allshear, **firstshear;
	double h, r0Sq;
	Vector3d x0i, x0j, dx0;

	// nlocal may include atoms added since last neigh build

	int nlocal = atom->nlocal;

	// zero npartner for all current atoms
	// clear 2 page data structures

	//for (i = 0; i < nlocal; i++) npartner[i] = 0;

	// 1st loop over neighbor list
	// calculate npartner for each owned atom
	// nlocal_neigh = nlocal when neigh list was built, may be smaller than nlocal

	double *radius = atom->radius;
	double **x0 = atom->x0;
	double **tlsph_stress = atom->tlsph_stress;
	tagint *tag = atom->tag;
	NeighList *list = pair->list;
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;
	Matrix3d *average_stress;
	double *shepard_weight;
	Matrix3d stress;
	double weight;

	average_stress = new Matrix3d[nlocal];
	shepard_weight = new double[nlocal];

	for (i = 0; i < nlocal; i++) {

		stress(0, 0) = tlsph_stress[i][0];
		stress(0, 1) = tlsph_stress[i][1];
		stress(0, 2) = tlsph_stress[i][2];
		stress(1, 1) = tlsph_stress[i][3];
		stress(1, 2) = tlsph_stress[i][4];
		stress(2, 2) = tlsph_stress[i][5];

		weight = 1.0;

		average_stress[i] = weight * stress;
		shepard_weight[i] = weight;
	}

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		x0i << x0[i][0], x0[i][1], x0[i][2];

		for (jj = 0; jj < jnum; jj++) {

			j = jlist[jj];
			j &= NEIGHMASK;

			x0j << x0[j][0], x0[j][1], x0[j][2];

			dx0 = x0j - x0i;
			r0Sq = dx0.squaredNorm();

			h = radius[i] + radius[j];

			if (r0Sq < h * h) {

				weight = 1.0; //(h * h - r0Sq) / (h * h);

				stress(0, 0) = tlsph_stress[j][0];
				stress(0, 1) = tlsph_stress[j][1];
				stress(0, 2) = tlsph_stress[j][2];
				stress(1, 1) = tlsph_stress[j][3];
				stress(1, 2) = tlsph_stress[j][4];
				stress(2, 2) = tlsph_stress[j][5];

				average_stress[i] += weight * stress;
				shepard_weight[i] += weight;

				//printf("r0sq = %f\n", r0Sq);

				if (j < nlocal) {

					stress(0, 0) = tlsph_stress[i][0];
					stress(0, 1) = tlsph_stress[i][1];
					stress(0, 2) = tlsph_stress[i][2];
					stress(1, 1) = tlsph_stress[i][3];
					stress(1, 2) = tlsph_stress[i][4];
					stress(2, 2) = tlsph_stress[i][5];

					average_stress[j] += weight * stress;
					shepard_weight[j] += weight;

				}
			}
		}

		for (i = 0; i < nlocal; i++) {

			average_stress[i] /= shepard_weight[i];

			stress(0, 0) = tlsph_stress[i][0];
			stress(0, 1) = tlsph_stress[i][1];
			stress(0, 2) = tlsph_stress[i][2];
			stress(1, 1) = tlsph_stress[i][3];
			stress(1, 2) = tlsph_stress[i][4];
			stress(2, 2) = tlsph_stress[i][5];

//			cout << "original stress " << endl << stress << endl << endl;
//			cout << "average  stress " << endl << average_stress[i] << endl << endl;
//			cout << "------------------------------------------" << endl;

			tlsph_stress[i][0] = average_stress[i](0, 0);
			tlsph_stress[i][1] = average_stress[i](0, 1);
			tlsph_stress[i][2] = average_stress[i](0, 2);
			tlsph_stress[i][3] = average_stress[i](1, 1);
			tlsph_stress[i][4] = average_stress[i](1, 2);
			tlsph_stress[i][5] = average_stress[i](2, 2);

//			tlsph_stress[i][0] = stress(0, 0);
//			tlsph_stress[i][1] = stress(0, 1);
//			tlsph_stress[i][2] = stress(0, 2);
//			tlsph_stress[i][3] = stress(1, 1);
//			tlsph_stress[i][4] = stress(1, 2);
//			tlsph_stress[i][5] = stress(2, 2);

		}

	}

	delete[] average_stress;
	delete[] shepard_weight;

}
