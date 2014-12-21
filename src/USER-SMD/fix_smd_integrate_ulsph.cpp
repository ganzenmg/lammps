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
		error->all(FLERR, "fix sph_fluid command requires atom_style with both energy and density");

	if (narg < 4)
		error->all(FLERR, "Illegal number of arguments for fix sph_fluid command");

	vlimit = force->numeric(FLERR, arg[3]);
	if (vlimit > 0.0) {
		if (comm->me == 0) {
			error->message(FLERR, "*** fix sph/fluid will cap velocities ***");
		}
	}

	adjust_radius_flag = false;
	xsphFlag = false;
	int iarg = 4;
	while (true) {
		if (strcmp(arg[iarg], "xsph") == 0) {
			xsphFlag = true;
			if (comm->me == 0) {
				error->message(FLERR, "*** fix sph_fluid will use XSPH time integration ***");
			}
		} else if (strcmp(arg[iarg], "adjust_radius") == 0) {
			adjust_radius_flag = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following adjust_radius");
			}

			adjust_radius_factor = force->numeric(FLERR, arg[iarg]);
			if (comm->me == 0) {
				printf("adjust_radius factor is %f\n", adjust_radius_factor);
				error->message(FLERR, "*** fix sph_fluid will adjust smoothing length dynamically ***");
			}
		}

		iarg++;

		if (iarg >= narg) {
			break;
		}

	}

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
	int i;
	double dtfm, vsq, scale;

	int itmp = 0;
	Vector3d *smoothVel = (Vector3d *) force->pair->extract("smd/ulsph/smoothVel_ptr", itmp);

	if (xsphFlag) {
		if (smoothVel == NULL) {
			error->one(FLERR, "fix sph_fluid failed to access smoothVel array");
		}
	}

	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			dtfm = dtf / rmass[i];

			e[i] += dtf * de[i]; // half-step update of particle internal energy
			rho[i] += dtf * drho[i]; // ... and density

			v[i][0] += dtfm * f[i][0];
			v[i][1] += dtfm * f[i][1];
			v[i][2] += dtfm * f[i][2];

			// extrapolate velocity from half- to full-step
			vest[i][0] = v[i][0] + dtfm * f[i][0];
			vest[i][1] = v[i][1] + dtfm * f[i][1];
			vest[i][2] = v[i][2] + dtfm * f[i][2];

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

			if (xsphFlag) {
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

void FixSMDIntegrateUlsph::final_integrate() {

	// update v, rho, and e of atoms in group

	double **v = atom->v;
	double **f = atom->f;
	double *e = atom->e;
	double *de = atom->de;
	double *rho = atom->rho;
	double *drho = atom->drho;
	double *radius = atom->radius;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;
	double dtfm, vsq, scale;
	double *rmass = atom->rmass;

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
				radius[i] = adjust_radius_factor * pow(rmass[i] / rho[i], 1./domain->dimension); // Monaghan approach for setting the radius
			}
		}
	}
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateUlsph::reset_dt() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
}
