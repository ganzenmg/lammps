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

/* ----------------------------------------------------------------------
 Contributing author: Mike Parks (SNL)
 ------------------------------------------------------------------------- */

#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "pair_smd_hertz.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define SQRT2 1.414213562e0

/* ---------------------------------------------------------------------- */

PairHertz::PairHertz(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;
	bulkmodulus = NULL;
	kn = NULL;
	scale = 1.0;
}

/* ---------------------------------------------------------------------- */

PairHertz::~PairHertz() {

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(bulkmodulus);
		memory->destroy(kn);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;
	}
}

/* ---------------------------------------------------------------------- */

void PairHertz::compute(int eflag, int vflag) {
	int i, j, ii, jj, inum, jnum, itype, jtype;
	double xtmp, ytmp, ztmp, delx, dely, delz;
	double rsq, r, evdwl, fpair;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double rcut, r_geom, delta, ri, rj;

	evdwl = 0.0;
	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	double **f = atom->f;
	double **x = atom->x;
	double **v = atom->vest;
	double *rmass = atom->rmass;
	double *de = atom->de;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	double *radius = atom->contact_radius;
	double q1, rho0, c0, mu_ij, visc_magnitude, delVdotDelR, f_visc, wfd, rcutSq, hrSq, deltaE;

	int newton_pair = force->newton_pair;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	// loop over neighbors of my atoms
	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		itype = type[i];
		ri = 1.1 * radius[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			jtype = type[j];
			delx = xtmp - x[j][0];
			dely = ytmp - x[j][1];
			delz = ztmp - x[j][2];
			if (periodic) {
				domain->minimum_image(delx, dely, delz);
			}
			rsq = delx * delx + dely * dely + delz * delz;

			rj = 1.1 * radius[j];


			rcut = ri + rj;
			rcutSq = rcut * rcut;

			if (rsq < rcutSq) {

				r = sqrt(rsq);

				// Hertzian short-range forces
				delta = rcut - r; // overlap distance
				r_geom = ri * rj / rcut;
				if (domain->dimension == 3) {
					//assuming poisson ratio = 1/4 for 3d
					fpair = 1.066666667e0 * bulkmodulus[itype][jtype] * delta * sqrt(delta * r_geom) / (r + 1.0e-2 * rcut); //  units:
					evdwl = r * fpair * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy
				} else {
					//assuming poisson ratio = 1/3 for 2d -- one factor of delta missing compared to 3d
					fpair = 0.16790413e0 * bulkmodulus[itype][jtype] * sqrt(delta * r_geom) / (r + 1.0e-2 * rcut);
					evdwl = r * fpair * 0.6666666666667e0 * delta;
				}

				/*
				 * contact viscosity
				 */

				q1 = 1.0;
				rho0 = 2700.0e-9;
				c0 = sqrt(bulkmodulus[itype][jtype] / rho0);

				hrSq = rcutSq - rsq; // [m^2]
				wfd = -14.0323944878e0 * hrSq / (rcutSq * rcutSq * rcutSq); // [1/m^4] ==> correct for dW/dr in 3D

				delVdotDelR = delx * (v[j][0] - v[i][0]) + dely * (v[j][1] - v[i][1]) + delz * (v[j][2] - v[i][2]); // units: m^2/s
				mu_ij = rcut * delVdotDelR / (r * r + 0.1 * rcut * rcut); // m * m/s * m / m*m ==> units m/s
				visc_magnitude = q1 * c0 * mu_ij / rho0; // units: m^5 / (s^-2 kg)
				f_visc = -rmass[i] * rmass[j] * visc_magnitude * wfd / (r + 1.0e-1 * rcut); // units: kg * s^-2

				//printf("fpair = %f, fvisc = %f, mu_ij, \n", fpair, f_visc);

				deltaE = 0.5 * f_visc * delVdotDelR;

				fpair += f_visc;

				if (evflag) {
					ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
				}

				f[i][0] += delx * fpair;
				f[i][1] += dely * fpair;
				f[i][2] += delz * fpair;
				de[i] += deltaE;

				if (newton_pair || j < nlocal) {
					f[j][0] -= delx * fpair;
					f[j][1] -= dely * fpair;
					f[j][2] -= delz * fpair;
					de[j] += deltaE;
				}

			}
		}
	}
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairHertz::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(bulkmodulus, n + 1, n + 1, "pair:kspring");
	memory->create(kn, n + 1, n + 1, "pair:kn");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairHertz::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal number of args for pair_style hertz");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairHertz::coeff(int narg, char **arg) {
	if (narg != 3)
		error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	double bulkmodulus_one = atof(arg[2]);

	// set short-range force constant
	double kn_one = 0.0;
	if (domain->dimension == 3) {
		kn_one = (16. / 15.) * bulkmodulus_one; //assuming poisson ratio = 1/4 for 3d
	} else {
		kn_one = 0.251856195 * (2. / 3.) * bulkmodulus_one; //assuming poisson ratio = 1/3 for 2d
	}

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			bulkmodulus[i][j] = bulkmodulus_one;
			kn[i][j] = kn_one;
			setflag[i][j] = 1;
			count++;
		}
	}

	if (count == 0)
		error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairHertz::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	bulkmodulus[j][i] = bulkmodulus[i][j];
	kn[j][i] = kn[i][j];

	// cutoff = sum of max I,J radii for
	// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);

	if (comm->me == 0) {
		printf("cutoff for pair hertz = %f\n", cutoff);
	}
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairHertz::init_style() {
	int i;

	// error checks

	if (!atom->contact_radius_flag)
		error->all(FLERR, "Pair style tlsph/ipc_hertz requires atom style with contact_radius");

	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;

	// set maxrad_dynamic and maxrad_frozen for each type
	// include future Fix pour particles as dynamic

	for (i = 1; i <= atom->ntypes; i++)
		onerad_dynamic[i] = onerad_frozen[i] = 0.0;

	double *radius = atom->radius;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	for (i = 0; i < nlocal; i++) {
		onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);
	}

	MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
	MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairHertz::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairHertz::memory_usage() {

	return 0.0;
}

