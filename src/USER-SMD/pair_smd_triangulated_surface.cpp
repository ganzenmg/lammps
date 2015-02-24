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
#include "pair_smd_triangulated_surface.h"
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
#include <Eigen/Eigen>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace LAMMPS_NS;
using namespace Eigen;

#define SQRT2 1.414213562e0

/* ---------------------------------------------------------------------- */

PairTriSurf::PairTriSurf(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;
	bulkmodulus = NULL;
	kn = NULL;
	scale = 1.0;
}

/* ---------------------------------------------------------------------- */

PairTriSurf::~PairTriSurf() {

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

void PairTriSurf::compute(int eflag, int vflag) {
	int i, j, ii, jj, inum, jnum, itype, jtype;
	double rsq, r, evdwl, fpair;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double rcut, r_geom, delta, ri, rj, touch_distance;
	int tri, particle;
	Vector3d normal, x1, x2, x3, x4, x13, x23, x43, w, cp, x4cp, vnew, v_old;;
	Vector3d xi, x_center, dx;
	Matrix2d C;
	Vector2d w2d, rhs;

	evdwl = 0.0;
	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	int *mol = atom->molecule;
	double **f = atom->f;
	double **tlsph_fold = atom->tlsph_fold;
	double **x = atom->x;
	double **x0 = atom->x0;
	double **v = atom->v;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	double *radius = atom->contact_radius;
	double rcutSq;
	Vector3d offset;

	int newton_pair = force->newton_pair;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	int max_neighs = 0;

	// loop over neighbors of my atoms using a full neighbor list
	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];

		// we are only interested in particles, not in triangles.
		if (mol[i] >= 65535) {
			continue;
		}

		// now we know that i is a particle
		particle = i;
		x4 << x[i][0], x[i][1], x[i][2];
		itype = type[i];
		ri = scale * radius[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		max_neighs = MAX(max_neighs, jnum);

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];

			j &= NEIGHMASK;

			jtype = type[j];

			if (mol[j] < 65535) {
				error->one(FLERR, "j has mol id < 65535 and is not a triangle. This should not happen.");
			}
			tri = j;

			x_center << x[j][0], x[j][1], x[j][2]; // center of triangle
			dx = x_center - x4; //
			if (periodic) {
				domain->minimum_image(dx(0), dx(1), dx(2));
			}
			rsq = dx.squaredNorm();

			rj = scale * radius[j];
			rcut = ri + rj;
			rcutSq = rcut * rcut;

			//printf("type i=%d, type j=%d, r=%f, ri=%f, rj=%f\n", itype, jtype, sqrt(rsq), ri, rj);

			if (rsq < rcutSq) {

				/*
				 * decide which one of i, j is triangle and which is particle
				 */
				if (mol[particle] >= 65535) {
					error->one(FLERR, "both i and j have mol id >= 65535. This should not happen.");
				}

				/*
				 * gather triangle information
				 */
				normal(0) = x0[tri][0];
				normal(1) = x0[tri][1];
				normal(2) = x0[tri][2];

				/*
				 * distance check: is particle closer than its radius to the triangle plane?
				 */
				if (fabs(dx.dot(normal)) < radius[particle]) {
					/*
					 * get other two triangle vertices
					 */
					x1(0) = tlsph_fold[tri][0];
					x1(1) = tlsph_fold[tri][1];
					x1(2) = tlsph_fold[tri][2];
					x2(0) = tlsph_fold[tri][3];
					x2(1) = tlsph_fold[tri][4];
					x2(2) = tlsph_fold[tri][5];
					x3(0) = tlsph_fold[tri][6];
					x3(1) = tlsph_fold[tri][7];
					x3(2) = tlsph_fold[tri][8];

					/*
					 * pre-compute stuff
					 */
					x13 = x1 - x3;
					x23 = x2 - x3;
					x43 = x4 - x3;

					/*
					 * get barycentric coordinates, see
					 * "Robust Treatment of Collisions, Contact and Friction for Cloth Animation",
					 * Robert Bridson, Ronald Fedkiw, John Anderson
					 */
					C(0, 0) = x13.dot(x13);
					C(0, 1) = x13.dot(x23);
					C(1, 0) = x13.dot(x23);
					C(1, 1) = x23.dot(x23);

					rhs(0) = x13.dot(x43);
					rhs(1) = x23.dot(x43);

					w2d = C.inverse() * rhs;

					w(0) = w2d(0);
					w(1) = w2d(1);
					w(2) = 1.0 - (w(0) + w(1));

					/*
					 * skip this interaction if barycentric coordinate is outside triangle
					 * this is important for collissions which occur in or nearly in the plane of the triangle,
					 * i.e., a node to edge contact.
					 */

					if ((w(0) > 1.0) || (w(0) < 0.0)) {
						continue;
					}

					if ((w(1) > 1.0) || (w(1) < 0.0)) {
						continue;
					}

					if ((w(2) > 1.0) || (w(2) < 0.0)) {
						continue;
					}

					/*
					 * determine closest point in triangle plane
					 * This is actually the closest point _inside_ the triangle due to the checks above
					 */
					cp = w(0) * x1 + w(1) * x2 + w(2) * x3;

					/*
					 * distance to closest point
					 */
					x4cp = x4 - cp;

					/*
					 * flip normal to point in direction of x4cp
					 */

					if (x4cp.dot(normal) < 0.0) {
						normal *= -1.0;
					}

					/*
					 * distance to closest point in triangle
					 */

					r = x4cp.norm();

					/*
					 * penalty force pushes particle away from triangle
					 */
					if (r < 1.0 * radius[particle]) {

						delta = radius[particle] - r; // overlap distance
						r_geom = radius[particle];
						fpair = 1.066666667e0 * bulkmodulus[itype][jtype] * delta * sqrt(delta * r_geom)
								/ (r + 1.0e-2 * radius[particle]); //  units:
						evdwl = r * fpair * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy

						if (evflag) {
							ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, x4cp(0), x4cp(1), x4cp(2));
						}

						f[i][0] += x4cp(0) * fpair;
						f[i][1] += x4cp(1) * fpair;
						f[i][2] += x4cp(2) * fpair;

					}

					/*
					 * if particle comes too close to triangle, reflect its velocity and explicitely move it away
					 */

					touch_distance = 0.5 * radius[particle];
					if (r < touch_distance) {

						/*
						 * reflect velocity if it points toward triangle
						 */

						v_old << v[particle][0], v[particle][1], v[particle][2];
						if (v_old.dot(normal) < 0.0) {
							//printf("flipping velocity\n");
							vnew = 1.0 * (-2.0 * v_old.dot(normal) * normal + v_old);
							v[particle][0] = vnew(0);
							v[particle][1] = vnew(1);
							v[particle][2] = vnew(2);
						}

						x[particle][0] = cp(0) + touch_distance * normal(0);
						x[particle][1] = cp(1) + touch_distance * normal(1);
						x[particle][2] = cp(2) + touch_distance * normal(2);

					}

				}
			}
		}
	}

//	int max_neighs_all = 0;
//	MPI_Allreduce(&max_neighs, &max_neighs_all, 1, MPI_INT, MPI_MAX, world);
//	if (comm->me == 0) {
//		printf("max. neighs in tri pair is %d\n", max_neighs_all);
//	}
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairTriSurf::allocate() {
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

void PairTriSurf::settings(int narg, char **arg) {
	if (narg != 1)
		error->all(FLERR, "Illegal number of args for pair_style smd/tri_surface");

	scale = force->numeric(FLERR, arg[0]);
	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("SMD/TRI_SURFACE CONTACT SETTINGS:\n");
		printf("... effective contact radius is scaled by %f\n", scale);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
	}

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairTriSurf::coeff(int narg, char **arg) {
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

double PairTriSurf::init_one(int i, int j) {

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
		printf("cutoff for pair smd/smd/tri_surface = %f\n", cutoff);
	}
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairTriSurf::init_style() {
	int i;

	// error checks

	if (!atom->contact_radius_flag)
		error->all(FLERR, "Pair style smd/smd/tri_surface requires atom style with contact_radius");

	// old: half list
//	int irequest = neighbor->request(this);
//	neighbor->requests[irequest]->half = 0;
//	neighbor->requests[irequest]->gran = 1;

	// need a full neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;

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

void PairTriSurf::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairTriSurf::memory_usage() {

	return 0.0;
}

