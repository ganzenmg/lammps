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
 Contributing authors: Mike Parks (SNL), Ezwanur Rahman, J.T. Foster (UTSA)
 ------------------------------------------------------------------------- */

#include "math.h"
#include "fix_smd_wall_surface.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "update.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "lattice.h"
#include "memory.h"
#include "error.h"
#include <Eigen/Eigen>
#include <stdio.h>
#include <iostream>
#include "atom_vec.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace Eigen;
using namespace std;
#define DELTA 16384
#define EPSILON 1.0e-6
enum {
	LAYOUT_UNIFORM, LAYOUT_NONUNIFORM, LAYOUT_TILED
};
// several files

/* ---------------------------------------------------------------------- */

FixSMDWallSurface::FixSMDWallSurface(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	restart_global = 1;
	restart_peratom = 1;
	first = 1;

	// perform initial allocation of atom-based arrays
	// register with atom class
	// set maxpartner = 1 as placeholder

	maxpartner = 1;
	npartner = NULL;
	partner = NULL;
	r0 = plastic_stretch = NULL;
	vinter = NULL;

	maxTrianglePairs = 1;
	nTrianglePairs = NULL;
	trianglePairs = NULL;
	trianglePairAngle0 = NULL;
	trianglePairPlasticAngle = NULL;

	nmax = atom->nmax;
	grow_arrays(atom->nmax);

	atom->add_callback(0);
	atom->add_callback(1);
	atom->add_callback(2); // for border communication

	// initialize npartner to 0 so atom migration is OK the 1st time

	int nlocal = atom->nlocal;
	for (int i = 0; i < nlocal; i++) {
		npartner[i] = 0;
		nTrianglePairs[i] = 0;
	}

	// set comm sizes needed by this fix

	comm_forward = 1;
	comm_border = 1;
}

/* ---------------------------------------------------------------------- */

FixSMDWallSurface::~FixSMDWallSurface() {
	// unregister this fix so atom class doesn't invoke it any more

	atom->delete_callback(id, 0);
	atom->delete_callback(id, 1);

	// delete locally stored arrays

	memory->destroy(npartner);
	memory->destroy(partner);
	memory->destroy(r0);
	memory->destroy(plastic_stretch);
	memory->destroy(vinter);

	memory->destroy(nTrianglePairs);
	memory->destroy(trianglePairs);
	memory->destroy(trianglePairAngle0);
	memory->destroy(trianglePairPlasticAngle);

}

/* ---------------------------------------------------------------------- */

int FixSMDWallSurface::setmask() {
	int mask = 0;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDWallSurface::init() {
	if (!first)
		return;

	int irequest = neighbor->request((void *) this);

	// can utilize gran neighbor list
	neighbor->requests[irequest]->pair = 0;
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;
	neighbor->requests[irequest]->fix = 1;
	neighbor->requests[irequest]->full = 0;
	neighbor->requests[irequest]->occasional = 1;

}

/* ---------------------------------------------------------------------- */

void FixSMDWallSurface::init_list(int id, NeighList *ptr) {
	list = ptr;
}

/* ----------------------------------------------------------------------
 For minimization: setup as with dynamics
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::min_setup(int vflag) {
	setup(vflag);
}

/* ----------------------------------------------------------------------
 create initial list of neighbor partners via call to neighbor->build()
 must be done in setup (not init) since fix init comes before neigh init
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::setup(int vflag) {
	int i, j, ii, jj, itype, jtype, inum, jnum;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq, cutsq;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;

	double **x = atom->x;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	int *type = atom->type;
	tagint *tag = atom->tag;
	tagint *molecule = atom->molecule;
	int nlocal = atom->nlocal;
	int maxall;

	// only build list of bonds on very first run

	if (!first)
		return;
	first = 0;

	// set bounds for my proc
	// if periodic and I am lo/hi proc, adjust bounds by EPSILON
	// insures all data atoms will be owned even with round-off

	int triclinic = domain->triclinic;

	double epsilon[3];
	if (triclinic)
		epsilon[0] = epsilon[1] = epsilon[2] = EPSILON;
	else {
		epsilon[0] = domain->prd[0] * EPSILON;
		epsilon[1] = domain->prd[1] * EPSILON;
		epsilon[2] = domain->prd[2] * EPSILON;
	}

	if (triclinic == 0) {
		sublo[0] = domain->sublo[0];
		subhi[0] = domain->subhi[0];
		sublo[1] = domain->sublo[1];
		subhi[1] = domain->subhi[1];
		sublo[2] = domain->sublo[2];
		subhi[2] = domain->subhi[2];
	} else {
		sublo[0] = domain->sublo_lamda[0];
		subhi[0] = domain->subhi_lamda[0];
		sublo[1] = domain->sublo_lamda[1];
		subhi[1] = domain->subhi_lamda[1];
		sublo[2] = domain->sublo_lamda[2];
		subhi[2] = domain->subhi_lamda[2];
	}

	if (comm->layout != LAYOUT_TILED) {
		if (domain->xperiodic) {
			if (comm->myloc[0] == 0)
				sublo[0] -= epsilon[0];
			if (comm->myloc[0] == comm->procgrid[0] - 1)
				subhi[0] += epsilon[0];
		}
		if (domain->yperiodic) {
			if (comm->myloc[1] == 0)
				sublo[1] -= epsilon[1];
			if (comm->myloc[1] == comm->procgrid[1] - 1)
				subhi[1] += epsilon[1];
		}
		if (domain->zperiodic) {
			if (comm->myloc[2] == 0)
				sublo[2] -= epsilon[2];
			if (comm->myloc[2] == comm->procgrid[2] - 1)
				subhi[2] += epsilon[2];
		}

	} else {
		if (domain->xperiodic) {
			if (comm->mysplit[0][0] == 0.0)
				sublo[0] -= epsilon[0];
			if (comm->mysplit[0][1] == 1.0)
				subhi[0] += epsilon[0];
		}
		if (domain->yperiodic) {
			if (comm->mysplit[1][0] == 0.0)
				sublo[1] -= epsilon[1];
			if (comm->mysplit[1][1] == 1.0)
				subhi[1] += epsilon[1];
		}
		if (domain->zperiodic) {
			if (comm->mysplit[2][0] == 0.0)
				sublo[2] -= epsilon[2];
			if (comm->mysplit[2][1] == 1.0)
				subhi[2] += epsilon[2];
		}
	}

	read_triangles(0);

//	// build full neighbor list, will copy or build as necessary
//
//	neighbor->build_one(list->index);
//
//	inum = list->inum;
//	ilist = list->ilist;
//	numneigh = list->numneigh;
//	firstneigh = list->firstneigh;
//
//	// scan neighbor list to set maxpartner
//
//	for (ii = 0; ii < inum; ii++) {
//		i = ilist[ii];
//
//		if (molecule[i] == 1000) {
//
//			xtmp = x[i][0];
//			ytmp = x[i][1];
//			ztmp = x[i][2];
//			itype = type[i];
//			jlist = firstneigh[i];
//			jnum = numneigh[i];
//
//			for (jj = 0; jj < jnum; jj++) {
//				j = jlist[jj];
//				j &= NEIGHMASK;
//
//				if (molecule[i] == molecule[j]) {
//
//					delx = xtmp - x[j][0];
//					dely = ytmp - x[j][1];
//					delz = ztmp - x[j][2];
//					rsq = delx * delx + dely * dely + delz * delz;
//					jtype = type[j];
//
//					cutsq = (radius[i] + radius[j]) * (radius[i] + radius[j]);
//
//					if (rsq <= cutsq) {
//						npartner[i]++;
//
//						if (j < nlocal) {
//							npartner[j]++;
//						}
//					} // end check for distance
//				} // end check for molecule[i] == molecule[j]
//			}
//		} // end check for molecule[i] == 1000
//	}
//
//	maxpartner = 0;
//	for (i = 0; i < nlocal; i++)
//		maxpartner = MAX(maxpartner, npartner[i]);
//	MPI_Allreduce(&maxpartner, &maxall, 1, MPI_INT, MPI_MAX, world);
//	maxpartner = maxall;
//
//	/*
//	 * scan triangles to get maximum number of triangle pairs per atom
//	 */
//	read_triangles(0); // first pass: get number of local triangles
//	maxTrianglePairs = 0;
//	for (i = 0; i < nlocal; i++) {
//		maxTrianglePairs = MAX(maxTrianglePairs, nTrianglePairs[i]);
//	}
//	printf("maxTrianglePairs before MPI reduce %d\n", maxTrianglePairs);
//	MPI_Allreduce(&maxTrianglePairs, &maxall, 1, MPI_INT, MPI_MAX, world);
//	maxTrianglePairs = maxall;
//	printf("maxTrianglePairs after second pass is %d\n", maxTrianglePairs);
//
//	// realloc arrays with correct value for maxpartner
//
//	memory->destroy(partner);
//	memory->destroy(r0);
//	memory->destroy(plastic_stretch);
//	memory->destroy(npartner);
//	memory->destroy(trianglePairs);
//	memory->destroy(nTrianglePairs);
//	memory->destroy(trianglePairAngle0);
//	memory->destroy(trianglePairPlasticAngle);
//
//	npartner = NULL;
//	partner = NULL;
//	r0 = plastic_stretch = NULL;
//	trianglePairs = NULL;
//	nTrianglePairs = NULL;
//	trianglePairAngle0 = NULL;
//	trianglePairPlasticAngle = NULL;
//	nmax = atom->nmax;
//	grow_arrays(nmax);
//
//	/*
//	 * read triangle pairs
//	 */
//	read_triangles(1); // second pass
//	setup_bending_triangles();
//
//	// create partner list and r0 values from neighbor list
//	// compute vinter for each atom
//
//	for (i = 0; i < nmax; i++) {
//		npartner[i] = 0;
//		vinter[i] = 0.0;
//	}
//
//	for (ii = 0; ii < inum; ii++) {
//		i = ilist[ii];
//
//		if (molecule[i] == 1000) {
//
//			xtmp = x[i][0];
//			ytmp = x[i][1];
//			ztmp = x[i][2];
//			itype = type[i];
//			jlist = firstneigh[i];
//			jnum = numneigh[i];
//
//			for (jj = 0; jj < jnum; jj++) {
//				j = jlist[jj];
//				j &= NEIGHMASK;
//
//				if (molecule[i] == molecule[j]) {
//
//					delx = xtmp - x[j][0];
//					dely = ytmp - x[j][1];
//					delz = ztmp - x[j][2];
//					rsq = delx * delx + dely * dely + delz * delz;
//					jtype = type[j];
//
//					cutsq = (radius[i] + radius[j]) * (radius[i] + radius[j]);
//
//					if (rsq <= cutsq) {
//						partner[i][npartner[i]] = tag[j];
//						r0[i][npartner[i]] = sqrt(rsq);
//						plastic_stretch[i][npartner[i]] = 0.0;
//						npartner[i]++;
//						vinter[i] += vfrac[j];
//
//						if (j < nlocal) {
//							partner[j][npartner[j]] = tag[i];
//							r0[j][npartner[j]] = sqrt(rsq);
//							plastic_stretch[j][npartner[j]] = 0.0;
//							npartner[j]++;
//							vinter[j] += vfrac[i];
//						}
//					} // end check for distance
//				} // end check for molecule[i] == molecule[j]
//			} // end check for molecule[i] == 1000
//		}
//	}
//
//	// sanity check: does any atom appear twice in any neigborlist?
//	// should only be possible if using pbc and domain < 2*delta
//
//	if (domain->xperiodic || domain->yperiodic || domain->zperiodic) {
//		for (i = 0; i < nlocal; i++) {
//			jnum = npartner[i];
//			//printf("\n\n partners of atom tag %d:\n", tag[i]);
//			for (jj = 0; jj < jnum; jj++) {
//				//printf("%d\n", partner[i][jj]);
//				for (int kk = jj + 1; kk < jnum; kk++) {
//					if (partner[i][jj] == partner[i][kk])
//						error->one(FLERR, "Duplicate particle in PeriDynamic bond - "
//								"simulation box is too small");
//				}
//			}
//		}
//	}
//
//	// communicate vinter to ghosts, needed in pair
//	comm->forward_comm_fix(this);
//
//	// bond statistics
//
//	int n = 0;
//	for (i = 0; i < nlocal; i++)
//		n += npartner[i];
//	int nall;
//	MPI_Allreduce(&n, &nall, 1, MPI_INT, MPI_SUM, world);
//
//	if (comm->me == 0) {
//		if (screen) {
//			fprintf(screen, "Peridynamic bonds:\n");
//			fprintf(screen, "  maximum # of bonds per particle = %d\n", maxpartner);
//			fprintf(screen, "  total # of bonds = %d\n", nall);
//			fprintf(screen, "  bonds/atom = %g\n", (double) nall / atom->natoms);
//
//			fprintf(screen, "  maximum # of triangle pairs per particle = %d\n", maxTrianglePairs);
//		}
//		if (logfile) {
//			fprintf(logfile, "Peridynamic bonds:\n");
//			fprintf(logfile, "  total # of bonds = %d\n", nall);
//			fprintf(logfile, "  bonds/atom = %g\n", (double) nall / atom->natoms);
//		}
//	}
}

/* ----------------------------------------------------------------------
 set up triangles for bending

 convention: dihedral atoms with indices 0, 1: common nodes
 dihedral atoms with indices 2, 3: non-common nodes
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::setup_bending_triangles() {

	double **x = atom->x0;
	int *type = atom->type;

	int nlocal = atom->nlocal;
	int i, i1, i2, i3, i4;
	int tnum, t;
	int itype, jtype, itmp;
	double sign, angle;
	double N1_normSq, N2_normSq, E_normSq, E_norm;

	Vector3d E, n1, n2, E_normed;
	Vector3d x31, x41, x42, x32, N1, N2, u1, u2, u3, u4;

	double **kbend = (double **) force->pair->extract("pdgcg/shells/kbend_ptr", itmp);
	if (kbend == NULL) {
		error->all(FLERR, "fix pdgcg/shells failed to access kbend array");
	}

	for (i = 0; i < nmax; i++) {
		for (t = 0; t < maxTrianglePairs; t++) {
			trianglePairAngle0[i][t] = 0.0;
			trianglePairPlasticAngle[i][t] = 0.0;

		}
	}

	for (i = 0; i < nlocal; i++) {

		tnum = nTrianglePairs[i];

		for (t = 0; t < tnum; t++) {

			i3 = atom->map(trianglePairs[i][t][0]);
			i4 = atom->map(trianglePairs[i][t][1]);
			i1 = atom->map(trianglePairs[i][t][2]);
			i2 = atom->map(trianglePairs[i][t][3]);

			itype = type[i1]; // types are taken from non-common nodes
			jtype = type[i2];

			if (i3 != i) {
				printf("hurz\n");
				char str[128];
				sprintf(str, "triangle index cn1=%d does not match up with local index=%d", i3, i);
				error->one(FLERR, str);
			}

			// common bond from cn1 to cn2
			E << (x[i4][0] - x[i3][0]), (x[i4][1] - x[i3][1]), (x[i4][2] - x[i3][2]);
			E_norm = E.norm();
			E_normSq = E_norm * E_norm;
			E_normed = E / E.norm();

			x31 << (x[i1][0] - x[i3][0]), (x[i1][1] - x[i3][1]), (x[i1][2] - x[i3][2]);
			x41 << (x[i1][0] - x[i4][0]), (x[i1][1] - x[i4][1]), (x[i1][2] - x[i4][2]);
			x42 << (x[i2][0] - x[i4][0]), (x[i2][1] - x[i4][1]), (x[i2][2] - x[i4][2]);
			x32 << (x[i2][0] - x[i3][0]), (x[i2][1] - x[i3][1]), (x[i2][2] - x[i3][2]);

			N1 = x31.cross(x41);
			N1_normSq = N1.squaredNorm();

			N2 = x42.cross(x32);
			N2_normSq = N2.squaredNorm();

			u1 = (E_norm / N1_normSq) * N1;
			u2 = (E_norm / N2_normSq) * N2;

			u3 = x41.dot(E_normed) * N1 / N1_normSq + x42.dot(E_normed) * N2 / N2_normSq;
			u4 = -x31.dot(E_normed) * N1 / N1_normSq - x32.dot(E_normed) * N2 / N2_normSq;

			// normal of triangle 1
			n1 = N1 / N1.norm();

			// normal of triangle 2
			n2 = N2 / N2.norm();

			// determine sin(phi) / 2
			sign = (n1.cross(n2)).dot(E_normed);

			angle = 0.5 * (1.0 - n1.dot(n2));
			if (angle < 0.0)
				angle = 0.0;
			angle = sqrt(angle);
			if (sign * angle < 0.0) {
				angle = -angle;
			}

			trianglePairAngle0[i][t] = angle;

//			cout << "cb =" << cb << endl;
//			cout << "b11 =" << b11 << endl;
//			cout << "n1 =" << n1 << endl;
//			cout << "n2 =" << n2 << endl;
//			cout << "angle is " << trianglePairAngle0[i][t] << endl << endl;

		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double FixSMDWallSurface::memory_usage() {
	int nmax = atom->nmax;
	int bytes = 0;
	bytes += nmax * sizeof(tagint); // npartner
	bytes += nmax * maxpartner * sizeof(tagint); // partners
	bytes += nmax * maxpartner * sizeof(double); // r0
	bytes += nmax * maxpartner * sizeof(double); // r1
	bytes += nmax * sizeof(double); // vinter

	bytes += nmax * sizeof(int); // nTrianglePairs
	bytes += nmax * maxTrianglePairs * 4 * sizeof(tagint); // trianglePairs
	bytes += nmax * maxTrianglePairs * sizeof(double); // trianglePairAngle0
	bytes += nmax * maxTrianglePairs * sizeof(double); // trianglePairPlasticAngle
	return bytes;
}

/* ----------------------------------------------------------------------
 allocate local atom-based arrays
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::grow_arrays(int nmax) {
	printf("in FixSMDWallSurface::grow_arrays, nmax=%d, maxTrianglePairs=%d\n", nmax, maxTrianglePairs);
	memory->grow(npartner, nmax, "peri_neigh:npartner");
	memory->grow(partner, nmax, maxpartner, "peri_neigh:partner");

	memory->grow(nTrianglePairs, nmax, "pdgcg_shells:nTrianglePairs");
	memory->grow(trianglePairs, nmax, maxTrianglePairs, 4, "pdgcg_shells:trianglePairs");
	memory->grow(trianglePairAngle0, nmax, maxTrianglePairs, "pdgcg_shells:trianglePairAngle0");
	memory->grow(trianglePairPlasticAngle, nmax, maxTrianglePairs, "pdgcg_shells:trianglePairPlasticAngle");

	memory->grow(r0, nmax, maxpartner, "peri_neigh:r0");
	memory->grow(plastic_stretch, nmax, maxpartner, "peri_neigh:plastic_stretch");
	memory->grow(vinter, nmax, "peri_neigh:vinter");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::copy_arrays(int i, int j, int delflag) {

	int m;

	npartner[j] = npartner[i];
	for (m = 0; m < npartner[j]; m++) {
		partner[j][m] = partner[i][m];
		r0[j][m] = r0[i][m];
		plastic_stretch[j][m] = plastic_stretch[i][m];
	}
	vinter[j] = vinter[i];

	nTrianglePairs[j] = nTrianglePairs[i];
	for (m = 0; m < nTrianglePairs[j]; m++) {
		trianglePairs[j][m][0] = trianglePairs[i][m][0];
		trianglePairs[j][m][1] = trianglePairs[i][m][1];
		trianglePairs[j][m][2] = trianglePairs[i][m][2];
		trianglePairs[j][m][3] = trianglePairs[i][m][3];
		trianglePairAngle0[j][m] = trianglePairAngle0[i][m];
		trianglePairPlasticAngle[j][m] = trianglePairPlasticAngle[i][m];
	}

}

/* ----------------------------------------------------------------------
 pack values for border communication at re-neighboring
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::pack_border(int n, int *list, double *buf) {
	int i, j;

	//printf("FixSMDWallSurface::pack_border\n");

	int m = 0;

	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = vinter[j];
	}

	return m;
}

/* ----------------------------------------------------------------------
 unpack values for border communication at re-neighboring
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::unpack_border(int n, int first, double *buf) {
	int i, last;

	int m = 0;
	last = first + n;
	for (i = first; i < last; i++)
		vinter[i] = buf[m++];

	return m;
}

/* ---------------------------------------------------------------------- */

int FixSMDWallSurface::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;

	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = vinter[j];
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void FixSMDWallSurface::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++)
		vinter[i] = buf[m++];
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::pack_exchange(int i, double *buf) {
	// compact list by eliminating partner = 0 entries
	// set buf[0] after compaction

	//printf("in FixSMDWallSurface::pack_exchange ------------------------------------------\n");
	//tagint *tag = atom->tag;

	int m = 0;
	int count = 0;
	buf[m++] = npartner[i];
	for (int n = 0; n < npartner[i]; n++) {
		//if (partner[i][n] == 0)
		//	continue;
		buf[m++] = partner[i][n];
		//printf("SND[%d]: atom %d with tag id %d has partner with tag id %d with r0=%f\n", comm->me, i, tag[i], partner[i][n],
		//		r0[i][n]);
		buf[m++] = r0[i][n];
		buf[m++] = plastic_stretch[i][n];
		count++;
	}
	//buf[0] = count;
	buf[m++] = vinter[i];

	buf[m++] = nTrianglePairs[i];
	for (int n = 0; n < nTrianglePairs[i]; n++) {
		buf[m++] = trianglePairs[i][n][0];
		buf[m++] = trianglePairs[i][n][1];
		buf[m++] = trianglePairs[i][n][2];
		buf[m++] = trianglePairs[i][n][3];
		buf[m++] = trianglePairAngle0[i][n];
		buf[m++] = trianglePairPlasticAngle[i][n];
	}

	return m;
}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::unpack_exchange(int nlocal, double *buf) {

	if (nlocal == nmax) {

		//printf("nlocal=%d, nmax=%d\n", nlocal, nmax);

		nmax = nmax / DELTA * DELTA;
		nmax += DELTA;
		grow_arrays(nmax);

		error->message(FLERR,
				"in FixSMDWallSurface::unpack_exchange: local arrays too small for receiving partner information; growing arrays");
	}
	//printf("nlocal=%d, nmax=%d\n", nlocal, nmax);

	int m = 0;
	npartner[nlocal] = static_cast<int>(buf[m++]);
	for (int n = 0; n < npartner[nlocal]; n++) {
		partner[nlocal][n] = static_cast<tagint>(buf[m++]);
		r0[nlocal][n] = buf[m++];
		plastic_stretch[nlocal][n] = buf[m++];
	}
	vinter[nlocal] = buf[m++];

	nTrianglePairs[nlocal] = static_cast<int>(buf[m++]);
	for (int n = 0; n < nTrianglePairs[nlocal]; n++) {
		trianglePairs[nlocal][n][0] = static_cast<int>(buf[m++]);
		trianglePairs[nlocal][n][1] = static_cast<int>(buf[m++]);
		trianglePairs[nlocal][n][2] = static_cast<int>(buf[m++]);
		trianglePairs[nlocal][n][3] = static_cast<int>(buf[m++]);
		trianglePairAngle0[nlocal][n] = buf[m++];
		trianglePairPlasticAngle[nlocal][n] = buf[m++];
	}

	return m;
}

/* ----------------------------------------------------------------------
 pack entire state of Fix into one write
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::write_restart(FILE *fp) {
	int n = 0;
	double list[2];
	list[n++] = first;
	list[n++] = maxpartner;

	if (comm->me == 0) {
		int size = n * sizeof(double);
		fwrite(&size, sizeof(int), 1, fp);
		fwrite(list, sizeof(double), n, fp);
	}
}

/* ----------------------------------------------------------------------
 use state info from restart file to restart the Fix
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::restart(char *buf) {
	int n = 0;
	double *list = (double *) buf;

	first = static_cast<int>(list[n++]);
	maxpartner = static_cast<int>(list[n++]);

	// grow 2D arrays now, cannot change size of 2nd array index later

	grow_arrays(atom->nmax);
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for restart file
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::pack_restart(int i, double *buf) {
	int m = 0;
	buf[m++] = 3 * npartner[i] + 3;
	buf[m++] = npartner[i];
	for (int n = 0; n < npartner[i]; n++) {
		buf[m++] = partner[i][n];
		buf[m++] = r0[i][n];
		buf[m++] = plastic_stretch[i][n];
	}
	buf[m++] = vinter[i];
	return m;
}

/* ----------------------------------------------------------------------
 unpack values from atom->extra array to restart the fix
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::unpack_restart(int nlocal, int nth) {

	double **extra = atom->extra;

	// skip to Nth set of extra values

	int m = 0;
	for (int i = 0; i < nth; i++)
		m += static_cast<int>(extra[nlocal][m]);
	m++;

	npartner[nlocal] = static_cast<int>(extra[nlocal][m++]);
	for (int n = 0; n < npartner[nlocal]; n++) {
		partner[nlocal][n] = static_cast<tagint>(extra[nlocal][m++]);
		r0[nlocal][n] = extra[nlocal][m++];
		plastic_stretch[nlocal][n] = extra[nlocal][m++];
	}
	vinter[nlocal] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
 maxsize of any atom's restart data
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::maxsize_restart() {
	return 2 * maxpartner + 3;
}

/* ----------------------------------------------------------------------
 size of atom nlocal's restart data
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::size_restart(int nlocal) {
	return 2 * npartner[nlocal] + 3;
}

/* ----------------------------------------------------------------------
 function to determine number of values in a text line
 ------------------------------------------------------------------------- */

int FixSMDWallSurface::count_words(const char *line) {
	int n = strlen(line) + 1;
	char *copy;
	memory->create(copy, n, "atom:copy");
	strcpy(copy, line);

	char *ptr;
	if ((ptr = strchr(copy, '#')))
		*ptr = '\0';

	if (strtok(copy, " \t\n\r\f") == NULL) {
		memory->destroy(copy);
		return 0;
	}
	n = 1;
	while (strtok(NULL, " \t\n\r\f"))
		n++;

	memory->destroy(copy);
	return n;
}

/* ----------------------------------------------------------------------
 size of atom nlocal's restart data
 ------------------------------------------------------------------------- */

void FixSMDWallSurface::read_triangles(int pass) {

	double *radius = atom->radius;
	double coord[3];

	int nlocal = atom->nlocal;
	int nlocal_previous = atom->nlocal;
	int ilocal = nlocal_previous;
	int cn1, cn2, ncn1, ncn2;
	int n, m, i;
	int me;

	bigint natoms_previous = atom->natoms;
	Vector3d *vert;
	vert = new Vector3d[3];
	Vector3d normal, center;

	char *file = "bucket1.stl";
	FILE *fp = fopen(file, "r");
	if (fp == NULL) {
		char str[128];
		sprintf(str, "Cannot open file %s", file);
		error->one(FLERR, str);
	}

	MPI_Comm_rank(world, &me);
	if (me == 0) {
		if (screen) {
			if (pass == 0) {
				fprintf(screen, "  scanning triangle pairs ...\n");
			} else {
				fprintf(screen, "  reading triangle pairs ...\n");
			}
		}
		if (logfile) {
			if (pass == 0) {
				fprintf(logfile, "  scanning triangle pairs ...\n");
			} else {
				fprintf(logfile, "  reading triangle pairs ...\n");
			}
		}
	}

	char str[128];
	char line[256];
	char *retpointer;
	char **values;
	int nwords;

	// read STL solid name
	retpointer = fgets(line, sizeof(line), fp);
	if (retpointer == NULL) {
		sprintf(str, "error reading number of triangle pairs");
		error->one(FLERR, str);
	}

	nwords = count_words(line);
	if (nwords != 2) {
		sprintf(str, "first line of file does not name a STL solid");
		error->one(FLERR, str);
	}

	values = new char*[nwords];
	values[0] = strtok(line, " \t\n\r\f");
	if (values[0] == NULL)
		error->all(FLERR, "Incorrect atom format in data file");
	for (m = 1; m < nwords; m++) {
		values[m] = strtok(NULL, " \t\n\r\f");
		if (values[m] == NULL)
			error->all(FLERR, "Incorrect atom format in data file");
	}
	delete[] values;

	cout << "STL file contains solid body with name: " << values[1] << endl;

	// iterate over STL facets util end of body is reached

	while (fgets(line, sizeof(line), fp)) { // read a line, should be the facet line

		// evaluate facet line
		nwords = count_words(line);
		if (nwords == 2) {
			sprintf(str, "found end solid line");
			error->message(FLERR, str);
			break;
		} else if (nwords != 5) {
			sprintf(str, "wrng format in STL line");
			error->one(FLERR, str);
		}

		values = new char*[nwords];
		values[0] = strtok(line, " \t\n\r\f");
		if (values[0] == NULL)
			error->all(FLERR, "Incorrect atom format in data file");
		for (m = 1; m < nwords; m++) {
			values[m] = strtok(NULL, " \t\n\r\f");
			if (values[m] == NULL)
				error->all(FLERR, "Incorrect atom format in data file");
		}

		normal << force->numeric(FLERR, values[2]), force->numeric(FLERR, values[3]), force->numeric(FLERR, values[4]);
		//cout << "normal is " << normal << endl;

		delete[] values;

		// read outer loop line
		retpointer = fgets(line, sizeof(line), fp);
		if (retpointer == NULL) {
			sprintf(str, "error reading outer loop");
			error->one(FLERR, str);
		}

		nwords = count_words(line);
		if (nwords != 2) {
			sprintf(str, "error reading outer loop");
			error->one(FLERR, str);
		}

		// read vertex lines

		for (int k = 0; k < 3; k++) {
			retpointer = fgets(line, sizeof(line), fp);
			if (retpointer == NULL) {
				sprintf(str, "error reading vertex line");
				error->one(FLERR, str);
			}

			nwords = count_words(line);
			if (nwords != 4) {
				sprintf(str, "error reading vertex line");
				error->one(FLERR, str);
			}

			values = new char*[nwords];
			values[0] = strtok(line, " \t\n\r\f");
			if (values[0] == NULL)
				error->all(FLERR, "Incorrect vertex line");
			for (m = 1; m < nwords; m++) {
				values[m] = strtok(NULL, " \t\n\r\f");
				if (values[m] == NULL)
					error->all(FLERR, "Incorrect vertex line");
			}

			vert[k] << force->numeric(FLERR, values[1]), force->numeric(FLERR, values[2]), force->numeric(FLERR, values[3]);
			//cout << "vertex is " << vert[k] << endl;
			delete[] values;

		}

		// read end loop line
		retpointer = fgets(line, sizeof(line), fp);
		if (retpointer == NULL) {
			sprintf(str, "error reading endloop");
			error->one(FLERR, str);
		}

		nwords = count_words(line);
		if (nwords != 1) {
			sprintf(str, "error reading endloop");
			error->one(FLERR, str);
		}

		// read end facet line
		retpointer = fgets(line, sizeof(line), fp);
		if (retpointer == NULL) {
			sprintf(str, "error reading endfacet");
			error->one(FLERR, str);
		}

		nwords = count_words(line);
		if (nwords != 1) {
			sprintf(str, "error reading endfacet");
			error->one(FLERR, str);
		}

		// now we have a normal and three vertices ... proceed with adding triangle

		center = (vert[0] + vert[1] + vert[2]) / 3.0;

	//	cout << "center is " << center << endl;

		double r1 = (center - vert[0]).norm();
		double r2 = (center - vert[1]).norm();
		double r3 = (center - vert[2]).norm();
		double r = MAX(r1,r2);
		r = MAX(r,r3);

		// if atom/molecule is in my subbox, create it
	//	printf("coord: %f %f %f\n", coord[0], coord[1], coord[2]);
	//	printf("sublo: %f %f %f\n", sublo[0], sublo[1], sublo[2]);
	//	printf("subhi: %f %f %f\n", subhi[0], subhi[1], subhi[2]);
		if (center(0) >= sublo[0] && center(0) < subhi[0] && center(1) >= sublo[1] && center(1) < subhi[1] && center(2) >= sublo[2]
				&& center(2) < subhi[2]) {
			//printf("******* KERATIN nlocal=%d ***\n", nlocal);
			coord[0] = center(0);
			coord[1] = center(1);
			coord[2] = center(2);
			atom->avec->create_atom(2, coord);
			radius[ilocal] = r; //ilocal;
			ilocal++;
		}

	}

	// set new total # of atoms and error check

	  bigint nblocal = atom->nlocal;
	  MPI_Allreduce(&nblocal,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
	  if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
	    error->all(FLERR,"Too many total atoms");

	  // add IDs for newly created atoms
	  // check that atom IDs are valid

	  if (atom->tag_enable) atom->tag_extend();
	  atom->tag_check();

	  // create global mapping of atoms
	  // zero nghost in case are adding new atoms to existing atoms

	  if (atom->map_style) {
	    atom->nghost = 0;
	    atom->map_init();
	    atom->map_set();
	  }


	  // print status

	  if (comm->me == 0) {
	    if (screen)
	      fprintf(screen,"Created " BIGINT_FORMAT " atoms\n",
	              atom->natoms-natoms_previous);
	    if (logfile)
	      fprintf(logfile,"Created " BIGINT_FORMAT " atoms\n",
	              atom->natoms-natoms_previous);
	  }


	delete[] vert;
	fclose(fp);
}

