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
#include "pair_peri_defgrad.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_peri_neigh_gcg.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <Eigen/Eigen>
using namespace Eigen;
using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */

PairPeriDefgrad::PairPeriDefgrad(LAMMPS *lmp) :
		Pair(lmp) {
	for (int i = 0; i < 6; i++)
		virial[i] = 0.0;
	no_virial_fdotr_compute = 1;

	ifix_peri = -1;

	bulkmodulus = NULL;
	smax = syield = NULL;
	G0 = NULL;
	alpha = NULL;
	nmax = 1;
	F = K = PK1 = PK1_as = NULL;

}

/* ---------------------------------------------------------------------- */

PairPeriDefgrad::~PairPeriDefgrad() {
	if (ifix_peri >= 0)
		modify->delete_fix("PERI_NEIGH_GCG");

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(bulkmodulus);
		memory->destroy(smax);
		memory->destroy(syield);
		memory->destroy(G0);
		memory->destroy(alpha);
		memory->destroy(cutsq);
	}

	delete[] F;
	delete[] K;
	delete[] PK1;
	delete[] PK1_as;
}

/* ---------------------------------------------------------------------- */

void PairPeriDefgrad::compute(int eflag, int vflag) {
	int i, j, jj, jnum, itype;
	double evdwl, weight, h, r0, wf, wfd, cutoff, r;
	// ---------------------------------------------------------------------------------
	double **f = atom->f;
	double **x = atom->x;
	double **x0 = atom->x0;
	int *type = atom->type;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	int nlocal = atom->nlocal;
	tagint **partner = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->partner;
	int *npartner = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->npartner;
	Vector2d x0i, x0j, xi, xj, dx, dx0, f_stress, f_artstress;
	Matrix2d E, tau, eye, tmp, sigma;

	double Ey = 1.0;
	double nu = 0.45;
	double lambda = Ey * nu / ((1.+nu)*(1.-2*nu));
	double mu = Ey/(2*(1.+nu));

	eye.setIdentity();
	evdwl = 0.0;
	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
		nmax = atom->nmax;
		delete[] F;
		F = new Matrix2d[nmax];
		delete[] K;
		K = new Matrix2d[nmax];
		delete[] PK1;
		PK1 = new Matrix2d[nmax];
		delete[] PK1_as;
		PK1_as = new Matrix2d[nmax];
	}

	/* ----------------------- COMPUTE SHAPE MATRIX AND DEFORMATION GRADIENT --------------------- */

	for (i = 0; i < nlocal; i++) {
		F[i].setZero();
		K[i].setZero();
		PK1[i].setZero();
		PK1_as[i].setZero();
	}

	for (i = 0; i < nlocal; i++) {
		x0i << x0[i][0], x0[i][1];
		xi << x[i][0], x[i][1];
		itype = type[i];
		jnum = npartner[i];

		for (jj = 0; jj < jnum; jj++) {

			if (partner[i][jj] == 0)
				continue;
			j = atom->map(partner[i][jj]);

			// check if lost a partner without first breaking bond
			if (j < 0) {
				partner[i][jj] = 0;
				continue;
			}

			x0j << x0[j][0], x0[j][1];
			xj << x[j][0], x[j][1];

			dx0 = x0j - x0i;
			dx = xj - xi;

			h = radius[i] + radius[j];
			r0 = dx0.norm();
			kernel_and_derivative(h, r0, wf, wfd);
			weight = wf * vfrac[j];

			K[i] += weight * dx0 * dx0.transpose();
			F[i] += weight * dx * dx0.transpose();
		}

		tmp = K[i].inverse();
		K[i] = tmp;

		tmp = F[i] * K[i]; // now we have a corrected deformation gradient
		F[i] = tmp;

		/*
		 * material model
		 */

		double J = F[i].determinant();
		double logJ = log(J);
		Matrix2d b;
		b = F[i] * F[i].transpose();
		sigma = (mu / J) * (b - eye) + (lambda / J) * logJ * eye;

		PK1[i] = J * sigma * F[i].inverse().transpose();
		PK1[i] = PK1[i] * K[i];

		/*
		 * compute Eigenvalues and Eigenvectors of stress matrix
		 */
		SelfAdjointEigenSolver<Matrix2d> es;
		es.compute(sigma);
		Vector2d eigenvalues = es.eigenvalues();
		Matrix2d V = es.eigenvectors();

		// diagonalize stress matrix
		Matrix2d sigma_diag = V.inverse() * sigma * V;

		Matrix2d artificial_stress_diag, artificial_stress;
		artificial_stress.setZero();
		if (sigma_diag(0, 0) > 0.0) {
			artificial_stress_diag(0, 0) = -sigma_diag(0, 0);
		}
		if (sigma_diag(1, 1) > 0.0) {
			artificial_stress_diag(1, 1) = -sigma_diag(1, 1);
		}

		// undiagonalize stress matrix
		artificial_stress = V * artificial_stress_diag * V.inverse();

		// transform artificial stress into PK1 stress measure
		PK1_as[i] = J * artificial_stress * F[i].inverse().transpose();
		PK1_as[i] = PK1_as[i] * K[i];

//		if (i==1) {
//			cout << "\nF:\n" << F[i] << endl;
//			cout << "PK1as:\n" << PK1_as[i] << endl;
//		}
	}

	/* ----------------------- PERIDYNAMIC BOND FORCES --------------------- */

	for (i = 0; i < nlocal; i++) {

		x0i << x0[i][0], x0[i][1];
		xi << x[i][0], x[i][1];
		itype = type[i];
		jnum = npartner[i];
		//cout << "num neighs: " << jnum << endl;

		for (jj = 0; jj < jnum; jj++) {

			if (partner[i][jj] == 0)
				continue;
			j = atom->map(partner[i][jj]);

			// check if lost a partner without first breaking bond
			if (j < 0) {
				partner[i][jj] = 0;
				continue;
			}

			x0j << x0[j][0], x0[j][1];
			xj << x[j][0], x[j][1];
			dx0 = x0j - x0i;
			dx = xj - xi;

			h = radius[i] + radius[j];
			r0 = dx0.norm();
			r = dx.norm();
			kernel_and_derivative(h, r0, wf, wfd);

			f_stress = wf * vfrac[i] * vfrac[j] * (PK1[i] + PK1[j]) * dx0;

			/*
			 * artificial stress
			 */
//			double wf0;
//			kernel_and_derivative(h, 0.75, wf0, wfd);
////
//			weight = pow(wf/wf0,4.0);
//			f_artstress = 0.01 * weight * vfrac[i] * vfrac[j] * (PK1_as[i] + PK1_as[j]) * dx0;
//			f_stress += f_artstress;

			f[i][0] += f_stress(0);
			f[i][1] += f_stress(1);
		}

	}

}

void PairPeriDefgrad::kernel_and_derivative(double h, double r, double &wf, double &wfd) {

	double hr = h - r; // [m]
//	double n = 0.3141592654e0 * h * h * h * h * h; // [m^5]
//	wfd = -3.0e0 * hr * hr / n; // [m*m/m^5] = [1/m^3] ==> correct for dW/dr in 2D
//	wf = -0.333333333333e0 * hr * wfd; // [m/m^3] ==> [1/m^2] correct for W in 2D
//	wf = wfd;

	//wf = pow(hr/h,4); // works well for the tensile instability problem

	// Wendland C5
	double q = 2.0 * r/h;
	wf = pow(1.0 - 0.5*q, 4) * (2.0 * q + 1.0);


//	double arg = (1.570796327 * (r + h)) / h;
//	double hsq = h * h;
//	if (r > h) {
//		char msg[128];
//		sprintf(msg, "r = %f > h = %f in kernel function", r, h);
//		error->one(FLERR, msg);
//	}
//	wf = (1.680351548 * (cos(arg) + 1.)) / hsq;
//	wfd = -2.639490040 * sin(arg) / (hsq * h);

}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(bulkmodulus, n + 1, n + 1, "pair:kspring");
	memory->create(smax, n + 1, n + 1, "pair:smax");
	memory->create(syield, n + 1, n + 1, "pair:syield");
	memory->create(G0, n + 1, n + 1, "pair:G0");
	memory->create(alpha, n + 1, n + 1, "pair:alpha");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::coeff(int narg, char **arg) {
	if (narg != 7)
		error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	double bulkmodulus_one = atof(arg[2]);
	double smax_one = atof(arg[3]);
	double G0_one = atof(arg[4]);
	double alpha_one = atof(arg[5]);
	double syield_one = atof(arg[6]);

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			bulkmodulus[i][j] = bulkmodulus_one;
			smax[i][j] = smax_one;
			syield[i][j] = syield_one;
			G0[i][j] = G0_one;
			alpha[i][j] = alpha_one;
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

double PairPeriDefgrad::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	bulkmodulus[j][i] = bulkmodulus[i][j];
	alpha[j][i] = alpha[i][j];
	smax[j][i] = smax[i][j];
	syield[j][i] = syield[i][j];
	G0[j][i] = G0[i][j];

	return cutoff_global;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::init_style() {
	int i;

// error checks

	if (!atom->x0_flag)
		error->all(FLERR, "Pair style peri requires atom style with x0");
	if (atom->map_style == 0)
		error->all(FLERR, "Pair peri requires an atom map, see atom_modify");

// if first init, create Fix needed for storing fixed neighbors

	if (ifix_peri == -1) {
		char **fixarg = new char*[3];
		fixarg[0] = (char *) "PERI_NEIGH_GCG";
		fixarg[1] = (char *) "all";
		fixarg[2] = (char *) "PERI_NEIGH_GCG";
		modify->add_fix(3, fixarg);
		delete[] fixarg;
	}

// find associated PERI_NEIGH fix that must exist
// could have changed locations in fix list since created

	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "PERI_NEIGH_GCG") == 0)
			ifix_peri = i;
	if (ifix_peri == -1)
		error->all(FLERR, "Fix peri neigh GCG does not exist");

// request a granular neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;

	double *radius = atom->radius;
	int nlocal = atom->nlocal;
	double maxrad_one = 0.0;

	for (i = 0; i < nlocal; i++)
		maxrad_one = MAX(maxrad_one, 2 * radius[i]);

	printf("proc %d has maxrad %f\n", comm->me, maxrad_one);

	MPI_Allreduce(&maxrad_one, &cutoff_global, atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::write_restart(FILE *fp) {
	int i, j;
	for (i = 1; i <= atom->ntypes; i++)
		for (j = i; j <= atom->ntypes; j++) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&bulkmodulus[i][j], sizeof(double), 1, fp);
				fwrite(&smax[i][j], sizeof(double), 1, fp);
				fwrite(&syield[i][j], sizeof(double), 1, fp);
				fwrite(&alpha[i][j], sizeof(double), 1, fp);
			}
		}
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairPeriDefgrad::read_restart(FILE *fp) {
	allocate();

	int i, j;
	int me = comm->me;
	for (i = 1; i <= atom->ntypes; i++)
		for (j = i; j <= atom->ntypes; j++) {
			if (me == 0)
				fread(&setflag[i][j], sizeof(int), 1, fp);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (me == 0) {
					fread(&bulkmodulus[i][j], sizeof(double), 1, fp);
					fread(&smax[i][j], sizeof(double), 1, fp);
					fread(&syield[i][j], sizeof(double), 1, fp);
					fread(&alpha[i][j], sizeof(double), 1, fp);
				}
				MPI_Bcast(&bulkmodulus[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&smax[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&syield[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&alpha[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairPeriDefgrad::memory_usage() {

	return 0.0;
}

