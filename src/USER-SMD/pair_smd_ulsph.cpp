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

#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "pair_smd_ulsph.h"
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
#include <stdio.h>
#include <iostream>
#include "smd_material_models.h"
#include "smd_math.h"

using namespace std;
using namespace LAMMPS_NS;
using namespace SMD_Math;

#include <Eigen/SVD>
#include <Eigen/Eigen>
using namespace Eigen;

#define COMPUTE_CORRECTED_DERIVATIVES false

PairULSPH::PairULSPH(LAMMPS *lmp) :
		Pair(lmp) {

	Q1 = NULL;
	eos = NULL;
	pressure = NULL;
	c0 = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = K = NULL;
	artStress = NULL;
	delete_flag = NULL;
	shepardWeight = NULL;
	smoothVel = NULL;
	numNeighs = NULL;

	comm_forward = 9; // this pair style communicates 9 doubles to ghost atoms
}

/* ---------------------------------------------------------------------- */

PairULSPH::~PairULSPH() {
	if (allocated) {
		//printf("... deallocating\n");
		memory->destroy(Q1);
		memory->destroy(rho0);
		memory->destroy(eos);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;

		delete[] K;
		delete[] delete_flag;
		delete[] shepardWeight;
		delete[] c0;
		delete[] smoothVel;
		delete[] stressTensor;
		delete[] L;
		delete[] artStress;
		delete[] numNeighs;
	}
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairULSPH::PreCompute_DensitySummation() {
	double *radius = atom->radius;
	double **x = atom->x;
	double *rmass = atom->rmass;
	double *rho = atom->rho;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, itype, jtype, j;
	double h, irad, hsq, rSq, wf;
	Vector3d dx, xi, xj;

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		numNeighs[i] = 0;
	}

	/*
	 * only recompute mass density if density summation is used.
	 * otherwise, change in mass density is time-integrated
	 */
	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			// initialize particle density with self-contribution.
			h = 2.0 * radius[i];
			hsq = h * h;
			Poly6Kernel(hsq, h, 0.0, wf);
			rho[i] = wf * rmass[i];
		}
	}

	// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];

		xi << x[i][0], x[i][1], x[i][2];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			xj << x[j][0], x[j][1], x[j][2];
			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = irad + radius[j];
			hsq = h * h;
			if (rSq < hsq) {

				jtype = type[j];
				Poly6Kernel(hsq, h, rSq, wf);

				if (setflag[itype][itype] == 1) {
					rho[i] += wf * rmass[j];
				}
				numNeighs[i] += 1;

				if (j < nlocal) {
					if (setflag[jtype][jtype] == 1) {
						rho[j] += wf * rmass[i];
					}
					numNeighs[j] += 1;
				}
			}
		} // end if check distance
	} // end loop over j
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairULSPH::PreCompute() {
	double *radius = atom->radius;
	double **x = atom->x;
	double **v = atom->vest;
	double *rmass = atom->rmass;
	double *rho = atom->rho;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, itype, jtype, j, iDim;
	double wfd, h, irad, r, rSq, wf, ivol, jvol;
	Vector3d dx, dv, g;
	Matrix3d Ktmp, Ltmp;
	Vector3d xi, xj, vi, vj;

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		delete_flag[i] = 0.0;
		L[i].setZero();
		numNeighs[i] = 0;
		K[i].setZero();
	}

// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];
		ivol = rmass[i] / rho[i];

		// initialize Eigen data structures from LAMMPS data structures
		for (iDim = 0; iDim < 3; iDim++) {
			xi(iDim) = x[i][iDim];
			vi(iDim) = v[i][iDim];
		}

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			for (iDim = 0; iDim < 3; iDim++) {
				xj(iDim) = x[j][iDim];
				vj(iDim) = v[j][iDim];
			}

			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = irad + radius[j];
			if (rSq < h * h) {

				r = sqrt(rSq);
				jtype = type[j];
				jvol = rmass[j] / rho[j];

				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;

				// kernel and derivative
				kernel_and_derivative(h, r, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r) * dx;

				/* build correction matrix for kernel derivatives */
				if (COMPUTE_CORRECTED_DERIVATIVES) {
					Ktmp = g * dx.transpose();
					K[i] += jvol * Ktmp;
				}

				// velocity gradient L
				Ltmp = -dv * g.transpose();
				L[i] += jvol * Ltmp;

				numNeighs[i] += 1;

				if (j < nlocal) {

					if (COMPUTE_CORRECTED_DERIVATIVES) {
						K[j] += ivol * Ktmp;
					}

					L[j] += ivol * Ltmp;
					numNeighs[j] += 1;
				}
			} // end if check distance
		} // end loop over j
	} // end loop over i

	/*
	 * invert shape matrix and compute corrected quantities
	 */

	if (COMPUTE_CORRECTED_DERIVATIVES) {
		for (i = 0; i < nlocal; i++) {
			itype = type[i];

			if (domain->dimension == 2) {
				Matrix2d K2d;
				K2d(0, 0) = K[i](0, 0);
				K2d(0, 1) = K[i](0, 1);
				K2d(1, 0) = K[i](1, 0);
				K2d(1, 1) = K[i](1, 1);

				if ((K2d.determinant() > 1.0e-1) && (numNeighs[i] > 4)) {
					Matrix2d K2di;
					K2di = K2d.inverse();

					K[i].setIdentity();
					K[i](0, 0) = K2di(0, 0);
					K[i](0, 1) = K2di(0, 1);
					K[i](1, 0) = K2di(1, 0);
					K[i](1, 1) = K2di(1, 1);
				} else {
					K[i].setIdentity();
				}

			} else { // 3d
				if (K[i].determinant() > 1.0e-4) {
					K[i] = pseudo_inverse_SVD(K[i]);
				} else {
					K[i].setIdentity();
				}
			}

			L[i] *= K[i];

		} // end loop over i = 0 to nlocal
	}
}

/* ---------------------------------------------------------------------- */

void PairULSPH::compute(int eflag, int vflag) {
	double **x = atom->x;
	double **v = atom->vest;
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	double **f = atom->f;
	double *de = atom->de;
	double *drho = atom->drho;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *rho = atom->rho;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, jtype, iDim, inum;
	double r, wf, wfd, h, rSq, ivol, jvol;
	double mu_ij, c_ij, rho_ij;
	double delVdotDelR, visc_magnitude, deltaE;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	Vector3d fi, fj, dx, dv, f_stress, g, vinti, vintj, dvint;
	Vector3d xi, xj, vi, vj, f_visc, sumForces, f_stress_new;

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
		//printf("... allocating in compute with nmax = %d\n", atom->nmax);
		nmax = atom->nmax;
		delete[] K;
		K = new Matrix3d[nmax];
		delete[] delete_flag;
		delete_flag = new double[nmax];
		delete[] shepardWeight;
		shepardWeight = new double[nmax];
		delete[] pressure;
		pressure = new double[nmax];
		delete[] c0;
		c0 = new double[nmax];
		delete[] smoothVel;
		smoothVel = new Vector3d[nmax];
		delete[] stressTensor;
		stressTensor = new Matrix3d[nmax];
		delete[] L;
		L = new Matrix3d[nmax];
		delete[] artStress;
		artStress = new Matrix3d[nmax];
		delete[] numNeighs;
		numNeighs = new int[nmax];
	}

// zero accumulators
	for (i = 0; i < nlocal; i++) {
		shepardWeight[i] = 0.0;
		smoothVel[i].setZero();
		numNeighs[i] = 0;
	}

	//PairULSPH::PreCompute_DensitySummation();
	//PairULSPH::PreCompute();
	PairULSPH::ComputePressure();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);

	/*
	 * iterate over pairs of particles i, j and assign forces using pre-computed pressure
	 */

// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		ivol = rmass[i] / rho[i];

		//printf("i = %d, nj=%d\n", i, jnum);

		// initialize Eigen data structures from LAMMPS data structures
		for (iDim = 0; iDim < 3; iDim++) {
			xi(iDim) = x[i][iDim];
			vi(iDim) = v[i][iDim];
			vinti(iDim) = vint[i][iDim];
		}

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			xj << x[j][0], x[j][1], x[j][2];

			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = radius[i] + radius[j];
			if (rSq < h * h) {

				// initialize Eigen data structures from LAMMPS data structures
				for (iDim = 0; iDim < 3; iDim++) {
					vj(iDim) = v[j][iDim];
					vintj(iDim) = vint[j][iDim];
				}

				r = sqrt(rSq);
				jtype = type[j];
				jvol = rmass[j] / rho[j];

				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;
				dvint = vintj - vinti;

				// kernel and derivative
				kernel_and_derivative(h, r, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r) * dx;

				/*
				 * force -- the classical SPH way
				 */

				f_stress = ivol * jvol * (stressTensor[i] + stressTensor[j]) * g;

				/*
				 * artificial viscosity -- alpha is dimensionless
				 * Monaghan–Balsara form of the artificial viscosity
				 */

				delVdotDelR = dx.dot(dv) / (r + 0.1 * h); // project relative velocity onto unit particle distance vector [m/s]
				c_ij = 0.5 * (c0[i] + c0[j]);

				LimitDoubleMagnitude(delVdotDelR, 0.1 * c_ij);

				//if (fabs(delVdotDelR) > 0.5 * c_ij) { // limit delVdotDelR to a fraction of speed of sound
				//	double s = copysign(1, delVdotDelR);
				//	delVdotDelR = s * 0.5 * c_ij;
				//}

				mu_ij = h * delVdotDelR / (r + 0.1 * h); // units: [m * m/s / m = m/s]
				rho_ij = 0.5 * (rho[i] + rho[j]);
				visc_magnitude = 0.5 * (Q1[itype] + Q1[jtype]) * c_ij * mu_ij / rho_ij;
				f_visc = rmass[i] * rmass[j] * visc_magnitude * g;

				sumForces = f_stress + f_visc;

				// energy rate -- project velocity onto force vector
				deltaE = 0.5 * sumForces.dot(dv);

				// change in mass density
				drho[i] -= rmass[j] * wfd * delVdotDelR;

				// apply forces to pair of particles
				f[i][0] += sumForces(0);
				f[i][1] += sumForces(1);
				f[i][2] += sumForces(2);
				de[i] += deltaE;

				// accumulate smooth velocities
				shepardWeight[i] += jvol * wf;
				smoothVel[i] += jvol * wf * dvint;
				numNeighs[i] += 1;

				if (j < nlocal) {
					f[j][0] -= sumForces(0);
					f[j][1] -= sumForces(1);
					f[j][2] -= sumForces(2);
					de[j] += deltaE;
					drho[j] -= rmass[i] * wfd * delVdotDelR;

					shepardWeight[j] += ivol * wf;
					smoothVel[j] -= ivol * wf * dvint;
					numNeighs[j] += 1;
				}

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
				}
			}

		}
	}

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {

			/*
			 * ghost particle for normal: y
			 */

//			xi << x[i][0], x[i][1], x[i][2];
//			h = 2 * radius[i];
//
//			if (xi(1) < h) {
//
//				ivol = rmass[i] / rho[i];
//				xj << x[i][0], x[i][1] - h / 3.0, x[i][2];
//				vj << v[i][0], -v[i][1], v[i][2];
//				vintj << vint[i][0], -vint[i][1], vint[i][2];
//				dx = xj - xi;
//				rSq = dx.squaredNorm();
//				r = sqrt(rSq);
//				dv = vj - vi;
//				dvint = vintj - vinti;
//
//				// kernel and derivative
//				kernel_and_derivative(h, r, wf, wfd);
//
//				// uncorrected kernel gradient
//				g = (wfd / r) * dx;
//
//				/*
//				 * force -- the classical SPH way
//				 */
//
//				f_stress = 4.0 * ivol * ivol * stressTensor[i] * g;
//
//				sumForces = f_stress;
//
//				// energy rate -- project velocity onto force vector
//
//				deltaE = 0.5 * sumForces.dot(dv);
//
//				// change in mass density
//				delVdotDelR = dx.dot(dv) / (r + 0.1 * h);
//				drho[i] -= 2 * rmass[i] * wfd * delVdotDelR;
//
//				// apply forces to pair of particles
//				f[i][0] += sumForces(0);
//				f[i][1] += sumForces(1);
//				f[i][2] += sumForces(2);
//				de[i] += deltaE;
//
//			}

			if (shepardWeight[i] != 0.0) {
				smoothVel[i] /= shepardWeight[i];
			} else {
				smoothVel[i].setZero();
			}
		} // end check if particle is SPH-type
	} // end loop over i = 0 to nlocal

	if (vflag_fdotr)
		virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
 compute pressure
 ------------------------------------------------------------------------- */
void PairULSPH::ComputePressure() {
	double *radius = atom->radius;
	double *rho = atom->rho;
	double *rmass = atom->rmass;
	double *e = atom->e;
	int *type = atom->type;
	double pFinal;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d D, W, V, sigma_diag;
	Matrix3d eye;

	dtCFL = 1.0e22;
	eye.setIdentity();

	double var1, var2, var3, vol;

	/*
	 * iterate over particle types first, only use SafeLookup (slow!) once for each type, store the result.
	 * The iterate over nlocal and check if type matches!
	 */

	int ntypes = atom->ntypes;
	for (itype = 1; itype < ntypes + 1; itype++) {
		if (setflag[itype][itype] == 1) {
			switch (eos[itype]) {
			case EOS_TAIT:
				var1 = SafeLookup("tait_exponent", itype);
				var2 = SafeLookup("c_ref", itype);
				var3 = SafeLookup("rho_ref", itype);

				for (i = 0; i < nlocal; i++) {
					if (type[i] == itype) {
						TaitEOS_density(var1, var2, var3, rho[i], pFinal, c0[i]);
						stressTensor[i].setZero();
						stressTensor[i](0, 0) = -pFinal;
						stressTensor[i](1, 1) = -pFinal;
						stressTensor[i](2, 2) = -pFinal;
					}
				}
				break;
			case EOS_PERFECT_GAS:
				var1 = SafeLookup("perfect_gas_gamma", itype);

				for (i = 0; i < nlocal; i++) {
					if (type[i] == itype) {
						vol = rmass[i] / rho[i];
						PerfectGasEOS(var1, vol, rmass[i], e[i], pFinal, c0[i]);
						stressTensor[i].setZero();
						stressTensor[i](0, 0) = pFinal;
						stressTensor[i](1, 1) = pFinal;
						stressTensor[i](2, 2) = pFinal;
					}
				}
				break;

			case NONE:
				pFinal = 0.0;
				c0[i] = 1.0;
				break;
			default:
				error->one(FLERR, "unknown EOS.");
				break;
			}
		}
	}

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			/*
			 * kernel gradient correction
			 */
			if (COMPUTE_CORRECTED_DERIVATIVES) {
				stressTensor[i] = stressTensor[i] * K[i];
			}

			pressure[i] = stressTensor[i].trace() / 3.0;

			/*
			 * minimum kernel Radius
			 */

			dtCFL = MIN(radius[i] / c0[i], dtCFL);
		}
	}

//printf("stable timestep = %g\n", 0.1 * hMin * MaxBulkVelocity);
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairULSPH::allocate() {

//printf("in allocate\n");

	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(Q1, n + 1, "pair:Q1");
	memory->create(rho0, n + 1, "pair:Q2");
	memory->create(eos, n + 1, "pair:eosmodel");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];

//printf("end of allocate\n");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairULSPH::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairULSPH::coeff(int narg, char **arg) {
	int ioffset, iarg, iNextKwd, itype, jtype;
	char str[128];
	std::string s, t;

	if (narg < 3) {
		sprintf(str, "number of arguments for pair ulsph is too small!");
		error->all(FLERR, str);
	}
	if (!allocated)
		allocate();

	/*
	 * if parameters are give in i,i form, i.e., no a cross interaction, set material parameters
	 */

	if (force->inumeric(FLERR, arg[0]) == force->inumeric(FLERR, arg[1])) {
		//sprintf(str, "ULSPH coefficients can only be specified between particles of same type!");
		//error->all(FLERR, str);

		itype = force->inumeric(FLERR, arg[0]);

		if (comm->me == 0)
			printf("\n******************SMD / ULSPH PROPERTIES OF PARTICLE TYPE %d **************************\n", itype);

		/*
		 * read parameters which are common -- regardless of material / eos model
		 */

		ioffset = 2;
		if (strcmp(arg[ioffset], "*COMMON") != 0) {
			sprintf(str, "common keyword missing!");
			error->all(FLERR, str);
		} else {
			printf("common keyword found\n");
		}

		t = string("*");
		iNextKwd = -1;
		for (iarg = ioffset + 1; iarg < narg; iarg++) {
			s = string(arg[iarg]);
			if (s.compare(0, t.length(), t) == 0) {
				iNextKwd = iarg;
				break;
			}
		}

		printf("keyword following *COMMON is %s\n", arg[iNextKwd]);

		if (iNextKwd < 0) {
			sprintf(str, "no *KEYWORD terminates *COMMON");
			error->all(FLERR, str);
		}

		if (iNextKwd - ioffset != 4 + 1) {
			sprintf(str, "expected 7 arguments following *COMMON but got %d\n", iNextKwd - ioffset - 1);
			error->all(FLERR, str);
		}

		matProp2[std::make_pair("rho_ref", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
		matProp2[std::make_pair("c_ref", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
		matProp2[std::make_pair("viscosity_q1", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
		matProp2[std::make_pair("heat_capacity", itype)] = force->numeric(FLERR, arg[ioffset + 4]);

		matProp2[std::make_pair("bulk_modulus", itype)] = SafeLookup("c_ref", itype) * SafeLookup("c_ref", itype)
				* SafeLookup("rho_ref", itype);

		if (comm->me == 0) {
			printf("\n material unspecific properties for SMD/ULSPH definition of particle type %d:\n", itype);
			printf("%40s : %g\n", "reference density", SafeLookup("rho_ref", itype));
			printf("%40s : %g\n", "reference speed of sound", SafeLookup("c_ref", itype));
			printf("%40s : %g\n", "linear viscosity coefficient", SafeLookup("viscosity_q1", itype));
			printf("%40s : %g\n", "heat capacity [energy / (mass * temperature)]", SafeLookup("heat_capacity", itype));
			printf("%40s : %g\n", "bulk modulus", SafeLookup("bulk_modulus", itype));
		}

		/*
		 * read following material cards
		 */

		if (comm->me == 0) {
			printf("next kwd is %s\n", arg[iNextKwd]);
		}
		eos[itype] = NONE;

		while (true) {
			if (strcmp(arg[iNextKwd], "*END") == 0) {
				if (comm->me == 0) {
					sprintf(str, "found *END");
					error->message(FLERR, str);
				}
				break;
			}

			ioffset = iNextKwd;
			if (strcmp(arg[ioffset], "*EOS_TAIT") == 0) {

				/*
				 * Tait EOS
				 */

				eos[itype] = EOS_TAIT;
				printf("reading *EOS_TAIT\n");

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *EOS_TAIT");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *EOS_TAIT but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				matProp2[std::make_pair("tait_exponent", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf("\n%60s\n", "Tait EOS");
					printf("%60s : %g\n", "Exponent", SafeLookup("tait_exponent", itype));
				}
			} // end Tait EOS

			else if (strcmp(arg[ioffset], "*EOS_PERFECT_GAS") == 0) {

				/*
				 * Tait EOS
				 */

				eos[itype] = EOS_PERFECT_GAS;
				printf("reading *EOS_PERFECT_GAS\n");

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *EOS_PERFECT_GAS");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *EOS_PERFECT_GAS but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				matProp2[std::make_pair("perfect_gas_gamma", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf("\n%60s\n", "Perfect Gas EOS");
					printf("%60s : %g\n", "Heat Capacity Ratio Gamma", SafeLookup("perfect_gas_gamma", itype));
				}
			} // end Perfect Gas EOS

			else {
				sprintf(str, "unknown *KEYWORD: %s", arg[ioffset]);
				error->all(FLERR, str);
			}

		}

		/*
		 * copy data which is looked up in inner pairwise loops from slow maps to fast arrays
		 */

		Q1[itype] = SafeLookup("viscosity_q1", itype);
		rho0[itype] = SafeLookup("rho_ref", itype);

		setflag[itype][itype] = 1;
	} else {
		/*
		 * we are reading a cross-interaction line for particle types i, j
		 */

		itype = force->inumeric(FLERR, arg[0]);
		jtype = force->inumeric(FLERR, arg[1]);

		if (strcmp(arg[2], "*CROSS") != 0) {
			sprintf(str, "ulsph cross interaction between particle type %d and %d requested, however, *CROSS keyword is missing",
					itype, jtype);
			error->all(FLERR, str);
		}

		if (setflag[itype][itype] != 1) {
			sprintf(str,
					"ulsph cross interaction between particle type %d and %d requested, however, properties of type %d  have not yet been specified",
					itype, jtype, itype);
			error->all(FLERR, str);
		}

		if (setflag[jtype][jtype] != 1) {
			sprintf(str,
					"ulsph cross interaction between particle type %d and %d requested, however, properties of type %d  have not yet been specified",
					itype, jtype, jtype);
			error->all(FLERR, str);
		}

		setflag[itype][jtype] = 1;
		setflag[jtype][itype] = 1;

	}
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairULSPH::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
//printf("cutoff for pair sph/fluid = %f\n", cutoff);
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairULSPH::init_style() {
	int i;

//printf(" in init style\n");
// request a granular neighbor list
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

	for (i = 0; i < nlocal; i++)
		onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);

	MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes,
	MPI_DOUBLE, MPI_MAX, world);
	MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes,
	MPI_DOUBLE, MPI_MAX, world);

}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairULSPH::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairULSPH::memory_usage() {

//printf("in memory usage\n");

	return 11 * nmax * sizeof(double);

}

/* ---------------------------------------------------------------------- */

int PairULSPH::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	double *rho = atom->rho;
	int i, j, m;

//printf("packing comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = pressure[j];
		buf[m++] = rho[j];
		buf[m++] = c0[j];

		buf[m++] = stressTensor[j](0, 0);
		buf[m++] = stressTensor[j](1, 1);
		buf[m++] = stressTensor[j](2, 2);
		buf[m++] = stressTensor[j](0, 1);
		buf[m++] = stressTensor[j](0, 2);
		buf[m++] = stressTensor[j](1, 2);

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairULSPH::unpack_forward_comm(int n, int first, double *buf) {
	double *rho = atom->rho;
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		pressure[i] = buf[m++];
		rho[i] = buf[m++];
		c0[i] = buf[m++];

		stressTensor[i](0, 0) = buf[m++];
		stressTensor[i](1, 1) = buf[m++];
		stressTensor[i](2, 2) = buf[m++];
		stressTensor[i](0, 1) = buf[m++];
		stressTensor[i](0, 2) = buf[m++];
		stressTensor[i](1, 2) = buf[m++];
		stressTensor[i](1, 0) = stressTensor[i](0, 1);
		stressTensor[i](2, 0) = stressTensor[i](0, 2);
		stressTensor[i](2, 1) = stressTensor[i](1, 2);
	}
}

/*
 * compute a normalized smoothing kernel and its derivative
 */

void PairULSPH::kernel_and_derivative(const double h, const double r, double &wf, double &wfd) {

	/*
	 * Spiky kernel
	 */

	double n;
	if (domain->dimension == 2) {
		n = 0.3141592654e0 * h * h * h * h * h;
	} else {
		n = 0.2094395103e0 * h * h * h * h * h * h;
	}

	double hr = h - r;
	wfd = 3.0e0 * hr * hr / n;
	wf = 0.333333333333e0 * hr * wfd;

	/*
	 * cubic spline - to do
	 */

}

/*
 * compute a normalized smoothing kernel only
 */
void PairULSPH::Poly6Kernel(const double hsq, const double h, const double rsq, double &wf) {

	double tmp = hsq - rsq;
	if (domain->dimension == 2) {
		wf = tmp * tmp * tmp / (0.7853981635e0 * hsq * hsq * hsq * hsq);
	} else {
		wf = tmp * tmp * tmp / (0.6382918409e0 * hsq * hsq * hsq * hsq * h);
	}
}

/*
 * Pseudo-inverse via SVD
 */

Matrix3d PairULSPH::pseudo_inverse_SVD(Matrix3d M) {

	JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV);

	Vector3d singularValuesInv;
	Vector3d singularValues = svd.singularValues();
	Matrix3d U = svd.matrixU();
	Matrix3d V = svd.matrixV();

//cout << "Here is the matrix V:" << endl << V * singularValues.asDiagonal() * U << endl;
//cout << "Its singular values are:" << endl << singularValues << endl;

	double pinvtoler = 1.0e-6;
	for (long row = 0; row < 3; row++) {
		if (singularValues(row) > pinvtoler) {
			singularValuesInv(row) = 1.0 / singularValues(row);
		} else {
			singularValuesInv(row) = 0.0;
		}
	}

	Matrix3d pInv;
	pInv = V * singularValuesInv.asDiagonal() * U.transpose();

	return pInv;
}

/*
 * EXTRACT
 */

void *PairULSPH::extract(const char *str, int &i) {
//printf("in extract\n");
	if (strcmp(str, "smd/ulsph/smoothVel_ptr") == 0) {
		return (void *) smoothVel;
	} else if (strcmp(str, "smd/ulsph/stressTensor_ptr") == 0) {
		return (void *) stressTensor;
	} else if (strcmp(str, "smd/ulsph/velocityGradient_ptr") == 0) {
		return (void *) L;
	} else if (strcmp(str, "smd/ulsph/numNeighs_ptr") == 0) {
		return (void *) numNeighs;
	} else if (strcmp(str, "smd/ulsph/dtCFL_ptr") == 0) {
		//printf("dtcfl = %f\n", dtCFL);
		return (void *) &dtCFL;
	}

	return NULL;
}

/*
 * deviator of a tensor
 */
Matrix3d PairULSPH::Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}

double PairULSPH::SafeLookup(std::string str, int itype) {
//cout << "string passed to lookup: " << str << endl;
	char msg[128];
	if (matProp2.count(std::make_pair(str, itype)) == 1) {
		//cout << "returning look up value %d " << matProp2[std::make_pair(str, itype)] << endl;
		return matProp2[std::make_pair(str, itype)];
	} else {
		//sprintf(msg, "failed to lookup indentifier [%s] for particle type %d", str, itype);
		error->all(FLERR, msg);
	}
	return 1.0;
}

bool PairULSPH::CheckKeywordPresent(std::string str, int itype) {
	int count = matProp2.count(std::make_pair(str, itype));
	if (count == 0) {
		return false;
	} else if (count == 1) {
		return true;
	} else {
		char msg[128];
		cout << "keyword: " << str << endl;
		sprintf(msg, "ambiguous count for keyword: %d times present\n", count);
		error->all(FLERR, msg);
	}
	return false;
}
