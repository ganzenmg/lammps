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
#include "pair_smd_tlsph.h"
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
#include "math_special.h"
#include <map>
#include "fix_smd_integrate_tlsph.h"
#include <Eigen/SVD>
#include <Eigen/Eigen>
#include "smd_material_models.h"
using namespace Eigen;
using namespace std;
using namespace LAMMPS_NS;

#define JAUMANN false
#define DETF_MIN 0.1 // maximum compression deformation allowed
#define DETF_MAX 40.0 // maximum tension deformation allowed
#define TLSPH_DEBUG 1
#define PLASTIC_STRAIN_AVERAGE_WINDOW 100.0

/* ---------------------------------------------------------------------- */

PairTlsph::PairTlsph(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;

	hg_coeff = NULL;
	Q1 = Q2 = NULL;
	strengthModel = eos = NULL;
	signal_vel0 = NULL; // signal velocity basedon on p-wave speed, used for artificial viscosity
	rho0 = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	Fdot = Fincr = K = PK1 = NULL;
	d = R = FincrInv = W = D = NULL;
	detF = NULL;
	smoothVelDifference = NULL;
	shearFailureFlag = NULL;
	numNeighsRefConfig = NULL;
	shepardWeight = NULL;
	CauchyStress = NULL;
	hourglass_error = NULL;

	updateFlag = 0;
	not_first = 0;

	comm_forward = 20; // this pair style communicates 20 doubles to ghost atoms : PK1 tensor + F tensor + shepardWeight
}

/* ---------------------------------------------------------------------- */

PairTlsph::~PairTlsph() {
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(youngsmodulus);
		memory->destroy(hg_coeff);
		memory->destroy(Q1);
		memory->destroy(Q2);
		memory->destroy(strengthModel);
		memory->destroy(eos);
		memory->destroy(signal_vel0);
		memory->destroy(rho0);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;

		delete[] Fdot;
		delete[] Fincr;
		delete[] K;
		delete[] detF;
		delete[] PK1;
		delete[] smoothVelDifference;
		delete[] d;
		delete[] R;
		delete[] FincrInv;
		delete[] W;
		delete[] D;
		delete[] shearFailureFlag;
		delete[] numNeighsRefConfig;
		delete[] shepardWeight;
		delete[] CauchyStress;
		delete[] hourglass_error;
	}
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairTlsph::PreCompute() {
	int *mol = atom->molecule;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double **x0 = atom->x0;
	double **x = atom->x;
	double **v = atom->vest; // extrapolated velocities corresponding to current positions
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	int *tag = atom->tag;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, j, itype;
	double r0, r0Sq, wf, wfd, h, irad, voli, volj;
	Vector3d dx0, dx, dv, g;
	Matrix3d Ktmp, Fdottmp, Ftmp, L, Fold, U, eye;
	Vector3d xi, xj, vi, vj, vinti, vintj, x0i, x0j, dvint, du;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	eye.setIdentity();

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		K[i].setZero();
		Fincr[i].setZero();
		Fdot[i].setZero();
		shearFailureFlag[i] = false;
		shepardWeight[i] = 0.0;
		numNeighsRefConfig[i] = 0;
		smoothVelDifference[i].setZero();
		hourglass_error[i] = 0.0;
	}

	// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];

		if (mol[i] < 0) { // valid SPH particle have mol > 0
			continue;
		}

		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];
		voli = vfrac[i];

		// initialize Eigen data structures from LAMMPS data structures
		x0i << x0[i][0], x0[i][1], x0[i][2];
		xi << x[i][0], x[i][1], x[i][2];
		vi << v[i][0], v[i][1], v[i][2];
		vinti << vint[i][0], vint[i][1], vint[i][2];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (mol[j] < 0) { // particle has failed. do not include it for computing any property
				continue;
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			x0j << x0[j][0], x0[j][1], x0[j][2];
			dx0 = x0j - x0i;

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			r0Sq = dx0.squaredNorm();
			h = irad + radius[j];
			if (r0Sq < h * h) {

				// initialize Eigen data structures from LAMMPS data structures
				xj << x[j][0], x[j][1], x[j][2];
				vj << v[j][0], v[j][1], v[j][2];
				vintj << vint[j][0], vint[j][1], vint[j][2];

				r0 = sqrt(r0Sq);
				volj = vfrac[j];

				// distance vectors in current and reference configuration, velocity difference
				dx = xj - xi;
				if (periodic)
					domain->minimum_image(dx(0), dx(1), dx(2));
				dv = vj - vi;
				dvint = vintj - vinti;

				// kernel function
				barbara_kernel_and_derivative(h, r0, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r0) * dx0;

				/* build matrices */
				Ktmp = g * dx0.transpose();
				Ftmp = (dx - dx0) * g.transpose();
				Fdottmp = dv * g.transpose();

				K[i] += volj * Ktmp;
				Fincr[i] += volj * Ftmp;
				Fdot[i] += volj * Fdottmp;
				shepardWeight[i] += volj * wf;
				smoothVelDifference[i] += volj * wf * dvint;
				numNeighsRefConfig[i]++;

				if (j < nlocal) {
					K[j] += voli * Ktmp;
					Fincr[j] += voli * Ftmp;
					Fdot[j] += voli * Fdottmp;
					shepardWeight[j] += voli * wf;
					smoothVelDifference[j] -= voli * wf * dvint;
					numNeighsRefConfig[j]++;
				}

			} // end if check distance
		} // end loop over j
	} // end loop over i

	/*
	 * invert shape matrix and compute corrected quantities
	 */

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype] == 1) { // we only do the subsequent calculation if pair style is defined for this particle

			if ((numNeighsRefConfig[i] < domain->dimension + 2) && (mol[i] > 0)) { // cannot possibly invert shape matrix
				printf("deleting particle [%d] because number of neighbors=%d is too small\n", tag[i], numNeighsRefConfig[i]);
				mol[i] = -1;
			}

			if (mol[i] > 0) {

				if (domain->dimension == 2) {
					K[i](0, 2) = 0.0;
					K[i](1, 2) = 0.0;
					K[i](2, 0) = 0.0;
					K[i](2, 1) = 0.0;
					K[i](2, 2) = 1.0;
					K[i] = K[i].inverse().eval();
					//K[i] = PairTlsph::pseudo_inverse_SVD(K[i]);
					K[i](2, 2) = 1.0; // make inverse of K well defined even when it is rank-deficient (3d matrix, only 2d information)
				} else {
					K[i] = K[i].inverse().eval();
					//K[i] = PairTlsph::pseudo_inverse_SVD(K[i]);
				}

				Fincr[i] *= K[i]; // use shape matrix to obtain first-order corrected SPH approximation
				Fincr[i] += eye;
				Fdot[i] *= K[i];

				/*
				 * we need to be able to recover from a potentially planar (2d) configuration of particles
				 */
				if (domain->dimension == 2) {
					Fincr[i](0, 2) = 0.0;
					Fincr[i](1, 2) = 0.0;
					Fincr[i](2, 0) = 0.0;
					Fincr[i](2, 1) = 0.0;
					Fincr[i](2, 2) = 1.0;
					Fdot[i](0, 2) = 0.0;
					Fdot[i](1, 2) = 0.0;
					Fdot[i](2, 0) = 0.0;
					Fdot[i](2, 1) = 0.0;
					Fdot[i](2, 2) = 0.0;
				}
				detF[i] = Fincr[i].determinant();
				//FincrInv[i] = PairTlsph::pseudo_inverse_SVD(Fincr[i]);;
				FincrInv[i] = Fincr[i].inverse();

				/*
				 * check that F * F^-1 is unit matrix
				 */
				Matrix3d Mdiff = Fincr[i] * FincrInv[i] - eye;
				if (Mdiff.norm() > 1.0e-10) {
					printf("Inversion of F is inaccurate\n");
					cout << "Here is the difference matrix to unity:" << endl << Mdiff << endl;
				}

//				if (tag[i] == 101) {
//					printf("J=%f, nn = %d\n", detF[i], numNeighsRefConfig[i]);
//					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
//					cout << "Here is matrix F-1:" << endl << FincrInv[i] << endl;
//					cout << "Here is matrix K-1:" << endl << K[i] << endl << endl;
//				}

				/*
				 * make sure F stays within some limits
				 */

				if (detF[i] < DETF_MIN) {
					printf("deleting particle [%d] because det(F)=%f is smaller than limit=%f\n", tag[i], Fincr[i].determinant(),
					DETF_MIN);
					printf("nn = %d\n", numNeighsRefConfig[i]);
					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
					cout << "Here is matrix F-1:" << endl << FincrInv[i] << endl;
					cout << "Here is matrix K-1:" << endl << K[i] << endl;
					mol[i] = -1;
					//error->one(FLERR, "");
				} else if (detF[i] > DETF_MAX) {
					printf("deleting particle [%d] because det(F)=%f is larger than limit=%f\n", tag[i], Fincr[i].determinant(),
					DETF_MAX);
					printf("nn = %d\n", numNeighsRefConfig[i]);
					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
					cout << "Here is matrix F-1:" << endl << FincrInv[i] << endl;
					cout << "Here is matrix K-1:" << endl << K[i] << endl;
					mol[i] = -1;
					//error->one(FLERR, "");
				}

				if (mol[i] > 0) {
					// velocity gradient, see Pronto2d, eqn.(2.1.3)
					L = Fdot[i] * FincrInv[i];

					// symmetric (D) and asymmetric (W) parts of L
					D[i] = 0.5 * (L + L.transpose());
					W[i] = 0.5 * (L - L.transpose());					// spin tensor:: need this for Jaumann rate

					if (JAUMANN) {
						d[i] = D[i];
						R[i].setIdentity(); // for Jaumann stress rate, we do not need a subsequent rotation back into the reference configuration
					} else {

						// convention: unrotated frame is that one, where the true rotation of an integration point has been subtracted.
						// stress in the unrotated frame of reference is denoted sigma (stress seen by an observer doing rigid body rotations along with the material)
						// stress in the true frame of reference (a stationary observer) is denoted by T, "true stress"

						// polar decomposition of the deformation gradient, F = R * U

						PairTlsph::PolDec(Fincr[i], &R[i], &U);

						// unrotated rate-of-deformation tensor d, see right side of Pronto2d, eqn.(2.1.7)
						d[i] = R[i].transpose() * D[i] * R[i];
					}

					// normalize average velocity field around an integration point
					smoothVelDifference[i] /= shepardWeight[i];

				} // end if mol[i] > 0

			} // end if mol[i] > 0

			if (mol[i] < 0) {

				d[i].setZero();
				D[i].setZero();
				Fdot[i].setZero();
				Fincr[i].setIdentity();
				//smoothVel[i].setZero();
				detF[i] = 1.0;
				K[i].setIdentity();
			}
		}  // end if setflag[itype]
	} // end loop over i = 0 to nlocal
}

/* ---------------------------------------------------------------------- */

void PairTlsph::compute(int eflag, int vflag) {

	if (atom->nmax > nmax) {
		nmax = atom->nmax;
		delete[] Fdot;
		Fdot = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] Fincr;
		Fincr = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] K;
		K = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] PK1;
		PK1 = new Matrix3d[nmax]; // memory usage: 9 doubles; total 5*9=45 doubles
		delete[] detF;
		detF = new double[nmax]; // memory usage: 1 double; total 46 doubles
		delete[] smoothVelDifference;
		smoothVelDifference = new Vector3d[nmax]; // memory usage: 3 doubles; total 49 doubles
		delete[] d;
		d = new Matrix3d[nmax]; // memory usage: 9 doubles; total 58 doubles
		delete[] R;
		R = new Matrix3d[nmax]; // memory usage: 9 doubles; total 67 doubles
		delete[] FincrInv;
		FincrInv = new Matrix3d[nmax]; // memory usage: 9 doubles; total 85 doubles
		delete[] W;
		W = new Matrix3d[nmax]; // memory usage: 9 doubles; total 94 doubles
		delete[] D;
		D = new Matrix3d[nmax]; // memory usage: 9 doubles; total 103 doubles
		delete[] shearFailureFlag;
		shearFailureFlag = new bool[nmax]; // memory usage: 1 double; total 104 doubles
		delete[] numNeighsRefConfig;
		numNeighsRefConfig = new int[nmax]; // memory usage: 1 int; total 108 doubles
		delete[] shepardWeight;
		shepardWeight = new double[nmax]; // memory usage: 1 double; total 109 doubles
		delete[] CauchyStress;
		CauchyStress = new Matrix3d[nmax]; // memory usage: 9 doubles; total 118 doubles
		delete[] hourglass_error;
		hourglass_error = new double[nmax];
	}

//	if (update->ntimestep % 100 == 0) {
//		SmoothField();
//	}

	PairTlsph::PreCompute();
	PairTlsph::AssembleStress();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);

	ComputeForces(eflag, vflag);
}

void PairTlsph::ComputeForces(int eflag, int vflag) {
	int *mol = atom->molecule;
	double **x = atom->x;
	double **v = atom->vest;
	double **x0 = atom->x0;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *de = atom->de;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *plastic_strain = atom->eff_plastic_strain;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, iDim, inum;
	double r, hg_mag, wf, wfd, h, r0, r0Sq, voli, volj;
	double delVdotDelR, visc_magnitude, deltaE, mu_ij, hg_err;
	double delta, stress_force_magnitude, hourglass_force_magnitude;
	char str[128];
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	Vector3d fi, fj, dx0, dx, dv, f_stress, f_hg, dxp_i, dxp_j, gamma, g, gamma_i, gamma_j;
	Vector3d xi, xj, vi, vj, f_visc, sumForces;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	/*
	 * iterate over pairs of particles i, j and assign forces using PK1 stress tensor
	 */

	// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	updateFlag = 0;
	hMin = 1.0e22;
	dtRelative = 1.0e22;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];

		if (mol[i] < 0) {
			continue; // Particle i is not a valid SPH particle (anymore). Skip all interactions with this particle.
		}

		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		// initialize Eigen data structures from LAMMPS data structures
		for (iDim = 0; iDim < 3; iDim++) {
			xi(iDim) = x[i][iDim];
			vi(iDim) = v[i][iDim];
		}

		voli = vfrac[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (mol[j] < 0) {
				continue; // Particle j is not a valid SPH particle (anymore). Skip all interactions with this particle.
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			if (type[j] != itype) {
				sprintf(str, "particle pair is not of same type!");
				error->all(FLERR, str);
			}

			// check that distance between i and j (in the reference config) is less than cutoff
			dx0(0) = x0[j][0] - x0[i][0];
			dx0(1) = x0[j][1] - x0[i][1];
			dx0(2) = x0[j][2] - x0[i][2];

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			r0Sq = dx0.squaredNorm();
			h = radius[i] + radius[j];
			if (r0Sq < h * h) {

				hMin = MIN(hMin, h);
				r0 = sqrt(r0Sq);

				// initialize Eigen data structures from LAMMPS data structures
				for (iDim = 0; iDim < 3; iDim++) {
					xj(iDim) = x[j][iDim];
					vj(iDim) = v[j][iDim];
				}

				volj = vfrac[j];

				// distance vectors in current and reference configuration, velocity difference
				dx = xj - xi;
				dv = vj - vi;

				// derivative of kernel function and reference distance
				// kernel function
				barbara_kernel_and_derivative(h, r0, wf, wfd);
				//printf("wf = %f, wfd = %f\n", wf, wfd);

				// current distance
				r = dx.norm();

				// uncorrected kernel gradient
				g = (wfd / r0) * dx0;

				/*
				 * force contribution -- note that the kernel gradient correction has been absorbed into PK1
				 */

				f_stress = voli * volj * (PK1[i] + PK1[j]) * g;

				/*
				 * artificial viscosity
				 */

				// because hourglass control and artificial viscosity are evaluated using the current coordinates,
				// call kernel and kernel gradient weights with current separation
				spiky_kernel_and_derivative(h, r, wf, wfd);

				delVdotDelR = dx.dot(dv);
				mu_ij = h * delVdotDelR / (r * r + 0.1 * h * h); // m * m/s * m / m*m ==> units m/s
				visc_magnitude = (-Q1[itype] * signal_vel0[itype] * mu_ij + Q2[itype] * mu_ij * mu_ij) / rho0[itype]; // this has units of pressure
				//printf("visc_magnitude = %f, c_ij=%f, mu_ij=%f\n", visc_magnitude, c_ij, mu_ij);
				f_visc = rmass[i] * rmass[j] * visc_magnitude * wfd * dx / (r + 1.0e-2 * h);

				/*
				 * hourglass deviation of particles i and j
				 */

				gamma = 0.5 * (Fincr[i] + Fincr[j]) * dx0 - dx;
				hg_err = gamma.norm() / r0;

				hourglass_error[i] += volj * wf * hg_err;

				/*
				 * friction-like hourglass formulation
				 */

//				hg_mag = hg_coeff[itype] * hg_err; // hg_mag has no dimensions
//				hg_mag *= voli * volj * wf * youngsmodulus[itype] / h; // hg_mag has dimensions [J/m] == [N]
//				f_hg = hg_mag * gamma / (gamma.norm() + 0.1 * h);
//				if (gamma.dot(dv) != 0.0) {
//					hg_mag = hg_coeff[itype] * hg_err * gamma.dot(dv) / (gamma.norm() * dv.norm()); // hg_mag has no dimensions
//					hg_mag *= voli * volj * wf * youngsmodulus[itype] / h; // hg_mag has dimensions [J/m] == [N]
//					if (dv.norm() > 1.0e-16) {
//						f_hg = hg_mag * dv / dv.norm();
//					} else {
//						f_hg.setZero();
//					}
//				} else {
//					f_hg.setZero();
//				}
				/* SPH-like hourglass formulation */

				if (MAX(plastic_strain[i], plastic_strain[j]) > 1.0e-3) {

					/*
					 * viscous hourglass formulation
					 */

					delta = gamma.dot(dx);
					if (delVdotDelR * delta < 0.0) {
						hg_mag = -hg_err * hg_coeff[itype] * signal_vel0[itype] * mu_ij / rho0[itype]; // this has units of pressure
					} else {
						hg_mag = 0.0;
					}
					f_hg = rmass[i] * rmass[j] * hg_mag * wfd * dx / (r + 1.0e-2 * h);

				} else {
					/*
					 * stiffness hourglass formulation
					 */

					delta = 0.5 * gamma.dot(dx) / (r + 0.1 * h); // delta has dimensions of [m]
					hg_mag = hg_coeff[itype] * delta / (r0Sq + 0.01 * h * h); // hg_mag has dimensions [m^(-1)]
					hg_mag *= -voli * volj * wf * youngsmodulus[itype]; // hg_mag has dimensions [J*m^(-1)] = [N]
					f_hg = (hg_mag / r) * dx;
					//printf("delta = %g, hg_mag = %g, coeff=%g, wf=%g, vi=%g, vj=%g, Ei=%f, Ej-%f\n",delta, hg_mag, hg_coeff[itype][jtype], wf, voli, volj,
					//		youngsmodulus[itype], youngsmodulus[jtype]);

				}

				/*
				 * scale hourglass forces if neccessary
				 */

				stress_force_magnitude = f_stress.norm();
				hourglass_force_magnitude = f_hg.norm();
				if (hourglass_force_magnitude > 100.0 * stress_force_magnitude) {
					f_hg = 100.0 * stress_force_magnitude * (f_hg / hourglass_force_magnitude);
				}

				// sum stress, viscous, and hourglass forces
				sumForces = f_stress + f_visc + f_hg;

				// energy rate -- project velocity onto force vector
				deltaE = 0.5 * sumForces.dot(dv);

				// apply forces to pair of particles
				f[i][0] += sumForces(0);
				f[i][1] += sumForces(1);
				f[i][2] += sumForces(2);
				de[i] += deltaE;

				if (j < nlocal) {
					f[j][0] -= sumForces(0);
					f[j][1] -= sumForces(1);
					f[j][2] -= sumForces(2);
					de[j] += deltaE;
					hourglass_error[j] += voli * wf * hg_err;
				}

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
				}

				// check if a particle has moved too much w.r.t another particles
				if (r > r0) {
					//if ((r - r0) > 0.75 * neighbor->skin) { works reasonably well, but let's base the update criterion on a per-particle length-scale
					if (r > h + 0.75 * neighbor->skin) {
						//printf("current distance is %f, r0 distance is %f\n", r, r0);
						updateFlag = 1;
					}
				}
			}
		}
	}

	for (i = 0; i < nlocal; i++) {
		if (shepardWeight[i] != 0.0) {
			hourglass_error[i] /= shepardWeight[i];
		}
	}

	if (vflag_fdotr)
		virial_fdotr_compute();
}



/* ----------------------------------------------------------------------
 assemble unrotated stress tensor using deviatoric and pressure components.
 Convert to corotational Cauchy stress, then to PK1 stress and apply
 shape matrix correction
 ------------------------------------------------------------------------- */
void PairTlsph::AssembleStress() {
	int *mol = atom->molecule;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	double **tlsph_stress = atom->tlsph_stress;
	int *type = atom->type;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double *rmass = atom->rmass;
	double *vfrac = atom->vfrac;
	double *e = atom->e;
	double pInitial, d_iso, pFinal, p_rate, plastic_strain_increment;
	int i, itype;
	int nlocal = atom->nlocal;
	double dt = update->dt;
	double M, p_wave_speed, mass_specific_energy, vol_specific_energy, rho, damage_increment;
	Matrix3d sigma_rate, eye, sigmaInitial, sigmaFinal, T, Jaumann_rate, sigma_rate_check;
	Matrix3d d_dev, sigmaInitial_dev, sigmaFinal_dev, sigma_dev_rate, strain, sigma_damaged;
	Vector3d x0i, xi, xp;

	eye.setIdentity();
	dtCFL = 1.0e22;
	pFinal = 0.0;

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype] == 1) {
			if (mol[i] > 0) { // only do the following if particle has not failed -- mol < 0 means particle has failed

				/*
				 * initial stress state: given by the unrotateted Cauchy stress.
				 * Assemble Eigen 3d matrix from stored stress state
				 */
				sigmaInitial(0, 0) = tlsph_stress[i][0];
				sigmaInitial(0, 1) = tlsph_stress[i][1];
				sigmaInitial(0, 2) = tlsph_stress[i][2];
				sigmaInitial(1, 1) = tlsph_stress[i][3];
				sigmaInitial(1, 2) = tlsph_stress[i][4];
				sigmaInitial(2, 2) = tlsph_stress[i][5];
				sigmaInitial(1, 0) = sigmaInitial(0, 1);
				sigmaInitial(2, 0) = sigmaInitial(0, 2);
				sigmaInitial(2, 1) = sigmaInitial(1, 2);

				pInitial = sigmaInitial.trace() / 3.0; // initial pressure, isotropic part of initial stress
				sigmaInitial_dev = PairTlsph::Deviator(sigmaInitial);
				d_iso = d[i].trace(); // volumetric part of stretch rate
				d_dev = PairTlsph::Deviator(d[i]); // deviatoric part of stretch rate
				strain = 0.5 * (Fincr[i].transpose() * Fincr[i] - eye);

				mass_specific_energy = e[i] / rmass[i]; // energy per unit mass
				rho = rmass[i] / (detF[i] * vfrac[i]); // current mass density
				vol_specific_energy = mass_specific_energy * rho; // energy per current volume

				/*
				 * pressure: compute pressure rate p_rate and final pressure pFinal
				 */

				ComputePressure(i, pInitial, d_iso, pFinal, p_rate);

				/*
				 * material strength
				 */

				if (damage[i] < 1.0) {
					ComputeStressDeviator(i, sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);

					eff_plastic_strain[i] += plastic_strain_increment;

					eff_plastic_strain_rate[i] -= eff_plastic_strain_rate[i] / PLASTIC_STRAIN_AVERAGE_WINDOW;
					eff_plastic_strain_rate[i] += (plastic_strain_increment / dt) / PLASTIC_STRAIN_AVERAGE_WINDOW;
				} else {
					sigmaFinal_dev.setZero(); // damage[i] >= 1, therefore it cannot carry shear stress
					sigma_dev_rate.setZero();
				}

				/*
				 *  assemble total stress from pressure and deviatoric stress
				 */
				sigmaFinal = pFinal * eye + sigmaFinal_dev; // this is the stress that is kept

				/*
				 *  Damage due to failure criteria.
				 */

				if (damage[i] < 1.0) {

					ComputeDamage(i, strain, pFinal, sigmaFinal, sigmaFinal_dev, sigma_damaged, damage_increment);

					damage[i] += damage_increment;
					damage[i] = MIN(damage[i], 1.0);

				} // end if damage[i] < 1.0

				/*
				 * Apply damage
				 */

				if (pFinal < 0.0) { // compression: particle can carry compressive load but reduced shear
					sigma_damaged = pFinal * eye + (1.0 - damage[i]) * sigmaFinal_dev;
				} else { // tension: particle has reduced tensile and shear load bearing capability
					sigma_damaged = (1.0 - damage[i]) * (pFinal * eye + sigmaFinal_dev);
				}

				if (JAUMANN) {
					/*
					 * sigma is already the co-rotated Cauchy stress.
					 * The stress rate, however, needs to be made objective.
					 */

					sigma_rate = (1.0 / dt) * (sigmaFinal - sigmaInitial);

					Jaumann_rate = sigma_rate + W[i] * sigmaInitial + sigmaInitial * W[i].transpose();
					sigmaFinal = sigmaInitial + dt * Jaumann_rate;
					T = sigmaFinal;
				} else {
					/*
					 * sigma is the unrotated stress.
					 * need to do forward rotation of the unrotated stress sigma to the current configuration
					 */
					T = R[i] * sigma_damaged * R[i].transpose();
				}

				/*
				 * store unrotated stress in atom vector
				 * symmetry is exploited
				 */
				tlsph_stress[i][0] = sigmaFinal(0, 0);
				tlsph_stress[i][1] = sigmaFinal(0, 1);
				tlsph_stress[i][2] = sigmaFinal(0, 2);
				tlsph_stress[i][3] = sigmaFinal(1, 1);
				tlsph_stress[i][4] = sigmaFinal(1, 2);
				tlsph_stress[i][5] = sigmaFinal(2, 2);

				// store rotated, "true" Cauchy stress
				CauchyStress[i] = T;

				/*
				 * We have the corotational Cauchy stress.
				 * Convert to PK1. Note that reference configuration used for computing the forces is linked via
				 * the incremental deformation gradient, not the full deformation gradient.
				 */
				PK1[i] = detF[i] * T * FincrInv[i].transpose();

				/*
				 * pre-multiply stress tensor with shape matrix to save computation in force loop
				 */
				PK1[i] = PK1[i] * K[i];

				//cout << "F:" << Fincr[i] << endl;
				//cout << "T:" << T << endl;
				//cout << "PK1:" << PK1[i] << endl << endl;

				/*
				 * compute stable time step according to Pronto 2d
				 */

				Matrix3d deltaSigma;
				deltaSigma = sigma_damaged - sigmaInitial;
				p_rate = deltaSigma.trace() / (3.0 * dt);
				sigma_dev_rate = Deviator(deltaSigma) / dt;

				M = effective_longitudinal_modulus(itype, dt, d_iso, p_rate, d_dev, sigma_dev_rate, damage[i]);
				p_wave_speed = sqrt(M / (rmass[i] / vfrac[i]));
				//printf("c0 = %f\n", c0[i]);
				dtCFL = MIN(dtCFL, radius[i] / p_wave_speed);

			} else { // end if delete_flag == 0
				PK1[i].setZero();
				K[i].setIdentity();
				CauchyStress[i].setZero();

				sigma_rate.setZero();
				tlsph_stress[i][0] = 0.0;
				tlsph_stress[i][1] = 0.0;
				tlsph_stress[i][2] = 0.0;
				tlsph_stress[i][3] = 0.0;
				tlsph_stress[i][4] = 0.0;
				tlsph_stress[i][5] = 0.0;
			}

		} else { // setflag[i] is not checked
			PK1[i].setZero();
			K[i].setIdentity();
			CauchyStress[i].setZero();
		}
	}
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairTlsph::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(youngsmodulus, n + 1, "pair:youngsmodulus");
	memory->create(hg_coeff, n + 1, "pair:hg_magnitude");
	memory->create(Q1, n + 1, "pair:q1");
	memory->create(Q2, n + 1, "pair:q2");
	memory->create(strengthModel, n + 1, "pair:strengthmodel");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(signal_vel0, n + 1, "pair:signal_vel0");
	memory->create(rho0, n + 1, "pair:rho0");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairTlsph::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairTlsph::coeff(int narg, char **arg) {
	int ioffset, iarg, iNextKwd, itype;
	char str[128];
	std::string s, t;

	if (narg < 3) {
		sprintf(str, "number of arguments for pair tlsph is too small!");
		error->all(FLERR, str);
	}
	if (!allocated)
		allocate();

	/*
	 * check that TLSPH parameters are given only in i,i form
	 */
	if (force->inumeric(FLERR, arg[0]) != force->inumeric(FLERR, arg[1])) {
		sprintf(str, "TLSPH coefficients can only be specified between particles of same type!");
		error->all(FLERR, str);
	}
	itype = force->inumeric(FLERR, arg[0]);

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("SMD / TLSPH PROPERTIES OF PARTICLE TYPE %d:\n", itype);
	}

	/*
	 * read parameters which are common -- regardless of material / eos model
	 */

	ioffset = 2;
	if (strcmp(arg[ioffset], "*COMMON") != 0) {
		sprintf(str, "common keyword missing!");
		error->all(FLERR, str);
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

	//printf("keyword following *COMMON is %s\n", arg[iNextKwd]);

	if (iNextKwd < 0) {
		sprintf(str, "no *KEYWORD terminates *COMMON");
		error->all(FLERR, str);
	}

	if (iNextKwd - ioffset != 7 + 1) {
		sprintf(str, "expected 7 arguments following *COMMON but got %d\n", iNextKwd - ioffset - 1);
		error->all(FLERR, str);
	}

	matProp2[std::make_pair("rho_ref", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
	matProp2[std::make_pair("youngs_modulus", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
	matProp2[std::make_pair("poisson_ratio", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
	matProp2[std::make_pair("viscosity_q1", itype)] = force->numeric(FLERR, arg[ioffset + 4]);
	matProp2[std::make_pair("viscosity_q2", itype)] = force->numeric(FLERR, arg[ioffset + 5]);
	matProp2[std::make_pair("hourglass_amp", itype)] = force->numeric(FLERR, arg[ioffset + 6]);
	matProp2[std::make_pair("heat_capacity", itype)] = force->numeric(FLERR, arg[ioffset + 7]);

	matProp2[std::make_pair("lame_lambda", itype)] = SafeLookup("youngs_modulus", itype) * SafeLookup("poisson_ratio", itype)
			/ ((1.0 + SafeLookup("poisson_ratio", itype) * (1.0 - 2.0 * SafeLookup("poisson_ratio", itype))));
	matProp2[std::make_pair("lame_mu", itype)] = SafeLookup("youngs_modulus", itype)
			/ (2.0 * (1.0 + SafeLookup("poisson_ratio", itype)));
	matProp2[std::make_pair("signal_vel", itype)] = sqrt(
			(SafeLookup("lame_lambda", itype) + 2.0 * SafeLookup("lame_mu", itype)) / SafeLookup("rho_ref", itype));
	matProp2[std::make_pair("bulk_modulus", itype)] = SafeLookup("lame_lambda", itype) + 2.0 * SafeLookup("lame_mu", itype) / 3.0;
	matProp2[std::make_pair("m_modulus", itype)] = SafeLookup("lame_lambda", itype) + 2.0 * SafeLookup("lame_mu", itype);

	if (comm->me == 0) {
		printf("\n material unspecific properties for SMD/TLSPH definition of particle type %d:\n", itype);
		printf("%60s : %g\n", "reference density", SafeLookup("rho_ref", itype));
		printf("%60s : %g\n", "Young's modulus", SafeLookup("youngs_modulus", itype));
		printf("%60s : %g\n", "Poisson ratio", SafeLookup("poisson_ratio", itype));
		printf("%60s : %g\n", "linear viscosity coefficient", SafeLookup("viscosity_q1", itype));
		printf("%60s : %g\n", "quadratic viscosity coefficient", SafeLookup("viscosity_q2", itype));
		printf("%60s : %g\n", "hourglass control coefficient", SafeLookup("hourglass_amp", itype));
		printf("%60s : %g\n", "heat capacity [energy / (mass * temperature)]", SafeLookup("heat_capacity", itype));
		printf("%60s : %g\n", "Lame constant lambda", SafeLookup("lame_lambda", itype));
		printf("%60s : %g\n", "shear modulus", SafeLookup("lame_mu", itype));
		printf("%60s : %g\n", "bulk modulus", SafeLookup("bulk_modulus", itype));
		printf("%60s : %g\n", "signal velocity", SafeLookup("signal_vel", itype));

	}

	/*
	 * read following material cards
	 */

	//printf("next kwd is %s\n", arg[iNextKwd]);
	eos[itype] = EOS_NONE;
	strengthModel[itype] = STRENGTH_NONE;

	while (true) {
		if (strcmp(arg[iNextKwd], "*END") == 0) {
			if (comm->me == 0) {
				printf("found *END keyword");
				printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n\n");
			}
			break;
		}

		/*
		 * Linear Elasticity model based on deformation gradient
		 */
		ioffset = iNextKwd;
		if (strcmp(arg[ioffset], "*LINEAR_DEFGRAD") == 0) {
			strengthModel[itype] = LINEAR_DEFGRAD;

			if (comm->me == 0) {
				printf("reading *LINEAR_DEFGRAD\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_DEFGRAD");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *LINEAR_DEFGRAD but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "Linear Elasticity model based on deformation gradient");
			}
		} else if (strcmp(arg[ioffset], "*LINEAR_STRENGTH") == 0) {

			/*
			 * Linear Elasticity strength only model based on strain rate
			 */

			strengthModel[itype] = LINEAR_STRENGTH;
			if (comm->me == 0) {
				printf("reading *LINEAR_STRENGTH\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_STRENGTH");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *LINEAR_STRENGTH but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("%60s\n", "Linear Elasticity strength based on strain rate");
			}
		} // end Linear Elasticity strength only model based on strain rate

		else if (strcmp(arg[ioffset], "*LINEAR_PLASTICITY") == 0) {

			/*
			 * Linear Elastic / perfectly plastic strength only model based on strain rate
			 */

			strengthModel[itype] = LINEAR_PLASTICITY;
			if (comm->me == 0) {
				printf("reading *LINEAR_PLASTICITY\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_PLASTICITY");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *LINEAR_PLASTICITY but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("yield_stress0", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("%60s\n", "Linear elastic / perfectly plastic strength based on strain rate");
				printf("%60s : %g\n", "Young's modulus", SafeLookup("youngs_modulus", itype));
				printf("%60s : %g\n", "Poisson ratio", SafeLookup("poisson_ratio", itype));
				printf("%60s : %g\n", "shear modulus", SafeLookup("lame_mu", itype));
				printf("%60s : %g\n", "constant yield stress", SafeLookup("yield_stress0", itype));
			}
		} // end Linear Elastic / perfectly plastic strength only model based on strain rate

		else if (strcmp(arg[ioffset], "*JOHNSON_COOK") == 0) {

			/*
			 * JOHNSON - COOK
			 */

			strengthModel[itype] = STRENGTH_JOHNSON_COOK;
			if (comm->me == 0) {
				printf("reading *JOHNSON_COOK\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *JOHNSON_COOK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 8 + 1) {
				sprintf(str, "expected 8 arguments following *JOHNSON_COOK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("JC_A", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
			matProp2[std::make_pair("JC_B", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
			matProp2[std::make_pair("JC_a", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
			matProp2[std::make_pair("JC_C", itype)] = force->numeric(FLERR, arg[ioffset + 4]);
			matProp2[std::make_pair("JC_epdot0", itype)] = force->numeric(FLERR, arg[ioffset + 5]);
			matProp2[std::make_pair("JC_T0", itype)] = force->numeric(FLERR, arg[ioffset + 6]);
			matProp2[std::make_pair("JC_Tmelt", itype)] = force->numeric(FLERR, arg[ioffset + 7]);
			matProp2[std::make_pair("JC_M", itype)] = force->numeric(FLERR, arg[ioffset + 8]);

			if (comm->me == 0) {
				printf("%60s\n", "Johnson Cook material strength model");
				printf("%60s : %g\n", "A: initial yield stress", SafeLookup("JC_A", itype));
				printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", SafeLookup("JC_B", itype));
				printf("%60s : %g\n", "a : exponent for plastic strain dependency", SafeLookup("JC_a", itype));
				printf("%60s : %g\n", "C : proportionality factor for logarithmic plastic strain rate dependency",
						SafeLookup("JC_C", itype));
				printf("%60s : %g\n", "epdot0 : dimensionality factor for plastic strain rate dependency",
						SafeLookup("JC_epdot0", itype));
				printf("%60s : %g\n", "T0 : reference (room) temperature", SafeLookup("JC_T0", itype));
				printf("%60s : %g\n", "Tmelt : melting temperature", SafeLookup("JC_Tmelt", itype));
				printf("%60s : %g\n", "M : exponent for temperature dependency", SafeLookup("JC_M", itype));
			}

		} else if (strcmp(arg[ioffset], "*EOS_NONE") == 0) {

			/*
			 * no eos
			 */

			eos[itype] = EOS_NONE;
			if (comm->me == 0) {
				printf("reading *EOS_NONE\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_NONE");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *EOS_NONE but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "no EOS selected");
			}

		} else if (strcmp(arg[ioffset], "*EOS_LINEAR") == 0) {

			/*
			 * linear eos
			 */

			eos[itype] = EOS_LINEAR;
			if (comm->me == 0) {
				printf("reading *EOS_LINEAR\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_LINEAR");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *EOS_LINEAR but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "linear EOS based on strain rate");
				printf("%60s : %g\n", "bulk modulus", SafeLookup("bulk_modulus", itype));
			}
		} // end linear eos
		else if (strcmp(arg[ioffset], "*EOS_SHOCK") == 0) {

			/*
			 * shock eos
			 */

			eos[itype] = EOS_SHOCK;
			if (comm->me == 0) {
				printf("reading *EOS_SHOCK\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_SHOCK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 3 + 1) {
				sprintf(str, "expected 3 arguments following *EOS_SHOCK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("eos_shock_c0", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
			matProp2[std::make_pair("eos_shock_s", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
			matProp2[std::make_pair("eos_shock_gamma", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
			if (comm->me == 0) {
				printf("\n%60s\n", "shock EOS based on strain rate");
				printf("%60s : %g\n", "reference speed of sound", SafeLookup("eos_shock_c0", itype));
				printf("%60s : %g\n", "Hugoniot parameter S", SafeLookup("eos_shock_s", itype));
				printf("%60s : %g\n", "Grueneisen Gamma", SafeLookup("eos_shock_gamma", itype));
			}
		} // end shock eos

		else if (strcmp(arg[ioffset], "*EOS_POLYNOMIAL") == 0) {
			/*
			 * polynomial eos
			 */

			eos[itype] = EOS_POLYNOMIAL;
			if (comm->me == 0) {
				printf("reading *EOS_POLYNOMIAL\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_POLYNOMIAL");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 7 + 1) {
				sprintf(str, "expected 7 arguments following *EOS_POLYNOMIAL but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("eos_polynomial_c0", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
			matProp2[std::make_pair("eos_polynomial_c1", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
			matProp2[std::make_pair("eos_polynomial_c2", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
			matProp2[std::make_pair("eos_polynomial_c3", itype)] = force->numeric(FLERR, arg[ioffset + 4]);
			matProp2[std::make_pair("eos_polynomial_c4", itype)] = force->numeric(FLERR, arg[ioffset + 5]);
			matProp2[std::make_pair("eos_polynomial_c5", itype)] = force->numeric(FLERR, arg[ioffset + 6]);
			matProp2[std::make_pair("eos_polynomial_c6", itype)] = force->numeric(FLERR, arg[ioffset + 7]);
			if (comm->me == 0) {
				printf("\n%60s\n", "polynomial EOS based on strain rate");
				printf("%60s : %g\n", "parameter c0", SafeLookup("eos_polynomial_c0", itype));
				printf("%60s : %g\n", "parameter c1", SafeLookup("eos_polynomial_c1", itype));
				printf("%60s : %g\n", "parameter c2", SafeLookup("eos_polynomial_c2", itype));
				printf("%60s : %g\n", "parameter c3", SafeLookup("eos_polynomial_c3", itype));
				printf("%60s : %g\n", "parameter c4", SafeLookup("eos_polynomial_c4", itype));
				printf("%60s : %g\n", "parameter c5", SafeLookup("eos_polynomial_c5", itype));
				printf("%60s : %g\n", "parameter c6", SafeLookup("eos_polynomial_c6", itype));
			}
		} // end polynomial eos

		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PLASTIC_STRAIN") == 0) {

			/*
			 * maximum plastic strain failure criterion
			 */

			if (comm->me == 0) {
				printf("reading *FAILURE_MAX_PLASTIC_SRTRAIN\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PLASTIC_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PLASTIC_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_max_plastic_strain", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("\n%60s\n", "maximum plastic strain failure criterion");
				printf("%60s : %g\n", "failure occurs when plastic strain reaches limit",
						SafeLookup("failure_max_plastic_strain", itype));
			}
		} // end maximum plastic strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRAIN") == 0) {

			/*
			 * maximum principal strain failure criterion
			 */
			if (comm->me == 0) {
				printf("reading *FAILURE_MAX_PRINCIPAL_STRAIN\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_max_principal_strain", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("\n%60s\n", "maximum principal strain failure criterion");
				printf("%60s : %g\n", "failure occurs when principal strain reaches limit",
						SafeLookup("failure_max_principal_strain", itype));
			}
		} // end maximum principal strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_JOHNSON_COOK") == 0) {

			if (comm->me == 0) {
				printf("reading *FAILURE_JOHNSON_COOK\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_JOHNSON_COOK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 5 + 1) {
				sprintf(str, "expected 5 arguments following *FAILURE_JOHNSON_COOK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_JC_d1", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
			matProp2[std::make_pair("failure_JC_d2", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
			matProp2[std::make_pair("failure_JC_d3", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
			matProp2[std::make_pair("failure_JC_d4", itype)] = force->numeric(FLERR, arg[ioffset + 4]);
			matProp2[std::make_pair("failure_JC_epdot0", itype)] = force->numeric(FLERR, arg[ioffset + 5]);

			if (comm->me == 0) {
				printf("\n%60s\n", "Johnson-Cook failure criterion");
				printf("%60s : %g\n", "parameter d1", SafeLookup("failure_JC_d1", itype));
				printf("%60s : %g\n", "parameter d2", SafeLookup("failure_JC_d2", itype));
				printf("%60s : %g\n", "parameter d3", SafeLookup("failure_JC_d3", itype));
				printf("%60s : %g\n", "parameter d4", SafeLookup("failure_JC_d4", itype));
				printf("%60s : %g\n", "reference plastic strain rate", SafeLookup("failure_JC_epdot0", itype));
			}

		} else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRESS") == 0) {

			/*
			 * maximum principal stress failure criterion
			 */

			if (comm->me == 0) {
				printf("reading *FAILURE_MAX_PRINCIPAL_STRESS\n");
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

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRESS");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRESS but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_max_principal_stress", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("\n%60s\n", "maximum principal stress failure criterion");
				printf("%60s : %g\n", "failure occurs when principal stress reaches limit",
						SafeLookup("failure_max_principal_stress", itype));
			}
		} // end maximum principal stress failure criterion
		else {
			sprintf(str, "unknown *KEYWORD: %s", arg[ioffset]);
			error->all(FLERR, str);
		}

	}

	/*
	 * copy data which is looked up in inner pairwise loops from slow maps to fast arrays
	 */

	hg_coeff[itype] = SafeLookup("hourglass_amp", itype);
	Q1[itype] = SafeLookup("viscosity_q1", itype);
	Q2[itype] = SafeLookup("viscosity_q2", itype);
	youngsmodulus[itype] = SafeLookup("youngs_modulus", itype);
	signal_vel0[itype] = SafeLookup("signal_vel", itype);
	rho0[itype] = SafeLookup("rho_ref", itype);

	setflag[itype][itype] = 1;

}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairTlsph::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	if (force->newton == 1)
		error->all(FLERR, "Pair style tlsph requires newton off");

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
//printf("cutoff for pair pair tlsph = %f\n", cutoff);
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairTlsph::init_style() {
	int i;

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

void PairTlsph::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairTlsph::memory_usage() {

	return 118 * nmax * sizeof(double);
}

/* ----------------------------------------------------------------------
 extract method to provide access to this class' data structures
 ------------------------------------------------------------------------- */

void *PairTlsph::extract(const char *str, int &i) {
//printf("in PairTlsph::extract\n");
	if (strcmp(str, "smd/tlsph/Fincr_ptr") == 0) {
		return (void *) Fincr;
	} else if (strcmp(str, "smd/tlsph/detF_ptr") == 0) {
		return (void *) detF;
	} else if (strcmp(str, "smd/tlsph/PK1_ptr") == 0) {
		return (void *) PK1;
	} else if (strcmp(str, "smd/tlsph/smoothVel_ptr") == 0) {
		return (void *) smoothVelDifference;
	} else if (strcmp(str, "smd/tlsph/numNeighsRefConfig_ptr") == 0) {
		return (void *) numNeighsRefConfig;
	} else if (strcmp(str, "smd/tlsph/stressTensor_ptr") == 0) {
		return (void *) CauchyStress;
	} else if (strcmp(str, "smd/tlsph/updateFlag_ptr") == 0) {
		return (void *) &updateFlag;
	} else if (strcmp(str, "smd/tlsph/strain_rate_ptr") == 0) {
		return (void *) D;
	} else if (strcmp(str, "smd/tlsph/hMin_ptr") == 0) {
		return (void *) &hMin;
	} else if (strcmp(str, "smd/tlsph/dtCFL_ptr") == 0) {
		return (void *) &dtCFL;
	} else if (strcmp(str, "smd/tlsph/dtRelative_ptr") == 0) {
		return (void *) &dtRelative;
	} else if (strcmp(str, "smd/tlsph/hourglass_error_ptr") == 0) {
		return (void *) hourglass_error;
	}

	return NULL;
}

/* ---------------------------------------------------------------------- */

int PairTlsph::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	int *mol = atom->molecule;

//printf("in PairTlsph::pack_forward_comm\n");

	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = PK1[j](0, 0); // PK1 is not symmetric
		buf[m++] = PK1[j](0, 1);
		buf[m++] = PK1[j](0, 2);
		buf[m++] = PK1[j](1, 0);
		buf[m++] = PK1[j](1, 1);
		buf[m++] = PK1[j](1, 2);
		buf[m++] = PK1[j](2, 0);
		buf[m++] = PK1[j](2, 1);
		buf[m++] = PK1[j](2, 2); // 9

		buf[m++] = Fincr[j](0, 0); // Fincr is not symmetric
		buf[m++] = Fincr[j](0, 1);
		buf[m++] = Fincr[j](0, 2);
		buf[m++] = Fincr[j](1, 0);
		buf[m++] = Fincr[j](1, 1);
		buf[m++] = Fincr[j](1, 2);
		buf[m++] = Fincr[j](2, 0);
		buf[m++] = Fincr[j](2, 1);
		buf[m++] = Fincr[j](2, 2); // 9 + 9 = 18

		buf[m++] = shepardWeight[j]; // 19
		buf[m++] = mol[j];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairTlsph::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	int *mol = atom->molecule;

//printf("in PairTlsph::unpack_forward_comm\n");

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {

		PK1[i](0, 0) = buf[m++]; // PK1 is not symmetric
		PK1[i](0, 1) = buf[m++];
		PK1[i](0, 2) = buf[m++];
		PK1[i](1, 0) = buf[m++];
		PK1[i](1, 1) = buf[m++];
		PK1[i](1, 2) = buf[m++];
		PK1[i](2, 0) = buf[m++];
		PK1[i](2, 1) = buf[m++];
		PK1[i](2, 2) = buf[m++];

		Fincr[i](0, 0) = buf[m++];
		Fincr[i](0, 1) = buf[m++];
		Fincr[i](0, 2) = buf[m++];
		Fincr[i](1, 0) = buf[m++];
		Fincr[i](1, 1) = buf[m++];
		Fincr[i](1, 2) = buf[m++];
		Fincr[i](2, 0) = buf[m++];
		Fincr[i](2, 1) = buf[m++];
		Fincr[i](2, 2) = buf[m++];

		shepardWeight[i] = buf[m++];
		mol[i] = static_cast<int>(buf[m++]);
	}
}

void PairTlsph::spiky_kernel_and_derivative(const double h, const double r, double &wf, double &wfd) {

	/*
	 * Spiky kernel
	 */

	//error->one(FLERR, "should not be here");
	if (r > h) {
		wfd = 0.0;
		wf = 0.0;
		return;
	}

	if (domain->dimension == 2) {
		double hr = h - r; // [m]
		double n = 0.3141592654e0 * h * h * h * h * h; // [m^5]
		wfd = -3.0e0 * hr * hr / n; // [m*m/m^5] = [1/m^3] ==> correct for dW/dr in 2D
		wf = -0.333333333333e0 * hr * wfd; // [m/m^3] ==> [1/m^2] correct for W in 2D
	} else {
		double hr = h - r; // [m]
		wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h); // [1/m^4] ==> correct for dW/dr in 3D
		wf = -0.333333333333e0 * hr * wfd; // [m/m^4] ==> [1/m^3] correct for W in 3D
	}

}

void PairTlsph::barbara_kernel_and_derivative(const double h, const double r, double &wf, double &wfd) {

	/*
	 * Barbara kernel
	 */

	double arg = (1.570796327 * (r + h)) / h;
	double hsq = h * h;

	if (r > h) {
		char msg[128];
		sprintf(msg, "r = %f > h = %f in kernel function", r, h);
		error->one(FLERR, msg);
	}

	if (domain->dimension == 2) {
		wf = (1.680351548 * (cos(arg) + 1.)) / hsq;
		wfd = -2.639490040 * sin(arg) / (hsq * h);
	} else {
		wf = 2.051578323 * (cos(arg) + 1.) / (hsq * h);
		wfd = -3.222611694 * sin(arg) / (hsq * hsq);
//error->one(FLERR, "not implemented yet");
	}

}

/*
 * Polar Decomposition via SVD, M = R * T
 * where R is a rotation and T a pure translation/stretch
 */

void PairTlsph::PolDec(Matrix3d &M, Matrix3d *R, Matrix3d *T) {

	JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV); // SVD(A) = U S V*
	Matrix3d S = svd.singularValues().asDiagonal();
	Matrix3d U = svd.matrixU();
	Matrix3d V = svd.matrixV();
	Matrix3d eye;
	eye.setIdentity();

// now do polar decomposition into M = R * T, where R is rotation
// and T is translation matrix
	*R = U * V.transpose();
	*T = V * S * V.transpose();

// now we check the polar decomposition
#ifdef TLSPH_DEBUG
	Matrix3d Mcheck = *R * R->transpose();
	Matrix3d Mdiff = Mcheck - eye;
	if (Mdiff.norm() > 1.0e-8) {
		printf("R is not orthogonal\n");
		cout << "Here is the Rotation matrix:" << endl << *R << endl;
	}

	if (fabs(R->determinant() - 1.0) > 1.0e-8) {
		printf("determinant of R=%f is not unity!\n", R->determinant());
	}

//    cout << "Here is the difference between M and its reconstruction from the polar decomp:" << endl << Mdiff << endl;
//    cout << "Here is the Rotation matrix:" << endl << *R << endl;
//    cout << "Here is the Translation Matrix:" << endl << *T << endl;
//    cout << "is U unitary? Her is U^T * U:" << endl << R->transpose()* *R << endl;
#endif
}

/*
 * deviator of a tensor
 */
Matrix3d PairTlsph::Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}

/*
 * Pseudo-inverse via SVD
 */

Matrix3d PairTlsph::pseudo_inverse_SVD(Matrix3d M) {

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
			singularValuesInv(row) = 1.0;
		}
	}

	Matrix3d pInv;
	pInv = V * singularValuesInv.asDiagonal() * U.transpose();

	return pInv;
}

/*
 * test if two matrices are equal
 */
double PairTlsph::TestMatricesEqual(Matrix3d A, Matrix3d B, double eps) {
	Matrix3d diff;
	diff = A - B;
	double norm = diff.norm();
	if (norm > eps) {
		printf("Matrices A and B are not equal! The L2-norm difference is: %g\n", norm);
		cout << "Here is matrix A:" << endl << A << endl;
		cout << "Here is matrix B:" << endl << B << endl;
	}
	return norm;
}


/* ----------------------------------------------------------------------
 compute effectiveP-wave speed
 determined by longitudinal modulus
 ------------------------------------------------------------------------- */

double PairTlsph::effective_longitudinal_modulus(int itype, double dt, double d_iso, double p_rate, Matrix3d d_dev,
		Matrix3d sigma_dev_rate, double damage) {
	double K3; // 3 times the effective bulk modulus, see Pronto 2d eqn 3.4.6
	double mu2; // 2 times the effective shear modulus, see Pronto 2d eq. 3.4.7
	double shear_rate_sq;
	double M; //effective longitudinal modulus
	double M0; // initial longitudinal modulus

	M0 = SafeLookup("m_modulus", itype);

	if (dt * d_iso > 1.0e-6) {
		K3 = 3.0 * p_rate / d_iso;
	} else {
		K3 = 3.0 * M0;
	}

	if (domain->dimension == 3) {
		mu2 = sigma_dev_rate(0, 1) / (d_dev(0, 1)) + sigma_dev_rate(0, 2) / (d_dev(0, 2)) + sigma_dev_rate(1, 2) / (d_dev(1, 2));
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2);
	} else {
		mu2 = sigma_dev_rate(0, 1) / (d_dev(0, 1));
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1);
	}

	if (dt * dt * shear_rate_sq < 1.0e-8) {
		mu2 = 0.5 * (3.0 * M0 - K3); // Formel stimmt
	}

	M = (K3 + 2.0 * mu2) / 3.0; // effective dilational modulus, see Pronto 2d eqn 3.4.8

	if (M < M0) { // do not allow effective dilatational modulus to decrease beyond its initial value
		M = M0;
	}

	/*
	 * damaged particles potentially have a very high dilatational modulus, even though damage degradation scales down the
	 * effective stress. we simply use the initial modulus for damaged particles.
	 */

	if (damage > 0.99) {
		M = M0; //
	}

	return M;

}

/* ----------------------------------------------------------------------
 smooth stress field
 ------------------------------------------------------------------------- */

void PairTlsph::SmoothField() {
	int *mol = atom->molecule;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double **x0 = atom->x0;
	double **tlsph_stress = atom->tlsph_stress;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, j;
	double r0, r0Sq, wf, wfd, h, irad, voli, volj;
	Vector3d dx0;
	Matrix3d stress;
	Matrix3d *average_stress;
	double *norm;
	Vector3d xi, xj, vi, vj, vinti, vintj, x0i, x0j, dvint, du;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

// allocate storage
	average_stress = new Matrix3d[nlocal];
	norm = new double[nlocal];

// zero accumulators
	stress.setZero();
	for (i = 0; i < nlocal; i++) {
		stress(0, 0) = tlsph_stress[i][0];
		stress(0, 1) = tlsph_stress[i][1];
		stress(0, 2) = tlsph_stress[i][2];
		stress(1, 1) = tlsph_stress[i][3];
		stress(1, 2) = tlsph_stress[i][4];
		stress(2, 2) = tlsph_stress[i][5];

		h = 2.0 * radius[i];
		r0 = 0.0;

		barbara_kernel_and_derivative(h, r0, wf, wfd);
		average_stress[i] = wf * vfrac[i] * stress;
		norm[i] = wf * vfrac[i];
	}

// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];

		if (mol[i] < 0) { // valid SPH particle have mol > 0
			continue;
		}

		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];
		voli = vfrac[i];
		x0i << x0[i][0], x0[i][1], x0[i][2];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (mol[j] < 0) { // particle has failed. do not include it for computing any property
				continue;
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			x0j << x0[j][0], x0[j][1], x0[j][2];
			dx0 = x0j - x0i;

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			r0Sq = dx0.squaredNorm();
			h = irad + radius[j];
			if (r0Sq < h * h) {

				r0 = sqrt(r0Sq);
				volj = vfrac[j];

				stress(0, 0) = tlsph_stress[j][0];
				stress(0, 1) = tlsph_stress[j][1];
				stress(0, 2) = tlsph_stress[j][2];
				stress(1, 1) = tlsph_stress[j][3];
				stress(1, 2) = tlsph_stress[j][4];
				stress(2, 2) = tlsph_stress[j][5];

				// kernel function
				barbara_kernel_and_derivative(h, r0, wf, wfd);

				average_stress[i] += wf * volj * stress;
				norm[i] += volj * wf;

				if (j < nlocal) {

					stress(0, 0) = tlsph_stress[i][0];
					stress(0, 1) = tlsph_stress[i][1];
					stress(0, 2) = tlsph_stress[i][2];
					stress(1, 1) = tlsph_stress[i][3];
					stress(1, 2) = tlsph_stress[i][4];
					stress(2, 2) = tlsph_stress[i][5];

					average_stress[j] += wf * voli * stress;
					norm[j] += voli * wf;
				}

			} // end if check distance
		} // end loop over j
	} // end loop over i

	for (i = 0; i < nlocal; i++) {
		stress = average_stress[i] / norm[i];

		tlsph_stress[i][0] = stress(0, 0);
		tlsph_stress[i][1] = stress(0, 1);
		tlsph_stress[i][2] = stress(0, 2);
		tlsph_stress[i][3] = stress(1, 1);
		tlsph_stress[i][4] = stress(1, 2);
		tlsph_stress[i][5] = stress(2, 2);

	}

	delete[] average_stress;
	delete[] norm;

}

/* ----------------------------------------------------------------------
 XSPH-like smoothing of stress field
 ------------------------------------------------------------------------- */

void PairTlsph::SmoothFieldXSPH() {
	int *mol = atom->molecule;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double **x0 = atom->x0;
	double **tlsph_stress = atom->tlsph_stress;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, j;
	double r0, r0Sq, wf, wfd, h, irad, voli, volj;
	Vector3d dx0;
	Matrix3d stress_i, stress_j;
	Matrix3d *stress_difference;
	double *norm;
	Vector3d xi, xj, vi, vj, vinti, vintj, x0i, x0j, dvint, du;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

// allocate storage
	stress_difference = new Matrix3d[nlocal];
	norm = new double[nlocal];

// zero accumulators
	stress_i.setZero();
	for (i = 0; i < nlocal; i++) {
		stress_difference[i].setZero();
		norm[i] = 0.0;
	}

// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];

		if (mol[i] < 0) { // valid SPH particle have mol > 0
			continue;
		}

		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];
		voli = vfrac[i];
		x0i << x0[i][0], x0[i][1], x0[i][2];

		stress_i(0, 0) = tlsph_stress[i][0];
		stress_i(0, 1) = tlsph_stress[i][1];
		stress_i(0, 2) = tlsph_stress[i][2];
		stress_i(1, 1) = tlsph_stress[i][3];
		stress_i(1, 2) = tlsph_stress[i][4];
		stress_i(2, 2) = tlsph_stress[i][5];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (mol[j] < 0) { // particle has failed. do not include it for computing any property
				continue;
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			x0j << x0[j][0], x0[j][1], x0[j][2];
			dx0 = x0j - x0i;

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			r0Sq = dx0.squaredNorm();
			h = irad + radius[j];
			if (r0Sq < h * h) {

				r0 = sqrt(r0Sq);
				volj = vfrac[j];

				stress_j(0, 0) = tlsph_stress[j][0];
				stress_j(0, 1) = tlsph_stress[j][1];
				stress_j(0, 2) = tlsph_stress[j][2];
				stress_j(1, 1) = tlsph_stress[j][3];
				stress_j(1, 2) = tlsph_stress[j][4];
				stress_j(2, 2) = tlsph_stress[j][5];

				// kernel function
				barbara_kernel_and_derivative(h, r0, wf, wfd);

				stress_difference[i] += wf * volj * (stress_j - stress_i);
				norm[i] += volj * wf;

				if (j < nlocal) {
					stress_difference[j] += wf * voli * (stress_i - stress_j);
					norm[j] += voli * wf;
				}

			} // end if check distance
		} // end loop over j
	} // end loop over i

	for (i = 0; i < nlocal; i++) {
		if (norm[i] > 0.0) {
			tlsph_stress[i][0] += stress_difference[i](0, 0) / norm[i];
			tlsph_stress[i][1] += stress_difference[i](0, 1) / norm[i];
			tlsph_stress[i][2] += stress_difference[i](0, 2) / norm[i];
			tlsph_stress[i][3] += stress_difference[i](1, 1) / norm[i];
			tlsph_stress[i][4] += stress_difference[i](1, 2) / norm[i];
			tlsph_stress[i][5] += stress_difference[i](2, 2) / norm[i];
		}
	}

	delete[] stress_difference;
	delete[] norm;

}

double PairTlsph::SafeLookup(std::string str, int itype) {
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

bool PairTlsph::CheckKeywordPresent(std::string str, int itype) {
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

/* ----------------------------------------------------------------------
 Limit eigenvalues of a matrix to upper and lower bounds.
 ------------------------------------------------------------------------- */

Matrix3d PairTlsph::LimitEigenvalues(Matrix3d S, double limitEigenvalue) {

	/*
	 * compute Eigenvalues of matrix S
	 */
	SelfAdjointEigenSolver<Matrix3d> es;
	es.compute(S);

	double max_eigenvalue = es.eigenvalues().maxCoeff();
	double min_eigenvalue = es.eigenvalues().minCoeff();
	double amax_eigenvalue = fabs(max_eigenvalue);
	double amin_eigenvalue = fabs(min_eigenvalue);

	if ((amax_eigenvalue > limitEigenvalue) || (amin_eigenvalue > limitEigenvalue)) {
		if (amax_eigenvalue > amin_eigenvalue) { // need to scale with max_eigenvalue
			double scale = amax_eigenvalue / limitEigenvalue;
			Matrix3d V = es.eigenvectors();
			Matrix3d S_diag = V.inverse() * S * V; // diagonalized input matrix
			S_diag /= scale;
			Matrix3d S_scaled = V * S_diag * V.inverse(); // undiagonalize matrix
			return S_scaled;
		} else { // need to scale using min_eigenvalue
			double scale = amin_eigenvalue / limitEigenvalue;
			Matrix3d V = es.eigenvectors();
			Matrix3d S_diag = V.inverse() * S * V; // diagonalized input matrix
			S_diag /= scale;
			Matrix3d S_scaled = V * S_diag * V.inverse(); // undiagonalize matrix
			return S_scaled;
		}
	} else { // limiting does not apply
		return S;
	}
}

/* ----------------------------------------------------------------------
 compute pressure. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputePressure(const int i, const double pInitial, const double d_iso, double &pFinal, double &p_rate) {
	double *rmass = atom->rmass;
	double *vfrac = atom->vfrac;
	double *e = atom->e;
	int *type = atom->type;
	double dt = update->dt;

	int itype;

	itype = type[i];

	double mass_specific_energy = e[i] / rmass[i]; // energy per unit mass
	double rho = rmass[i] / (detF[i] * vfrac[i]); // current mass density
	double vol_specific_energy = mass_specific_energy * rho; // energy per current volume

	switch (eos[itype]) {
	case EOS_LINEAR:
		LinearEOS(SafeLookup("bulk_modulus", itype), pInitial, d_iso, dt, pFinal, p_rate);
		break;
	case EOS_NONE:
		pFinal = 0.0;
		p_rate = 0.0;
		break;
	case EOS_SHOCK:
		//  rho,  rho0,  e,  e0,  c0,  S,  Gamma,  pInitial,  dt,  &pFinal,  &p_rate);
		ShockEOS(rho, rho0[itype], mass_specific_energy, 0.0, SafeLookup("eos_shock_c0", itype), SafeLookup("eos_shock_s", itype),
				SafeLookup("eos_shock_gamma", itype), pInitial, dt, pFinal, p_rate);
		break;
	case EOS_POLYNOMIAL:
		polynomialEOS(rho, rho0[itype], vol_specific_energy, SafeLookup("eos_polynomial_c0", itype),
				SafeLookup("eos_polynomial_c1", itype), SafeLookup("eos_polynomial_c2", itype),
				SafeLookup("eos_polynomial_c3", itype), SafeLookup("eos_polynomial_c4", itype),
				SafeLookup("eos_polynomial_c5", itype), SafeLookup("eos_polynomial_c6", itype), pInitial, dt, pFinal, p_rate);

		break;
	default:
		error->one(FLERR, "unknown EOS.");
		break;
	}
}

/* ----------------------------------------------------------------------
 Compute stress deviator. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeStressDeviator(const int i, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev,
		Matrix3d &sigma_dev_rate, double &plastic_strain_increment) {
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *type = atom->type;
	double *rmass = atom->rmass;
	//double *vfrac = atom->vfrac;
	double *e = atom->e;
	double dt = update->dt;
	int itype;

	double mass_specific_energy = e[i] / rmass[i]; // energy per unit mass
	plastic_strain_increment = 0.0;
	itype = type[i];

	switch (strengthModel[itype]) {
	case LINEAR_STRENGTH:
		//printf("mu0 is %f\n", mu0[itype]);
		//LinearStrength(SafeLookup("lame_mu", itype), sigmaInitial_dev, d[i], dt, &sigmaFinal_dev, &sigma_dev_rate);

		sigma_dev_rate = 2.0 * SafeLookup("lame_mu", itype) * d_dev;
		sigmaFinal_dev = sigmaInitial_dev + dt * sigma_dev_rate;

		break;
	case LINEAR_DEFGRAD:
		//LinearStrengthDefgrad(SafeLookup("lame_lambda", itype), SafeLookup("lame_mu", itype), Fincr[i], &sigmaFinal_dev);
		//eff_plastic_strain[i] = 0.0;
		//p_rate = pInitial - sigmaFinal_dev.trace() / 3.0;
		//sigma_dev_rate = sigmaInitial_dev - Deviator(sigmaFinal_dev);
		error->one(FLERR, "LINEAR_DEFGRAD is only for debugging purposes and currently deactivated.");
		R[i].setIdentity();
		break;
	case LINEAR_PLASTICITY:
		LinearPlasticStrength(SafeLookup("lame_mu", itype), SafeLookup("yield_stress0", itype), sigmaInitial_dev, d_dev, dt,
				sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);
		break;
	case STRENGTH_JOHNSON_COOK:
		JohnsonCookStrength(SafeLookup("lame_mu", itype), SafeLookup("heat_capacity", itype), mass_specific_energy,
				SafeLookup("JC_A", itype), SafeLookup("JC_B", itype), SafeLookup("JC_a", itype), SafeLookup("JC_C", itype),
				SafeLookup("JC_epdot0", itype), SafeLookup("JC_T0", itype), SafeLookup("JC_Tmelt", itype),
				SafeLookup("JC_M", itype), dt, eff_plastic_strain[i], eff_plastic_strain_rate[i], sigmaInitial_dev, d_dev,
				sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);
		break;
	case STRENGTH_NONE:
		sigmaFinal_dev.setZero();
		sigma_dev_rate.setZero();
		break;
	default:
		error->one(FLERR, "unknown strength model.");
		break;
	}

}

/* ----------------------------------------------------------------------
 Compute damage. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeDamage(const int i, const Matrix3d strain, const double pFinal, const Matrix3d sigmaFinal, const Matrix3d sigmaFinal_dev, Matrix3d &sigma_damaged, double &damage_increment) {
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *type = atom->type;
	double *radius = atom->radius;
	double dt = update->dt;
	int itype = type[i];
	bool damage_flag = false;

	if (CheckKeywordPresent("failure_max_principal_stress", itype)) {
		/*
		 * maximum stress failure criterion:
		 */
		damage_flag = IsotropicMaxStressDamage(strain, SafeLookup("failure_max_principal_stress", itype));
	}
	if (CheckKeywordPresent("failure_max_principal_strain", itype)) {
		/*
		 * maximum strain failure criterion:
		 */
		damage_flag = IsotropicMaxStrainDamage(strain, SafeLookup("failure_max_principal_strain", itype));
	}

	if (CheckKeywordPresent("failure_max_plastic_strain", itype)) {

		if (eff_plastic_strain[i] >= 0.99 * SafeLookup("failure_max_plastic_strain", itype)) {
			damage_flag = true;
		}

	}

	if (CheckKeywordPresent("failure_JC_d1", itype)) {

		double jc_failure_strain = JohnsonCookFailureStrain(pFinal, sigmaFinal_dev, SafeLookup("failure_JC_d1", itype),
				SafeLookup("failure_JC_d2", itype), SafeLookup("failure_JC_d3", itype), SafeLookup("failure_JC_d4", itype),
				SafeLookup("failure_JC_epdot0", itype), eff_plastic_strain_rate[i]);

		//cout << "plastic strain increment is " << plastic_strain_increment << "  jc fs is " << jc_failure_strain << endl;
		if (eff_plastic_strain[i] / jc_failure_strain > 1.0) {
			damage_flag = true;
		}
	}

	/*
	 * Damage does not happen instantaneously but rather takes time which signal_velocity takes to go over a distance h.
	 * Therefore compute a damage increment which is time-integrated in AssembleStress().
	 */

	if (damage_flag == true) {
		damage_increment = dt * signal_vel0[itype] / (10.0 * radius[i]);
	} else {
		damage_increment = 0.0;
	}
}
