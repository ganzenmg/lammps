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
#include "pair_sph2_tlsph.h"
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

using namespace std;
using namespace LAMMPS_NS;

#define JAUMANN 0
#define DETF_MIN 0.1 // maximum compression deformation allowed
#define DETF_MAX 40.0 // maximum tension deformation allowdw
#include <Eigen/SVD>
#include <Eigen/Eigen>
using namespace Eigen;

#define TLSPH_DEBUG 0

/* ---------------------------------------------------------------------- */

PairTlsph::PairTlsph(LAMMPS *lmp) :
		Pair(lmp) {
	maxstrain = maxstress = NULL;
	c0_type = NULL;
	youngsmodulus = NULL;
	poissonr = NULL;
	hg_coeff = NULL;
	Q1 = Q2 = NULL;
	yieldStress = NULL;
	strengthModel = eos = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	F = Fdot = Fincr = K = PK1 = NULL;
	d = R = FincrInv = W = D = NULL;
	detF = NULL;
	smoothVel = NULL;
	shearFailureFlag = NULL;
	numNeighsRefConfig = NULL;
	shepardWeight = NULL;
	CauchyStress = NULL;

	updateFlag = 0;

	comm_forward = 20; // this pair style communicates 20 doubles to ghost atoms : PK1 tensor + F tensor + shepardWeight
}

/* ---------------------------------------------------------------------- */

PairTlsph::~PairTlsph() {
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(c0_type);
		memory->destroy(youngsmodulus);
		memory->destroy(poissonr);
		memory->destroy(hg_coeff);
		memory->destroy(Q1);
		memory->destroy(Q2);
		memory->destroy(yieldStress);
		memory->destroy(strengthModel);
		memory->destroy(eos);
		memory->destroy(maxstrain);
		memory->destroy(maxstress);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;

		delete[] F;
		delete[] Fdot;
		delete[] Fincr;
		delete[] K;
		delete[] detF;
		delete[] PK1;
		delete[] smoothVel;
		delete[] d;
		delete[] R;
		delete[] FincrInv;
		delete[] W;
		delete[] D;
		delete[] shearFailureFlag;
		delete[] numNeighsRefConfig;
		delete[] shepardWeight;
		delete[] CauchyStress;
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
	double *damage = atom->damage;
	double **x0 = atom->x0;
	double **x = atom->x;
	double **v = atom->vest;
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	double **defgrad0 = atom->tlsph_fold;
	int *tag = atom->tag;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, j, iDim, itype;
	double r0, r0Sq, wf, wfd, h, irad;
	double pairweight;
	Vector3d dx0, dx, dv, g;
	Matrix3d Ktmp, Fdottmp, Ftmp, L, Fold, U, eye;
	Vector3d xi, xj, vi, vj, vinti, vintj, x0i, x0j, dvint;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);
	double damage_factor;

	eye.setIdentity();

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		K[i].setZero();
		F[i].setZero();
		Fincr[i].setZero();
		Fdot[i].setZero();
		shearFailureFlag[i] = false;
		shepardWeight[i] = 0.0;
		numNeighsRefConfig[i] = 0;
		smoothVel[i].setZero();
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

		// initialize Eigen data structures from LAMMPS data structures
		for (iDim = 0; iDim < 3; iDim++) {
			x0i(iDim) = x0[i][iDim];
			xi(iDim) = x[i][iDim];
			vi(iDim) = v[i][iDim];
			vinti(iDim) = vint[i][iDim];
		}

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
				for (iDim = 0; iDim < 3; iDim++) {
					xj(iDim) = x[j][iDim];
					vj(iDim) = v[j][iDim];
					vintj(iDim) = vint[j][iDim];
				}

				r0 = sqrt(r0Sq);

				// distance vectors in current and reference configuration, velocity difference
				dx = xj - xi;
				dv = vj - vi;
				dvint = vintj - vinti;

				// kernel function
				kernel_and_derivative(h, r0, wf, wfd);

				// apply damage model if in tension
				if (dx.dot(dv) > 0.0) {
					damage_factor = sqrt(damage[i] * damage[j]);
					damage_factor = 1.0 - pow(damage_factor, 3.0);
				} else {
					damage_factor = 1.0;
				}
				//wf *= damage_factor;
				//wfd *= damage_factor;

				// uncorrected kernel gradient
				g = (wfd / r0) * dx0;

				/* build matrices */
				Ktmp = g * dx0.transpose();
				Ftmp = dx * g.transpose();
				Fdottmp = dv * g.transpose();

				K[i] += vfrac[j] * Ktmp;
				Fincr[i] += vfrac[j] * Ftmp;
				Fdot[i] += vfrac[j] * Fdottmp;
				shepardWeight[i] += vfrac[j] * wf;
				smoothVel[i] += vfrac[j] * wf * dvint;
				numNeighsRefConfig[i]++;

				if (j < nlocal) {
					K[j] += vfrac[i] * Ktmp;
					Fincr[j] += vfrac[i] * Ftmp;
					Fdot[j] += vfrac[i] * Fdottmp;
					shepardWeight[j] += vfrac[i] * wf;
					smoothVel[j] -= vfrac[i] * wf * dvint;
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

			if ((numNeighsRefConfig[i] < domain->dimension) && (mol[i] > 0)) { // cannot possibly invert shape matrix
				printf("deleting particle [%d] because number of neighbors=%d is too small\n", tag[i], numNeighsRefConfig[i]);
				mol[i] = -1;
			}

			if (mol[i] > 0) {
				K[i] = PairTlsph::pseudo_inverse_SVD(K[i]);

				if (domain->dimension == 2) {
					K[i](2, 2) = 1.0; // make inverse of K well defined even when it is rank-deficient (3d matrix, only 2d information)
				}

				Fincr[i] *= K[i];
				Fdot[i] = Fdot[i] * K[i];

				/*
				 * we need to be able to recover from a potentially planar (2d) configuration of particles
				 */
				if ((domain->dimension == 2) || (Fincr[i](2, 2) < 1.0e-2)) {
					Fincr[i](2, 2) = 1.0;
				}

				// build total deformation gradient with Eigen data structures
				Fold(0, 0) = defgrad0[i][0];
				Fold(0, 1) = defgrad0[i][1];
				Fold(0, 2) = defgrad0[i][2];
				Fold(1, 0) = defgrad0[i][3];
				Fold(1, 1) = defgrad0[i][4];
				Fold(1, 2) = defgrad0[i][5];
				Fold(2, 0) = defgrad0[i][6];
				Fold(2, 1) = defgrad0[i][7];
				Fold(2, 2) = defgrad0[i][8];

				F[i] = Fold * Fincr[i]; // this is the total deformation gradient: reference deformation times incremental deformation

				/*
				 * make sure F stays within some limits
				 */

				if (F[i].determinant() < DETF_MIN) {
					printf("deleting particle [%d] because det(F)=%f is smaller than limit=%f\n", tag[i], F[i].determinant(),
					DETF_MIN);
					cout << "Here is matrix F:" << endl << F[i] << endl;
					mol[i] = -1;
				} else if (F[i].determinant() > DETF_MAX) {
					printf("deleting particle [%d] because det(F)=%f is larger than limit=%f\n", tag[i], F[i].determinant(),
					DETF_MAX);
					cout << "Here is matrix F:" << endl << F[i] << endl;
					mol[i] = -1;
				}

				if (mol[i] > 0) {
					detF[i] = F[i].determinant();
					FincrInv[i] = PairTlsph::pseudo_inverse_SVD(Fincr[i]);

					// velocity gradient, see Pronto2d, eqn.(2.1.3)
					// I think that the incremental defgrad should be used here as we describe the motion relative to the reference configuration
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
						PairTlsph::PolDec(F[i], &R[i], &U);

						// unrotated rate-of-deformation tensor d, see right side of Pronto2d, eqn.(2.1.7)
						d[i] = R[i].transpose() * D[i] * R[i];
					}

					// normalize average velocity field aroudn an integration point
					smoothVel[i] /= shepardWeight[i];

				} // end if mol[i] > 0

			} // end if mol[i] > 0

			if (mol[i] < 0) {
				F[i].setIdentity();
				Fdot[i].setZero();
				Fincr[i].setIdentity();
				smoothVel[i].setZero();
				detF[i] = 1.0;
			}
		}  // end if setflage[itype]
	} // end loop over i = 0 to nlocal
}

/* ---------------------------------------------------------------------- */

void PairTlsph::compute(int eflag, int vflag) {
	int *mol = atom->molecule;
	double **x = atom->x;
	double **v = atom->vest;
	double **x0 = atom->x0;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *de = atom->de;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, jtype, iDim, inum;
	double r, hg_mag, wf, wfd, h, r0, r0Sq;
	double delVdotDelR, visc_magnitude, deltaE, mu_ij, c_ij, rho_ij;
	double delta, damage_factor;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	Vector3d fi, fj, dx0, dx, dv, f_stress, f_hg, dxp_i, dxp_j, gamma, g, gamma_i, gamma_j;
	Vector3d xi, xj, vi, vj, f_visc, sumForces;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
		nmax = atom->nmax;
		delete[] F;
		F = new Matrix3d[nmax]; // memory usage: 9 doubles
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
		delete[] smoothVel;
		smoothVel = new Vector3d[nmax]; // memory usage: 3 doubles; total 49 doubles
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
	}

	PairTlsph::PreCompute();
	PairTlsph::AssembleStress();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);

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

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (mol[j] < 0) {
				continue; // Particle j is not a valid SPH particle (anymore). Skip all interactiions with this particle.
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			jtype = type[j];

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

				jtype = type[j];

				// distance vectors in current and reference configuration, velocity difference
				dx = xj - xi;
				dv = vj - vi;

				// derivative of kernel function and reference distance
				// kernel function
				kernel_and_derivative(h, r0, wf, wfd);
				//printf("wf = %f, wfd = %f\n", wf, wfd);

				// current distance
				r = dx.norm();

				// uncorrected kernel gradient
				g = (wfd / r0) * dx0;

				/*
				 * force contribution -- note that the kernel gradient correction has been absorbed into PK1
				 */

				f_stress = vfrac[i] * vfrac[j] * (PK1[i] + PK1[j]) * g;

				/*
				 * hourglass deviation of particles i and j
				 */

				gamma = 0.5 * (Fincr[i] + Fincr[j]) * dx0 - dx;

				if (r > 0.0) { // we divide by r, so guard against divide by zero
					/* SPH-like formulation */
					delta = 0.5 * gamma.dot(dx) / (r + 0.1 * h); // delta has dimensions of [m]
					hg_mag = -hg_coeff[itype][jtype] * delta / (r0Sq); // hg_mag has dimensions [m^(-1)]
					hg_mag *= vfrac[i] * vfrac[j] * wf * (youngsmodulus[itype] + youngsmodulus[jtype]); // hg_mag has dimensions [J*m^(-1)] = [N]

					/* scale hourglass correction to enable plastic flow */
					if ( MAX(eff_plastic_strain[i], eff_plastic_strain[j]) > 1.0e-2) {
						hg_mag = hg_mag * yieldStress[itype] / youngsmodulus[itype];
					}

					f_hg = (hg_mag / (r + 0.1 * h)) * dx;

				} else {
					f_hg.setZero();
				}

				/*
				 * maximum (symmetric damage factor)
				 */
				damage_factor = MAX(damage[i], damage[j]);

				/*
				 * artificial viscosity with linear and quadratic terms
				 * note that artificial viscosity is enhanced if particles are damaged
				 */

				delVdotDelR = dx.dot(dv);
				if (delVdotDelR != 0.0) {

					double hr = h - r; // [m]
					wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h); // [1/m^4] ==> correct for dW/dr in 3D

					mu_ij = h * delVdotDelR / (r * r + 0.1 * h * h);
					c_ij = c0_type[itype][jtype];
					rho_ij = 0.5 * (rmass[i] / vfrac[i] + rmass[j] / vfrac[j]);
					visc_magnitude = (-(Q1[itype][jtype] + damage_factor) * c_ij * mu_ij
							+ (Q2[itype][jtype] + 2.0 * damage_factor) * mu_ij * mu_ij) / rho_ij;
					f_visc = rmass[i] * rmass[j] * visc_magnitude * wfd * dx / r;

				} else {
					f_visc.setZero();
				}

				// sum stress, viscous, and hourglass forces, and apply (nonlinear) failure model only to hourglass correction force

				// apply damage model if in tension
				if (delVdotDelR > 0.0) {
					damage_factor = sqrt(damage[i] * damage[j]);
					damage_factor = 1.0 - pow(damage_factor, 3.0);
				} else {
					damage_factor = 1.0;
				}


				sumForces = damage_factor * (f_visc + f_stress + f_hg);

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
				}

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
				}

				// check if a particle has moved too much w.r.t another particles
				if (r > r0) {
					if ((r - r0) > neighbor->skin) {
						//printf("current distance is %f, r0 distance is %f\n", r, r0);
						updateFlag = 1;
					}
				}

				// update relative velocity
				dtRelative = MIN(dtRelative, r / (dv.norm() + 0.1 * c0_type[itype][jtype]));

			}

		}
	}

	if (vflag_fdotr)
		virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
 linear EOS for use with linear elasticity
 input: initial pressure pInitial, isotropic part of the strain rate d, time-step dt
 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void PairTlsph::LinearEOS(double &lambda, double &pInitial__, double &d__, double &dt__, double &pFinal__, double &p_rate__) {
	double pLimit;

	/*
	 * pressure rate
	 */
	p_rate__ = lambda * d__;

	pFinal__ = pInitial__ + dt__ * p_rate__; // increment pressure using pressure rate

	/*
	 * limit tensile pressure
	 */

	pLimit = lambda * (DETF_MAX - 1.0);
	if (pFinal__ > pLimit) {
		pFinal__ = pLimit;

		/* do not allow pressure to increase */
		if (p_rate__ > 0.0) { // same signs
			p_rate__ = 0.0;
		}

		printf("limiting pressure max\n");
	}

	/*
	 * limit compressive pressure
	 */
	pLimit = lambda * (DETF_MIN - 1.0);
	if (pFinal__ < pLimit) {

		printf("limiting pressure min, pIni=%f, pFin=%f, dt=%f, d_iso=%f\n", pInitial__, pFinal__, dt__, d__);

		pFinal__ = pLimit;

		/* do not allow negative pressure to become more gegative */
		if (p_rate__ < 0.0) { // same signs
			p_rate__ = 0.0;
		}

	}

}

/* ----------------------------------------------------------------------
 linear EOS for use with linear elasticity
 This EOS cuts off pressure at a given limit
 input: initial pressure pInitial, isotropic part of the strain rate d, time-step dt, maxPressure
 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void PairTlsph::LinearCutoffEOS(double &lambda, double &maxPressure, double &pInitial, double &d, double &dt, double &pFinal,
		double &p_rate) {

	/*
	 * pressure rate
	 */
	p_rate = lambda * d;

	pFinal = pInitial + dt * p_rate; // increment pressure using pressure rate

	if (pFinal < maxPressure) {
		pFinal = maxPressure;
		p_rate = (pFinal - pInitial) / dt;
	}

}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void PairTlsph::LinearStrength(double mu, Matrix3d sigmaInitial_dev, Matrix3d d_dev, double dt, Matrix3d *sigmaFinal_dev__,
		Matrix3d *sigma_dev_rate__) {

	/*
	 * deviatoric rate of unrotated stress
	 */
	*sigma_dev_rate__ = 2.0 * mu * d_dev;

	/*
	 * elastic update to the deviatoric stress
	 */
	*sigmaFinal_dev__ = sigmaInitial_dev + dt * *sigma_dev_rate__;
}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: F: deformation gradient
 output:  total stress tensor, deviator + pressure
 ------------------------------------------------------------------------- */
void PairTlsph::LinearStrengthDefgrad(double lambda, double mu, Matrix3d F, Matrix3d *T) {
	Matrix3d E, PK2, eye, sigma, S, tau;

	eye.setIdentity();

	E = 0.5 * (F * F.transpose() - eye); // strain measure E = 0.5 * (B - I) = 0.5 * (F * F^T - I)
	tau = lambda * E.trace() * eye + 2.0 * mu * E; // Kirchhoff stress, work conjugate to above strain
	sigma = tau / F.determinant(); // convert Kirchhoff stress to Cauchy stress

//printf("l=%f, mu=%f, sigma xy = %f\n", lambda, mu, sigma(0,1));

//    E = 0.5 * (F.transpose() * F - eye); // Green-Lagrange Strain E = 0.5 * (C - I)
//    S = lambda * E.trace() * eye + 2.0 * mu * Deviator(E); // PK2 stress
//    tau = F * S * F.transpose(); // convert PK2 to Kirchhoff stress
//    sigma = tau / F.determinant();

	*T = sigma;

	/*
	 * neo-hookean model due to Bonet
	 */
//    lambda = mu = 100.0;
//    // left Cauchy-Green Tensor, b = F.F^T
//    double J = F.determinant();
//    double logJ = log(J);
//    Matrix3d b;
//    b = F * F.transpose();
//
//    sigma = (mu / J) * (b - eye) + (lambda / J) * logJ * eye;
//    *T = sigma;
}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void PairTlsph::LinearPlasticStrength(double G, double yieldStress, Matrix3d sigmaInitial_dev, Matrix3d d_dev, double dt,
		Matrix3d *sigmaFinal_dev__, Matrix3d *sigma_dev_rate__, double *plastic_strain_increment) {

	Matrix3d sigmaTrial_dev, dev_rate;
	double J2;

	/*
	 * deviatoric rate of unrotated stress
	 */
	dev_rate = 2.0 * G * d_dev;

	/*
	 * perform a trial elastic update to the deviatoric stress
	 */
	sigmaTrial_dev = sigmaInitial_dev + dt * dev_rate; // increment stress deviator using deviatoric rate

	/*
	 * check yield condition
	 */
	J2 = sqrt(3. / 2.) * sigmaTrial_dev.norm();

	if (J2 < yieldStress) {
		/*
		 * no yielding has occured.
		 * final deviatoric stress is trial deviatoric stress
		 */
		*sigma_dev_rate__ = dev_rate;
		*sigmaFinal_dev__ = sigmaTrial_dev;
		*plastic_strain_increment = 0.0;
		//printf("no yield\n");

	} else {
		//printf("yiedl\n");
		/*
		 * yielding has occured
		 */
		*plastic_strain_increment = (J2 - yieldStress) / (3.0 * G);

		/*
		 * new deviatoric stress:
		 * obtain by scaling the trial stress deviator
		 */
		*sigmaFinal_dev__ = (yieldStress / J2) * sigmaTrial_dev;

		/*
		 * new deviatoric stress rate
		 */
		*sigma_dev_rate__ = *sigmaFinal_dev__ - sigmaInitial_dev;
		//printf("yielding has occured.\n");
	}
}

/* ----------------------------------------------------------------------
 assemble unrotated stress tensor using deviatoric and pressure components.
 Convert to corotational Cauchy stress, then to PK1 stress and apply
 shape matrix correction
 ------------------------------------------------------------------------- */
void PairTlsph::AssembleStress() {
	int *mol = atom->molecule;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double **tlsph_stress = atom->tlsph_stress;
	int *type = atom->type;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double factor, lambda, mu;
	double pInitial, d_iso, pFinal, p_rate, plastic_strain_increment;
	int i, itype;
	int nlocal = atom->nlocal;
	double dt = update->dt;
	double bulkmodulus;
	Matrix3d sigma_rate, eye, sigmaInitial, sigmaFinal, T, Jaumann_rate, sigma_rate_check;
	Matrix3d d_dev, sigmaInitial_dev, sigmaFinal_dev, sigma_dev_rate, E, sigma_damaged;
	Vector3d x0i, xi, xp;

	eye.setIdentity();
	c0Max = 0.0;
	dtCFL = 1.0e22;

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype] == 1) {
			if (mol[i] > 0) { // only do the following if particle has not failed -- mol < 0 means particle has failed

				/*
				 * initial stress state: given by the unrotateted Kirchhoff stress.
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
				d_iso = d[i].trace();
				d_dev = PairTlsph::Deviator(d[i]);

				/*
				 * pressure
				 */

				//printf("eos = %d\n", eos[itype]);
				switch (eos[itype]) {
				case LINEAR:
					lambda = youngsmodulus[itype] * poissonr[itype] / ((1.0 + poissonr[itype]) * (1.0 - 2.0 * poissonr[itype]));
					mu = youngsmodulus[itype] / (2.0 * (1.0 + poissonr[itype]));
					bulkmodulus = lambda + 2.0 * mu / 3.0;
					LinearEOS(bulkmodulus, pInitial, d_iso, dt, pFinal, p_rate);
					break;
				case LINEAR_CUTOFF:
					lambda = youngsmodulus[itype] * poissonr[itype] / ((1.0 + poissonr[itype]) * (1.0 - 2.0 * poissonr[itype]));
					mu = youngsmodulus[itype] / (2.0 * (1.0 + poissonr[itype]));
					bulkmodulus = lambda + 2.0 * mu / 3.0;
					LinearCutoffEOS(bulkmodulus, yieldStress[itype], pInitial, d_iso, dt, pFinal, p_rate);
					break;
				case NONE:
					pFinal = 0.0;
					p_rate = 0.0;
					break;
				default:
					error->one(FLERR, "unknown EOS.");
					break;
				}

				dtCFL = MIN(dtCFL, radius[i] / c0_type[itype][itype]);

				/*
				 * material strength
				 */

				switch (strengthModel[itype]) {
				case LINEAR:
					mu = youngsmodulus[itype] / (2.0 * (1.0 + poissonr[itype]));
					LinearStrength(mu, sigmaInitial_dev, d[i], dt, &sigmaFinal_dev, &sigma_dev_rate);
					break;
				case LINEAR_DEFGRAD:
					lambda = youngsmodulus[itype] * poissonr[itype] / ((1.0 + poissonr[itype]) * (1.0 - 2.0 * poissonr[itype]));
					mu = youngsmodulus[itype] / (2.0 * (1.0 + poissonr[itype]));
					LinearStrengthDefgrad(lambda, mu, F[i], &sigmaFinal_dev);
					eff_plastic_strain[i] = 0.0;
					p_rate = pInitial - sigmaFinal_dev.trace() / 3.0;
					sigma_dev_rate = sigmaInitial_dev - Deviator(sigmaFinal_dev);
					R[i].setIdentity();
					break;
				case LINEAR_PLASTIC:
					mu = youngsmodulus[itype] / (2.0 * (1.0 + poissonr[itype]));
					LinearPlasticStrength(mu, yieldStress[itype], sigmaInitial_dev, d_dev, dt, &sigmaFinal_dev, &sigma_dev_rate,
							&plastic_strain_increment);
					eff_plastic_strain[i] += plastic_strain_increment;
					break;
				case NONE:
					sigmaFinal_dev.setZero();
					sigma_dev_rate.setZero();
					break;
				default:
					error->one(FLERR, "unknown strength model.");
					break;
				}

				if (eff_plastic_strain[i] > 1.0) {
					shearFailureFlag[i] = true;
				}

				/*
				 *  assemble total stress from pressure and deviatoric stress
				 */
				sigmaFinal = pFinal * eye + sigmaFinal_dev;

				/*
				 *  failure criteria
				 */
				if (maxstress[itype] != 0.0) {
					/*
					 * maximum stress failure criterion:
					 */
					IsotropicMaxStressDamage(sigmaFinal, maxstress[itype], dt, c0_type[itype][itype], radius[i], damage[i],
							sigma_damaged);
				} else if (maxstrain[itype] != 0.0) {
					/*
					 * maximum strain failure criterion:
					 */
					E = 0.5 * (F[i] * F[i].transpose() - eye);
					IsotropicMaxStrainDamage(E, sigmaFinal, maxstrain[itype], dt, c0_type[itype][itype], radius[i], damage[i],
							sigma_damaged);
				} else {
					sigma_damaged = sigmaFinal;
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

				// store rotated, "true" Cauchy stress
				CauchyStress[i] = T;

				/*
				 * We have the corotational Cauchy stress.
				 * Convert to PK1. Note that reference configuration used for computing the forces is linked via
				 * the incremental deformation gradient, not the full deformation gradient.
				 */
				PK1[i] = Fincr[i].determinant() * T * Fincr[i].inverse().transpose();

				/*
				 * correct stress tensor with shape matrix
				 */
				PK1[i] = PK1[i] * K[i];

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

	memory->create(c0_type, n + 1, n + 1, "pair:soundspeed");
	memory->create(youngsmodulus, n + 1, "pair:youngsmodulus");
	memory->create(poissonr, n + 1, "pair:poissonratio");
	memory->create(hg_coeff, n + 1, n + 1, "pair:hg_magnitude");
	memory->create(Q1, n + 1, n + 1, "pair:q1");
	memory->create(Q2, n + 1, n + 1, "pair:q2");
	memory->create(yieldStress, n + 1, "pair:yieldstress");
	memory->create(strengthModel, n + 1, "pair:strengthmodel");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(maxstrain, n + 1, "pair:maxstrain");
	memory->create(maxstress, n + 1, "pair:maxstress");

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
	if (narg != 13)
		error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	int mat_one;
	if (strcmp(arg[2], "linear") == 0) {
		mat_one = LINEAR;
	} else if (strcmp(arg[2], "linearplastic") == 0) {
		mat_one = LINEAR_PLASTIC;
	} else if (strcmp(arg[2], "linear_defgrad") == 0) {
		mat_one = LINEAR_DEFGRAD;
	} else if (strcmp(arg[2], "none") == 0) {
		mat_one = NONE;
	} else {
		error->all(FLERR, "unknown material strength model selected");
	}

	int eos_one;
	if (strcmp(arg[3], "linear") == 0) {
		eos_one = LINEAR;
	} else if (strcmp(arg[3], "linear_cutoff") == 0) {
		eos_one = LINEAR_CUTOFF;
	} else if (strcmp(arg[3], "none") == 0) {
		eos_one = NONE;
	} else {
		error->all(FLERR, "unknown EOS model selected");
	}

	double c0_one = atof(arg[4]);
	double youngsmodulus_one = atof(arg[5]);
	double poissonr_one = atof(arg[6]);
	double hg_one = atof(arg[7]);
	double yieldstress_one = atof(arg[8]);
	double maxstrain_one = atof(arg[9]);
	double maxstress_one = atof(arg[10]);
	double q1_one = atof(arg[11]);
	double q2_one = atof(arg[12]);

	if (maxstrain_one * maxstress_one != 0.0) {
		error->all(FLERR, "both maximum strain or stress damage models are set. only one damage model is allowed to be active.");
	}

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {

// set all properties which are defined per type:
		strengthModel[i] = mat_one;
		eos[i] = eos_one;
		yieldStress[i] = yieldstress_one;
		youngsmodulus[i] = youngsmodulus_one;
		poissonr[i] = poissonr_one;
		maxstrain[i] = maxstrain_one;
		maxstress[i] = maxstress_one;

		for (int j = MAX(jlo, i); j <= jhi; j++) {
			c0_type[i][j] = c0_one;
			hg_coeff[i][j] = hg_one;
			Q1[i][j] = q1_one;
			Q2[i][j] = q2_one;
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

double PairTlsph::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	c0_type[j][i] = c0_type[i][j];
	hg_coeff[j][i] = hg_coeff[i][j];
	Q1[j][i] = Q1[i][j];
	Q2[j][i] = Q2[i][j];

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
//printf("cutoff for pair peri/gcg = %f\n", cutoff);
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
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairTlsph::write_restart(FILE *fp) {
	int i, j;
	for (i = 1; i <= atom->ntypes; i++) {
		fwrite(&youngsmodulus[i], sizeof(double), 1, fp);
		fwrite(&poissonr[i], sizeof(double), 1, fp);
		for (j = i; j <= atom->ntypes; j++) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&Q1[i][j], sizeof(double), 1, fp);
			}
		}
	}
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairTlsph::read_restart(FILE *fp) {
	allocate();

	int i, j;
	int me = comm->me;
	for (i = 1; i <= atom->ntypes; i++) {
		if (me == 0) {
			fread(&youngsmodulus[i], sizeof(double), 1, fp);
			fread(&poissonr[i], sizeof(double), 1, fp);
		}

		MPI_Bcast(&youngsmodulus[i], 1, MPI_DOUBLE, 0, world);
		MPI_Bcast(&poissonr[i], 1, MPI_DOUBLE, 0, world);

		for (j = i; j <= atom->ntypes; j++) {
			if (me == 0) {
				fread(&setflag[i][j], sizeof(int), 1, fp);
			}
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);

			if (setflag[i][j]) {
				if (me == 0) {
					fread(&Q1[i][j], sizeof(double), 1, fp);
				}
				MPI_Bcast(&Q1[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
	}
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
	if (strcmp(str, "sph2/tlsph/F_ptr") == 0) {
		return (void *) F;
	} else if (strcmp(str, "sph2/tlsph/Fincr_ptr") == 0) {
		return (void *) Fincr;
	} else if (strcmp(str, "sph2/tlsph/detF_ptr") == 0) {
		return (void *) detF;
	} else if (strcmp(str, "sph2/tlsph/PK1_ptr") == 0) {
		return (void *) PK1;
	} else if (strcmp(str, "sph2/tlsph/smoothVel_ptr") == 0) {
		return (void *) smoothVel;
	} else if (strcmp(str, "sph2/tlsph/numNeighsRefConfig_ptr") == 0) {
		return (void *) numNeighsRefConfig;
	} else if (strcmp(str, "sph2/tlsph/stressTensor_ptr") == 0) {
		return (void *) CauchyStress;
	} else if (strcmp(str, "sph2/tlsph/updateFlag_ptr") == 0) {
		return (void *) &updateFlag;
	} else if (strcmp(str, "sph2/tlsph/strain_rate_ptr") == 0) {
		return (void *) D;
	} else if (strcmp(str, "sph2/tlsph/hMin_ptr") == 0) {
		return (void *) &hMin;
	} else if (strcmp(str, "sph2/tlsph/dtCFL_ptr") == 0) {
		return (void *) &dtCFL;
	} else if (strcmp(str, "sph2/tlsph/dtRelative_ptr") == 0) {
		return (void *) &dtRelative;
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

void PairTlsph::kernel_and_derivative(const double h, const double r, double &wf, double &wfd) {

	/*
	 * Spiky kernel
	 */

	if (domain->dimension == 2) {
		double hr = h - r; // [m]
		double n = 0.3141592654e0 * h * h * h * h * h; // [m^5]
		wfd = -3.0e0 * hr * hr / n; // [m*m/m^5] = [1/m^3] ==> correct for dW/dr in 2D
		wf = -0.333333333333e0 * hr * wfd; // [m/m^3] ==> [1/m^2] correct for W in 2D
	} else {
		double hr = h - r; // [m]
		wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h); // [1/m^4] ==> correct for dW/dr in 3D
		wfd = -1.0;
		wf = -0.333333333333e0 * hr * wfd; // [m/m^4] ==> [1/m^3] correct for W in 3D
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
			singularValuesInv(row) = 0.0;
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
 isotropic damage model based on max strain
 ------------------------------------------------------------------------- */

void PairTlsph::IsotropicMaxStressDamage(Matrix3d S, double maxStress, double dt, double soundspeed, double characteristicLength,
		double &damage, Matrix3d &S_damaged) {

	/*
	 * compute Eigenvalues of stress matrix
	 */
	SelfAdjointEigenSolver<Matrix3d> es;
	es.compute(S);

	double max_eigenvalue = es.eigenvalues().maxCoeff();

	if (max_eigenvalue > maxStress) {
		damage = damage + dt * 0.4 * soundspeed / characteristicLength;
		damage = MIN(damage, 1.0);
	}

	/* apply damage model */
	if (damage > 0.0) {
		Matrix3d V = es.eigenvectors();

// diagonalize stress matrix
		Matrix3d S_diag = V.inverse() * S * V;

		for (int dim = 0; dim < 3; dim++) {
			if (S_diag(dim, dim) > 0.0) {
				S_diag(dim, dim) *= (1.0 - pow(damage, 3.0));
			}
		}

		// undiagonalize stress matrix
		S_damaged = V * S_diag * V.inverse();
	} else {
		S_damaged = S;
	}
}

void PairTlsph::IsotropicMaxStrainDamage(Matrix3d E, Matrix3d S, double maxStrain, double dt, double soundspeed,
		double characteristicLength, double &damage, Matrix3d &S_damaged) {

	/*
	 * compute Eigenvalues of strain matrix
	 */
	SelfAdjointEigenSolver<Matrix3d> es;
	es.compute(E); // compute eigenvalue and eigenvectors of strain

	double max_eigenvalue = es.eigenvalues().maxCoeff();

	if (max_eigenvalue > maxStrain) {
		damage = damage + dt * 0.04 * soundspeed / characteristicLength;
		//printf("failing strain at %f\n", max_eigenvalue);
		damage = MIN(damage, 1.0);
	}

	/* apply damage model to stress matrix */
	if (damage > 0.0) {
		es.compute(S);
		Matrix3d V = es.eigenvectors(); // compute eigenvalue and eigenvectors of stress

		// diagonalize stress matrix
		Matrix3d S_diag = V.inverse() * S * V;

		// apply damage to diagonalized  matrix if in tension
		for (int dim = 0; dim < 3; dim++) {
			if (S_diag(dim, dim) > 0.0) {
				S_diag(dim, dim) *= (1.0 - pow(damage, 3.0));
			}
		}

		// undiagonalize strain matrix
		S_damaged = V * S_diag * V.inverse();
	} else {
		S_damaged = S;
	}
}
