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
#include "math_special.h"
#include <map>
#include "fix_sph2_integrate_tlsph.h"

using namespace std;
using namespace LAMMPS_NS;

#define JAUMANN 0
#define DETF_MIN 0.1 // maximum compression deformation allowed
#define DETF_MAX 40.0 // maximum tension deformation allowdw
#include <Eigen/SVD>
#include <Eigen/Eigen>
using namespace Eigen;
using namespace MathSpecial;

#define TLSPH_DEBUG 0
#define PLASTIC_STRAIN_AVERAGE_WINDOW 100.0

/* ---------------------------------------------------------------------- */

PairTlsph::PairTlsph(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;

	poissonr = NULL;
	hg_coeff = NULL;
	Q1 = Q2 = NULL;
	strengthModel = eos = NULL;
	lmbda0 = mu0 = NULL;
	signal_vel0 = NULL; // signal velocity basedon on p-wave speed, used for artificial viscosity
	rho0 = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	Fdot = Fincr = K = PK1 = NULL;
	d = R = FincrInv = W = D = NULL;
	detF = NULL;
	smoothVel = NULL;
	shearFailureFlag = NULL;
	numNeighsRefConfig = NULL;
	shepardWeight = NULL;
	CauchyStress = NULL;

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
		memory->destroy(poissonr);
		memory->destroy(hg_coeff);
		memory->destroy(Q1);
		memory->destroy(Q2);
		memory->destroy(strengthModel);
		memory->destroy(eos);
		memory->destroy(lmbda0);
		memory->destroy(mu0);
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
	double **x0 = atom->x0;
	double **x = atom->x;
	double **v = atom->vest;
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	int *tag = atom->tag;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, j, iDim, itype;
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
		smoothVel[i].setZero();

//		x0i << x0[i][0], x0[i][1], x0[i][2];
//		xi << x[i][0], x[i][1], x[i][2];
//		if ((xi - x0i).norm() > 1.0e-8) {
//			printf("particle positions deviate in zeroth time step\n");
//			error->one(FLERR, "d");
//		}

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

//				if (r0 < 0.1) {
//					printf("distance is %f\n", r0);
//				}

				volj = vfrac[j];

				// distance vectors in current and reference configuration, velocity difference
				dx = xj - xi;
				dv = vj - vi;
				dvint = vintj - vinti;
				du = xj - x0j - (xi - x0i);

				// kernel function
				kernel_and_derivative(h, r0, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r0) * dx0;

				/* build matrices */
				Ktmp = g * dx0.transpose();

//				if (tag[i] == 9575) {
//					cout << "displacement vector is " << dx << endl;
//					cout << "current pos is " << xi << endl;
//					cout << "pos0 is " << x0i << endl;
//
//					cout << "displacement of i is " << x0i - xi << endl;
//					cout << "displacement of j is " << x0j - xj << endl << endl;
//
//
//				}

				//Ftmp = du * g.transpose(); // only use displacement here, turns out to be numerically more stable
				Ftmp = dx * g.transpose();
				Fdottmp = dv * g.transpose();

				K[i] += volj * Ktmp;
				Fincr[i] += volj * Ftmp;
				Fdot[i] += volj * Fdottmp;
				shepardWeight[i] += volj * wf;
				smoothVel[i] += volj * wf * dvint;
				numNeighsRefConfig[i]++;

				if (j < nlocal) {
					K[j] += voli * Ktmp;
					Fincr[j] += voli * Ftmp;
					Fdot[j] += voli * Fdottmp;
					shepardWeight[j] += voli * wf;
					smoothVel[j] -= voli * wf * dvint;
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

//				if (tag[i] == 9575) {
//					cout << "Here is matrix F before corection:" << endl << Fincr[i] << endl << endl;
//				}

				if (domain->dimension == 2) {
					K[i] = PairTlsph::pseudo_inverse_SVD(K[i]);
					K[i](2, 2) = 1.0; // make inverse of K well defined even when it is rank-deficient (3d matrix, only 2d information)
				} else {
					K[i] = K[i].inverse().eval();
				}

				Fincr[i] *= K[i];
				//Fincr[i] += eye; // need to add identity matrix to comple the deformation gradient
				Fdot[i] *= K[i];

				/*
				 * we need to be able to recover from a potentially planar (2d) configuration of particles
				 */
				if ((domain->dimension == 2)) {
					Fincr[i](2, 2) = 1.0;
				}

				/*
				 * make sure F stays within some limits
				 */

				if (Fincr[i].determinant() < DETF_MIN) {
					printf("deleting particle [%d] because det(F)=%f is smaller than limit=%f\n", tag[i], Fincr[i].determinant(),
					DETF_MIN);
					printf("nn = %d\n", numNeighsRefConfig[i]);
					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
					cout << "Here is matrix K-1:" << endl << K[i] << endl;
					mol[i] = -1;
				} else if (Fincr[i].determinant() > DETF_MAX) {
					printf("deleting particle [%d] because det(F)=%f is larger than limit=%f\n", tag[i], Fincr[i].determinant(),
					DETF_MAX);
					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
					mol[i] = -1;
				}

				if (mol[i] > 0) {
					detF[i] = Fincr[i].determinant();
					//FincrInv[i] = PairTlsph::pseudo_inverse_SVD(Fincr[i]);
					FincrInv[i] = Fincr[i].inverse();

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

						PairTlsph::PolDec(Fincr[i], &R[i], &U);

						// unrotated rate-of-deformation tensor d, see right side of Pronto2d, eqn.(2.1.7)
						d[i] = R[i].transpose() * D[i] * R[i];
					}

					// normalize average velocity field aroudn an integration point
					smoothVel[i] /= shepardWeight[i];

				} // end if mol[i] > 0

			} // end if mol[i] > 0

			if (mol[i] < 0) {
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

	if (atom->nmax > nmax) {

		printf("******************************** REALLOC\n");

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

	ComputeForces(eflag, vflag);

	/*
	 * initialize neighbor list pointer for time integration fix.
	 * this needs to be done here beacuse fix is created after pair.
	 */
	if (update->ntimestep == 1) {

		ifix_tlsph = -1;
		for (int i = 0; i < modify->nfix; i++) {
			printf("fix %d\n", i);
			if (strcmp(modify->fix[i]->style, "sph2/integrate_tlsph") == 0)
				ifix_tlsph = i;
		}
		if (ifix_tlsph == -1)
			error->all(FLERR, "Fix ifix_tlsph does not exist");

		fix_tlsph_time_integration = (FixSph2IntegrateTlsph *) modify->fix[ifix_tlsph];
		fix_tlsph_time_integration->pair = this;
	}

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
	double *damage = atom->damage;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, jtype, iDim, inum;
	double r, hg_mag, wf, wfd, h, r0, r0Sq, voli, volj;
	double delVdotDelR, visc_magnitude, deltaE, mu_ij, c_ij, rho_ij;
	double delta, hr;
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
				volj = vfrac[j];

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

				f_stress = voli * volj * (PK1[i] + PK1[j]) * g;

				/*
				 * hourglass deviation of particles i and j
				 */

				gamma = 0.5 * (Fincr[i] + Fincr[j]) * dx0 - dx;

				/* SPH-like formulation */

				delta = 0.5 * gamma.dot(dx) / (r + 0.1 * h); // delta has dimensions of [m]
				hg_mag = -hg_coeff[itype][jtype] * delta / (r0Sq + 0.01 * h * h); // hg_mag has dimensions [m^(-1)]
				hg_mag *= voli * volj * wf * (youngsmodulus[itype] + youngsmodulus[jtype]); // hg_mag has dimensions [J*m^(-1)] = [N]

				/* scale hourglass correction to enable plastic flow */
				//if ( MAX(eff_plastic_strain[i], eff_plastic_strain[j]) > 1.0e-2) {
				//	hg_mag = 0.1 * hg_mag;
				//}
				f_hg = (hg_mag / (r + 0.1 * h)) * dx;

				/*
				 * maximum (symmetric damage factor)
				 */
				//damage_factor = MAX(damage[i], damage[j]);
				/*
				 * artificial viscosity with linear and quadratic terms
				 * note that artificial viscosity is enhanced if particles are damaged
				 * note that quadratic term in viscosity is not useful at all apparently
				 */

				delVdotDelR = dx.dot(dv);
				//hr = h - r; // [m]
				//wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h); // [1/m^4] ==> correct for dW/dr in 3D

				mu_ij = h * delVdotDelR / (r * r + 0.1 * h * h);
				c_ij = 0.5 * (signal_vel0[itype] + signal_vel0[jtype]);
				//printf("c_ij = %f\n", c_ij);
				rho_ij = 0.5 * (rho0[itype] + rho0[jtype]);
				//visc_magnitude = (-(Q1[itype][jtype] + damage_factor) * c_ij * mu_ij
				//		+ (Q2[itype][jtype] + 2.0 * damage_factor) * mu_ij * mu_ij) / rho_ij;
				visc_magnitude = -(Q1[itype][jtype]) * c_ij * mu_ij / rho_ij; // this has units of pressure
				//printf("visc_magnitude = %f, c_ij=%f, mu_ij=%f\n", visc_magnitude, c_ij, mu_ij);
				f_visc = rmass[i] * rmass[j] * visc_magnitude * wfd * dx / (r + 1.0e-2 * h);

				// sum stress, viscous, and hourglass forces
				//cout << "fstress: " << f_stress << endl;
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
				}

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
				}

				// check if a particle has moved too much w.r.t another particles
				if (r > r0) {
					if ((r - r0) > 0.5 * neighbor->skin) {
						//printf("current distance is %f, r0 distance is %f\n", r, r0);
						updateFlag = 1;
					}
				}

				// update relative velocity based timestep
				//dtRelative = MIN(dtRelative, r / (dv.norm() + 0.1 * c_ij));

			}

		}
	}

	if (vflag_fdotr)
		virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 shock EOS
 input:
 current density rho
 reference density rho0
 current energy density e
 reference energy density e0
 reference speed of sound c0
 shock Hugoniot parameter S
 Grueneisen parameter Gamma
 initial pressure pInitial
 time step dt

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void PairTlsph::ShockEOS(double rho, double rho0, double e, double e0, double c0, double S, double Gamma, double pInitial,
		double dt, double &pFinal, double &p_rate) {

	double mu = rho / rho0 - 1.0;
	double pH = rho0 * square(c0) * mu * (1.0 + mu) / square(1.0 - (S - 1.0) * mu);

	pFinal = -(pH + rho * Gamma * (e - e0));

	//printf("shock EOS: rho = %g, rho0 = %g, Gamma=%f, c0=%f, S=%f, e=%f, e0=%f\n", rho, rho0, Gamma, c0, S, e, e0);
	//printf("pFinal = %f\n", pFinal);
	p_rate = (pFinal - pInitial) / dt;

}

/* ----------------------------------------------------------------------
 polynomial EOS
 input:
 current density rho
 reference density rho0
 coefficients 0 .. 6
 initial pressure pInitial
 time step dt

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void PairTlsph::polynomialEOS(double rho, double rho0, double e, double C0, double C1, double C2, double C3, double C4, double C5,
		double C6, double pInitial, double dt, double &pFinal, double &p_rate) {

	double mu = rho / rho0 - 1.0;

	pFinal = C0 + C1 * mu + C2 * mu * mu + C3 * mu * mu * mu + (C4 + C5 * mu + C6 * mu * mu) * e;
	pFinal = -pFinal;

	//printf("pFinal = %f\n", pFinal);
	p_rate = (pFinal - pInitial) / dt;

}

/* ----------------------------------------------------------------------
 linear EOS for use with linear elasticity
 input: initial pressure pInitial, isotropic part of the strain rate d, time-step dt
 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void PairTlsph::LinearEOS(double lambda, double pInitial, double d, double dt, double &pFinal, double &p_rate) {

	/*
	 * pressure rate
	 */
	p_rate = lambda * d;

	pFinal = pInitial + dt * p_rate; // increment pressure using pressure rate

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
 linear strength model for use with linear elasticity
 input:
 G : shear modulus
 cp : heat capacity
 espec : energy / mass
 A : initial yield stress under quasi-static / room temperature conditions
 B : proportionality factor for plastic strain dependency
 a : exponent for plastic strain dpendency
 C : proportionality factor for logarithmic plastic strain rate dependency
 epdot0 : dimensionality factor for plastic strain rate dependency
 T : current temperature
 T0 : reference (room) temperature
 Tmelt : melting temperature
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void PairTlsph::JohnsonCookStrength(double G, double cp, double espec, double A, double B, double a, double C, double epdot0,
		double T0, double Tmelt, double M, double dt, double ep, double epdot, Matrix3d sigmaInitial_dev, Matrix3d d_dev,
		Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment) {

	Matrix3d sigmaTrial_dev, dev_rate;
	double J2, yieldStress;

	double deltaT = espec / cp;
	double TH = deltaT / (Tmelt - T0);
	TH = MAX(TH, 0.0);
	double epdot_ratio = epdot / epdot0;
	epdot_ratio = MAX(epdot_ratio, 1.0);
	//printf("current temperature delta is %f, TH=%f\n", deltaT, TH);

	yieldStress = (A + B * pow(ep, a)) * (1.0 + C * log(epdot_ratio)) * (1.0 - pow(TH, M));

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
		sigma_dev_rate__ = dev_rate;
		sigmaFinal_dev__ = sigmaTrial_dev;
		plastic_strain_increment = 0.0;
		//printf("no yield\n");

	} else {
		//printf("yiedl\n");
		/*
		 * yielding has occured
		 */
		plastic_strain_increment = (J2 - yieldStress) / (3.0 * G);

		/*
		 * new deviatoric stress:
		 * obtain by scaling the trial stress deviator
		 */
		sigmaFinal_dev__ = (yieldStress / J2) * sigmaTrial_dev;

		/*
		 * new deviatoric stress rate
		 */
		sigma_dev_rate__ = sigmaFinal_dev__ - sigmaInitial_dev;
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
	double bulkmodulus, M, p_wave_speed, mass_specific_energy, vol_specific_energy, rho;
	Matrix3d sigma_rate, eye, sigmaInitial, sigmaFinal, T, Jaumann_rate, sigma_rate_check;
	Matrix3d d_dev, sigmaInitial_dev, sigmaFinal_dev, sigma_dev_rate, E, sigma_damaged;
	Vector3d x0i, xi, xp;

	eye.setIdentity();
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

				mass_specific_energy = e[i] / rmass[i];
				rho = rmass[i] / (detF[i] * vfrac[i]);
				vol_specific_energy = mass_specific_energy * rho;

				/*
				 * pressure
				 */

				//printf("eos = %d\n", eos[itype]);
				switch (eos[itype]) {
				case LINEAR:
					bulkmodulus = lmbda0[itype] + 2.0 * mu0[itype] / 3.0;
					//printf("bulk modulus is %f\n", bulkmodulus);
					LinearEOS(bulkmodulus, pInitial, d_iso, dt, pFinal, p_rate);
					break;
				case LINEAR_CUTOFF:
					bulkmodulus = lmbda0[itype] + 2.0 * mu0[itype] / 3.0;
					LinearCutoffEOS(bulkmodulus, EOSProps["cutoff_pressure"][itype], pInitial, d_iso, dt, pFinal, p_rate);
					break;
				case NONE:
					pFinal = 0.0;
					p_rate = 0.0;
					break;
				case SHOCK_EOS:
					// check that required parameters are known
					if (EOSProps.count("c0") != 1) {
						error->one(FLERR, "c0 for shock EOS is missing in EOS property map");
					}
					if (EOSProps.count("S_Hugoniot") != 1) {
						error->one(FLERR, "S_Hugoniot for shock EOS is missing in EOS property map");
					}
					if (EOSProps.count("Gamma_Grueneisen") != 1) {
						error->one(FLERR, "Gamma_Grueneisen for shock EOS is missing in EOS property map");
					}

					//  rho,  rho0,  e,  e0,  c0,  S,  Gamma,  pInitial,  dt,  &pFinal,  &p_rate);
					ShockEOS(rho, rho0[itype], mass_specific_energy, 0.0, EOSProps["c0"][itype], EOSProps["S_Hugoniot"][itype],
							EOSProps["Gamma_Grueneisen"][itype], pInitial, dt, pFinal, p_rate);
					break;
				case POLYNOMIAL_EOS:
					polynomialEOS(rho, rho0[itype], vol_specific_energy, EOSProps["C0"][itype], EOSProps["C1"][itype],
							EOSProps["C2"][itype], EOSProps["C3"][itype], EOSProps["C4"][itype], EOSProps["C5"][itype],
							EOSProps["C6"][itype], pInitial, dt, pFinal, p_rate);

					break;

				default:
					error->one(FLERR, "unknown EOS.");
					break;
				}

				/*
				 * material strength
				 */

				switch (strengthModel[itype]) {
				case LINEAR:
					//printf("mu0 is %f\n", mu0[itype]);
					LinearStrength(mu0[itype], sigmaInitial_dev, d[i], dt, &sigmaFinal_dev, &sigma_dev_rate);
					break;
				case LINEAR_DEFGRAD:
					LinearStrengthDefgrad(lmbda0[itype], mu0[itype], Fincr[i], &sigmaFinal_dev);
					eff_plastic_strain[i] = 0.0;
					p_rate = pInitial - sigmaFinal_dev.trace() / 3.0;
					sigma_dev_rate = sigmaInitial_dev - Deviator(sigmaFinal_dev);
					R[i].setIdentity();
					break;
				case LINEAR_PLASTIC:
					// check that required parameters are known
					if (strengthProps.count("yield_stress0") != 1) {
						error->one(FLERR, "yield stress for linear plasticity is missing in strength property map");
					}

					LinearPlasticStrength(mu0[itype], strengthProps["yield_stress0"][itype], sigmaInitial_dev, d_dev, dt,
							&sigmaFinal_dev, &sigma_dev_rate, &plastic_strain_increment);
					eff_plastic_strain[i] += plastic_strain_increment;

					eff_plastic_strain_rate[i] -= eff_plastic_strain_rate[i] / PLASTIC_STRAIN_AVERAGE_WINDOW;
					eff_plastic_strain_rate[i] += (plastic_strain_increment / dt) / PLASTIC_STRAIN_AVERAGE_WINDOW;
					break;
				case JOHNSON_COOK:
//					strengthProps["JC_A"][i] = materialCoeffs_one[12];
//								strengthProps["JC_B"][i] = materialCoeffs_one[13];
//								strengthProps["JC_a"][i] = materialCoeffs_one[14];
//								strengthProps["JC_C"][i] = materialCoeffs_one[15];
//								strengthProps["JC_epdot0"][i] = materialCoeffs_one[16];
//								strengthProps["JC_T0"][i] = materialCoeffs_one[17];
//								strengthProps["JC_Tmelt"][i] = materialCoeffs_one[18];

					JohnsonCookStrength(mu0[itype], commonProps["heat_capacity"][itype], mass_specific_energy,
							strengthProps["JC_A"][itype], strengthProps["JC_B"][itype], strengthProps["JC_a"][itype],
							strengthProps["JC_C"][itype], strengthProps["JC_epdot0"][itype], strengthProps["JC_T0"][itype],
							strengthProps["JC_Tmelt"][itype], strengthProps["JC_M"][itype], dt, eff_plastic_strain[i],
							eff_plastic_strain_rate[i], sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate,
							plastic_strain_increment);

					eff_plastic_strain[i] += plastic_strain_increment;

					eff_plastic_strain_rate[i] -= eff_plastic_strain_rate[i] / PLASTIC_STRAIN_AVERAGE_WINDOW;
					eff_plastic_strain_rate[i] += (plastic_strain_increment / dt) / PLASTIC_STRAIN_AVERAGE_WINDOW;

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
				eff_plastic_strain[i] = MIN(eff_plastic_strain[i], 2.0);

				/*
				 *  assemble total stress from pressure and deviatoric stress
				 */
				sigmaFinal = pFinal * eye + sigmaFinal_dev;

				/*
				 *  failure criteria
				 */
				if (strengthProps["max_tensile_stress"][itype] != 0.0) {
					/*
					 * maximum stress failure criterion:
					 */
					IsotropicMaxStressDamage(sigmaFinal, strengthProps["max_tensile_stress"][itype], dt, signal_vel0[itype],
							radius[i], damage[i], sigma_damaged);
				} else if (strengthProps["max_tensile_strain"][itype] != 0.0) {
					/*
					 * maximum strain failure criterion:
					 */
					E = 0.5 * (Fincr[i] * Fincr[i].transpose() - eye);
					IsotropicMaxStrainDamage(E, sigmaFinal, strengthProps["max_tensile_strain"][itype], dt, signal_vel0[itype],
							radius[i], damage[i], sigma_damaged);
				} else {
					sigma_damaged = sigmaFinal;
				}

				if (shearFailureFlag[i]) {
					if (pFinal < 0.0) {
						sigma_damaged = pFinal * eye;
					} else {
						sigma_damaged	.setZero();
					}
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
				PK1[i] = Fincr[i].determinant() * T * FincrInv[i].transpose();

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
	memory->create(poissonr, n + 1, "pair:poissonratio");
	memory->create(hg_coeff, n + 1, n + 1, "pair:hg_magnitude");
	memory->create(Q1, n + 1, n + 1, "pair:q1");
	memory->create(Q2, n + 1, n + 1, "pair:q2");
	memory->create(strengthModel, n + 1, "pair:strengthmodel");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(lmbda0, n + 1, "pair:lmbda0");
	memory->create(mu0, n + 1, "pair:mu0");
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

	if (narg != 31) {
		char str[128];
		sprintf(str, "number of arguments for pair tlsph is %d but 31 are required!", narg);
		error->all(FLERR, str);
	}
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi, k;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	int mat_one = NONE;
	if (strcmp(arg[2], "linear") == 0) {
		mat_one = LINEAR;
	} else if (strcmp(arg[2], "linearplastic") == 0) {
		mat_one = LINEAR_PLASTIC;
	} else if (strcmp(arg[2], "linear_defgrad") == 0) {
		mat_one = LINEAR_DEFGRAD;
	} else if (strcmp(arg[2], "johnson_cook") == 0) {
		mat_one = JOHNSON_COOK;
	} else if (strcmp(arg[2], "none") == 0) {
		mat_one = NONE;
	} else {
		error->all(FLERR, "unknown material strength model selected");
	}

	int eos_one = NONE;
	if (strcmp(arg[3], "linear") == 0) {
		eos_one = LINEAR;
	} else if (strcmp(arg[3], "linear_cutoff") == 0) {
		eos_one = LINEAR_CUTOFF;
	} else if (strcmp(arg[3], "shock_eos") == 0) {
		eos_one = SHOCK_EOS;
	} else if (strcmp(arg[3], "polynomial_eos") == 0) {
		eos_one = POLYNOMIAL_EOS;
	} else if (strcmp(arg[3], "none") == 0) {
		eos_one = NONE;
	} else {
		error->all(FLERR, "unknown EOS model selected");
	}

	/*
	 * read in common data defined for all material & EOS models:
	 */
	double rho0_one = atof(arg[4]);
	double youngsmodulus_one = atof(arg[5]);
	double poissonr_one = atof(arg[6]);
	double q1_one = atof(arg[7]);
	double q2_one = atof(arg[8]);
	double hg_one = atof(arg[9]);
	double cp_one = atof(arg[10]); // heat capacity

	/*
	 * read in EOS ans strength parameters
	 */
	double *materialCoeffs_one = new double[20];
	for (k = 0; k < 20; k++) {
		materialCoeffs_one[k] = atof(arg[11 + k]);
	}

	/*
	 * check stress and strain failure criteria
	 */

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {

		if (comm->me == 0)
			printf("\n******************SPH2 / TLSPH PROPERTIES OF PARTICLE TYPE %d **************************\n", i);

		/*
		 * assign properties which are defined for all material & eos models
		 */
		strengthModel[i] = mat_one;
		eos[i] = eos_one;
		lmbda0[i] = youngsmodulus_one * poissonr_one / ((1.0 + poissonr_one) * (1.0 - 2.0 * poissonr_one));
		rho0[i] = rho0_one;
		mu0[i] = youngsmodulus_one / (2.0 * (1.0 + poissonr_one));
		signal_vel0[i] = sqrt((lmbda0[i] + 2.0 * mu0[i]) / rho0[i]);
		commonProps["heat_capacity"][i] = cp_one;

		EOSProps["bulk_modulus"][i] = lmbda0[i] + 2.0 * mu0[i] / 3.0;

		if (comm->me == 0) {
			printf("\n material unspecific properties for SPH2/TLSPH definition of particle type %d:\n", i);
			printf("%40s : %g\n", "Young's modulus", youngsmodulus_one);
			printf("%40s : %g\n", "Poisson ratio", poissonr_one);
			printf("%40s : %g\n", "Lame constants lambda", lmbda0[i]);
			printf("%40s : %g\n", "shear modulus", mu0[i]);
			printf("%40s : %g\n", "reference density", rho0[i]);
			printf("%40s : %g\n", "signal velocity", signal_vel0[i]);
			printf("%40s : %g\n", "linear viscosity coefficient", q1_one);
			printf("%40s : %g\n", "quadratic viscosity coefficient", q2_one);
			printf("%40s : %g\n", "heat capacity [energy / (mass * temperature)]", commonProps["heat_capacity"][i]);
		}

		/*
		 * assign pairwise defined quantities which are defined for all material & eos models
		 */
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			hg_coeff[i][j] = hg_one;
			Q1[i][j] = q1_one;
			Q2[i][j] = q2_one;
			setflag[i][j] = 1;
			count++;
		}

		/*
		 * assign data specific to EOS model
		 */
		if (comm->me == 0)
			printf("\nEOS specific properties for SPH2/TLSPH definition of particle type %d:\n", i);

		if (eos[i] == LINEAR) {
			if (comm->me == 0)
				printf("%60s\n", "linear EOS");
		} else if (eos[i] == LINEAR_CUTOFF) {
			if (comm->me == 0)
				printf("%60s\n", "linear EOS");
			EOSProps["pressure_cutoff"][i] = materialCoeffs_one[0];
			if (comm->me == 0)
				printf("%60s : %g\n", "pressure cutoff for linear EOS", EOSProps["pressure_cutoff"][i]);
		} else if (eos[i] == SHOCK_EOS) {
			if (comm->me == 0)
				printf("\n%60s\n", "shock EOS");
			EOSProps["c0"][i] = materialCoeffs_one[0];
			EOSProps["S_Hugoniot"][i] = materialCoeffs_one[1];
			EOSProps["Gamma_Grueneisen"][i] = materialCoeffs_one[2];
			if (comm->me == 0) {
				printf("%60s : %g\n", "reference speed of sound", EOSProps["c0"][i]);
				printf("%60s : %g\n", "Hugoniot parameter S", EOSProps["S_Hugoniot"][i]);
				printf("%60s : %g\n", "Grueneisen Gamma", EOSProps["Gamma_Grueneisen"][i]);
			}
		} else if (eos[i] == POLYNOMIAL_EOS) {
			if (comm->me == 0)
				printf("\n%60s\n", "polynomial EOS");
			EOSProps["C0"][i] = materialCoeffs_one[0];
			EOSProps["C1"][i] = materialCoeffs_one[1];
			EOSProps["C2"][i] = materialCoeffs_one[2];
			EOSProps["C3"][i] = materialCoeffs_one[3];
			EOSProps["C4"][i] = materialCoeffs_one[4];
			EOSProps["C5"][i] = materialCoeffs_one[5];
			EOSProps["C6"][i] = materialCoeffs_one[6];
			if (comm->me == 0) {
				printf("%60s : %g\n", "coefficient 0", EOSProps["C0"][i]);
				printf("%60s : %g\n", "coefficient 1", EOSProps["C1"][i]);
				printf("%60s : %g\n", "coefficient 2", EOSProps["C2"][i]);
				printf("%60s : %g\n", "coefficient 3", EOSProps["C3"][i]);
				printf("%60s : %g\n", "coefficient 4", EOSProps["C4"][i]);
				printf("%60s : %g\n", "coefficient 5", EOSProps["C5"][i]);
				printf("%60s : %g\n", "coefficient 6", EOSProps["C6"][i]);
			}
		}

		/*
		 * assign data specific to Strength model
		 */
		if (comm->me == 0)
			printf("\nStrength specific properties for SPH2/TLSPH definition of particle type %d:\n", i);
		if (materialCoeffs_one[10] * materialCoeffs_one[11] != 0.0) {
			error->all(FLERR,
					"both maximum strain or stress damage models are set. only one damage model is allowed to be active.");
		}

		strengthProps["max_tensile_strain"][i] = materialCoeffs_one[10];
		strengthProps["max_tensile_stress"][i] = materialCoeffs_one[11];
		if (comm->me == 0) {
			printf("%40s : %g\n", "maximum tensile strain (not active if zero)", strengthProps["max_tensile_strain"][i]);
			printf("%40s : %g\n", "maximum tensile stress (not active if zero)", strengthProps["max_tensile_stress"][i]);
		}

		if (strengthModel[i] == LINEAR) {
			// nothing to be done here
		} else if (strengthModel[i] == LINEAR_PLASTIC) {
			strengthProps["yield_stress0"][i] = materialCoeffs_one[12];
			if (comm->me == 0) {
				printf("\n%40s\n", "linear plastic material strength model");
				printf("%40s : %g\n", "initial plastic yield stress", strengthProps["yield_stress0"][i]);
			}
		} else if (strengthModel[i] == JOHNSON_COOK) {
			strengthProps["JC_A"][i] = materialCoeffs_one[12];
			strengthProps["JC_B"][i] = materialCoeffs_one[13];
			strengthProps["JC_a"][i] = materialCoeffs_one[14];
			strengthProps["JC_C"][i] = materialCoeffs_one[15];
			strengthProps["JC_epdot0"][i] = materialCoeffs_one[16];
			strengthProps["JC_T0"][i] = materialCoeffs_one[17];
			strengthProps["JC_Tmelt"][i] = materialCoeffs_one[18];
			strengthProps["JC_M"][i] = materialCoeffs_one[19];

			if (comm->me == 0) {
				printf("\n%60s\n", "Johnson Cook material strength model");
				printf("%60s : %g\n", "A: initial yield stress", strengthProps["JC_A"][i]);
				printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", strengthProps["JC_B"][i]);
				printf("%60s : %g\n", "a : exponent for plastic strain dependency", strengthProps["JC_a"][i]);
				printf("%60s : %g\n", "C : proportionality factor for logarithmic plastic strain rate dependency",
						strengthProps["JC_C"][i]);
				printf("%60s : %g\n", "epdot0 : dimensionality factor for plastic strain rate dependency",
						strengthProps["JC_epdot0"][i]);
				printf("%60s : %g\n", "T0 : reference (room) temperature", strengthProps["JC_T0"][i]);
				printf("%60s : %g\n", "Tmelt : melting temperature", strengthProps["JC_Tmelt"][i]);
				printf("%60s : %g\n", "M : exponent for temperature dependency", strengthProps["JC_M"][i]);
			}
		}

		if (comm->me == 0)
			printf("***************************************************************************************\n");
	}

	if (count == 0)
		error->all(FLERR, "Incorrect args for pair coefficients");

	delete[] materialCoeffs_one;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairTlsph::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

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
	if (strcmp(str, "sph2/tlsph/Fincr_ptr") == 0) {
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

	if (dt * d_iso > 1.0e-1) {
		K3 = 3.0 * p_rate / (dt * d_iso);
	} else {
		K3 = 3.0 * lmbda0[itype] + 2.0 * mu0[itype];
	}

	if (domain->dimension == 3) {
		mu2 = sigma_dev_rate(0, 1) / (dt * d_dev(0, 1)) + sigma_dev_rate(0, 2) / (dt * d_dev(0, 2))
				+ sigma_dev_rate(1, 2) / (dt * d_dev(1, 2));
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2);
	} else {
		mu2 = sigma_dev_rate(0, 1) / (dt * d_dev(0, 1));
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1);
	}

	if (dt * dt * shear_rate_sq < 1.0e-8) {
		mu2 = 0.5 * (3.0 * (lmbda0[itype] + 2.0 * mu0[itype]) - K3);
	}

	M = (K3 + 2.0 * mu2) / 3.0; // effective dilational modulus, see Pronto 2d eqn 3.4.8

	if (M < lmbda0[itype] + 2.0 * mu0[itype]) { // do not allow effective dilatational modulus to decrease beyond its initial value
		M = lmbda0[itype] + 2.0 * mu0[itype];
	}

	/*
	 * damaged particles potentially have a very high dilatational modulus, even though damage degradation scales down the
	 * effective stress. we simply use the initial modulus for damaged particles.
	 */

	if (damage > 0.99) {
		M = lmbda0[itype] + 2.0 * mu0[itype]; //
	}

	return M;

}
