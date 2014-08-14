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
#include "pair_sph.h"
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

#include <Eigen/SVD>
#include <Eigen/Eigen>
using namespace Eigen;

#define COMPUTE_CORRECTED_DERIVATIVES false
#define DENSITY_SUMMATION true
/* ---------------------------------------------------------------------- */

PairSphFluid::PairSphFluid(LAMMPS *lmp) :
		Pair(lmp) {

	C1 = NULL;
	C2 = NULL;
	C3 = NULL;
	C4 = NULL;
	Q1 = NULL;
	Q2 = NULL;
	eos = NULL;
	pressure = NULL;
	c0 = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = K = NULL;
	artStress = NULL;
	delete_flag = NULL;
	shepardWeight = NULL;
	smoothVel = NULL;

	comm_forward = 15; // this pair style communicates 9 doubles to ghost atoms
}

/* ---------------------------------------------------------------------- */

PairSphFluid::~PairSphFluid() {
	if (allocated) {
		//printf("... deallocating\n");
		memory->destroy(C1);
		memory->destroy(C2);
		memory->destroy(C3);
		memory->destroy(C4);
		memory->destroy(Q1);
		memory->destroy(Q2);
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
	}
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairSphFluid::PreCompute() {
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double **x = atom->x;
	double **v = atom->vest;
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	double *rmass = atom->rmass;
	double *rho = atom->rho;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int *mol = atom->molecule;
	int nmax = atom->nmax;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, itype, jtype, j, iDim;
	double wfd, wShep, h, norm, irad, r, rSq, wf, ih, ihsq;
	Vector3d dx, dv, g;
	Matrix3d Ktmp, Ltmp;
	Vector3d xi, xj, vi, vj, vinti, vintj, dvint;

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		itype = type[i];

		shepardWeight[i] = 0.0;
		delete_flag[i] = 0.0;
		smoothVel[i].setZero();
		L[i].setZero();

		if (DENSITY_SUMMATION) {
			if (setflag[itype][itype] == 1) {
				rho[i] = 0.0;
			}
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

		// initialize Eigen data structures from LAMMPS data structures
		for (iDim = 0; iDim < 3; iDim++) {
			xi(iDim) = x[i][iDim];
			vi(iDim) = v[i][iDim];
			vinti(iDim) = vint[i][iDim];
		}

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			for (iDim = 0; iDim < 3; iDim++) {
				xj(iDim) = x[j][iDim];
				vj(iDim) = v[j][iDim];
				vintj(iDim) = vint[j][iDim];
			}

			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = irad + radius[j];
			if (rSq < h * h) {

				r = sqrt(rSq);

				jtype = type[j];
				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;
				dvint = vintj - vinti;

				// kernel and derivative
				kernel_and_derivative(h, r, wf, wfd);
				wfd /= r; // we divide by r and mutliply with the full (non-normalized) distance vector

				/* build correction matric for kernel derivatives */
				if (COMPUTE_CORRECTED_DERIVATIVES) {
					Ktmp = wfd * dx * dx.transpose();
					K[i] += vfrac[j] * Ktmp;
				}

				if (DENSITY_SUMMATION) {
					if (setflag[itype][itype] == 1) {
						rho[i] += wf * rmass[j];
					}
				}

				// velocity gradient L
				Ltmp = wfd * dv * dx.transpose();
				L[i] += vfrac[j] * Ltmp;

				shepardWeight[i] += vfrac[j] * wf;
				smoothVel[i] += vfrac[j] * wf * dvint;

				if (j < nlocal) {
					if (COMPUTE_CORRECTED_DERIVATIVES) {
						K[j] += vfrac[i] * Ktmp;
					}

					if (DENSITY_SUMMATION) {
						if (setflag[jtype][jtype] == 1) {
							rho[j] += wf * rmass[i];
						}
					}

					L[j] += vfrac[i] * Ltmp;

					shepardWeight[j] += vfrac[i] * wf;
					smoothVel[j] -= vfrac[i] * wf * dvint;
				}
			} // end if check distance
		} // end loop over j
	} // end loop over i

	/*
	 * invert shape matrix and compute corrected quantities
	 */

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {

			if (DENSITY_SUMMATION) {
				// add self contribution to particle density
				h = 2.0 * radius[i];
				kernel_and_derivative(h, 0.0, wf, wfd);
				rho[i] += wf * rmass[i];
			}

			if (COMPUTE_CORRECTED_DERIVATIVES) {
				K[i] = pseudo_inverse_SVD(K[i]);
			} else {
				K[i].setIdentity();
			}

			if (shepardWeight[i] != 0.0) {
				smoothVel[i] /= shepardWeight[i];
			} else {
				smoothVel[i].setZero();
			}

			//printf("rho = %g\n", rho[i]);
		} // end check if particle is SPH-type
	} // end loop over i = 0 to nlocal
}

/* ---------------------------------------------------------------------- */

void PairSphFluid::compute(int eflag, int vflag) {
	double **x = atom->x;
	double **v = atom->vest;
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *de = atom->de;
	double *drho = atom->drho;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *contact_radius = atom->contact_radius;
	double *e = atom->e;
	double *rho = atom->rho;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, jtype, iDim, inum;
	double evdwl, r, hg_mag, wf, wfd, h, rSq;
	double mu_ij, c_ij, rho_ij;
	double delVdotDelR, visc_magnitude, deltaE;
	double delta, sigv;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	Vector3d fi, fj, dx, dv, f_stress, g, vinti, vintj, dvint;
	Vector3d xi, xj, vi, vj, f_visc, sumForces, f_stress_new;
	double rcut2, wf2;

	//printf("in compute\n");

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
		printf("... allocating in compute with nmax = %d\n", atom->nmax);
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
	}

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		shepardWeight[i] = 0.0;
		smoothVel[i].setZero();
	}

	PairSphFluid::PreCompute();
	PairSphFluid::ComputePressure();

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

				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;
				dvint = vintj - vinti;

				// kernel and derivative
				kernel_and_derivative(h, r, wf, wfd);
				wfd /= r; // we divide by r and mutliply with the full (non-normalized) distance vector

				// uncorrected kernel gradient
				g = wfd * dx;

				/*
				 * force -- the classical SPH way
				 */

//				f_stress = -rmass[i] * rmass[j]
//						* (pressure[i] / (rho[i] * rho[i])
//								+ pressure[j] / (rho[j] * rho[j])) * g;
				/*
				 * force -- like for solids with stress tensor
				 */

				f_stress = vfrac[i] * vfrac[j]
						* (stressTensor[i] + stressTensor[j]) * g;

				rcut2 = 1.5 * (contact_radius[i] + contact_radius[j]);
				if (r < rcut2) {
					wf2 = (rcut2 - r) / rcut2;
					wf2 = wf2;
					f_stress += wf2 * vfrac[i] * vfrac[j]
							* (artStress[i] + artStress[j]) * g;
				}

				/*
				 * artificial viscosity -- alpha is dimensionless
				 * Monaghan–Balsara form of the artificial viscosity
				 * quadratic part is not used because it makes the simulation unstable
				 */

				delVdotDelR = dx.dot(dv);
				if (delVdotDelR != 0.0) { // we divide by r, so guard against divide by zero
					mu_ij = h * delVdotDelR / (rSq + 0.1 * h * h);
					c_ij = 0.5 * (c0[i] + c0[j]);
					rho_ij = 0.5 * (rho[i] + rho[j]);

					visc_magnitude = (-Q1[itype][jtype] * c_ij * mu_ij
							+ Q2[itype][jtype] * mu_ij * mu_ij) / rho_ij;
					f_visc = rmass[i] * rmass[j] * visc_magnitude * g;

					//printf("mu=%g, c=%g, rho=%g, magnitude=%g\n", mu_ij, c_ij, rho_ij, f_visc.norm());
				} else {
					f_visc.setZero();
				}

				sumForces = f_stress + f_visc;

				// energy rate -- project velocity onto force vector
				deltaE = 0.5 * sumForces.dot(dv);

				// change in mass density
				drho[i] += rmass[j] * g.dot(dv);

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
					drho[j] += rmass[i] * g.dot(dv);
				}

				// accumulate smooth velocities
				shepardWeight[i] += vfrac[j] * wf;
				smoothVel[i] += vfrac[j] * wf * dvint;

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0,
					                        0.0, 0.0,
					                        sumForces(0), sumForces(1), sumForces(2),
					                        dx(0), dx(1), dx(2));
				}
			}

		}
	}

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {

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
 perfect gas EOS for use with linear elasticity
 input: gamma -- adiabatic index (ratio of specific heats)
 J -- determinant of deformation gradient
 volume0 -- reference configuration volume of particle
 energy -- energy of particle
 pInitial -- initial pressure of the particle
 d -- isotropic part of the strain rate tensor,
 dt -- time-step size

 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
//PerfectGasEOS(lambda, rho[i], vfrac[i], e[i], &pFinal);
void PairSphFluid::PerfectGasEOS(double gamma, double vol, double mass,
		double energy, double *pFinal__, double *c0) {

	/*
	 * perfect gas EOS is p = (gamma - 1) rho e
	 */

	*pFinal__ = (1.0 - gamma) * energy / vol;
	//printf("gamma = %f, vol%f, e=%g ==> p=%g\n", gamma, vol, energy, *pFinal__/1.0e-9);

	if (energy > 0.0) {
		*c0 = sqrt((gamma - 1.0) * energy / mass);
	} else {
		*c0 = 0.0;
	}

}

/* ----------------------------------------------------------------------
 Tait EOS

 input: (1) reference sound speed
 (2) equilibrium mass density
 (3) current mass density

 output:(1) pressure
 (2) current speed of sound
 ------------------------------------------------------------------------- */
void PairSphFluid::TaitEOS(const double exponent, const double c0_reference,
		const double rho_reference, const double rho_current, double &pressure,
		double &sound_speed) {

	double B = rho_reference * c0_reference * c0_reference / exponent;
	pressure = B * (pow(rho_current / rho_reference, exponent) - 1.0);
	sound_speed = c0_reference;
}

/* ----------------------------------------------------------------------
 compute pressure
 ------------------------------------------------------------------------- */
void PairSphFluid::ComputePressure() {
	double *radius = atom->radius;
	double *rho = atom->rho;
	double *vfrac = atom->vfrac;
	double *rmass = atom->rmass;
	double *e = atom->e;
	double **tlsph_stress = atom->tlsph_stress;
	double dt = update->dt;
	int *type = atom->type;
	double pFinal;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d D, W, elaStress, elaStressDot, viscStress, V, sigma_diag,
			artStressDiag;
	double mu_e, mu_s, eta, lambda1, lambda2, kinvisc, epsilon;
	double lambda, mu;
	Matrix3d Jaumann_rate, eye;
	SelfAdjointEigenSolver<Matrix3d> es;
	int flag;

	hMin = 1.0e22;
	eye.setIdentity();

	//printf("in assemble stress\n");

	for (i = 0; i < nlocal; i++) {
		stressTensor[i].setZero();
		itype = type[i];
		//printf("setflag: %d\n", setflag[itype][itype]);
		if (setflag[itype][itype] == 1) {

			/*
			 * initial elastic stress state:
			 */

			elaStress(0, 0) = tlsph_stress[i][0];
			elaStress(0, 1) = tlsph_stress[i][1];
			elaStress(0, 2) = tlsph_stress[i][2];
			elaStress(1, 1) = tlsph_stress[i][3];
			elaStress(1, 2) = tlsph_stress[i][4];
			elaStress(2, 2) = tlsph_stress[i][5];
			elaStress(1, 0) = elaStress(0, 1);
			elaStress(2, 0) = elaStress(0, 2);
			elaStress(2, 1) = elaStress(1, 2);

			//printf("eos = %d\n", eos[itype]);
			switch (eos[itype][itype]) {
			case PERFECT_GAS:
				PerfectGasEOS(C1[itype][itype], vfrac[i], rmass[i], e[i],
						&pFinal, &c0[i]);
				break;
			case TAIT:
				//      Tait exponent,    c0_reference,     rho_reference
				TaitEOS(C1[itype][itype], C2[itype][itype], C3[itype][itype],
						rho[i], pFinal, c0[i]);

				stressTensor[i].setZero();
				stressTensor[i](0, 0) = pFinal;
				stressTensor[i](1, 1) = pFinal;
				stressTensor[i](2, 2) = pFinal;

				D = 0.5 * (L[i] + L[i].transpose());
//				W = 0.5 * (L[i] - L[i].transpose());
//
//				lambda1 = 0.02e3;
//				lambda2 = 0.02e3;
//
//				kinvisc = 0.0;
//				eta = kinvisc * C3[itype][itype]; // C3 is reference rho;
//				mu_s = eta * lambda2 / lambda1;
//				mu_e = eta - mu_s;
//
//				// elastic stress rate
//				elaStressDot = (2. * mu_e / lambda1) * D
//						- (1. / lambda1) * elaStress + W * elaStress
//						- elaStress * W + D * elaStress + elaStress * D
//						- L[i] * elaStress;
//
//				elaStress += dt * elaStressDot;
//
//				// Newtonian part of stress
//				viscStress = mu_s * Deviator(D);

				// total stress
				//stressTensor[i] += elaStress + viscStress;
				stressTensor[i] += 2.0 * C4[itype][itype] * D;

				//printf("atom %d: p=%f, c0=%f, rho[i]=%g\n", i, pFinal, c0[i], rho[i]);

				break;
			case LINEAR_ELASTIC:
				lambda = C1[itype][itype] * C2[itype][itype]
						/ ((1.0 + C2[itype][itype])
								* (1.0 - 2.0 * C2[itype][itype]));
				mu = C1[itype][itype] / (2.0 * (1.0 + C2[itype][itype]));

				D = 0.5 * (L[i] + L[i].transpose());
				W = 0.5 * (L[i] - L[i].transpose());

				elaStressDot = (lambda + 2.0 * mu / 3.0) * D.trace() * eye
						+ 2.0 * mu * Deviator(D);

				Jaumann_rate = elaStressDot + W * elaStress
						+ elaStress * W.transpose();
				elaStress += dt * Jaumann_rate;
				stressTensor[i] = elaStress;

			case NONE:
				pFinal = 0.0;
				c0[i] = 1.0;
				break;
			default:
				error->one(FLERR, "unknown EOS.");
				break;
			}

			// artificial stress
			artStress[i].setZero();
//			epsilon = 0.5;
//
//			es.compute(stressTensor[i]);
//			V = es.eigenvectors();
//
//			// diagonalize stress matrix
//			sigma_diag = V.inverse() * stressTensor[i] * V;
//
//			flag = 0;
//
//			artStressDiag.setZero();
//			for (int dim = 0; dim < 3; dim++) {
//				if (sigma_diag(dim, dim) > 0.0) {
//					artStressDiag(dim, dim) = -10*epsilon * sigma_diag(dim, dim);
//				}
//			}
//			// undiagonalize artificial stress matrix
//			artStress[i] = V * artStressDiag * V.inverse();

			pressure[i] = pFinal;

			/*
			 * minimum kernel Radius
			 */

			hMin = MIN(radius[i], hMin);

			/*
			 * store updated stress
			 */

			tlsph_stress[i][0] = elaStress(0, 0);
			tlsph_stress[i][1] = elaStress(0, 1);
			tlsph_stress[i][2] = elaStress(0, 2);

			tlsph_stress[i][3] = elaStress(1, 1);
			tlsph_stress[i][4] = elaStress(1, 2);
			tlsph_stress[i][5] = elaStress(2, 2);

		}
	}

	//printf("stable timestep = %g\n", 0.1 * hMin * MaxBulkVelocity);
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSphFluid::allocate() {

	//printf("in allocate\n");

	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(C1, n + 1, n + 1, "pair:C1");
	memory->create(C2, n + 1, n + 1, "pair:C2");
	memory->create(C3, n + 1, n + 1, "pair:C3");
	memory->create(C4, n + 1, n + 1, "pair:C4");
	memory->create(Q1, n + 1, n + 1, "pair:Q1");
	memory->create(Q2, n + 1, n + 1, "pair:Q2");
	memory->create(eos, n + 1, n + 1, "pair:eosmodel");

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

void PairSphFluid::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSphFluid::coeff(int narg, char **arg) {
	//printf("in coeff\n");

	double C1_one, C2_one, C3_one, C4_one, q1_one, q2_one;
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	int eos_one;
	if (strcmp(arg[2], "perfectgas") == 0) {
		eos_one = PERFECT_GAS;
	} else if (strcmp(arg[2], "tait") == 0) {
		eos_one = TAIT;
	} else if (strcmp(arg[2], "linear_elastic") == 0) {
		eos_one = LINEAR_ELASTIC;
	} else if (strcmp(arg[2], "none") == 0) {
		eos_one = NONE;
	} else {
		//printf(arg[2]);
		error->all(FLERR, "unknown EOS model selected");
	}

	switch (eos_one) {
	case PERFECT_GAS:
		// we want gamma, q1, q2
		if (narg != 6)
			error->all(FLERR, "Incorrect number of args for EOS perfectgas");
		C1_one = atof(arg[3]);
		q1_one = atof(arg[4]);
		q2_one = atof(arg[5]);
		break;
	case TAIT:
		if (narg != 9)
			error->all(FLERR, "Incorrect number of args for EOS Tait");
		C1_one = atof(arg[3]); // Tait exponent
		C2_one = atof(arg[4]); // reference speed of sound
		C3_one = atof(arg[5]); // reference density
		C4_one = atof(arg[6]); // dynamic shear viscosity
		q1_one = atof(arg[7]); // linear term in artificial viscosiy
		q2_one = atof(arg[8]); // quadratic term in artificial viscosiy
		break;
	case LINEAR_ELASTIC:
		if (narg != 7)
			error->all(FLERR,
					"Incorrect number of args for linear elastic material model");
		C1_one = atof(arg[3]); // Young's modulus
		C2_one = atof(arg[4]); // Poisson ratio
		q1_one = atof(arg[5]); // linear term in artificial viscosiy
		q2_one = atof(arg[6]); // quadratic term in artificial viscosiy
		break;
	case NONE:
		if (narg != 3)
			error->all(FLERR, "Incorrect number of args for EOS none");
		break;
	default:
		error->one(FLERR, "unknown EOS.");
		break;
	}

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			eos[i][j] = eos_one;
			C1[i][j] = C1_one;
			C2[i][j] = C2_one;
			C3[i][j] = C3_one;
			C4[i][j] = C4_one;
			Q1[i][j] = q1_one;
			Q2[i][j] = q2_one;
			setflag[i][j] = 1;
			count++;

			if (comm->me == 0) {
				printf("setting sph/fluid itype=%d, jtype = %d\n", i, j);
			}
		}
	}

	if (count == 0)
		error->all(FLERR, "Incorrect args for pair coefficients");

	//printf("end of coeff\n");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSphFluid::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	Q1[j][i] = Q1[i][j];
	Q2[j][i] = Q2[i][j];
	C1[j][i] = C1[i][j];
	C2[j][i] = C2[i][j];
	C3[j][i] = C3[i][j];
	C4[j][i] = C4[i][j];
	eos[j][i] = eos[i][j];

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
	printf("cutoff for pair sph/fluid = %f\n", cutoff);
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairSphFluid::init_style() {
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
	//printf("end of init style\n");

}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairSphFluid::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairSphFluid::write_restart(FILE *fp) {
	int i, j;
	for (i = 1; i <= atom->ntypes; i++) {
		fwrite(&C1[i], sizeof(double), 1, fp);
		fwrite(&C2[i], sizeof(double), 1, fp);
		fwrite(&Q1[i], sizeof(double), 1, fp);
		for (j = i; j <= atom->ntypes; j++) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {

			}
		}
	}
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairSphFluid::read_restart(FILE *fp) {
	allocate();

	int i, j;
	int me = comm->me;
	for (i = 1; i <= atom->ntypes; i++) {
		if (me == 0) {
			fread(&C1[i], sizeof(double), 1, fp);
			fread(&C2[i], sizeof(double), 1, fp);
			fread(&Q1[i], sizeof(double), 1, fp);
		}

		MPI_Bcast(&C1[i], 1, MPI_DOUBLE, 0, world);
		MPI_Bcast(&C2[i], 1, MPI_DOUBLE, 0, world);
		MPI_Bcast(&Q1[i], 1, MPI_DOUBLE, 0, world);

		for (j = i; j <= atom->ntypes; j++) {
			if (me == 0) {
				fread(&setflag[i][j], sizeof(int), 1, fp);
			}
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);

			if (setflag[i][j]) {
				if (me == 0) {
					//fread(&Q1[i][j], sizeof(double), 1, fp);
				}
				//MPI_Bcast(&Q1[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairSphFluid::memory_usage() {

	//printf("in memory usage\n");

	return 11 * nmax * sizeof(double);

}

/* ---------------------------------------------------------------------- */

int PairSphFluid::pack_comm(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
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

		buf[m++] = artStress[j](0, 0);
		buf[m++] = artStress[j](1, 1);
		buf[m++] = artStress[j](2, 2);
		buf[m++] = artStress[j](0, 1);
		buf[m++] = artStress[j](0, 2);
		buf[m++] = artStress[j](1, 2);
	}
	return 15;
}

/* ---------------------------------------------------------------------- */

void PairSphFluid::unpack_comm(int n, int first, double *buf) {
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

		artStress[i](0, 0) = buf[m++];
		artStress[i](1, 1) = buf[m++];
		artStress[i](2, 2) = buf[m++];
		artStress[i](0, 1) = buf[m++];
		artStress[i](0, 2) = buf[m++];
		artStress[i](1, 2) = buf[m++];
		artStress[i](1, 0) = artStress[i](0, 1);
		artStress[i](2, 0) = artStress[i](0, 2);
		artStress[i](2, 1) = artStress[i](1, 2);
	}
}

/*
 * compute a normalized smoothing kernel and its derivative
 */

void PairSphFluid::kernel_and_derivative(const double h, const double r,
		double &wf, double &wfd) {

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
	wfd = -3.0e0 * hr * hr / n;
	wf = -0.333333333333e0 * hr * wfd;

	/*
	 * cubic spline - to do
	 */

}

/*
 * Pseudo-inverse via SVD
 */

Matrix3d PairSphFluid::pseudo_inverse_SVD(Matrix3d M) {

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

void *PairSphFluid::extract(const char *str, int &i) {
	//printf("in extract\n");
	if (strcmp(str, "sphFluid_smoothVel_ptr") == 0) {
		return (void *) smoothVel;
	} else if (strcmp(str, "fluid_hMin_ptr") == 0) {
		//printf("min h exported is %f\n", hMin);
		return (void *) &hMin;
	} else if (strcmp(str, "fluid_stressTensor_ptr") == 0) {
		return (void *) stressTensor;
	} else if (strcmp(str, "fluid_velocityGradient_ptr") == 0) {
		return (void *) L;
	}

	return NULL;
}

/*
 * deviator of a tensor
 */
Matrix3d PairSphFluid::Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}
