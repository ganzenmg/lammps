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
#include "smd_kernels.h"

using namespace SMD_Kernels;
using namespace std;
using namespace LAMMPS_NS;
using namespace SMD_Math;

#include <Eigen/SVD>
#include <Eigen/Eigen>
using namespace Eigen;

#define ARTIFICIAL_STRESS false

PairULSPH::PairULSPH(LAMMPS *lmp) :
		Pair(lmp) {

	Q1 = NULL;
	eos = strength = NULL;
	c0_type = NULL;
	pressure = NULL;
	c0 = NULL;
	Lookup = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = K = NULL;
	artStress = NULL;
	delete_flag = NULL;
	shepardWeight = NULL;
	smoothVel = NULL;
	numNeighs = NULL;
	F = NULL;
	gradient_correction = NULL;
	hourglass_amplitude = NULL;

	artificial_stress_flag = false; // turn off artificial stress correction by default
	velocity_gradient_required = false; // turn off computation of velocity gradient by default
	gradient_correction_possible = NULL;

	comm_forward = 26; // this pair style communicates 20 doubles to ghost atoms
	updateFlag = 0;
}

/* ---------------------------------------------------------------------- */

PairULSPH::~PairULSPH() {
	if (allocated) {
		//printf("... deallocating\n");
		memory->destroy(Q1);
		memory->destroy(rho0);
		memory->destroy(eos);
		memory->destroy(strength);
		memory->destroy(c0_type);
		memory->destroy(gradient_correction);
		memory->destroy(hourglass_amplitude);
		memory->destroy(Lookup);

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
		delete[] F;
		delete[] gradient_correction_possible;

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
	double **atom_data9 = atom->tlsph_fold;
	double *radius = atom->radius;
	double **x = atom->x;
	double **x0 = atom->x0;
	double **v = atom->vest;
	double *rmass = atom->rmass;
	double *rho = atom->rho;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, itype, jtype, j, idim;
	double wfd, h, irad, r, rSq, wf, ivol, jvol;
	bool gradient_correction_flag;
	Vector3d dx, dv, g, du;
	Matrix3d Ktmp, Ltmp, Ftmp, K3di, D;
	Vector3d xi, xj, vi, vj, x0i, x0j, dx0;
	Matrix2d K2di, K2d;

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype]) {

			if (gradient_correction[itype]) {
				K[i].setZero();
			} else {
				K[i].setIdentity();
			}

			delete_flag[i] = 0.0;
			L[i].setZero();
			numNeighs[i] = 0;
			F[i].setZero();
			gradient_correction_possible[i] = false;
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
		ivol = rmass[i] / rho[i];

		gradient_correction_flag = gradient_correction[itype]; // perform kernel gradient correction for this particle type?

		// initialize Eigen data structures from LAMMPS data structures
		for (idim = 0; idim < 3; idim++) {
			x0i(idim) = x0[i][idim];
			xi(idim) = x[i][idim];
			vi(idim) = v[i][idim];
		}

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			for (idim = 0; idim < 3; idim++) {
				x0j(idim) = x0[j][idim];
				xj(idim) = x[j][idim];
				vj(idim) = v[j][idim];
			}

			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = irad + radius[j];
			if (rSq < h * h) {

				r = sqrt(rSq);
				jtype = type[j];
				jvol = rmass[j] / rho[j];

//				if (itype != jtype) {
//					error->one(FLERR, "type of particle i does not equal type of particle j. Something is very wrong.");
//				}

				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;
				dx0 = x0j - x0i;

				// kernel and derivative
				kernel_and_derivative(h, r, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r) * dx;

				/* build correction matrix for kernel derivatives */
				if (gradient_correction_flag) {
					Ktmp = -g * dx.transpose();
					K[i] += jvol * Ktmp;
				}

				// velocity gradient L
				Ltmp = dv * g.transpose();
				L[i] += jvol * Ltmp;

				// deformation gradient F in Eulerian frame
				du = dx - dx0;
				Ftmp = du * g.transpose();
				F[i] += jvol * Ftmp;

				numNeighs[i] += 1;

				if (j < nlocal) {

					if (gradient_correction_flag) {
						K[j] += ivol * Ktmp;
					}

					L[j] += ivol * Ltmp;
					F[j] += ivol * Ftmp;
					numNeighs[j] += 1;
				}
			} // end if check distance
		} // end loop over j
	} // end loop over i

	/*
	 * invert shape matrix and compute corrected quantities
	 */

	bool Shape_Matrix_Inversion_Success = false;
	SelfAdjointEigenSolver<Matrix2d> es;
	SelfAdjointEigenSolver<Matrix3d> es3d;

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype]) {
			if (gradient_correction[itype]) {
				Shape_Matrix_Inversion_Success = false;

				if (domain->dimension == 2) {
					Matrix2d K2d;
					K2d(0, 0) = K[i](0, 0);
					K2d(0, 1) = K[i](0, 1);
					K2d(1, 0) = K[i](1, 0);
					K2d(1, 1) = K[i](1, 1);

					if (fabs(K2d.determinant()) > 1.0e-16) {
						K2di = K2d.inverse();
						// check if inverse of K2d is reasonable
						es.compute(K2di);
						if ((fabs(es.eigenvalues()(0)) > 0.1) && (fabs(es.eigenvalues()(1)) > 0.1)) {
							Shape_Matrix_Inversion_Success = true;
						}
					}

					if (Shape_Matrix_Inversion_Success) {
						K[i].setZero();
						K[i](0, 0) = K2di(0, 0);
						K[i](0, 1) = K2di(0, 1);
						K[i](1, 0) = K2di(1, 0);
						K[i](1, 1) = K2di(1, 1);
						K[i](2, 2) = 1.0;
						gradient_correction_possible[i] = true;
					} else {
						cout << "we have a problem with K; this is K" << endl << K2d << endl;
						K[i].setIdentity();
					}

				} else { // 3d
					if (fabs(K[i].determinant()) > 1.0e-16) {
						// check if inverse of K is reasonable
						es3d.compute(K[i]);
						if ((fabs(es3d.eigenvalues()(0)) > 1.0e-3) && (fabs(es3d.eigenvalues()(1)) > 1.0e-3)
								&& (fabs(es3d.eigenvalues()(2)) > 1.0e-3)) {
							Shape_Matrix_Inversion_Success = true;
						} else {
							cout << endl << "we have a problem with K due to small eigenvalues; this is K" << endl << K[i] << endl;
							cout << "these are the eigenvalues of K " << es3d.eigenvalues() << endl;
						}
					} else {
						cout << endl << "we have a problem with K due to a small determinant; this is K" << endl << K[i] << endl;
					}

					if (Shape_Matrix_Inversion_Success) {
						K3di = K[i].inverse();
						K[i] = K3di;
						gradient_correction_possible[i] = true;
					} else {
						K[i].setIdentity();

					}
				} // end if 3d

//			if (i == 20) {
//				cout << endl << "this is L before mult with K" << endl << L[i] << endl << "-----------------------" << endl;
//				cout << "this is K" << endl << K[i] << endl;
//			}

				L[i] *= K[i];
				F[i] *= K[i];

//			if (i == 20) {
//				cout << "this is L after mult with K" << endl << L[i] << endl << "-----------------------" << endl;
//			}

				/*
				 * accumulate strain increments
				 * we abuse the atom array "tlsph_fold" for this purpose, which 3was originally designed to hold the deformation gradient.
				 */
				D = update->dt * 0.5 * (L[i] + L[i].transpose());
				atom_data9[i][0] += D(0, 0); // xx
				atom_data9[i][1] += D(1, 1); // yy
				atom_data9[i][2] += D(2, 2); // zz
				atom_data9[i][3] += D(0, 1); // xy
				atom_data9[i][4] += D(0, 2); // xz
				atom_data9[i][5] += D(1, 2); // yz

			} // end if (gradient_correction[itype]) {
		} // end if (setflag[itype][itype])
	} // end loop over i = 0 to nlocal

}

/* ---------------------------------------------------------------------- */

void PairULSPH::compute(int eflag, int vflag) {
	double **x = atom->x;
	double **x0 = atom->x0;
	double **v = atom->vest;
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	double **f = atom->f;
	double *de = atom->de;
	double *drho = atom->drho;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *plastic_strain = atom->eff_plastic_strain;
	double **atom_data9 = atom->tlsph_fold;

	double *rho = atom->rho;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, jtype, iDim, inum;
	double r, wf, wfd, h, rSq, r0Sq, ivol, jvol;
	double mu_ij, c_ij, rho_ij;
	double delVdotDelR, visc_magnitude, deltaE;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	Vector3d fi, fj, dx, dv, f_stress, g, vinti, vintj, dvint;
	Vector3d xi, xj, vi, vj, f_visc, sumForces, f_stress_new;
	Vector3d gamma, f_hg, dx0, du_est, du;
	double gamma_dot_dx, hg_mag;

	double ini_dist, weight;
	double *contact_radius = atom->contact_radius;
	Matrix3d S, D, V;
	int k;
	SelfAdjointEigenSolver<Matrix3d> es;

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
		delete[] F;
		F = new Matrix3d[nmax];
		delete[] gradient_correction_possible;
		gradient_correction_possible = new bool[nmax];
	}

// zero accumulators
	for (i = 0; i < nlocal; i++) {
		shepardWeight[i] = 0.0;
		smoothVel[i].setZero();
		numNeighs[i] = 0;
	}

	/*
	 * if this is the very first step, zero the array which holds the accumulated strain
	 */
	if (update->ntimestep == 0) {
		for (i = 0; i < nlocal; i++) {
			itype = type[i];
			if (setflag[itype][itype]) {
				for (j = 0; j < 9; j++) {
					atom_data9[i][j] = 0.0;
				}
			}
		}
	}

	PairULSPH::PreCompute(); // get velocity gradient
	PairULSPH::ComputePressure();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);

	updateFlag = 0;

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

			//xj << x[j][0], x[j][1], x[j][2];
			xj(0) = x[j][0];
			xj(1) = x[j][1];
			xj(2) = x[j][2];

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

				// symmetric stress
				S = stressTensor[i] + stressTensor[j];

				/*
				 * artificial stress to control tensile instability
				 */
//				if (true) { //(artificial_stress_flag == true) {
//					ini_dist = contact_radius[i] + contact_radius[j];
//					weight = Kernel_Cubic_Spline(r, h) / Kernel_Cubic_Spline(ini_dist, h);
//					weight = pow(weight, 4.0);
//
//					es.compute(S);
//					D = es.eigenvalues().asDiagonal();
//					for (k = 0; k < 3; k++) {
//						if (D(k, k) > 0.0) {
//							D(k, k) -= weight * 1.0 * D(k, k);
//						}
//					}
//					V = es.eigenvectors();
//					S = V * D * V.inverse();
//				}
				/*
				 * force -- the classical SPH way
				 */

				//f_stress = ivol * jvol * (stressTensor[i] * K[i] + stressTensor[j] * K[j]) * g;
				f_stress = ivol * jvol * S * g;

				/*
				 * artificial viscosity -- alpha is dimensionless
				 * MonaghanBalsara form of the artificial viscosity
				 */

				delVdotDelR = dx.dot(dv) / (r + 0.1 * h); // project relative velocity onto unit particle distance vector [m/s]
				c_ij = 0.5 * (c0[i] + c0[j]);

				LimitDoubleMagnitude(delVdotDelR, 1.1 * c_ij);

				mu_ij = h * delVdotDelR / (r + 0.1 * h); // units: [m * m/s / m = m/s]
				rho_ij = 0.5 * (rho[i] + rho[j]);
				visc_magnitude = 0.5 * (Q1[itype] + Q1[jtype]) * c_ij * mu_ij / rho_ij;
				f_visc = -rmass[i] * rmass[j] * visc_magnitude * g;

				/*
				 * hourglass treatment
				 */

				dx0(0) = x0[j][0] - x0[i][0];
				dx0(1) = x0[j][1] - x0[i][1];
				dx0(2) = x0[j][2] - x0[i][2];
				r0Sq = dx0.squaredNorm();
				du = dx - dx0;

				if ((hourglass_amplitude[itype] > 0.0) && (gradient_correction_possible[i] == true)
						&& (gradient_correction_possible[j] == true)) {

					du_est = 0.5 * (F[i] + F[j]) * dx;
					gamma = du_est - du;

//					if (du.norm() > 0.01) {
//						printf("du_est is %f %f %f\n", du_est(0), du_est(1), du_est(2));
//						printf("du actual is %f %f %f\n", du(0), du(1), du(2));
//						printf("gamma is %f %f %f\n", gamma(0), gamma(1), gamma(2));
//						printf("--------------\n");
//					}

					gamma_dot_dx = gamma.dot(dx) / (r + 0.1 * h); // project hourglass error vector onto normalized pair distance vector

					if (rSq - r0Sq < 0) { // compression mode, we limit hourglass correction force
						LimitDoubleMagnitude(gamma_dot_dx, 0.005 * r);
					} else {
						LimitDoubleMagnitude(gamma_dot_dx, 0.5 * r);
					}

					if (MAX(plastic_strain[i], plastic_strain[j]) > 1.0e-4) {
						hg_mag = Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * Lookup[YIELD_STRENGTH][itype] * gamma_dot_dx
								/ (rSq + 0.01 * h * h); // hg_mag has dimensions [Pa m^(-1)]
					} else {
						hg_mag = Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * Lookup[BULK_MODULUS][itype] * gamma_dot_dx
								/ (rSq + 0.01 * h * h); // hg_mag has dimensions [Pa m^(-1)]
					}

					hg_mag *= -ivol * jvol * wf; // hg_mag has dimensions [J*m^(-1)] = [N]
					f_hg = (hg_mag / (r + 0.01 * h)) * dx;
				} else {
					f_hg.setZero();
				}

				sumForces = f_stress + f_visc; // + f_hg;

				// energy rate -- project velocity onto force vector
				deltaE = 0.5 * sumForces.dot(dv);

				// change in mass density
				drho[i] -= rho[i] * jvol * wfd * delVdotDelR;

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
					drho[j] -= rho[j] * ivol * wfd * delVdotDelR;

					shepardWeight[j] += ivol * wf;
					smoothVel[j] -= ivol * wf * dvint;
					numNeighs[j] += 1;
				}

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
				}

				// check if a particle  has moved too much w.r.t another particle
				if (du.norm() > 10.5 * dx0.norm()) {
					updateFlag = 1;
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
 compute pressure
 ------------------------------------------------------------------------- */
void PairULSPH::ComputePressure() {
	double *radius = atom->radius;
	double *rho = atom->rho;
	double *rmass = atom->rmass;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double **tlsph_stress = atom->tlsph_stress;
	double *e = atom->e;
	int *type = atom->type;
	double pFinal;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d D, Ddev, W, V, sigma_diag;
	Matrix3d eye, stressRate, StressRateDevJaumann;
	Matrix3d sigmaInitial_dev, d_dev, sigmaFinal_dev, stressRateDev, oldStressDeviator, newStressDeviator;
	double plastic_strain_increment, yieldStress;
	double dt = update->dt;
	double vol, newPressure;

	dtCFL = 1.0e22;
	eye.setIdentity();
	newStressDeviator.setZero();
	newPressure = 0.0;

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			stressTensor[i].setZero();
			vol = rmass[i] / rho[i];

			switch (eos[itype]) {
			default:
				error->one(FLERR, "unknown EOS.");
				break;
			case NONE:
				pFinal = 0.0;
				c0[i] = 1.0;
				break;
			case EOS_TAIT:
				TaitEOS_density(Lookup[EOS_TAIT_EXPONENT][itype], Lookup[REFERENCE_SOUNDSPEED][itype],
						Lookup[REFERENCE_DENSITY][itype], rho[i], newPressure, c0[i]);
				//printf("new pressure =%f\n", newPressure);

				break;
			case EOS_PERFECT_GAS:

				PerfectGasEOS(Lookup[EOS_PERFECT_GAS_GAMMA][itype], vol, rmass[i], e[i], newPressure, c0[i]);
				break;
			}

			/*
			 * ******************************* STRENGTH MODELS ************************************************
			 */

			if (strength[itype] != NONE) {

				/*
				 * initial stress state: given by the unrotateted Cauchy stress.
				 * Assemble Eigen 3d matrix from stored stress state
				 */
				oldStressDeviator(0, 0) = tlsph_stress[i][0];
				oldStressDeviator(0, 1) = tlsph_stress[i][1];
				oldStressDeviator(0, 2) = tlsph_stress[i][2];
				oldStressDeviator(1, 1) = tlsph_stress[i][3];
				oldStressDeviator(1, 2) = tlsph_stress[i][4];
				oldStressDeviator(2, 2) = tlsph_stress[i][5];
				oldStressDeviator(1, 0) = oldStressDeviator(0, 1);
				oldStressDeviator(2, 0) = oldStressDeviator(0, 2);
				oldStressDeviator(2, 1) = oldStressDeviator(1, 2);

				D = 0.5 * (L[i] + L[i].transpose());
				W = 0.5 * (L[i] - L[i].transpose()); // spin tensor:: need this for Jaumann rate
				double d_iso = D.trace();
				d_dev = Deviator(D);

				switch (strength[itype]) {
				default:
					error->one(FLERR, "unknown strength model.");
					break;
				case STRENGTH_LINEAR:
					stressRateDev = -2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
					//cout << "stress rate deviator is " << endl << stressRateDev << endl;
					break;
				case STRENGTH_VISCOSITY_NEWTON:
					oldStressDeviator.setZero();
					stressRateDev = 2.0 * Lookup[VISCOSITY_MU][itype] * d_dev / dt;
					break;
				}

				//double m = effective_longitudinal_modulus(itype, dt, d_iso, p_rate, d_dev, stressRate_dev, damage);

				StressRateDevJaumann = stressRateDev - W * oldStressDeviator + oldStressDeviator * W;
				newStressDeviator = oldStressDeviator + dt * StressRateDevJaumann;

				tlsph_stress[i][0] = newStressDeviator(0, 0);
				tlsph_stress[i][1] = newStressDeviator(0, 1);
				tlsph_stress[i][2] = newStressDeviator(0, 2);
				tlsph_stress[i][3] = newStressDeviator(1, 1);
				tlsph_stress[i][4] = newStressDeviator(1, 2);
				tlsph_stress[i][5] = newStressDeviator(2, 2);
			} // end if (strength[itype] != NONE)

			/*
			 * assemble stress Tensor from pressure and deviatoric parts
			 */

			stressTensor[i] = -newPressure * eye + newStressDeviator;

			/*
			 * kernel gradient correction
			 */
			if (gradient_correction[itype]) {
				stressTensor[i] = stressTensor[i] * K[i];
			}
			pressure[i] = stressTensor[i].trace() / 3.0;

			/*
			 * stable timestep based on speed-of-sound
			 */

			dtCFL = MIN(radius[i] / c0[i], dtCFL);

		} // end if (setflag[itype][itype] == 1)
	} // end loop over nlocal

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
	memory->create(c0_type, n + 1, "pair:c0_type");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(strength, n + 1, "pair:strengthmodel");
	memory->create(gradient_correction, n + 1, "pair:gradient_correction");
	memory->create(hourglass_amplitude, n + 1, "pair:hourglass_correction");
	memory->create(Lookup, MAX_KEY_VALUE, n + 1, "pair:LookupTable");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq");		// always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];

	for (int i = 1; i <= n; i++) {
		gradient_correction[i] = false;
	}

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

		if (iNextKwd - ioffset != 5 + 1) {
			sprintf(str, "expected 5 arguments following *COMMON but got %d\n", iNextKwd - ioffset - 1);
			error->all(FLERR, str);
		}

		Lookup[REFERENCE_DENSITY][itype] = force->numeric(FLERR, arg[ioffset + 1]);
		Lookup[REFERENCE_SOUNDSPEED][itype] = force->numeric(FLERR, arg[ioffset + 2]);
		Q1[itype] = force->numeric(FLERR, arg[ioffset + 3]);
		Lookup[HEAT_CAPACITY][itype] = force->numeric(FLERR, arg[ioffset + 4]);
		Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] = force->numeric(FLERR, arg[ioffset + 5]);

		Lookup[BULK_MODULUS][itype] = Lookup[REFERENCE_SOUNDSPEED][itype] * Lookup[REFERENCE_SOUNDSPEED][itype]
				* Lookup[REFERENCE_DENSITY][itype];

		if (comm->me == 0) {
			printf("\n material unspecific properties for SMD/ULSPH definition of particle type %d:\n", itype);
			printf("%40s : %g\n", "reference density", Lookup[REFERENCE_DENSITY][itype]);
			printf("%40s : %g\n", "reference speed of sound", Lookup[REFERENCE_SOUNDSPEED][itype]);
			printf("%40s : %g\n", "linear viscosity coefficient", Q1[itype]);
			printf("%40s : %g\n", "heat capacity [energy / (mass * temperature)]", Lookup[HEAT_CAPACITY][itype]);
			printf("%40s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
			printf("%40s : %g\n", "hourglass control amplitude", Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]);
		}

		/*
		 * read following material cards
		 */

		if (comm->me == 0) {
			printf("next kwd is %s\n", arg[iNextKwd]);
		}
		eos[itype] = strength[itype] = NONE;

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
				//printf("reading *EOS_TAIT\n");

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

				Lookup[EOS_TAIT_EXPONENT][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf("\n%60s\n", "Tait EOS");
					printf("%60s : %g\n", "Exponent", Lookup[EOS_TAIT_EXPONENT][itype]);
				}
			} // end Tait EOS

			else if (strcmp(arg[ioffset], "*EOS_PERFECT_GAS") == 0) {

				/*
				 * Perfect Gas EOS
				 */

				eos[itype] = EOS_PERFECT_GAS;
				//printf("reading *EOS_PERFECT_GAS\n");

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

				Lookup[EOS_PERFECT_GAS_GAMMA][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf("\n%60s\n", "Perfect Gas EOS");
					printf("%60s : %g\n", "Heat Capacity Ratio Gamma", Lookup[EOS_PERFECT_GAS_GAMMA][itype]);
				}
			} // end Perfect Gas EOS
			else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR_PLASTIC") == 0) {

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				strength[itype] = STRENGTH_LINEAR_PLASTIC;
				artificial_stress_flag = true;
				velocity_gradient_required = true;
				//printf("reading *LINEAR_PLASTIC\n");

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
					sprintf(str, "no *KEYWORD terminates *STRENGTH_LINEAR_PLASTIC");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 4 + 1) {
					sprintf(str, "expected 4 arguments following *STRENGTH_LINEAR_PLASTIC but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[POISSON_RATIO][itype] = force->numeric(FLERR, arg[ioffset + 2]);
				Lookup[YIELD_STRENGTH][itype] = force->numeric(FLERR, arg[ioffset + 3]);
				Lookup[HARDENING_PARAMETER][itype] = force->numeric(FLERR, arg[ioffset + 4]);

				Lookup[LAME_LAMBDA][itype] = Lookup[YOUNGS_MODULUS][itype] * Lookup[POISSON_RATIO][itype]
						/ ((1.0 + Lookup[POISSON_RATIO][itype] * (1.0 - 2.0 * Lookup[POISSON_RATIO][itype])));
				Lookup[SHEAR_MODULUS][itype] = Lookup[YOUNGS_MODULUS][itype] / (2.0 * (1.0 + Lookup[POISSON_RATIO][itype]));
				Lookup[M_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype];

				if (comm->me == 0) {
					printf("\n%60s\n", "linear elastic / ideal plastic material mode");
					printf("%60s : %g\n", "Youngs modulus", Lookup[YOUNGS_MODULUS][itype]);
					printf("%60s : %g\n", "poisson_ratio", Lookup[POISSON_RATIO][itype]);
					printf("%60s : %g\n", "yield_strength", Lookup[YIELD_STRENGTH][itype]);
					printf("%60s : %g\n", "constant hardening parameter", Lookup[HARDENING_PARAMETER][itype]);
					printf("%60s : %g\n", "Lame constant lambda", Lookup[LAME_LAMBDA][itype]);
					printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
					printf("%60s : %g\n", "p-wave modulus", Lookup[M_MODULUS][itype]);
				}
			} // end *STRENGTH_LINEAR_PLASTIC
			else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR") == 0) {

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				strength[itype] = STRENGTH_LINEAR;
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
					sprintf(str, "no *KEYWORD terminates *STRENGTH_LINEAR");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *STRENGTH_LINEAR but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[SHEAR_MODULUS][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf("\n%60s\n", "linear elastic strength model");
					printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
				}
			} // end *STRENGTH_LINEAR
			else if (strcmp(arg[ioffset], "*STRENGTH_VISCOSITY_NEWTON") == 0) {

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				strength[itype] = STRENGTH_VISCOSITY_NEWTON;
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
					sprintf(str, "no *KEYWORD terminates *STRENGTH_VISCOSITY_NEWTON");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *STRENGTH_VISCOSITY_NEWTON but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[VISCOSITY_MU][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf("\n%60s\n", "Newton viscosity model");
					printf("%60s : %g\n", "viscosity mu", Lookup[VISCOSITY_MU][itype]);
				}
			} // end *STRENGTH_VISCOSITY_NEWTON

			else if (strcmp(arg[ioffset], "*GRADIENT_CORRECTION") == 0) {

				/*
				 * Gradient correction card
				 */

				//printf("reading *GRADIENT_CORRECTION\n");
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
					sprintf(str, "no *KEYWORD terminates *GRADIENT_CORRECTION");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 0) {
					sprintf(str, "expected 0 arguments following *GRADIENT_CORRECTION but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				gradient_correction[itype] = true;

				if (comm->me == 0) {
					printf("\n%60s\n", "Kernel gradient correction is active");
				}
			} // end gradient correction material card

			else {
				sprintf(str, "unknown *KEYWORD: %s", arg[ioffset]);
				error->all(FLERR, str);
			}

		}

		/*
		 * copy data which is looked up in inner pairwise loops from slow maps to fast arrays
		 */

		rho0[itype] = Lookup[REFERENCE_DENSITY][itype];
		c0_type[itype] = Lookup[REFERENCE_SOUNDSPEED][itype];
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
	double *eff_plastic_strain = atom->eff_plastic_strain;
	int i, j, m;

//printf("packing comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = pressure[j];
		buf[m++] = rho[j];
		buf[m++] = c0[j]; //3

		buf[m++] = stressTensor[j](0, 0);
		buf[m++] = stressTensor[j](1, 1);
		buf[m++] = stressTensor[j](2, 2);
		buf[m++] = stressTensor[j](0, 1);
		buf[m++] = stressTensor[j](0, 2);
		buf[m++] = stressTensor[j](1, 2); //9

		buf[m++] = F[j](0, 0); // F is not symmetric
		buf[m++] = F[j](0, 1);
		buf[m++] = F[j](0, 2);
		buf[m++] = F[j](1, 0);
		buf[m++] = F[j](1, 1);
		buf[m++] = F[j](1, 2);
		buf[m++] = F[j](2, 0);
		buf[m++] = F[j](2, 1);
		buf[m++] = F[j](2, 2); // 9 + 9 = 18

		buf[m++] = K[j](0, 0);
		buf[m++] = K[j](1, 1);
		buf[m++] = K[j](2, 2);
		buf[m++] = K[j](0, 1);
		buf[m++] = K[j](0, 2);
		buf[m++] = K[j](1, 2); //9

		buf[m++] = static_cast<double>(gradient_correction_possible[j]);
		buf[m++] = eff_plastic_strain[j];
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairULSPH::unpack_forward_comm(int n, int first, double *buf) {
	double *rho = atom->rho;
	double *eff_plastic_strain = atom->eff_plastic_strain;
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

		F[i](0, 0) = buf[m++];
		F[i](0, 1) = buf[m++];
		F[i](0, 2) = buf[m++];
		F[i](1, 0) = buf[m++];
		F[i](1, 1) = buf[m++];
		F[i](1, 2) = buf[m++];
		F[i](2, 0) = buf[m++];
		F[i](2, 1) = buf[m++];
		F[i](2, 2) = buf[m++];

		K[i](0, 0) = buf[m++];
		K[i](1, 1) = buf[m++];
		K[i](2, 2) = buf[m++];
		K[i](0, 1) = buf[m++];
		K[i](0, 2) = buf[m++];
		K[i](1, 2) = buf[m++];
		K[i](1, 0) = K[i](0, 1);
		K[i](2, 0) = K[i](0, 2);
		K[i](2, 1) = K[i](1, 2);

		gradient_correction_possible[i] = static_cast<bool>(buf[m++]);
		eff_plastic_strain[i] = buf[m++];
	}
}

/*
 * compute a normalized smoothing kernel and its derivative
 */

void PairULSPH::kernel_and_derivative(const double h, const double r, double &wf, double &wfd) {
	double hr = h - r;

	/*
	 * Spiky kernel
	 */

	if (domain->dimension == 2) {
		double h5 = h * h * h * h * h;
		wf = 3.183098861e0 * hr * hr * hr / h5;
		wfd = -9.549296583 * hr * hr / h5;

	} else {
		double h6 = h * h * h * h * h * h;
		wf = 4.774648292 * hr * hr * hr / h6;
		wfd = -14.32394487 * hr * hr / h6;
	}
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
	} else if (strcmp(str, "smd/ulsph/updateFlag_ptr") == 0) {
		return (void *) &updateFlag;
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
		sprintf(msg, "failed to lookup indentifier [%s] for particle type %d", str.c_str(), itype);
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

/* ----------------------------------------------------------------------
 compute effective P-wave speed
 determined by longitudinal modulus
 ------------------------------------------------------------------------- */

double PairULSPH::effective_longitudinal_modulus(int itype, double dt, double d_iso, double p_rate, Matrix3d d_dev,
		Matrix3d sigma_dev_rate, double damage) {
	double K3; // 3 times the effective bulk modulus, see Pronto 2d eqn 3.4.6
	double mu2; // 2 times the effective shear modulus, see Pronto 2d eq. 3.4.7
	double shear_rate_sq;
	double M; //effective longitudinal modulus
	double M0; // initial longitudinal modulus

	M0 = Lookup[M_MODULUS][itype];

	//printf("hury\n");

	if (dt * d_iso > 1.0e-6) {
		K3 = 3.0 * p_rate / d_iso;
	} else {
		K3 = 3.0 * M0;
	}

	if (domain->dimension == 3) {
		mu2 = sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16) + sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)
				+ sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16);
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

	//printf("initial M=%f, current M=%f\n", M0, M);

	/*
	 * damaged particles potentially have a very high dilatational modulus, even though damage degradation scales down the
	 * effective stress. we simply use the initial modulus for damaged particles.
	 */

	if (damage > 0.99) {
		M = M0; //
	}

	return M;

}
