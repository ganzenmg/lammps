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

#include "pair_pdgcg_shells.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
//#include "error.h"
//#include "float.h"
//#include "memory.h"
#include <cstdio>
#include <iostream>

#include <Eigen/Eigen>
#include "../../lib/sph2/Eigen/src/Core/CommaInitializer.h"
#include "../../lib/sph2/Eigen/src/Core/DenseBase.h"
#include "../../lib/sph2/Eigen/src/Core/Matrix.h"
#include "../../lib/sph2/Eigen/src/Core/util/ForwardDeclarations.h"
#include "../atom.h"
#include "../comm.h"
#include "../domain.h"
#include "../error.h"
#include "../force.h"
#include "../lmptype.h"
#include "../memory.h"
#include "../modify.h"
#include "../neigh_list.h"
#include "../neigh_request.h"
#include "../neighbor.h"
#include "../pointers.h"
#include "../STUBS/mpi.h"
//#include "fix.h"
#include "fix_pdgcg_shells_neigh.h"
//#include "update.h"

using namespace LAMMPS_NS;
using namespace Eigen;
using namespace std;

/* ---------------------------------------------------------------------- */

PairPDGCGShells::PairPDGCGShells(LAMMPS *lmp) :
		Pair(lmp) {
	for (int i = 0; i < 6; i++)
		virial[i] = 0.0;
	no_virial_fdotr_compute = 1;

	ifix_peri = -1;

	bulkmodulus = kbend = NULL;
	smax = syield = NULL;
	G0 = NULL;
	alpha = NULL;

	nBroken = 0;
	ncall = 0;
}

/* ---------------------------------------------------------------------- */

PairPDGCGShells::~PairPDGCGShells() {
	if (ifix_peri >= 0)
		modify->delete_fix("PDGCG_SHELLS_NEIGH");

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(bulkmodulus);
		memory->destroy(kbend);
		memory->destroy(smax);
		memory->destroy(syield);
		memory->destroy(G0);
		memory->destroy(alpha);
		memory->destroy(cutsq);
	}
}

/* ---------------------------------------------------------------------- */

void PairPDGCGShells::compute(int eflag, int vflag) {
	int i, j, ii, jj, inum, jnum, itype, jtype;
	double xtmp, ytmp, ztmp, delx, dely, delz;
	double rsq, r, dr, evdwl, fpair, fbond;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double delta, stretch, ivol, jvol;
	double vxtmp, vytmp, vztmp, delvelx, delvely, delvelz, delVdotDelR, fvisc, rcut, c0;
	double c;
	double r0cut, delx0, dely0, delz0;
	double r_geom, radius_factor;
	// ---------------------------------------------------------------------------------
	double **f = atom->f;
	double **x = atom->x;
	double **x0 = atom->x0;
	int *type = atom->type;
	double *rmass = atom->rmass;
	double *e = atom->e;
	double **v = atom->v;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double *radiusSR = atom->contact_radius;
	int *molecule = atom->molecule;
	int nlocal = atom->nlocal;
	int newton_pair = force->newton_pair;
	double **r0 = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->r0;
	double **plastic_stretch = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->r1;
	tagint **partner = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->partner;
	int *npartner = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->npartner;
	tagint ***trianglePairs = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->trianglePairs;
	int *nTrianglePairs = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->nTrianglePairs;
	double ***trianglePairAngle0 = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->trianglePairCoeffs;
	double *vinter = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->vinter;
	tagint *tag = atom->tag;

	evdwl = 0.0;
	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	ncall += 1;
	if (ncall < 2) {
		// return on first timestep because no bonds are defined.
		return;
	}

	/* ----------------------- PERIDYNAMIC SHORT RANGE FORCES --------------------- */

	// neighbor list variables -- note that we use a granular neighbor list
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	if (false) {

		for (ii = 0; ii < inum; ii++) {
			i = ilist[ii];
			xtmp = x[i][0];
			ytmp = x[i][1];
			ztmp = x[i][2];
			vxtmp = v[i][0];
			vytmp = v[i][1];
			vztmp = v[i][2];
			itype = type[i];
			ivol = vfrac[i];
			jlist = firstneigh[i];
			jnum = numneigh[i];

			for (jj = 0; jj < jnum; jj++) {
				j = jlist[jj];
				j &= NEIGHMASK;

				jtype = type[j];
				delx = xtmp - x[j][0];
				dely = ytmp - x[j][1];
				delz = ztmp - x[j][2];
				rsq = delx * delx + dely * dely + delz * delz;

				/* We only let this pair of particles interact via short-range if their peridynamic bond is broken.
				 * The reduced cutoff radius below ensures this.
				 * If particles were not bonded initially (molecule[i] != molecule[j]), no reduction is performed
				 */
				if (molecule[i] == molecule[j]) {
					radius_factor = 1.0 - smax[itype][jtype];
				} else {
					radius_factor = 1.0;
				}

				// initial distance in reference config
				delx0 = x0[j][0] - x0[i][0];
				dely0 = x0[j][1] - x0[i][1];
				delz0 = x0[j][2] - x0[i][2];
				r0cut = sqrt(delx0 * delx0 + dely0 * dely0 + delz0 * delz0);

				rcut = radius_factor * (radiusSR[i] + radiusSR[j]);
				//rcut = MIN(rcut, r0cut);

				if (rsq < rcut * rcut) {

					// Hertzian short-range forces
					r = sqrt(rsq);
					delta = rcut - r; // overlap distance
					r_geom = radius_factor * radius_factor * radiusSR[i] * radiusSR[j] / rcut;
					if (domain->dimension == 3) {
						//assuming poisson ratio = 1/4 for 3d
						fpair = 1.066666667e0 * bulkmodulus[itype][jtype] * delta * sqrt(delta * r_geom) / r;
						evdwl = r * fpair * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy
					} else {
						//assuming poisson ratio = 1/3 for 2d -- one factor of delta missing compared to 3d
						fpair = 0.16790413e0 * bulkmodulus[itype][jtype] * sqrt(delta * r_geom) / r;
						evdwl = r * fpair * 0.6666666666667e0 * delta;
					}

					if (evflag) {
						ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
					}

					// artificial viscosity -- alpha is dimensionless
					delvelx = vxtmp - v[j][0];
					delvely = vytmp - v[j][1];
					delvelz = vztmp - v[j][2];
					delVdotDelR = delx * delvelx + dely * delvely + delz * delvelz;

					jvol = vfrac[j];
					c0 = sqrt(bulkmodulus[itype][jtype] / (0.5 * (rmass[i] / ivol + rmass[j] / jvol))); // soundspeed
					fvisc = -alpha[itype][jtype] * c0 * 0.5 * (rmass[i] + rmass[j]) * delVdotDelR / (rsq * r);

					fpair = fpair + fvisc;

					f[i][0] += delx * fpair;
					f[i][1] += dely * fpair;
					f[i][2] += delz * fpair;

					if (newton_pair || j < nlocal) {
						f[j][0] -= delx * fpair;
						f[j][1] -= dely * fpair;
						f[j][2] -= delz * fpair;
					}

				}
			}
		}
	}

	/* ----------------------- PERIDYNAMIC BOND FORCES --------------------- */

	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	for (i = 0; i < nlocal; i++) {

		if (molecule[i] == 1000) {

			xtmp = x[i][0];
			ytmp = x[i][1];
			ztmp = x[i][2];
			vxtmp = v[i][0];
			vytmp = v[i][1];
			vztmp = v[i][2];
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

				if (molecule[i] != molecule[j]) {
					printf("ERROR: molecule[i] != molecule[j] :: itype=%d, imol=%d, jtype=%d, jmol=%d\n\n", itype, molecule[i],
							jtype, molecule[j]);
					error->all(FLERR, "molecule[i] != molecule[j]");
				}

				// initial distance in reference config
				delx0 = x0[j][0] - x0[i][0];
				dely0 = x0[j][1] - x0[i][1];
				delz0 = x0[j][2] - x0[i][2];
				double this_r0 = sqrt(delx0 * delx0 + dely0 * dely0 + delz0 * delz0);

				delx = xtmp - x[j][0];
				dely = ytmp - x[j][1];
				delz = ztmp - x[j][2];

				if (periodic)
					domain->minimum_image(delx, dely, delz); // we need this periodic check because j can be non-ghosted

				rsq = delx * delx + dely * dely + delz * delz;
				jtype = type[j];
				delta = radius[i] + radius[j];
				r = sqrt(rsq);
				dr = r - r0[i][jj];

				// avoid roundoff errors
				if (fabs(dr) < 2.2204e-016)
					dr = 0.0;

				// bond stretch
				stretch = dr / r0[i][jj]; // total stretch

				double se;
				if (stretch > syield[itype][jtype]) {
					plastic_stretch[i][jj] = syield[itype][jtype] - stretch;
					se = syield[itype][jtype]; // elastic part of stretch
				} else {
					se = stretch;
				}

				c = 4.5 * bulkmodulus[itype][jtype] * (1.0 / vinter[i] + 1.0 / vinter[j]);

				// force computation -- note we divide by a factor of r
				evdwl = 0.5 * c * se * se * vfrac[i] * vfrac[j];
				//printf("evdwl = %f\n", evdwl);
				fbond = -c * vfrac[i] * vfrac[j] * se / r0[i][jj];
				if (r > 0.0)
					fbond = fbond / r;
				else
					fbond = 0.0;

				// project force -- missing factor of r is recovered here as delx, dely ... are not unit vectors
				f[i][0] += delx * fbond;
				f[i][1] += dely * fbond;
				f[i][2] += delz * fbond;

				if (evflag) {
					// since I-J is double counted, set newton off & use 1/2 factor and I,I
					ev_tally(i, i, nlocal, 0, 0.5 * evdwl, 0.0, 0.5 * fbond, delx, dely, delz);
					//printf("broken evdwl=%f, norm = %f %f\n", evdwl, vinter[i], vinter[j]);
				}

				// bond-based plasticity

//			if (smax[itype][jtype] > 0.0) { // maximum-stretch based failure
				if ((r - r0[i][jj]) / r0[i][jj] > smax[itype][jtype]) {
					partner[i][jj] = 0;
					nBroken += 1;
					e[i] += 0.5 * evdwl;

//					printf("nlocal=%d, i=%d, j=%d\n", nlocal, i, j);
////
//					printf("broken evdwl=%f, k=%f, v=%f %f, norm = %f %f, s=%f, r0=%f, r=%f\n", evdwl, bulkmodulus[itype][jtype],
//							vfrac[i], vfrac[j], vinter[i], vinter[j], stretch, r0[i][jj], r);
//				printf("velocity i: %f %f %f \n", v[i][0], v[i][1], v[i][2]);
//				printf("velocity j: %f %f %f \n", v[j][0], v[j][1], v[j][2]);
//				printf("position i: %f %f %f \n", x[i][0], x[i][1], x[i][2]);
//				printf("position j: %f %f %f \n", x[j][0], x[j][1], x[j][2]);
//				printf("position 0 i: %f %f %f \n", x0[i][0], x0[i][1], x0[i][2]);
//				printf("position 0 j: %f %f %f \n", x0[j][0], x0[j][1], x0[j][2]);
//
//					printf("itype=%d, imol=%d, jtype=%d, jmol=%d\n\n", itype, molecule[i], jtype, molecule[j]);
//					printf("-----------------------------------------------------------------------------\n");
					//error->one(FLERR, "STOP");
					//<
				}
//			} else {
//				printf("smax[%d][%d] = %f\n", itype, jtype, smax[itype][jtype]);
//				printf("broken evdwl=%f, k=%f, v=%f %f, norm = %f %f, s=%f, r0=%f\n", evdwl, bulkmodulus[itype][jtype], vfrac[i],
//						vfrac[j], vinter[i], vinter[j], stretch, r0[i][jj]);
//				printf("itype=%d, imol=%d, jtype=%d, jmol=%d\n\n", itype, molecule[i], jtype, molecule[j]);
//
//				error->all(FLERR, "G0-bnased fracture not implemented for 3d");
//				double this_smax = sqrt(3.0 * G0[itype][jtype] / (2.0 * c * delta * delta * delta)); // factor 2 beacuse we are creating 2 fracture surfaces
//				//printf("G0=%g, smax=%g\n", G0[itype][jtype], this_smax);
//				if (stretch > this_smax) {
//					partner[i][jj] = 0;
//					nBroken += 1;
//					//printf("c = %g, delta=%g\n", c, delta);
//					//printf("G0-based failure: G0=%g, smax = %g\n", G0[itype][jtype], this_smax);
//				}
//
//			}

			}
		} //end if (molecule[i] == 1000)

	}

	/* ----------------------- SHELL BENDING FORCES --------------------- */

	bending_forces();
//
//	int t, tnum;
//	int i1, i2, i3, i4;
//	Vector3d E, n1, n2, E_normed;
//	Vector3d x31, x41, x42, x32, N1, N2, u1, u2, u3, u4;
//	double sign, angle, E_norm;
//	double N1_normSq, N2_normSq, E_normSq;
//	double k, force_magnitude;
//
//	int a, b, c, d;
//
//	for (i = 0; i < nlocal; i++) {
//
//		itype = type[i];
//		tnum = nTrianglePairs[i];
//
//		for (t = 0; t < tnum; t++) {

//			i3 = atom->map(trianglePairs[i][t][0]);
//			i4 = atom->map(trianglePairs[i][t][1]);
//			i1 = atom->map(trianglePairs[i][t][2]);
//			i2 = atom->map(trianglePairs[i][t][3]);
//
//			itype = type[i1]; // types are taken from non-common nodes
//			jtype = type[i2];
//
//			if (i3 != i) {
//				printf("hurz\n");
//				char str[128];
//				sprintf(str, "triangle index cn1=%d does not match up with local index=%d", i3, i);
//				error->one(FLERR, str);
//			}
//
//			// common bond from cn1 to cn2
//			E << (x[i4][0] - x[i3][0]), (x[i4][1] - x[i3][1]), (x[i4][2] - x[i3][2]);
//			E_norm = E.norm();
//			E_normSq = E_norm * E_norm;
//			E_normed = E / E.norm();
//
//			x31 << (x[i1][0] - x[i3][0]), (x[i1][1] - x[i3][1]), (x[i1][2] - x[i3][2]);
//			x41 << (x[i1][0] - x[i4][0]), (x[i1][1] - x[i4][1]), (x[i1][2] - x[i4][2]);
//			x42 << (x[i2][0] - x[i4][0]), (x[i2][1] - x[i4][1]), (x[i2][2] - x[i4][2]);
//			x32 << (x[i2][0] - x[i3][0]), (x[i2][1] - x[i3][1]), (x[i2][2] - x[i3][2]);
//
//			N1 = x31.cross(x41);
//			N1_normSq = N1.squaredNorm();
//
//			N2 = x42.cross(x32);
//			N2_normSq = N2.squaredNorm();
//
//			u1 = (E_norm / N1_normSq) * N1;
//			u2 = (E_norm / N2_normSq) * N2;
//
//			u3 = x41.dot(E_normed) * N1 / N1_normSq + x42.dot(E_normed) * N2 / N2_normSq;
//			u4 = -x31.dot(E_normed) * N1 / N1_normSq - x32.dot(E_normed) * N2 / N2_normSq;
//
//			// normal of triangle 1
//			n1 = N1 / N1.norm();
//
//			// normal of triangle 2
//			n2 = N2 / N2.norm();
//
//			// determine sin(phi) / 2
//			sign = (n1.cross(n2)).dot(E_normed);
//
//			angle = 0.5 * (1.0 - n1.dot(n2));
//			if (angle < 0.0) angle = 0.0;
//			angle = sqrt(angle);
//			if (sign * angle < 0.0) {
//				angle = -angle;
//			}
//
//			force_magnitude = angle * kbend[itype][jtype] * E_normSq / (sqrt(N1_normSq) + sqrt(N2_normSq));
//
//			//printf("angle is %f\n", angle);
//
//
//			Vector3d force = force_magnitude * u1;
//			double fmag = force.norm();
//
//			if (fabs(force_magnitude) > 1.0e3) {
//				cout << "angle is " << angle << "   sign is " << sign << "      n1 dot n2 is " << n1.dot(n2) << endl;
//				cout << "force_magnitude is " <<  force_magnitude << endl;
//				cout << "E is " << E << endl;
//				cout << "u1 is " << u1 << endl;
//				cout << "u2 is " << u2 << endl;
//				cout << "u3 is " << u3 << endl;
//				cout << "u4 is " << u4 << endl;
//				cout << "N1 is " << n1 << endl;
//				cout << "N2 is " << n2 << endl << endl;
//
//			}
//
//
//
//			f[i1][0] += force_magnitude * u1(0);
//			f[i1][1] += force_magnitude * u1(1);
//			f[i1][2] += force_magnitude * u1(2);
//
//			if (i2 < nlocal) {
//				f[i2][0] += force_magnitude * u2(0);
//				f[i2][1] += force_magnitude * u2(1);
//				f[i2][2] += force_magnitude * u2(2);
//			}
//
//			if (i3 < nlocal) {
//				f[i3][0] += force_magnitude * u3(0);
//				f[i3][1] += force_magnitude * u3(1);
//				f[i3][2] += force_magnitude * u3(2);
//			}
//
//			if (i4 < nlocal) {
//				f[i4][0] += force_magnitude * u4(0);
//				f[i4][1] += force_magnitude * u4(1);
//				f[i4][2] += force_magnitude * u4(2);
//			}

//		}
//	}

}

/* ----------------------------------------------------------------------
 bending forces
 ------------------------------------------------------------------------- */

void PairPDGCGShells::bending_forces() {
	// ---------------------------------------------------------------------------------
	double **f = atom->f;
	double **x = atom->x;
	double **v = atom->v;
	int nlocal = atom->nlocal;
	tagint ***trianglePairs = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->trianglePairs;
	int *nTrianglePairs = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->nTrianglePairs;
	double ***trianglePairCoeffs = ((FixPDGCGShellsNeigh *) modify->fix[ifix_peri])->trianglePairCoeffs;

	int a, b, c, d, i, tnum, t;
	Vector3d R, Pa, Pb, Pc, Pd;
	double force_magnitude;

	for (i = 0; i < nlocal; i++) {

		tnum = nTrianglePairs[i];

		for (t = 0; t < tnum; t++) {

			c = atom->map(trianglePairs[i][t][0]);
			d = atom->map(trianglePairs[i][t][1]);
			a = atom->map(trianglePairs[i][t][2]);
			b = atom->map(trianglePairs[i][t][3]);

			Pa << x[a][0], x[a][1], x[a][2];
			Pb << x[b][0], x[b][1], x[b][2];
			Pc << x[c][0], x[c][1], x[c][2];
			Pd << x[d][0], x[d][1], x[d][2];

			R = trianglePairCoeffs[i][t][0] * Pa + trianglePairCoeffs[i][t][1] * Pb + trianglePairCoeffs[i][t][2] * Pc
					+ trianglePairCoeffs[i][t][3] * Pd;

//			cout << "R is " << R << endl;
//			cout << "amplitude is " << trianglePairCoeffs[i][t][4] << endl << endl;

			force_magnitude = -trianglePairCoeffs[i][t][4] * trianglePairCoeffs[i][t][0];
			f[a][0] += force_magnitude * R(0);
			f[a][1] += force_magnitude * R(1);
			f[a][2] += force_magnitude * R(2);

			//cout << "force on atom a is " << force_magnitude * R << endl;

			if (b < nlocal) {
				force_magnitude = -trianglePairCoeffs[i][t][4] * trianglePairCoeffs[i][t][1];
			//	cout << "force on atom b is " << force_magnitude * R << endl;
				f[b][0] += force_magnitude * R(0);
				f[b][1] += force_magnitude * R(1);
				f[b][2] += force_magnitude * R(2);
			}

			if (c < nlocal) {
				force_magnitude = -trianglePairCoeffs[i][t][4] * trianglePairCoeffs[i][t][2];
			//	cout << "force on atom c is " << force_magnitude * R << endl;
				f[c][0] += force_magnitude * R(0);
				f[c][1] += force_magnitude * R(1);
				f[c][2] += force_magnitude * R(2);
			}

			if (d < nlocal) {
			//	cout << "force on atom d is " << force_magnitude * R << endl << endl;
				force_magnitude = -trianglePairCoeffs[i][t][4] * trianglePairCoeffs[i][t][3];
				f[d][0] += force_magnitude * R(0);
				f[d][1] += force_magnitude * R(1);
				f[d][2] += force_magnitude * R(2);
			}

		}
	}

}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairPDGCGShells::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(bulkmodulus, n + 1, n + 1, "pair:");
	memory->create(kbend, n + 1, n + 1, "pair:kbend");
	memory->create(smax, n + 1, n + 1, "pair:smax");
	memory->create(syield, n + 1, n + 1, "pair:syield");
	memory->create(G0, n + 1, n + 1, "pair:G0");
	memory->create(alpha, n + 1, n + 1, "pair:alpha");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairPDGCGShells::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairPDGCGShells::coeff(int narg, char **arg) {
	if (narg != 8)
		error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	double bulkmodulus_one = atof(arg[2]);
	double kbend_one = atof(arg[3]);
	double smax_one = atof(arg[4]);
	double G0_one = atof(arg[5]);
	double alpha_one = atof(arg[6]);
	double syield_one = atof(arg[7]);

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			bulkmodulus[i][j] = bulkmodulus_one;
			kbend[i][j] = kbend_one;
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

double PairPDGCGShells::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	bulkmodulus[j][i] = bulkmodulus[i][j];
	kbend[j][i] = kbend[i][j];
	alpha[j][i] = alpha[i][j];
	smax[j][i] = smax[i][j];
	syield[j][i] = syield[i][j];
	G0[j][i] = G0[i][j];

	return cutoff_global;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairPDGCGShells::init_style() {
	int i;

// error checks

	if (!atom->x0_flag)
		error->all(FLERR, "Pair style peri requires atom style with x0");
	if (atom->map_style == 0)
		error->all(FLERR, "Pair peri requires an atom map, see atom_modify");

// if first init, create Fix needed for storing fixed neighbors

	if (ifix_peri == -1) {
		char **fixarg = new char*[3];
		fixarg[0] = (char *) "PDGCG_SHELLS_NEIGH";
		fixarg[1] = (char *) "all";
		fixarg[2] = (char *) "PDGCG_SHELLS_NEIGH";
		modify->add_fix(3, fixarg);
		delete[] fixarg;
	}

// find associated PERI_NEIGH fix that must exist
// could have changed locations in fix list since created

	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "PDGCG_SHELLS_NEIGH") == 0)
			ifix_peri = i;
	if (ifix_peri == -1)
		error->all(FLERR, "Fix PDGCG_SHELLS_NEIGH does not exist");

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

void PairPDGCGShells::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairPDGCGShells::write_restart(FILE *fp) {
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

void PairPDGCGShells::read_restart(FILE *fp) {
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

double PairPDGCGShells::memory_usage() {

	return 0.0;
}

/* ----------------------------------------------------------------------
 extract method to provide access to this class' data structures
 ------------------------------------------------------------------------- */

void *PairPDGCGShells::extract(const char *str, int &i) {

	if (strcmp(str, "pdgcg/shells/kbend_ptr") == 0) {
		return (void *) kbend;
	}

	return NULL;
}
