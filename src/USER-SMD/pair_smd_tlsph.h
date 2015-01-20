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

#ifdef PAIR_CLASS

PairStyle(tlsph,PairTlsph)

#else

#ifndef LMP_TLSPH_NEW_H
#define LMP_TLSPH_NEW_H

#include "pair.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <map>

using namespace std;
using namespace Eigen;
namespace LAMMPS_NS {

class PairTlsph: public Pair {
public:

	PairTlsph(class LAMMPS *);
	virtual ~PairTlsph();
	virtual void compute(int, int);
	void settings(int, char **);
	void coeff(int, char **);
	double init_one(int, int);
	void init_style();
	void init_list(int, class NeighList *);
	void write_restart_settings(FILE *) {
	}
	void read_restart_settings(FILE *) {
	}
	virtual double memory_usage();
	void compute_shape_matrix(void);
	void material_model(void);
	void *extract(const char *, int &);
	int pack_forward_comm(int, int *, double *, int, int *);
	void unpack_forward_comm(int, int, double *);
	void spiky_kernel_and_derivative(const double h, const double r, double &wf, double &wfd);
	void barbara_kernel_and_derivative(const double h, const double r, double &wf, double &wfd);
	Matrix3d pseudo_inverse_SVD(Matrix3d);
	void PolDec(Matrix3d &, Matrix3d *, Matrix3d *);
	void AssembleStress();
	Matrix3d Deviator(Matrix3d);

	void CheckIntegration();
	void PreCompute();
	void ComputeForces(int eflag, int vflag);
	double TestMatricesEqual(Matrix3d, Matrix3d, double);
	double effective_longitudinal_modulus(int itype, double dt, double d_iso, double p_rate, Matrix3d d_dev, Matrix3d sigma_dev_rate, double damage);

	void SmoothField();
	void SmoothFieldXSPH();

	/*
	 * strength models
	 */
	void LinearStrength(double, Matrix3d, Matrix3d, double, Matrix3d *, Matrix3d*);
	void LinearPlasticStrength(double, double, Matrix3d, Matrix3d, double, Matrix3d *, Matrix3d*, double *);
	void LinearStrengthDefgrad(double, double, Matrix3d, Matrix3d *);
	void JohnsonCookStrength(double G, double cp, double espec,
			double A, double B, double a, double C, double epdot0, double T0, double Tmelt, double M, double dt,
			double ep, double epdot,
			Matrix3d sigmaInitial_dev, Matrix3d d_dev,
			Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment);

	/*
	 * EOS models
	 */
	void LinearEOS(double lambda, double pInitial, double d, double dt, double &pFinal, double &p_rate);
	//void LinearCutoffEOS(double &, double &, double &, double &, double &, double &, double &);
	void ShockEOS(double rho, double rho0, double e, double e0, double c0, double S, double Gamma,
			double pInitial, double dt, double &pFinal, double &p_rate);
	void polynomialEOS(double rho, double rho0, double e, double C0, double C1, double C2, double C3, double C4, double C5,
			double C6, double pInitial, double dt, double &pFinal, double &p_rate);

	/*
	 * damage models
	 */
	void IsotropicMaxStressDamage(Matrix3d S, double maxStress, double dt, double soundspeed, double characteristicLength,
			double &damage, Matrix3d &S_damaged);
	void IsotropicMaxStrainDamage(Matrix3d E, Matrix3d S, double maxStress, double dt, double soundspeed, double characteristicLength,
				double &damage, Matrix3d &S_damaged);
	double JohnsonCookFailureStrain(double p, Matrix3d Sdev, double d1, double d2, double d3, double d4, double epdot0,
			double epdot);

protected:
	void allocate();

	/*
	 * per-type arrays
	 */
	double *youngsmodulus, *signal_vel0, *rho0;
	int *strengthModel, *eos;
	double *onerad_dynamic, *onerad_frozen, *maxrad_dynamic, *maxrad_frozen;

	/*
	 * per type pair arrays
	 */
	double *Q1, *Q2;
	double *hg_coeff;


	/*
	 * per atom arrays
	 */


	Matrix3d *K, *PK1, *Fdot, *Fincr;
	Matrix3d *d; // unrotated rate-of-deformation tensor
	Matrix3d *R; // rotation matrix
	Matrix3d *FincrInv;
	Matrix3d *D, *W; // strain rate and spin tensor
	Vector3d *smoothVelDifference;
	Matrix3d *CauchyStress;
	double *detF, *p_wave_speed, *shepardWeight;
	double *hourglass_error;
	bool *shearFailureFlag;
	int *numNeighsRefConfig;

	int nmax; // max number of atoms on this proc
	double hMin; // minimum kernel radius for two particles
	double dtCFL;
	double dtRelative; // relative velocity of two particles, divided by sound speed
	int updateFlag;

	enum {
		LINEAR_DEFGRAD, LINEAR_STRENGTH, LINEAR_PLASTICITY, STRENGTH_JOHNSON_COOK,
		STRENGTH_NONE,
		EOS_LINEAR, EOS_SHOCK, EOS_POLYNOMIAL,
		EOS_NONE
	};

	// C++ std dictionary to hold material model settings per particle type
	typedef std::map<std::pair<std::string, int>, double> Dict;
	Dict matProp2;
	//typedef Dict::const_iterator It;

	int ifix_tlsph;
	int not_first;

	class FixSMDIntegrateTlsph *fix_tlsph_time_integration;

private:
	double SafeLookup(std::string str, int itype);
	bool CheckKeywordPresent(std::string str, int itype);
};

}

#endif
#endif

/*
 * materialCoeffs array for EOS parameters:
 * 1: rho0
 *
 *
 * materialCoeffs array for strength parameters:
 *
 * Common
 * 10: maximum strain threshold for damage model
 * 11: maximum stress threshold for damage model
 *
 * Linear Plasticity model:
 * 12: plastic yield stress
 *
 *
 * Blei: rho = 11.34e-6, c0=2000, s=1.46, Gamma=2.77
 * Stahl 1403: rho = 7.86e-3, c=4569, s=1.49, Gamma=2.17
 */





