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

PairStyle(ulsph,PairULSPH)

#else

#ifndef LMP_ULSPH_H
#define LMP_ULSPH_H

#include "pair.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <map>

using namespace Eigen;
using namespace std;
namespace LAMMPS_NS {

class PairULSPH: public Pair {
public:
    PairULSPH(class LAMMPS *);
    virtual ~PairULSPH();
    virtual void compute(int, int);
    void settings(int, char **);
    void coeff(int, char **);
    double init_one(int, int);
    void init_style();
    void init_list(int, class NeighList *);
    virtual double memory_usage();
    int pack_forward_comm(int, int *, double *, int, int *);
    void unpack_forward_comm(int, int, double *);
    void kernel_and_derivative(const double h, const double r, double &wf, double &wfd);
    void Poly6Kernel(const double hsq, const double h, const double rsq, double &wf);
    Matrix3d pseudo_inverse_SVD(Matrix3d);
    void ComputePressure();
    void *extract(const char *, int &);
    void PreCompute();
    void PreCompute_DensitySummation();
    Matrix3d Deviator(Matrix3d);

    /*
     * EOS models
     */
    void PerfectGasEOS(double, double, double, double, double *, double *);
    void TaitEOS(const double exponent, const double c0_reference,
                 const double rho_reference, const double rho_current,
                 double &pressure, double &sound_speed);

protected:

    double *rho0; // reference mass density per type
    double *Q1; // linear artificial viscosity coeff
    int *eos; // eos models

    double *onerad_dynamic, *onerad_frozen;
    double *maxrad_dynamic, *maxrad_frozen;

    void allocate();

    int nmax; // max number of atoms on this proc
    int *numNeighs;
    Matrix3d *K, *artStress;
    double *shepardWeight, *c0;
    Vector3d *smoothVel;
    Matrix3d *stressTensor, *L;

    enum {
        NONE, PERFECT_GAS, EOS_TAIT, VISCOSITY_LINEAR
    };

    double dtCFL;


private:
    double *delete_flag;
    double *pressure;
    typedef std::map<std::pair<std::string, int>, double> Dict;
    Dict matProp2;
	double SafeLookup(std::string str, int itype);
	bool CheckKeywordPresent(std::string str, int itype);

};

}

#endif
#endif

