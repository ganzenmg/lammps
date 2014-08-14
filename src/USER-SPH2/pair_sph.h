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

PairStyle(sph/fluid,PairSphFluid)

#else

#ifndef LMP_SPH_H
#define LMP_SPH_H

#include "pair.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
namespace LAMMPS_NS {

class PairSphFluid: public Pair {
public:
    PairSphFluid(class LAMMPS *);
    virtual ~PairSphFluid();
    virtual void compute(int, int);
    void settings(int, char **);
    void coeff(int, char **);
    double init_one(int, int);
    void init_style();
    void init_list(int, class NeighList *);
    void write_restart(FILE *);
    void read_restart(FILE *);
    void write_restart_settings(FILE *) {
    }
    void read_restart_settings(FILE *) {
    }
    virtual double memory_usage();
    int pack_comm(int, int *, double *, int, int *);
    void unpack_comm(int, int, double *);
    void kernel_and_derivative(const double h, const double r, double &wf, double &wfd);
    Matrix3d pseudo_inverse_SVD(Matrix3d);
    void ComputePressure();
    void *extract(const char *, int &);
    void PreCompute();
    Matrix3d Deviator(Matrix3d);

    /*
     * EOS models
     */
    void PerfectGasEOS(double, double, double, double, double *, double *);
    void TaitEOS(const double exponent, const double c0_reference,
                 const double rho_reference, const double rho_current,
                 double &pressure, double &sound_speed);

protected:
    double **C1, **C2, **C3, **C4; // coefficients for EOS
    double **Q1, **Q2; // linear and quadratic artificial viscosity coeffs
    int **eos; // strength (deviatoric) and pressure constitutive models

    double *onerad_dynamic, *onerad_frozen;
    double *maxrad_dynamic, *maxrad_frozen;

    void allocate();

    int nmax; // max number of atoms on this proc
    Matrix3d *K, *artStress;
    double *shepardWeight, *c0;
    Vector3d *smoothVel;
    Matrix3d *stressTensor, *L;

    enum {
        NONE, PERFECT_GAS, TAIT, LINEAR_ELASTIC
    };

private:
    double *delete_flag;
    double *pressure;

    double hMin;
};

}

#endif
#endif

