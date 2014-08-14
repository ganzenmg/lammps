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
    void write_restart(FILE *);
    void read_restart(FILE *);
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
    void kernel_and_derivative(const double h, const double r, double &wf, double &wfd);
    Matrix3d pseudo_inverse_SVD(Matrix3d);
    void PolDec(Matrix3d &, Matrix3d *, Matrix3d *);
    void AssembleStress();
    Matrix3d Deviator(Matrix3d);

    void CheckIntegration();
    void PreCompute();
    double TestMatricesEqual(Matrix3d, Matrix3d, double);

    /*
     * strength models
     */
    void LinearStrength(double, Matrix3d, Matrix3d, double, Matrix3d *, Matrix3d*);
    void LinearPlasticStrength(double, double, Matrix3d, Matrix3d, double, Matrix3d *, Matrix3d*, double *);
    void LinearStrengthDefgrad(double, double, Matrix3d, Matrix3d *);

    /*
     * EOS models
     */
    void LinearEOS(double &, double &, double &, double &, double &, double &);
    void LinearCutoffEOS(double &, double &, double &, double &, double &, double &, double &);

protected:
    double *youngsmodulus, *poissonr, *yieldStress, *maxstrain;
    double **alpha, **hg_coeff, **c0_type;
    int *strengthModel, *eos; // strength (deviatoric) and pressure constitutive models

    double *onerad_dynamic, *onerad_frozen;
    double *maxrad_dynamic, *maxrad_frozen;

    void allocate();

    int nmax; // max number of atoms on this proc
    Matrix3d *F, *K, *PK1, *Fdot, *Fincr;
    Matrix3d *d; // unrotated rate-of-deformation tensor
    Matrix3d *R; // rotation matrix
    Matrix3d *Edot; // rate of Green-Lagrange strain
    Matrix3d *FincrInv;
    Matrix3d *D, *W; // strain rate and spin tensor
    Vector3d *smoothVel, *smoothPos;
    Matrix3d *CauchyStress;
    double *detF;
    bool *shearFailureFlag;
    int *numNeighsRefConfig;
    double *shepardWeight;

    int updateFlag;

    enum {
        LINEAR, LINEAR_PLASTIC, NONE, LINEAR_DEFGRAD, LINEAR_CUTOFF
    };

    void init_x0_box(void);
    void minimum_image(double &, double &, double &);

    double boxlo[3], boxhi[3], prd[3], prd_half[3];
    double xprd, yprd, zprd, xprd_half, yprd_half, zprd_half;
    double xy, xz, yz;

};

}

#endif
#endif

