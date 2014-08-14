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

#ifdef FIX_CLASS

FixStyle(tlsph/integrate,FixTlsphIntegrate)

#else

#ifndef LMP_FIX_TLSPH_INTEGRATE_H
#define LMP_FIX_TLSPH_INTEGRATE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixTlsphIntegrate: public Fix {
public:
    FixTlsphIntegrate(class LAMMPS *, int, char **);
    virtual ~FixTlsphIntegrate() {
    }
    int setmask();
    virtual void init();
    virtual void initial_integrate(int);
    virtual void final_integrate();
    virtual void reset_dt();
    void updateReferenceConfiguration();
    int pack_comm(int, int *, double *, int, int *);
    void unpack_comm(int, int, double *);

protected:
    double dtv, dtf, vlimit, vlimitsq;
    int mass_require;
    bool updateReferenceConfigurationFlag, xsphFlag;
    int nRefConfigUpdates;
};

}

#endif
#endif

/* ERROR/WARNING messages:

 E: Illegal ... command

 Self-explanatory.  Check the input script syntax and compare to the
 documentation for the command.  You can use -echo screen as a
 command-line option when running LAMMPS to see the offending line.

 */
