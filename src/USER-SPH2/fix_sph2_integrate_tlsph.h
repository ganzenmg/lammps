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

FixStyle(sph2/integrate_tlsph,FixSph2IntegrateTlsph)

#else

#ifndef LMP_FIX_SPH2_INTEGRATE_TLSPH_H
#define LMP_FIX_SPH2_INTEGRATE_TLSPH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSph2IntegrateTlsph: public Fix {
	friend class Neighbor;
	friend class PairTlsph;
public:
    FixSph2IntegrateTlsph(class LAMMPS *, int, char **);
    virtual ~FixSph2IntegrateTlsph() {
    }
    int setmask();
    virtual void init();
    virtual void initial_integrate(int);
    virtual void final_integrate();
    virtual void reset_dt();
    void updateReferenceConfiguration();
    int pack_forward_comm(int, int *, double *, int, int *);
    void unpack_forward_comm(int, int, double *);
    void smooth_fields();

protected:
    double dtv, dtf, vlimit, vlimitsq;
    int mass_require;
    bool updateReferenceConfigurationFlag, xsphFlag;
    int smoothPeriod;
    int nRefConfigUpdates;

    class Pair *pair;
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
