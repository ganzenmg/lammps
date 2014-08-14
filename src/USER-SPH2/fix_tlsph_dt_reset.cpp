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
#include "stdlib.h"
#include "string.h"
#include "fix_tlsph_dt_reset.h"
#include "atom.h"
#include "update.h"
#include "integrate.h"
#include "domain.h"
#include "lattice.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "fix.h"
#include "output.h"
#include "dump.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixTlsphDtReset::FixTlsphDtReset(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg) {
    if (narg < 7)
        error->all(FLERR, "Illegal fix tlsph/dt/reset command");

    // set time_depend, else elapsed time accumulation can be messed up

    time_depend = 1;
    scalar_flag = 1;
    vector_flag = 1;
    size_vector = 2;
    global_freq = 1;
    extscalar = 0;
    extvector = 0;

    nevery = atoi(arg[3]);
    if (nevery <= 0)
        error->all(FLERR, "Illegal fix tlsph/dt/reset command");

    minbound = maxbound = 1;
    tmin = tmax = 0.0;
    if (strcmp(arg[4], "NULL") == 0)
        minbound = 0;
    else
        tmin = atof(arg[4]);
    if (strcmp(arg[5], "NULL") == 0)
        maxbound = 0;
    else
        tmax = atof(arg[5]);
    xmax = atof(arg[6]);

    if (minbound && tmin < 0.0)
        error->all(FLERR, "Illegal fix tlsph/dt/reset command");
    if (maxbound && tmax < 0.0)
        error->all(FLERR, "Illegal fix tlsph/dt/reset command");
    if (minbound && maxbound && tmin >= tmax)
        error->all(FLERR, "Illegal fix tlsph/dt/reset command");
    if (xmax <= 0.0)
        error->all(FLERR, "Illegal fix tlsph/dt/reset command");

    // initializations

    t_elapsed = t_laststep = 0.0;
    laststep = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

int FixTlsphDtReset::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= END_OF_STEP;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixTlsphDtReset::init() {
    ftm2v = force->ftm2v;
    dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixTlsphDtReset::setup(int vflag) {
    end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixTlsphDtReset::initial_integrate(int vflag) {
    // calculate elapsed time based on previous reset timestep

    t_elapsed = t_laststep + (update->ntimestep - laststep) * dt;
}

/* ---------------------------------------------------------------------- */

void FixTlsphDtReset::end_of_step() {
    double dtv, dtf, dtsq;
    double vsq, fsq, massinv;
    double delx, dely, delz, delr;

    // compute vmax and amax of any atom in group

    double **v = atom->v;
    double **f = atom->f;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double dtmin = BIG;

    for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
            if (rmass)
                massinv = 1.0 / rmass[i];
            else
                massinv = 1.0 / mass[type[i]];
            vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
            fsq = f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];
            dtv = dtf = BIG;
            if (vsq > 0.0)
                dtv = xmax / sqrt(vsq);
            if (fsq > 0.0)
                dtf = sqrt(2.0 * xmax / (ftm2v * sqrt(fsq) * massinv));
            dt = MIN(dtv, dtf);
            dtsq = dt * dt;
            delx = dt * v[i][0] + 0.5 * dtsq * massinv * f[i][0] * ftm2v;
            dely = dt * v[i][1] + 0.5 * dtsq * massinv * f[i][1] * ftm2v;
            delz = dt * v[i][2] + 0.5 * dtsq * massinv * f[i][2] * ftm2v;
            delr = sqrt(delx * delx + dely * dely + delz * delz);
            if (delr > xmax)
                dt *= xmax / delr;
            dtmin = MIN(dtmin, dt);
        }

    MPI_Allreduce(&dtmin, &dt, 1, MPI_DOUBLE, MPI_MIN, world);

    if (minbound)
        dt = MAX(dt, tmin);
    if (maxbound)
        dt = MIN(dt, tmax);

    /*
     * determine stable CFL timestep
     */

    int itmp = 0;

    // get minimum SPH smoothing length for this processor
    double *hMin = (double *) force->pair->extract("solid_hMin_ptr", itmp);
    if (hMin == NULL) {
        error->all(FLERR, "fix tlsph/dt/reset failed to access hMin");
    }

    // get maximum SPH particle sound velocity length for this processor
    double *c0Max = (double *) force->pair->extract("solid_c0Max_ptr", itmp);
    if (c0Max == NULL) {
        error->all(FLERR, "fix tlsph/dt/reset failed to access c0Max");
    }

    // get maximum SPH particle signal velocity for this processor
    double *signalVelMax = (double *) force->pair->extract("solid_signalVelocity_ptr", itmp);
    if (signalVelMax == NULL) {
        error->all(FLERR, "fix tlsph/dt/reset failed to access signalVelMax");
    }

    double thisDtCFL = 0.5 * 0.125 * *hMin / *c0Max;
    double dtCFL;
    MPI_Allreduce(&thisDtCFL, &dtCFL, 1, MPI_DOUBLE, MPI_MIN, world);
    //printf("CFL stable timestep is %f, max travel timestep is %f, c0 max is %f, sign.vel is %f\n", dtCFL, dt, *c0Max,
    //        *signalVelMax);
    dt = MIN(dt, dtCFL);

    // if timestep didn't change, just return
    // else reset update->dt and other classes that depend on it

    if (dt == update->dt)
        return;

    t_elapsed = t_laststep += (update->ntimestep - laststep) * update->dt;
    laststep = update->ntimestep;

    update->dt = dt;
    if (force->pair)
        force->pair->reset_dt();
    for (int i = 0; i < modify->nfix; i++)
        modify->fix[i]->reset_dt();
}

/* ---------------------------------------------------------------------- */

double FixTlsphDtReset::compute_scalar() {
    return update->dt;
}

/* ---------------------------------------------------------------------- */

double FixTlsphDtReset::compute_vector(int n) {
    if (n == 0)
        return t_elapsed;
    return (double) laststep;
}
