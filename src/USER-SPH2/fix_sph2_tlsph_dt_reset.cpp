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
#include "fix_sph2_tlsph_dt_reset.h"
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

FixSph2TlsphDtReset::FixSph2TlsphDtReset(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg != 5)
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

	safety_factor = atof(arg[4]);

	// initializations
	t_elapsed = t_laststep = 0.0;
	laststep = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

int FixSph2TlsphDtReset::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSph2TlsphDtReset::init() {
	dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixSph2TlsphDtReset::setup(int vflag) {
	end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixSph2TlsphDtReset::initial_integrate(int vflag) {
	// calculate elapsed time based on previous reset timestep

	t_elapsed = t_laststep + (update->ntimestep - laststep) * dt;
}

/* ---------------------------------------------------------------------- */

void FixSph2TlsphDtReset::end_of_step() {
	double dtmin = BIG;

	/*
	 * determine stable CFL timestep
	 */

	int itmp = 0;
	double *dtCFL = (double *) force->pair->extract("sph2/tlsph/dtCFL_ptr", itmp);
	if (dtCFL == NULL) {
		error->all(FLERR, "fix tlsph/dt/reset failed to access tlsph dtCFL");
	}

	dtmin = safety_factor * *dtCFL; // apply safety factor

	MPI_Allreduce(&dtmin, &dt, 1, MPI_DOUBLE, MPI_MIN, world);

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

double FixSph2TlsphDtReset::compute_scalar() {
	return update->dt;
}

/* ---------------------------------------------------------------------- */

double FixSph2TlsphDtReset::compute_vector(int n) {
	if (n == 0)
		return t_elapsed;
	return (double) laststep;
}
