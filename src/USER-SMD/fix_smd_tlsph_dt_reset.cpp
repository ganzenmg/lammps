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
#include "stdlib.h"
#include "string.h"
#include "fix_smd_tlsph_dt_reset.h"
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

FixSMDTlsphDtReset::FixSMDTlsphDtReset(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg != 5)
		error->all(FLERR, "Illegal fix smd/adjust_dt command");

	// set time_depend, else elapsed time accumulation can be messed up

	time_depend = 1;
	scalar_flag = 1;
	vector_flag = 1;
	size_vector = 2;
	global_freq = 1;
	extscalar = 0;
	extvector = 0;

	nevery = force->inumeric(FLERR, arg[3]);
	if (nevery <= 0)
		error->all(FLERR, "Illegal fix smd/adjust_dt command. nevery must be positive and > 0.");

	safety_factor = force->numeric(FLERR, arg[4]);
	if (safety_factor > 0.9)
		error->all(FLERR, "Illegal fix smd/adjust_dt command. pre-factor must be <= 0.9 (0.1 is a safe choice).");

	// initializations
	t_elapsed = t_laststep = 0.0;
	laststep = update->ntimestep;

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix smd/integrate_ulsph is active for group: %s \n", arg[1]);
		printf("... will update the stable time increment every %d timesteps\n", nevery);
		printf("... pre-factor for CFL-criterion is %f\n", safety_factor);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}
}

/* ---------------------------------------------------------------------- */

int FixSMDTlsphDtReset::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::init() {
	dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::setup(int vflag) {
	end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::initial_integrate(int vflag) {
	// calculate elapsed time based on previous reset timestep

	t_elapsed = t_laststep + (update->ntimestep - laststep) * update->dt;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::end_of_step() {
	double dtmin = BIG;
	int itmp = 0;

	/*
	 * extract minimum CFL timestep from TLSPH and ULSPH pair styles
	 */

	double *dtCFL_TLSPH = (double *) force->pair->extract("smd/tlsph/dtCFL_ptr", itmp);
	double *dtCFL_ULSPH = (double *) force->pair->extract("smd/ulsph/dtCFL_ptr", itmp);

	if ((dtCFL_TLSPH == NULL) && (dtCFL_ULSPH == NULL)) {
		error->all(FLERR, "fix smd/adjust_dt failed to access a valid dtCFL");
	}

	if (dtCFL_TLSPH != NULL) {
		dtmin = MIN(dtmin, *dtCFL_TLSPH);
	}

	if (dtCFL_ULSPH != NULL) {
		dtmin = MIN(dtmin, *dtCFL_ULSPH);
	}

	dtmin *= safety_factor; // apply safety factor

	MPI_Allreduce(&dtmin, &dt, 1, MPI_DOUBLE, MPI_MIN, world);

// if timestep didn't change, just return
// else reset update->dt and other classes that depend on it

	if (dt == update->dt)
		return;

	t_elapsed = t_laststep + (update->ntimestep - laststep) * update->dt;
	t_laststep = t_elapsed;
	laststep = update->ntimestep;

	update->dt = dt;
	if (force->pair)
		force->pair->reset_dt();
	for (int i = 0; i < modify->nfix; i++)
		modify->fix[i]->reset_dt();
}

/* ---------------------------------------------------------------------- */

double FixSMDTlsphDtReset::compute_scalar() {
	return update->dt;
}

/* ---------------------------------------------------------------------- */

double FixSMDTlsphDtReset::compute_vector(int n) {
	if (n == 0)
		return t_elapsed;
	return (double) laststep;
}
