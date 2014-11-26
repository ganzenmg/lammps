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

#include "string.h"
#include "compute_sph2_hourglass_error.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSph2HourglassError::ComputeSph2HourglassError(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute sph2/hourglass_error command");
	if (atom->rho_flag != 1)
		error->all(FLERR, "compute sph2/hourglass_error command requires atom_style with hourglass_error (e.g. sph2)");

	peratom_flag = 1;
	size_peratom_cols = 0;

	nmax = 0;
	hourglass_error_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSph2HourglassError::~ComputeSph2HourglassError() {
	memory->sfree(hourglass_error_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSph2HourglassError::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "sph2/hourglass_error") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute sph2/hourglass_error");
}

/* ---------------------------------------------------------------------- */

void ComputeSph2HourglassError::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow rhoVector array if necessary

	if (atom->nlocal > nmax) {
		memory->sfree(hourglass_error_vector);
		nmax = atom->nmax;
		hourglass_error_vector = (double *) memory->smalloc(nmax * sizeof(double), "atom:hourglass_error_vector");
		vector_atom = hourglass_error_vector;
	}

	int itmp = 0;
	double *hourglass_error = (double *) force->pair->extract("sph2/tlsph/hourglass_error_ptr", itmp);
	if (hourglass_error == NULL) {
		error->all(FLERR, "compute sph2/hourglass_error failed to access hourglass_error array");
	}

	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			hourglass_error_vector[i] = hourglass_error[i];
		} else {
			hourglass_error_vector[i] = 0.0;
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSph2HourglassError::memory_usage() {
	double bytes = nmax * sizeof(double);
	return bytes;
}
