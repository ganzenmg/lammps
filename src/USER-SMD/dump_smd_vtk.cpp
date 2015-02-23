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
#include "dump_smd_vtk.h"
#include "atom.h"
#include "group.h"
#include "error.h"
#include "memory.h"
#include "update.h"

using namespace LAMMPS_NS;

#define ONELINE 128
#define DELTA 1048576
#define NLINES 9 // number of data items per particle

/* ---------------------------------------------------------------------- */

DumpSMDVTK::DumpSMDVTK(LAMMPS *lmp, int narg, char **arg) :
		Dump(lmp, narg, arg) {
	if (narg != 5)
		error->all(FLERR, "Illegal dump xyz command");
	if (binary || multiproc)
		error->all(FLERR, "Invalid dump xyz filename");

	size_one = NLINES;

	buffer_allow = 1;
	buffer_flag = 0;
	sort_flag = 0;
	sortcol = 0;

	if (format_default)
		delete[] format_default;

	char *str = (char *) "%s %g %g %g";
	int n = strlen(str) + 1;
	format_default = new char[n];
	strcpy(format_default, str);

	ntypes = atom->ntypes;
	typenames = NULL;

	// allocate global array to hold particle info
	bigint ngroup = group->count(igroup);
	//printf("***** GROUP is %d for gpartdata\n", ngroup);
	if (ngroup > MAXSMALLINT / sizeof(float))
		error->all(FLERR, "Too many particles for dump smd/vtk");
	natoms = static_cast<int>(ngroup);

	//printf("***** ALLOCATED %d for gpartdata\n", natoms);
	global_particle_data = NULL;
	memory->create(global_particle_data, natoms, NLINES, "dump:coords");
	ntotal = 0;

}

/* ---------------------------------------------------------------------- */

DumpSMDVTK::~DumpSMDVTK() {
	delete[] format_default;
	format_default = NULL;

	if (typenames) {
		for (int i = 1; i <= ntypes; i++)
			delete[] typenames[i];
		delete[] typenames;
		typenames = NULL;
	}

	memory->destroy(global_particle_data);
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::init_style() {
	delete[] format;
	char *str;
	if (format_user)
		str = format_user;
	else
		str = format_default;

	int n = strlen(str) + 2;
	format = new char[n];
	strcpy(format, str);
	strcat(format, "\n");

	// initialize typenames array to be backward compatible by default
	// a 32-bit int can be maximally 10 digits plus sign

	if (typenames == NULL) {
		typenames = new char*[ntypes + 1];
		for (int itype = 1; itype <= ntypes; itype++) {
			typenames[itype] = new char[12];
			sprintf(typenames[itype], "%d", itype);
		}
	}

	if (multifile == 0)
		openfile();
}

/* ---------------------------------------------------------------------- */

int DumpSMDVTK::modify_param(int narg, char **arg) {
	if (strcmp(arg[0], "element") == 0) {
		if (narg < ntypes + 1)
			error->all(FLERR, "Dump modify element names do not match atom types");

		if (typenames) {
			for (int i = 1; i <= ntypes; i++)
				delete[] typenames[i];

			delete[] typenames;
			typenames = NULL;
		}

		typenames = new char*[ntypes + 1];
		for (int itype = 1; itype <= ntypes; itype++) {
			int n = strlen(arg[itype]) + 1;
			typenames[itype] = new char[n];
			strcpy(typenames[itype], arg[itype]);
		}

		return ntypes + 1;
	}

	return 0;
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_header(bigint n) {
	if (me == 0) {
		//fprintf(fp,BIGINT_FORMAT "\n",n);
		//fprintf(fp,"Atoms. Timestep: " BIGINT_FORMAT "\n",update->ntimestep);
		fprintf(fp, "# vtk DataFile Version 3.1\n");
		fprintf(fp, "Smooth Mach Dynamics snapshot file for time-step " BIGINT_FORMAT "\n", update->ntimestep);
		fprintf(fp, "ASCII\n");
		fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
		fprintf(fp, "\n");
	}
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::pack(tagint *ids) {
	int m, n;

	tagint *tag = atom->tag;
	int *type = atom->type;
	int *mask = atom->mask;
	double **x = atom->x;
	double **v = atom->v;
	double *rho = atom->rho;
	int nlocal = atom->nlocal;

	m = n = 0;
	for (int i = 0; i < nlocal; i++)
		if (mask[i] & groupbit) {

			buf[m++] = tag[i];
			buf[m++] = type[i];
			buf[m++] = x[i][0];
			buf[m++] = x[i][1];
			buf[m++] = x[i][2]; // 5
			buf[m++] = v[i][0];
			buf[m++] = v[i][1];
			buf[m++] = v[i][2]; // 8
			buf[m++] = rho[i];  // 9

		}
}

/* ----------------------------------------------------------------------
 convert mybuf of doubles to one big formatted string in sbuf
 return -1 if strlen exceeds an int, since used as arg in MPI calls in Dump
 ------------------------------------------------------------------------- */

int DumpSMDVTK::convert_string(int n, double *mybuf) {
	int offset = 0;
	int m = 0;
	for (int i = 0; i < n; i++) {
		if (offset + ONELINE > maxsbuf) {
			if ((bigint) maxsbuf + DELTA > MAXSMALLINT)
				return -1;
			maxsbuf += DELTA;
			memory->grow(sbuf, maxsbuf, "dump:sbuf");
		}

		offset += sprintf(&sbuf[offset], format, typenames[static_cast<int>(mybuf[m + 1])], mybuf[m + 2], mybuf[m + 3],
				mybuf[m + 4]);
		m += size_one;
	}

	return offset;
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_data(int n, double *mybuf) {

	// copy buf atom coords into 3 global arrays

	if (group->count(igroup) < natoms) {
		printf("warning: current number of atoms in group is %d, old number is %d\n", group->count(igroup), natoms);
		printf("resetting natoms\n");
		natoms = group->count(igroup);
	}

	int m = 0;
	for (int i = 0; i < n; i++) {

		//printf("ntotal is %d\n", ntotal);

		if (ntotal > natoms - 1) {
			error->one(FLERR, "allocated storage in dump smd/vtk is exhausted.");
		}

		for (int k = 0; k < NLINES; k++)

		global_particle_data[ntotal][k] = mybuf[m++];
		ntotal++;
	}

	// if last chunk of atoms in this snapshot, write global arrays to file

	if (ntotal == natoms) {
		write_pointdata();
		fprintf(fp, "POINT_DATA %d\n", natoms);
		write_pointdataset_id();
		write_pointdataset_type();
		write_pointdataset_rho();
		write_pointdataset_velocity();
		ntotal = 0;
	}
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_pointdata() {

	fprintf(fp, "POINTS %d FLOAT\n", natoms);

	for (int i = 0; i < natoms; i++) {
		fprintf(fp, "%g %g %g\n", global_particle_data[i][2], global_particle_data[i][3], global_particle_data[i][4]);
	}
	fprintf(fp, "\n");
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_pointdataset_id() {

	fprintf(fp, "SCALARS id FLOAT\n");
	fprintf(fp, "LOOKUP_TABLE default\n");

	for (int i = 0; i < natoms; i++) {
		fprintf(fp, "%g\n", global_particle_data[i][0]);
	}
	fprintf(fp, "\n");
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_pointdataset_type() {

	fprintf(fp, "SCALARS type FLOAT\n");
	fprintf(fp, "LOOKUP_TABLE default\n");

	for (int i = 0; i < natoms; i++) {
		fprintf(fp, "%g\n", global_particle_data[i][1]);
	}
	fprintf(fp, "\n");
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_pointdataset_velocity() {

	fprintf(fp, "VECTORS velocity FLOAT\n");
	fprintf(fp, "LOOKUP_TABLE default\n");

	for (int i = 0; i < natoms; i++) {
		fprintf(fp, "%f %f %f\n", global_particle_data[i][5], global_particle_data[i][6], global_particle_data[i][7]);
	}
	fprintf(fp, "\n");
}

/* ---------------------------------------------------------------------- */

void DumpSMDVTK::write_pointdataset_rho() {

	fprintf(fp, "SCALARS rho FLOAT\n");
	fprintf(fp, "LOOKUP_TABLE default\n");

	for (int i = 0; i < natoms; i++) {
		fprintf(fp, "%g\n", global_particle_data[i][8]);
	}
	fprintf(fp, "\n");
}

