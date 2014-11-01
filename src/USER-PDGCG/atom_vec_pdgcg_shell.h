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

#ifdef ATOM_CLASS

AtomStyle(pdgcg/shell,AtomVecPDGCGShell)

#else

#ifndef LMP_ATOM_VEC_PDGCG_SHELL_H
#define LMP_ATOM_VEC_PDGCG_SHELL_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecPDGCGShell: public AtomVec {
public:
	AtomVecPDGCGShell(class LAMMPS *);
	~AtomVecPDGCGShell() {
	}
	void init();
	void grow(int);
	void grow_reset();
	void copy(int, int, int);
	void force_clear(int, size_t);
	int pack_comm(int, int *, double *, int, int *);
	int pack_comm_vel(int, int *, double *, int, int *);
	int pack_comm_hybrid(int, int *, double *);
	void unpack_comm(int, int, double *);
	void unpack_comm_vel(int, int, double *);
	int unpack_comm_hybrid(int, int, double *);
	int pack_reverse(int, int, double *);
	int pack_reverse_hybrid(int, int, double *);
	void unpack_reverse(int, int *, double *);
	int unpack_reverse_hybrid(int, int *, double *);
	int pack_border(int, int *, double *, int, int *);
	int pack_border_vel(int, int *, double *, int, int *);
	int pack_border_hybrid(int, int *, double *);
	void unpack_border(int, int, double *);
	void unpack_border_vel(int, int, double *);
	int unpack_border_hybrid(int, int, double *);
	int pack_exchange(int, double *);
	int unpack_exchange(double *);
	int size_restart();
	int pack_restart(int, double *);
	int unpack_restart(double *);
	void create_atom(int, double *);
	void data_atom(double *, imageint, char **);
	int data_atom_hybrid(int, char **);
	void data_vel(int, char **);
	int data_vel_hybrid(int, char **);
	void pack_data(double **);
	int pack_data_hybrid(int, double *);
	void write_data(FILE *, int, double **);
	int write_data_hybrid(FILE *, double *);
	void pack_vel(double **);
	int pack_vel_hybrid(int, double *);
	void write_vel(FILE *, int, double **);
	int write_vel_hybrid(FILE *, double *);
	bigint memory_usage();

protected:
	tagint *tag;
	int *type, *mask;
	imageint *image;
	double **x, **v, **f;
	double *radius, *rmass;

	int *molecule;
	double *vfrac, **x0, *contact_radius, **tlsph_fold, *e, *de, **vest;
	double **tlsph_stress;
	double *eff_plastic_strain;
	double *rho, *drho;
	double *damage;

	int *num_dihedral;
	int **dihedral_type;
	tagint **dihedral_atom1, **dihedral_atom2, **dihedral_atom3,
			**dihedral_atom4;

};

}

#endif
#endif

/* ERROR/WARNING messages:

 E: Per-processor system is too big

 The number of owned atoms plus ghost atoms on a single
 processor must fit in 32-bit integer.

 E: Invalid atom type in Atoms section of data file

 Atom types must range from 1 to specified # of types.

 E: Invalid radius in Atoms section of data file

 Radius must be >= 0.0.

 E: Invalid density in Atoms section of data file

 Density value cannot be <= 0.0.

 */
