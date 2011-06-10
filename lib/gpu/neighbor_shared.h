/***************************************************************************
                              neighbor_shared.h
                             -------------------
                            W. Michael Brown (ORNL)

  Class for management of data shared by all neighbor lists

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : 
    email                : brownw@ornl.gov
 ***************************************************************************/

#ifndef LAL_NEIGHBOR_SHARED_H
#define LAL_NEIGHBOR_SHARED_H

#ifdef USE_OPENCL

#include "geryon/ocl_kernel.h"
#include "geryon/ocl_texture.h"
using namespace ucl_opencl;

#else

#include "geryon/nvd_kernel.h"
#include "geryon/nvd_texture.h"
using namespace ucl_cudadr;

#endif

class NeighborShared {
 public:
  NeighborShared() : _compiled(false) {}
  ~NeighborShared() { clear(); }
 
  /// Free all memory on host and device
  void clear();

  /// Texture for cached position/type access with CUDA
  UCL_Texture neigh_tex;

  /// Compile kernels for neighbor lists
  void compile_kernels(UCL_Device &dev, const bool gpu_nbor);

  // ----------------------------- Kernels
  UCL_Program *nbor_program, *build_program;
  UCL_Kernel k_nbor, k_cell_id, k_cell_counts, k_build_nbor;
  UCL_Kernel k_transpose, k_special;

 private:
  bool _compiled, _gpu_nbor;
};

#endif
