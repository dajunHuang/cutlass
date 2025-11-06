/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Implements streamk threadblock mapping blockIdx to GEMM problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/gemm/threadblock/index_remat.h"

#if !defined(__CUDACC_RTC__)
#include <iostream>
#include "cutlass/core_io.h"
#include "cutlass/trace.h"
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock mapping control for GEMMs
struct ThreadblockSwizzlePersistent {

  /// Advertise StreamkFeature
  using PersistentFeature = void;


  /// Kernel traits
  template <typename GemmKernel>
  struct KernelTraits {};

  /// The 3D value-extents of the GEMM computation volume (m,n,k)
  GemmCoord problem_size;

  /// CTA occupancy per SM
  int sm_occupancy;

  ThreadblockSwizzlePersistent() = default;

  /// Returns the GEMM volume in thread block tiles
  CUTLASS_HOST_DEVICE
  GemmCoord tiled_shape() const
  {
    return GemmCoord(
        static_cast<int>(1),
        static_cast<int>(1),
        1);
  }

  /// Constructor: *Gemm* problem size (m, n, k)
  ThreadblockSwizzlePersistent(
    GemmUniversalMode const mode_,
    GemmCoord const problem_size_,
    GemmCoord const tile_size_,
    int const sm_occupancy_,
    int const device_sms_,
    size_t const element_A_bytes_,
    size_t const element_B_bytes_,
    size_t const element_C_bytes_,
    int const epilogue_acc_fragments_)
  :
    problem_size(problem_size_),
    sm_occupancy(sm_occupancy_)
  {

  }


  /// Obtains grid extents in CTAs
  dim3 get_grid_dims() const
  {
    return dim3(1, 1, 1);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

