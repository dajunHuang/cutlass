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
    \brief Problem visitor for single GEMMs.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Helper for correctly representing problem sizes in single GEMM kernels 
template <
  typename ThreadblockShape,
  bool Transposed
>
struct GemmSingleProblemSizeHelper {

  static bool const kTransposed = Transposed;

  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return cutlass::gemm::GemmCoord(
      ((problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
      ((problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN),
      1);
  }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem) {
    if (kTransposed) {
      cutlass::swap(problem.m(), problem.n());
    }
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord& grid) {
    return grid.m() * grid.n();
  }
};

} // namespace detail

/// Visitor class to abstract away the algorithm for iterating over tiles for a single problem
template <typename ThreadblockShape_,
          bool Transposed = false>
struct GemmSingleProblemVisitor {

  using ThreadblockShape = ThreadblockShape_;
  static bool const kTransposed = Transposed;
  using ProblemSizeHelper = detail::GemmSingleProblemSizeHelper<ThreadblockShape, kTransposed>;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    int32_t tile_count;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): tile_count(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord problem_size_
    ): problem_size(problem_size_) {
      ProblemSizeHelper::possibly_transpose_problem(problem_size);
      auto grid = ProblemSizeHelper::grid_shape(problem_size);
      tile_count = ProblemSizeHelper::tile_count(grid);
    }
  };

  struct SharedStorage {};

  Params params;
  int32_t tile_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GemmSingleProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ):
  params(params_),
  tile_idx(block_idx)
  {}

  /// Returns true if the tile is within the problem
  CUTLASS_DEVICE
  bool next_tile() {
    return tile_idx < params.tile_count;
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int32_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return 0;
  }

  CUTLASS_HOST_DEVICE
  int32_t threadblock_idx() const {
    return tile_idx;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    return params.problem_size;
  }

  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return ProblemSizeHelper::grid_shape(problem);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
