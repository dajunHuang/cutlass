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
    \brief
*/

#pragma once

#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_single_problem_visitor.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,                 ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock mapping function
          bool Transposed = false>
struct GemmUniversalPersistent {
   public:
    //
    // Types and constants
    //

    using Mma = Mma_;           // third_party/cutlass/include/cutlass/gemm/kernel/default_gemm.h:973
    using Epilogue = Epilogue_; // third_party/cutlass/include/cutlass/gemm/kernel/default_gemm.h:1001
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static bool const kTransposed = Transposed;

    // Optional transpose
    using MapArguments = kernel::detail::MapArguments<
        typename Mma::IteratorA::Element, typename Mma::IteratorA::Layout, Mma::kTransformA,
        Mma::IteratorA::AccessType::kElements, typename Mma::IteratorB::Element,
        typename Mma::IteratorB::Layout, Mma::kTransformB, Mma::IteratorB::AccessType::kElements,
        typename Mma::LayoutC, kTransposed>;

    // Public-facing type definitions related to operand element type, layout, and complex conjugate
    // operation. Must interact with the 'kTransposed' notion.
    using ElementA = typename MapArguments::ElementA;
    using LayoutA = typename MapArguments::LayoutA;
    using ElementB = typename MapArguments::ElementB;
    using LayoutB = typename MapArguments::LayoutB;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename MapArguments::LayoutC;

    /// The per-thread tile of raw accumulators
    using AccumulatorTile = typename Mma::FragmentC;

    static ComplexTransform const kTransformA = MapArguments::kTransformA;
    static ComplexTransform const kTransformB = MapArguments::kTransformB;

    using Operator = typename Mma::Operator;
    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;
    static int const kAlignmentA = MapArguments::kAlignmentA;
    static int const kAlignmentB = MapArguments::kAlignmentB;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    using ProblemVisitor = GemmSingleProblemVisitor<ThreadblockShape, kTransposed>;

    //
    // Structures
    //

    /// Argument structure
    struct Arguments {
        //
        // Data members
        //

        GemmUniversalMode mode = GemmUniversalMode::kGemm;
        GemmCoord problem_size{};
        int batch_count{1};

        typename EpilogueOutputOp::Params epilogue{};

        ElementA* ptr_A = nullptr;
        ElementB* ptr_B = nullptr;
        ElementC* ptr_C = nullptr;
        ElementC* ptr_D = nullptr;

        typename LayoutA::Stride stride_a{0};
        typename LayoutB::Stride stride_b{0};
        typename LayoutC::Stride stride_c{0};
        typename LayoutC::Stride stride_d{0};

        typename LayoutA::Stride::LongIndex lda{0};
        typename LayoutB::Stride::LongIndex ldb{0};
        typename LayoutC::Stride::LongIndex ldc{0};
        typename LayoutC::Stride::LongIndex ldd{0};

        int threadblock_count{0};

        //
        // Methods
        //

        /// Default Constructor
        Arguments() = default;

        /// Constructor
        Arguments(GemmUniversalMode mode, GemmCoord problem_size, int threadblock_count,
                  typename EpilogueOutputOp::Params epilogue, ElementA* ptr_A, ElementB* ptr_B,
                  ElementC* ptr_C, ElementC* ptr_D, typename LayoutA::Stride stride_a,
                  typename LayoutB::Stride stride_b, typename LayoutC::Stride stride_c,
                  typename LayoutC::Stride stride_d)
            : mode(mode),
              problem_size(problem_size),
              threadblock_count(threadblock_count),
              epilogue(epilogue),
              ptr_A(ptr_A),
              ptr_B(ptr_B),
              ptr_C(ptr_C),
              ptr_D(ptr_D),
              stride_a(stride_a),
              stride_b(stride_b),
              stride_c(stride_c),
              stride_d(stride_d) {
            CUTLASS_TRACE_HOST(
                "GemmUniversalPersistent::Arguments::Arguments() - problem_size: " << problem_size);
        }

        /// Constructor
        Arguments(GemmUniversalMode mode, GemmCoord problem_size, int threadblock_count,
                  typename EpilogueOutputOp::Params epilogue, ElementA* ptr_A, ElementB* ptr_B,
                  ElementC* ptr_C, ElementC* ptr_D, typename LayoutA::Stride::LongIndex lda,
                  typename LayoutB::Stride::LongIndex ldb, typename LayoutC::Stride::LongIndex ldc,
                  typename LayoutC::Stride::LongIndex ldd)
            : mode(mode),
              problem_size(problem_size),
              threadblock_count(threadblock_count),
              epilogue(epilogue),
              ptr_A(ptr_A),
              ptr_B(ptr_B),
              ptr_C(ptr_C),
              ptr_D(ptr_D),
              lda(lda),
              ldb(ldb),
              ldc(ldc),
              ldd(ldd) {
            stride_a = make_Coord(lda);
            stride_b = make_Coord(ldb);
            stride_c = make_Coord(ldc);
            stride_d = make_Coord(ldd);
            CUTLASS_TRACE_HOST(
                "GemmUniversalPersistent::Arguments::Arguments() - problem_size: " << problem_size);
        }

        /// Returns arguments for the transposed problem
        Arguments transposed_problem() const {
            Arguments args(*this);

            std::swap(args.problem_size.m(), args.problem_size.n());
            std::swap(args.ptr_A, args.ptr_B);
            std::swap(args.lda, args.ldb);
            std::swap(args.stride_a, args.stride_b);

            return args;
        }
    };

    /// Parameters structure
    struct Params {
       public:
        //
        // Data members
        //
        typename ProblemVisitor::Params problem_visitor{};
        ElementA* ptr_A = nullptr;
        ElementB* ptr_B = nullptr;
        ElementC* ptr_C = nullptr;
        ElementC* ptr_D = nullptr;

        typename LayoutA::Stride::LongIndex lda{0};
        typename LayoutB::Stride::LongIndex ldb{0};
        typename LayoutC::Stride::LongIndex ldc{0};
        typename LayoutC::Stride::LongIndex ldd{0};

        GemmUniversalMode mode = GemmUniversalMode::kGemm;

        ThreadblockSwizzle block_mapping{};

        void* barrier_workspace = nullptr;
        void* partials_workspace = nullptr;

        typename EpilogueOutputOp::Params output_op{};

        int device_sms = 0;
        int sm_occupancy = 0;
        int threadblock_count{0};

       public:
        //
        // Host dispatch API
        //

        /// Default constructor
        Params() = default;

        /// Constructor
        Params(Arguments const& args,  /// GEMM application arguments
               int device_sms,         /// Number of SMs on the device
               int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
            : problem_visitor(args.problem_size),
              lda(args.lda),
              ldb(args.ldb),
              ldc(args.ldc),
              ldd(args.ldd),
              output_op(args.epilogue),
              mode(args.mode),
              ptr_A(args.ptr_A),
              ptr_B(args.ptr_B),
              ptr_C(args.ptr_C),
              ptr_D(args.ptr_D),
              device_sms(device_sms),
              sm_occupancy(sm_occupancy),
              threadblock_count(args.threadblock_count <= 0 ? device_sms + args.threadblock_count
                                                            : args.threadblock_count) {
            // Initialize the block mapping structure
            block_mapping = ThreadblockSwizzle(
                args.mode, args.problem_size,
                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, sm_occupancy,
                device_sms, sizeof(ElementA), sizeof(ElementB), sizeof(ElementC),
                Epilogue::kAccumulatorFragments);
        }

        /// Returns the workspace size (in bytes) needed for these parameters
        size_t get_workspace_size() const { return 0; }

        /// Assign and initialize the specified workspace buffer.  Assumes
        /// the memory allocated to workspace is at least as large as get_workspace_size().
        Status init_workspace(void* workspace, cudaStream_t stream = nullptr) {
            return Status::kSuccess;
        }

        /// Returns the GEMM volume in thread block tiles
        cutlass::gemm::GemmCoord get_tiled_shape() const { return {-1, -1, -1}; }

        /// Returns the total number of thread blocks to launch
        int get_grid_blocks() const {
            dim3 grid_dims = get_grid_dims();
            return grid_dims.x * grid_dims.y * grid_dims.z;
        }

        /// Returns the grid extents in thread blocks to launch
        dim3 get_grid_dims() const { return threadblock_count; }

        /// Lightweight update given a subset of arguments.
        void update(Arguments const& args) {
            CUTLASS_TRACE_HOST("GemmUniversalPersistent::Params::update()");
            mode = args.mode;

            problem_visitor = typename ProblemVisitor::Params(args.problem_size);
            threadblock_count = args.threadblock_count <= 0 ? device_sms + args.threadblock_count
                                                            : args.threadblock_count;

            lda = args.lda;
            ldb = args.ldb;
            ldc = args.ldc;
            ldd = args.ldd;

            output_op = args.epilogue;

            // Update input/output pointers
            ptr_A = args.ptr_A;
            ptr_B = args.ptr_B;
            ptr_C = args.ptr_C;
            ptr_D = args.ptr_D;

            block_mapping = ThreadblockSwizzle(
                args.mode, args.problem_size,
                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, sm_occupancy,
                device_sms, sizeof(ElementA), sizeof(ElementB), sizeof(ElementC),
                Epilogue::kAccumulatorFragments);
        }
    };

    /// Shared memory storage structure
    union SharedStorage {
        union {
            typename Mma::SharedStorage main_loop;
            typename Epilogue::SharedStorage epilogue;
        } kernel;
        // ProblemVisitor shared storage can't be overlapped with others
        typename ProblemVisitor::SharedStorage problem_visitor;
    };

   protected:
    //
    // Data members
    //

    /// GEMM problem parameters
    Params params;

    /// Shared storage reference
    SharedStorage& shared_storage;

    /// ID within the threadblock
    int thread_idx;

    /// ID of warp
    int warp_idx;

    /// ID of each thread within a warp
    int lane_idx;

    /// Threadblock scoped epilogue
    Epilogue epilogue;

   public:
    //
    // Host-only dispatch API
    //

    /// Determines whether the GEMM problem size satisfies this kernel's
    /// alignment requirements
    static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
        CUTLASS_TRACE_HOST("GemmUniversalPersistent::can_implement()");

        static int const kAlignmentA =
            (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value) ? 32
            : (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value)
                ? 64
                : Mma::IteratorA::AccessType::kElements;
        static int const kAlignmentB =
            (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value) ? 32
            : (platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value)
                ? 64
                : Mma::IteratorB::AccessType::kElements;
        static int const kAlignmentC =
            (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value) ? 32
            : (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value)
                ? 64
                : Epilogue::OutputTileIterator::kElementsPerAccess;

        bool isAMisaligned = false;
        bool isBMisaligned = false;
        bool isCMisaligned = false;

        if (platform::is_same<LayoutA, layout::RowMajor>::value) {
            isAMisaligned = problem_size.k() % kAlignmentA;
        } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
            isAMisaligned = problem_size.m() % kAlignmentA;
        } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value ||
                   platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
            isAMisaligned = problem_size.k() % kAlignmentA;
        }

        if (platform::is_same<LayoutB, layout::RowMajor>::value) {
            isBMisaligned = problem_size.n() % kAlignmentB;
        } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
            isBMisaligned = problem_size.k() % kAlignmentB;
        } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value ||
                   platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
            isBMisaligned = problem_size.k() % kAlignmentB;
        }

        if (platform::is_same<LayoutC, layout::RowMajor>::value) {
            isCMisaligned = problem_size.n() % kAlignmentC;
        } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
            isCMisaligned = problem_size.m() % kAlignmentC;
        } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value ||
                   platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
            isCMisaligned = problem_size.n() % kAlignmentC;
        }

        if (isAMisaligned) {
            CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
            return Status::kErrorMisalignedOperand;
        }

        if (isBMisaligned) {
            CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
            return Status::kErrorMisalignedOperand;
        }

        if (isCMisaligned) {
            CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
            return Status::kErrorMisalignedOperand;
        }

        CUTLASS_TRACE_HOST("  returning kSuccess");

        return Status::kSuccess;
    }

    /// Determines whether the GEMM problem satisfies this kernel's
    /// alignment requirements
    static Status can_implement(Arguments const& args) { return can_implement(args.problem_size); }

   protected:
    /// Executes one GEMM
    CUTLASS_DEVICE
    void gemm() {
        //
        // These types shadow the type-level definitions and support the ability to implement
        // a 'transposed' GEMM that computes the transposed problems.
        //
        using ElementA = typename Mma::IteratorA::Element;
        using LayoutA = typename Mma::IteratorA::Layout;
        using ElementB = typename Mma::IteratorB::Element;
        using LayoutB = typename Mma::IteratorB::Layout;
        using ElementC = typename Epilogue::OutputTileIterator::Element;
        using LayoutC = typename Epilogue::OutputTileIterator::Layout;

        //
        // Problem visitor.
        //
        ProblemVisitor problem_visitor(params.problem_visitor, shared_storage.problem_visitor,
                                       blockIdx.x);

        GemmCoord problem_size = problem_visitor.problem_size();
        GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

        while (problem_visitor.next_tile()) {
            int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

            cutlass::gemm::GemmCoord threadblock_offset(
                int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
                int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN, 0);

            // Load element pointers. Exchange pointers and strides if working on the transpose
            ElementA* ptr_A =
                reinterpret_cast<ElementA*>((kTransposed ? params.ptr_B : params.ptr_A));
            typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb : params.lda);

            ElementB* ptr_B =
                reinterpret_cast<ElementB*>((kTransposed ? params.ptr_A : params.ptr_B));
            typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda : params.ldb);

            // Compute initial location in logical coordinates
            cutlass::MatrixCoord tb_offset_A{
                threadblock_offset.m(),
                0,
            };

            cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

            // Compute position within threadblock
            int thread_idx = threadIdx.x;

            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(LayoutA(ldm_A), ptr_A,
                                               {problem_size.m(), problem_size.k()}, thread_idx,
                                               tb_offset_A);

            typename Mma::IteratorB iterator_B(LayoutB(ldm_B), ptr_B,
                                               {problem_size.k(), problem_size.n()}, thread_idx,
                                               tb_offset_B);

            typename Mma::FragmentC accumulators;

            accumulators.clear();

            // Broadcast the warp_id computed by lane 0 to ensure dependent code
            // is compiled as warp-uniform.
            int warp_idx = canonical_warp_idx_sync();

            int lane_idx = threadIdx.x % 32;

            //
            // Matrix multiply phase
            //

            // Construct thread-scoped matrix multiply
            Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

            // Compute threadblock-scoped matrix multiply-add
            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            // Wait for all threads to finish their epilogue phases from the previous tile.
            __syncthreads();

            // Compute threadblock-scoped matrix multiply-add
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

            //
            // Epilogue
            //

            EpilogueOutputOp output_op(params.output_op);

            typename Epilogue::OutputTileIterator::Params params_C(params.ldc);
            typename Epilogue::OutputTileIterator::Params params_D(params.ldd);

            // Tile iterator loading from source tensor.
            typename Epilogue::OutputTileIterator iterator_C(
                params_C, params.ptr_C, problem_size.mn(), thread_idx, threadblock_offset.mn());

            // Tile iterator writing to destination tensor.
            typename Epilogue::OutputTileIterator iterator_D(
                params_D, params.ptr_D, problem_size.mn(), thread_idx, threadblock_offset.mn());

            Epilogue epilogue(shared_storage.kernel.epilogue, thread_idx, warp_idx, lane_idx);

            // Execute the epilogue operator to update the destination tensor.
            epilogue(output_op, iterator_D, accumulators, iterator_C);

            // Next tile
            problem_visitor.advance(gridDim.x);
        }
    }

   public:
    //
    // Device-only API
    //

    // Factory invocation
    CUTLASS_DEVICE
    static void invoke(Params const& params, SharedStorage& shared_storage) {
        GemmUniversalPersistent op(params, shared_storage);
        op();
    }

    // Constructor
    CUTLASS_DEVICE
    GemmUniversalPersistent(Params const& params, SharedStorage& shared_storage)
        : params(params),
          shared_storage(shared_storage),
          thread_idx(threadIdx.x),
          warp_idx(
              __shfl_sync(0xffffffff, threadIdx.x / 32,
                          0)),  // broadcast the warp_id computed by lane 0 to ensure dependent code
          lane_idx(threadIdx.x % 32),
          epilogue(shared_storage.kernel.epilogue, thread_idx, warp_idx, lane_idx) {}

    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()() { gemm(); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
