/*
################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
*/


#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm.h"
#include "helper.h"

#include "functions.h"

////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"

#include "batch_gemm_softmax.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM types
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementCompute = float;
using ElementD = ElementC;       // bfloat16_t
/// Softmax types
using ElementSoftmax = ElementC; // bfloat16_t
using ElementSoftmaxCompute = float;
using ElementNorm = float;
using ElementSum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentSoftmax =
    128 / cutlass::sizeof_bits<ElementSoftmax>::value;

using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80;

// ApplyShape for final Softmax.
using ApplyShape = cutlass::MatrixShape<1, 1024>;
static int const kStages = 4;

/// Linear scaling operator
using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
>;

using BatchGemmSoftmax = cutlass::BatchGemmSoftmax<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    kStages,
    ApplyShape,
    AlignmentA,
    AlignmentB,
    AlignmentSoftmax,
    ElementNorm,
    ElementSum,
    ElementSoftmax,
    ElementSoftmaxCompute
>;

using LayoutC = typename BatchGemmSoftmax::LayoutC;
using LayoutN = typename BatchGemmSoftmax::LayoutN;
using LayoutS = typename BatchGemmSoftmax::LayoutS;
using MatrixCoord = typename LayoutC::TensorCoord;


void batch_gemm_softmax(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha = 1.0,
    float beta = 0.0
) {
    cutlass::gemm::GemmCoord problem = {m, n, k};

    int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    // fixed rowmajor for norm and sum
    int64_t ldn = problem.m();
    int64_t lds = problem.m();

    int block_num = (problem.n() + BatchGemmSoftmax::ThreadblockShape::kN - 1) / BatchGemmSoftmax::ThreadblockShape::kN;

    BatchGemmSoftmax::Arguments args(
      problem,
      batch_count,
      {reinterpret_cast<ElementA *>(A.data_ptr()), lda},
      {reinterpret_cast<ElementB *>(B.data_ptr()), ldb},
      {nullptr, ldc},
      {reinterpret_cast<ElementD *>(D.data_ptr()), ldc},
      {
        ElementCompute(alpha),
        ElementCompute(beta)
      },
      {reinterpret_cast<ElementNorm *>(Norm.data_ptr()), ldn},
      {reinterpret_cast<ElementSum *>(Sum.data_ptr()), lds},
      {reinterpret_cast<ElementSoftmax *>(Softmax.data_ptr()), ldc},
      problem.m() * problem.k(),
      problem.k() * problem.n(),
      problem.m() * problem.n(),
      problem.m() * problem.n(),
      block_num * problem.m(),
      block_num * problem.m(),
      problem.m() * problem.n()
    );

    BatchGemmSoftmax batch_gemm_softmax;

    CUTLASS_CHECK(batch_gemm_softmax.initialize(args));

    CUTLASS_CHECK(batch_gemm_softmax(at::cuda::getCurrentCUDAStream()));
}

using IndexedGemm = BatchGemmSoftmax::GemmKernel;
using IndexedReduction = BatchGemmSoftmax::ApplyFinalReductionKernel;
using IndexedSoftmax = BatchGemmSoftmax::SoftmaxApplyKernel;

__global__ void indexed_gemm(IndexedGemm::Params params, const int* slots,
                            const int* lengths, int heads, int64_t query_stride) {
    int active = blockIdx.z / heads;
    int slot = slots[active];
    int n = lengths[slot];
    if (n == 0 || blockIdx.y * ThreadblockShape::kN >= n) return;
    params.problem_size.n() = n;
    params.ptr_A = static_cast<ElementA*>(params.ptr_A) +
        int64_t(active) * (query_stride - heads * params.batch_stride_A);
    params.ptr_B = static_cast<ElementB*>(params.ptr_B) +
        int64_t(slot - active) * heads * params.batch_stride_B;
    extern __shared__ int shared[];
    IndexedGemm op;
    op(params, *reinterpret_cast<IndexedGemm::SharedStorage*>(shared));
}

__global__ void indexed_reduction(IndexedReduction::Params params,
                                 const int* slots, const int* lengths, int heads) {
    int n = lengths[slots[blockIdx.z / heads]];
    if (n == 0) return;
    params.args.problem_size.n() = n;
    extern __shared__ int shared[];
    IndexedReduction op;
    op(params, *reinterpret_cast<IndexedReduction::SharedStorage*>(shared));
}

__global__ void indexed_softmax(IndexedSoftmax::Params params,
                               const int* slots, const int* lengths, int heads) {
    int n = lengths[slots[blockIdx.z / heads]];
    if (n == 0) return;
    params.args.extent.column() = n;
    extern __shared__ int shared[];
    IndexedSoftmax op;
    op(params, *reinterpret_cast<IndexedSoftmax::SharedStorage*>(shared));
}

void batch_gemm_softmax_indexed(torch::Tensor A, torch::Tensor B,
    torch::Tensor D, torch::Tensor Norm, torch::Tensor Sum, torch::Tensor Softmax,
    torch::Tensor slots, torch::Tensor lengths, int heads, int m, int max_n) {
    int n = B.size(2), k = 128;
    int batches = slots.numel() * heads;
    int blocks = (n + 255) / 256;
    BatchGemmSoftmax::Arguments args(
        {m,max_n,k}, batches,
        {reinterpret_cast<ElementA*>(A.data_ptr()), k},
        {reinterpret_cast<ElementB*>(B.data_ptr()), k},
        {nullptr,n}, {reinterpret_cast<ElementD*>(D.data_ptr()),n},
        {float(1.0 / sqrt(128.0)),0.0f},
        {Norm.data_ptr<float>(),m}, {Sum.data_ptr<float>(),m},
        {reinterpret_cast<ElementSoftmax*>(Softmax.data_ptr()),n},
        m*k, int64_t(n)*k, m*n, m*n, blocks*m, blocks*m, m*n);
    BatchGemmSoftmax::Params params(args);
    auto stream = at::cuda::getCurrentCUDAStream();
    int smem = sizeof(IndexedGemm::SharedStorage);
    C10_CUDA_CHECK(cudaFuncSetAttribute(indexed_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    indexed_gemm<<<dim3((m+31)/32,(max_n+255)/256,batches),IndexedGemm::kThreadCount,smem,stream>>>(
        params.gemm, slots.data_ptr<int>(), lengths.data_ptr<int>(), heads, A.stride(0));
    indexed_reduction<<<dim3((m+31)/32,1,batches),32,sizeof(IndexedReduction::SharedStorage),stream>>>(
        params.reduction, slots.data_ptr<int>(), lengths.data_ptr<int>(), heads);
    // The original ApplySoftmax loops over N internally; one CTA per query row.
    indexed_softmax<<<dim3(m,1,batches),1024,sizeof(IndexedSoftmax::SharedStorage),stream>>>(
        params.softmax, slots.data_ptr<int>(), lengths.data_ptr<int>(), heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
