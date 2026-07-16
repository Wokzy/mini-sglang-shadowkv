#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "higgs_quantizer.h"

// Inverse of HeadsQuantizeKernel for the [2, 256] (4 bit) grid and the heads-major layout:
//   out[b, h, t, :] = hadamard(lattice[codes[b, h, t, :]].flatten()) * scales[b, h, t] * hadamard_scale
// codes are [B, n_kv_heads, T, head_dim / 2] uint8, scales [B, n_kv_heads, T] float,
// out [B, n_kv_heads, T, head_dim] bf16. No predictor add, just plain dequantization.
//
// Same warp-per-token enumeration through cumulativeLengthes as the quantize kernel, but a warp
// dequantizes two heads of its token at a time: each half-warp owns one head vector, lane l
// produces elements [8l, 8l + 8) of it (code bytes [4l, 4l + 4), one uint32 load). Compared to a
// whole warp per head this halves the shuffle count per row, doubles the load/store widths and
// amortizes the address math -- the kernel is instruction- rather than DRAM-bound. Padding
// tokens are never touched, lengths need no rounding on the python side.
template <size_t kThreadsPerBlock, int kLanesPerRow>
__global__ void __launch_bounds__(kThreadsPerBlock, kLanesPerRow == 16 ? 4 : 2) HeadsDequantizeKernel(
    const void *__restrict__ idxPtr,
    const void *__restrict__ scalesPtr,
    void *__restrict__ outPtr,
    const int *__restrict__ cumulativeLengthes,
    const void *__restrict__ latticePtr,
    size_t batchSize,
    size_t T,
    size_t nKvHeads,
    float hadamard_scale
) {
    static_assert(kLanesPerRow == 16 || kLanesPerRow == 8);
    constexpr int N = 256;
    constexpr int D = 2;
    constexpr int kHeadDim = 128;
    constexpr int kNCodesPerHead = kHeadDim / D;              // 64
    constexpr int kRowsPerWarp = kWarpSize / kLanesPerRow;    // heads of one token at a time
    constexpr int kElemsPerLane = kHeadDim / kLanesPerRow;    // 8 or 16
    constexpr int kCodesPerLane = kElemsPerLane / D;          // 4 or 8 -> one word load per lane
    using CodeWord = std::conditional_t<kCodesPerLane == 4, uint32_t, uint64_t>;

    // Lattice lookups are data-dependent here (indexed by code bytes), so unlike the quantize
    // kernel bank conflicts are possible. One entry is a single 4-byte bank word, so with one
    // copy per lane, laid out as lattice[entry * 32 + lane], lane l always hits bank l --
    // conflict-free without any swizzling. 256 * 32 * 4 bytes = 32KB.
    constexpr int kLatticeCopies = kWarpSize;
    __shared__ __nv_bfloat162 lattice[N * kLatticeCopies];

    auto latticeLoad = reinterpret_cast<const float4*>(latticePtr);
    constexpr int kNLoadsForLattice = D * N * sizeof(nv_bfloat16) / sizeof(float4);
    constexpr int kNLoadedValues = sizeof(float4) / sizeof(__nv_bfloat162); // lattice points per float4 = 4
    #pragma unroll
    for (int i = threadIdx.x; i < kNLoadsForLattice; i += blockDim.x) {
        float4 latticeValues = latticeLoad[i];
        auto latticeElems = reinterpret_cast<__nv_bfloat162*>(&latticeValues);
        #pragma unroll
        for (int j = 0; j < kNLoadedValues; ++j) {
            #pragma unroll
            for (int c = 0; c < kLatticeCopies; ++c) {
                lattice[(i * kNLoadedValues + j) * kLatticeCopies + c] = latticeElems[j];
            }
        }
    }
    __syncthreads();

    const int laneId = threadIdx.x % kWarpSize;
    const int laneInRow = laneId % kLanesPerRow;
    const int rowInWarp = laneId / kLanesPerRow;
    const int globalWarpIdx = (blockIdx.x * kThreadsPerBlock + threadIdx.x) / kWarpSize;
    const int nWarps = gridDim.x * (kThreadsPerBlock / kWarpSize);
    const int totalTokens = cumulativeLengthes[batchSize];

    auto idxLoad = reinterpret_cast<const CodeWord*>(idxPtr);
    auto scalesLoad = reinterpret_cast<const float*>(scalesPtr);
    auto outStore = reinterpret_cast<int4*>(outPtr);
    constexpr int kInt4PerLane = kElemsPerLane * sizeof(nv_bfloat16) / sizeof(int4); // 1 or 2

    int batchHint = 0;
    for (int currentToken = globalWarpIdx; currentToken < totalTokens; currentToken += nWarps) {
        const int batchIdx = FindBatchIdxFromHint(cumulativeLengthes, batchSize, currentToken, batchHint);
        batchHint = batchIdx;
        const int tokenWithinSample = currentToken - cumulativeLengthes[batchIdx];
        for (int headPair = 0; headPair < ceildiv(nKvHeads, kRowsPerWarp); ++headPair) {
            const int head = headPair * kRowsPerWarp + rowInWarp;
            const bool headValid = head < nKvHeads; // odd n_kv tail: the upper half-warp idles
            const size_t row = (size_t(batchIdx) * nKvHeads + head) * T + tokenWithinSample;

            CodeWord codes = 0;
            float mult = 0.0f;
            if (headValid) {
                codes = idxLoad[row * (kNCodesPerHead / kCodesPerLane) + laneInRow];
                mult = scalesLoad[row] * hadamard_scale;
            }

            float x[kElemsPerLane];
            #pragma unroll
            for (int c = 0; c < kCodesPerLane; ++c) {
                float2 v = __bfloat1622float2(lattice[unsigned((codes >> (8 * c)) & 0xFFu) * kLatticeCopies + laneId]);
                x[2 * c] = v.x;
                x[2 * c + 1] = v.y;
            }

            hadamard_thread(x);
            hadamard_warp<cilog2(kLanesPerRow)>(x); // uses threadIdx.x % kLanesPerRow, stays within the row's lanes

            if (headValid) {
                __align__(16) __nv_bfloat162 result[kElemsPerLane / 2];
                #pragma unroll
                for (int c = 0; c < kElemsPerLane / 2; ++c) {
                    result[c] = __float22bfloat162_rn({x[2 * c] * mult, x[2 * c + 1] * mult});
                }
                #pragma unroll
                for (int c = 0; c < kInt4PerLane; ++c) {
                    outStore[row * (kHeadDim * sizeof(nv_bfloat16) / sizeof(int4)) + laneInRow * kInt4PerLane + c] = reinterpret_cast<int4*>(result)[c];
                }
            }
        }
    }
}

template <size_t kThreadsPerBlock, int kLanesPerRow>
void HeadsDequantizeImpl(
    at::Tensor &idx,
    at::Tensor &scales,
    at::Tensor &cum_lengths,
    at::Tensor &lattice,
    at::Tensor &out,
    float hadamard_scale
) {
    at::cuda::CUDAGuard device_guard{(char)idx.get_device()};

    auto* kernelInstance = HeadsDequantizeKernel<kThreadsPerBlock, kLanesPerRow>;
    int numBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernelInstance, kThreadsPerBlock, 0);

    // raw cudaGetDeviceProperties takes ~1ms per call and would stall every launch, torch caches it
    int smCount = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    const dim3 gridDim{unsigned(numBlocks * smCount), 1, 1};
    const dim3 blockDim{unsigned(kThreadsPerBlock), 1, 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    kernelInstance<<<gridDim, blockDim, 0, stream>>>(
        idx.data_ptr(),
        scales.data_ptr(),
        out.data_ptr(),
        reinterpret_cast<int*>(cum_lengths.data_ptr()),
        lattice.data_ptr(),
        static_cast<size_t>(idx.size(0)),   // batchSize
        static_cast<size_t>(idx.size(2)),   // T (padded seq len)
        static_cast<size_t>(idx.size(1)),   // nKvHeads
        hadamard_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void HeadsDequantize(
    at::Tensor &idx,
    at::Tensor &scales,
    at::Tensor &cum_lengths,
    at::Tensor &lattice,
    at::Tensor &out,
    float hadamard_scale
) {
    // 8 lanes per row (4 heads per warp iteration) amortizes shuffles/loop overhead best, but
    // would idle half the warp when the token has fewer than 4 heads to process
    if (idx.size(1) % 4 == 0) {
        HeadsDequantizeImpl<512, 8>(idx, scales, cum_lengths, lattice, out, hadamard_scale);
    } else {
        HeadsDequantizeImpl<512, 16>(idx, scales, cum_lengths, lattice, out, hadamard_scale);
    }
}
