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
#include <cuda_bf16.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define SORT_OFFSET 1

#ifndef BLOCK_SIZE_CP
#define BLOCK_SIZE_CP 128
#endif

#include "copy.cuh"
#include "functions.h"
#include "map.cuh"

#if BLOCK_SIZE_CP == 128
#define PTYPE int4
#endif

#if BLOCK_SIZE_CP == 256
#define PTYPE int2
#endif

void gather_copy(
    torch::Tensor values, torch::Tensor v_cache_buffer, torch::Tensor position_ids,
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int map_size = 256)
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads;
    int maxSMBytes = CPY_SIZE*2*1024 + map_size*4; // must less than 160 KB

    // Cast bf16 data pointers to int2 or int4
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());

    // this only needs to run once
    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 256><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    }  else if (map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 128><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    } else if (map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 512><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    } else if (map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 1024><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    }
}

/// for keys
void gather_copy_d2d_with_offsets(
    torch::Tensor keys, 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    int batch_size, int heads, 
    int gpu_v_length, 
    int gpu_v_offset, 
    int gpu_v_stride, 
    int map_size = 256)
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads;
    int maxSMBytes = CPY_SIZE*2*1024 + map_size*4 + sizeof(PTYPE); // must less than 160 KB

    // Cast bf16 data pointers to int2 or int4
    PTYPE* keys_ptr = reinterpret_cast<PTYPE*>(keys.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());

    // this only needs to run once
    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);
    } else if (map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);    
    } else if (map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 512><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);
    } else if (map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 1024><<<numBlocks, blockSize, maxSMBytes, at::cuda::getCurrentCUDAStream()>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0, /*start*/
        cnts_ptr);
    }
}

// reorder position ids, and compute offsets and cnts by computing cache hits/misses
// call it before gather_copy_with_offsets
void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // inout, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks, int type
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks, int type
    int batch_size, int heads, 
    int map_size = 256)
{
    int blockSize = map_size;
    int numBlocks = batch_size * heads;

    int64_t* cached_pos = cached_pos_ids.data_ptr<int64_t>();
    int64_t* cur_pos = cur_pos_ids.data_ptr<int64_t>();
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());


    if(map_size == 256) {
        reorder_keys_and_mixed_offsets<int64_t, 256, 1024><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 128) {
        reorder_keys_and_mixed_offsets<int64_t, 128, 1024><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 512) {
        reorder_keys_and_mixed_offsets<int64_t, 512, 2048><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 1024) {
        reorder_keys_and_mixed_offsets<int64_t, 1024, 4096><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    }
}

// gather copy with offsets
// call it after reorder_keys_and_compute_offsets
void gather_copy_with_offsets(
    torch::Tensor values,           // input, cpu values
    torch::Tensor v_cache_buffer,   // inout, gpu values
    torch::Tensor temp,             // a temp gpu memory for copy, size same as single layer v_cache_buffer 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    torch::Tensor signals,          // extra internal signals, all zeros sizes as numBlocks, size as numBlocks
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int gpu_v_offset, int gpu_v_stride, int map_size = 256) // input, torch stream
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads * 2; // numBlocksBP = 2 * BLOCK_NUM
    int maxSMBytes = CPY_SIZE * 2 * 1024 + map_size * 4 + sizeof(PTYPE); // must less than 160 KB

    // Cast bf16 data pointers to int2 or int4
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());
    PTYPE* temp_ptr = reinterpret_cast<PTYPE*>(temp.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());
    unsigned int* signals_ptr = reinterpret_cast<unsigned int*>(signals.data_ptr<int32_t>());
    
    // Get cudaStream_t from torch::cuda::Stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    } else if(map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    }else if(map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 512><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    } else if(map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 1024><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    }
}


// Continuous batches: persistent data is slot-indexed; workspaces are compact.
template<int M, int LUT>
void indexed_cache_ops(torch::Tensor positions, torch::Tensor selected,
    torch::Tensor offsets, torch::Tensor counts, torch::Tensor slots,
    torch::Tensor lengths, int heads) {
    reorder_keys_and_mixed_offsets<int64_t, M, LUT>
        <<<slots.numel() * heads, M, 0, at::cuda::getCurrentCUDAStream()>>>(
            positions.data_ptr<int64_t>(), selected.data_ptr<int64_t>(),
            positions.data_ptr<int64_t>(), offsets.data_ptr<int>(), counts.data_ptr<int>(),
            slots.data_ptr<int>(), lengths.data_ptr<int>(), heads);
}

void reorder_keys_indexed(torch::Tensor positions, torch::Tensor selected,
    torch::Tensor offsets, torch::Tensor counts, torch::Tensor slots,
    torch::Tensor lengths, int heads, int map_size) {
    switch (map_size) {
        case 128: indexed_cache_ops<128,1024>(positions,selected,offsets,counts,slots,lengths,heads); break;
        case 256: indexed_cache_ops<256,1024>(positions,selected,offsets,counts,slots,lengths,heads); break;
        case 512: indexed_cache_ops<512,2048>(positions,selected,offsets,counts,slots,lengths,heads); break;
        case 1024: indexed_cache_ops<1024,4096>(positions,selected,offsets,counts,slots,lengths,heads); break;
        default: TORCH_CHECK(false, "Unsupported ShadowKV selection size");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int M>
void indexed_gather(torch::Tensor values, torch::Tensor cache, torch::Tensor temp,
    torch::Tensor offsets, torch::Tensor counts, torch::Tensor signals,
    torch::Tensor slots, torch::Tensor starts, torch::Tensor lengths,
    int heads, int cpu_stride, int cache_stride, bool key_only) {
    int smem = CPY_SIZE * 2 * 1024 + M * 4 + sizeof(PTYPE);
    if (key_only) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gather_copy_d2d<PTYPE,M>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        gather_copy_d2d<PTYPE,M><<<slots.numel()*heads, BLOCK_SIZE_CP, smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<PTYPE*>(cache.data_ptr<at::BFloat16>()), nullptr,
            M*8*128, 0, cache_stride, offsets.data_ptr<int>(), 0, counts.data_ptr<int>(),
            slots.data_ptr<int>(), starts.data_ptr<int>(), lengths.data_ptr<int>(), heads);
    } else {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE,M>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        gather_copy_var_midpoint_BP<PTYPE,M><<<slots.numel()*heads*2, BLOCK_SIZE_CP, smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(temp.data_ptr<at::BFloat16>()),
            reinterpret_cast<PTYPE*>(cache.data_ptr<at::BFloat16>()),
            cpu_stride, M*8*128, 0, cache_stride, offsets.data_ptr<int>(), counts.data_ptr<int>(),
            reinterpret_cast<unsigned int*>(signals.data_ptr<int>()), slots.data_ptr<int>(),
            starts.data_ptr<int>(), lengths.data_ptr<int>(), heads);
    }
}

void gather_copy_indexed(torch::Tensor values, torch::Tensor cache, torch::Tensor temp,
    torch::Tensor offsets, torch::Tensor counts, torch::Tensor signals,
    torch::Tensor slots, torch::Tensor starts, torch::Tensor lengths,
    int heads, int cpu_stride, int cache_stride, int map_size, bool key_only) {
    switch (map_size) {
#define LAUNCH(M) case M: indexed_gather<M>(values,cache,temp,offsets,counts,signals,slots,starts,lengths,heads,cpu_stride,cache_stride,key_only); break;
        LAUNCH(128) LAUNCH(256) LAUNCH(512) LAUNCH(1024)
#undef LAUNCH
        default: TORCH_CHECK(false, "Unsupported ShadowKV selection size");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
