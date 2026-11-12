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
#include "functions.h"

void pin_cpu_tensor(torch::Tensor tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "Tensor must be on CPU");

  void* base_ptr = tensor.storage().data_ptr().get();
  size_t size_bytes = tensor.storage().nbytes();

  if (!tensor.is_pinned()) {
    // Register the range in page-aligned chunks instead of one giant range.
    // Registering many GiB at once forces the NVIDIA driver to allocate large
    // *physically contiguous* kernel arrays for the DMA mapping (order-10
    // kzalloc for multi-GiB ranges), which sporadically fails with
    // cudaErrorInvalidValue ("invalid argument") on hosts with fragmented
    // physical memory. Small chunks only need small kernel allocations.
    constexpr uintptr_t kPageSize = 4096;
    constexpr uintptr_t kChunkBytes = uintptr_t(64) << 20;  // 64 MiB
    static_assert(kChunkBytes % kPageSize == 0);

    const uintptr_t begin =
        reinterpret_cast<uintptr_t>(base_ptr) & ~(kPageSize - 1);
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(base_ptr) + size_bytes + kPageSize - 1) &
        ~(kPageSize - 1);

    auto registered = std::make_shared<std::vector<void*>>();
    auto unregister_all = [](std::vector<void*>& chunks) {
      for (void* ptr : chunks) {
        cudaHostUnregister(ptr);
      }
      chunks.clear();
    };

    for (uintptr_t cur = begin; cur < end; cur += kChunkBytes) {
      const size_t chunk_size =
          std::min<uintptr_t>(kChunkBytes, end - cur);
      cudaError_t err = cudaHostRegister(reinterpret_cast<void*>(cur),
                                         chunk_size, cudaHostRegisterMapped);
      if (err == cudaErrorHostMemoryAlreadyRegistered) {
        (void)cudaGetLastError();
        continue;
      }
      if (err != cudaSuccess) {
        (void)cudaGetLastError();
        unregister_all(*registered);
        TORCH_CHECK(false, "cudaHostRegister: ", cudaGetErrorString(err));
      }
      registered->push_back(reinterpret_cast<void*>(cur));
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pin_cpu_tensor", &pin_cpu_tensor);
    m.def("batch_gemm_softmax_indexed", &batch_gemm_softmax_indexed);
    m.def("reorder_keys_indexed", &reorder_keys_indexed);
    m.def("gather_copy_indexed", &gather_copy_indexed);
    m.def("gather_copy", &gather_copy, "Gather-Copy (CUDA)");
    m.def("gather_copy_d2d_with_offsets", &gather_copy_d2d_with_offsets, "Gather-Copy D2D with offsets for keys (CUDA)");
    m.def("reorder_keys_and_compute_offsets", &reorder_keys_and_compute_offsets, "Reorder keys and compute offsets (CUDA)");
    m.def("gather_copy_with_offsets", &gather_copy_with_offsets, "Gather-Copy with offsets (CUDA)");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "Apply rotary position embedding (CUDA)");
    m.def("apply_rotary_pos_emb_new", &apply_rotary_pos_emb_new, "Apply rotary positional embeddings (CUDA)");
    m.def("apply_rotary_pos_emb_new_v2", &apply_rotary_pos_emb_new_v2, "Apply rotary positional embeddings version 2 (CUDA)");
    m.def("apply_rotary_pos_emb_push_cache", &apply_rotary_pos_emb_push_cache, "Apply rotary positional embeddings with cache pushing (CUDA)");
    m.def("apply_rotary_pos_emb_push_cache_opt", &apply_rotary_pos_emb_push_cache_opt, "Apply rotary positional embeddings with cache pushing (CUDA)");
    m.def("apply_rotary_pos_emb_push_cache_opt_glm", &apply_rotary_pos_emb_push_cache_opt_glm, "Apply rotary positional embeddings with cache pushing (CUDA) for GLM");
    m.def("batch_gather_gemm", &batch_gather_gemm, "Batch Gather GEMM (CUDA)");
    m.def("batch_gemm_softmax", &batch_gemm_softmax, "Batch GEMM Softmax (CUDA)");
}