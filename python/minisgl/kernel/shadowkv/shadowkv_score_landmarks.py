import torch

import triton
import triton.language as tl


@triton.jit
def __shadowkv_score_landmarks_kernel_triton(
    query_states,  # (BS, num_heads, 1, head_dim)
    query_states_stride,
    landmarks,  # (BS, num_heads, max_num_chunks, head_dim) we should transpose in mind
    landmarks_stride,
    scores,  # (BS, num_heads, max_num_chunks)
    scores_stride,
    num_chunks,
    num_chunks_stride,
    batch_indices,
    batch_indices_stride,
    HD: tl.constexpr,
):
    """
    All buffers must be contiguous.
    """
    pid_x, pid_y, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    batch_index = tl.load(batch_indices + pid_x * batch_indices_stride[0])
    mask = pid_z < tl.load(num_chunks + batch_index * num_chunks_stride[0])

    query_ptr = query_states + pid_x * query_states_stride[0] + pid_y * query_states_stride[1]
    landmark_ptr = (
        landmarks
        + batch_index * landmarks_stride[0]
        + pid_y * landmarks_stride[1]
        + pid_z * landmarks_stride[2]
    )
    score_ptr = scores + pid_x * scores_stride[0] + pid_y * scores_stride[1] + pid_z

    offsets = tl.arange(0, HD)
    x = tl.load(query_ptr + offsets, mask=mask)
    y = tl.load(landmark_ptr + offsets, mask=mask)

    tl.store(score_ptr, tl.sum(x * y), mask=mask)


def shadowkv_score_landmarks_kernel_hd128(
    query_states: torch.Tensor,
    landmarks: torch.Tensor,
    local_kv_heads: int,
    num_chunks: torch.Tensor,
    max_num_chunks: int,
    batch_indices: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    BS, HD = len(batch_indices), 128

    query_states = query_states.contiguous()
    landmarks = landmarks.contiguous()
    assert query_states.is_contiguous()
    assert query_states.shape == (BS, local_kv_heads, 1, HD), (
        query_states.shape,
        (BS, local_kv_heads, 1, HD),
    )

    scores = torch.empty(
        (BS, local_kv_heads, max_num_chunks), dtype=dtype, device=device
    ).contiguous()
    grid = lambda meta: (BS, local_kv_heads, max_num_chunks)

    __shadowkv_score_landmarks_kernel_triton[grid](
        query_states,
        query_states.stride(),
        landmarks,
        landmarks.stride(),
        scores,
        scores.stride(),
        num_chunks,
        num_chunks.stride(),
        batch_indices,
        batch_indices.stride(),
        HD=HD,
    )

    # scores *= (HD ** -.5)

    return scores
