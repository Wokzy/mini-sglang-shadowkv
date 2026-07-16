import os
import torch
import typing as tp

from collections import namedtuple

# from minisgl.shadowkv_kernels import higgs_quantize, higgs_dequantize, higgs_dequantize_full
import minisgl.shadowkv_kernels as higgs_quantization_cuda


def is_power_of_two(x: int) -> bool:
    return (x & (x - 1)) == 0


def quantize(x: torch.Tensor):
    assert is_power_of_two(x.shape[-1])


def get_2bit_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    file = f"{os.path.dirname(os.path.abspath(__file__))}/grids/EDEN4-256.pt"
    grid = torch.load(file, map_location="cpu").to(device=device, dtype=dtype)

    return grid

QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])

def higgs_quantize(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return higgs_quantization_cuda.higgs_quantize(x, grid)


def higgs_dequantize(quantized: QuantizedTensor, grid: torch.Tensor) -> torch.Tensor:
    return higgs_quantization_cuda.higgs_dequantize(
        quantized.idx, 
        quantized.scales,
        grid
    )

def higgs_dequantize_full(quantized: QuantizedTensor, grid: torch.Tensor, add_prediction: torch.Tensor, out: torch.Tensor, hadamard_scale) -> torch.Tensor:
    higgs_quantization_cuda.higgs_dequantize_full(
        quantized.idx, 
        quantized.scales,
        grid,
        add_prediction,
        out,
        hadamard_scale
    )

def _cum_lengths(lengths: torch.Tensor, device) -> torch.Tensor:
    lengths = lengths.to(torch.int64)
    cum_lengths = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=device)
    cum_lengths[1:] = torch.cumsum(lengths, 0).to(torch.int32)
    return cum_lengths


def higgs_quantize_heads(
        x: torch.Tensor,
        lengths: torch.Tensor,
        grid: torch.Tensor,
        # out: tp.Optional[QuantizedTensor] = None,
        out_idx: tp.Optional[QuantizedTensor] = None,
        out_scales: tp.Optional[QuantizedTensor] = None,
        heads_first: bool = False
) -> QuantizedTensor:
    # x: [B, padded_T, n_kv_heads, head_dim] bf16 keys ([B, n_kv_heads, padded_T, head_dim] if heads_first).
    # `grid`: [256, 4] for 2-bit or [256, 2] for 4-bit quantization.
    # `lengths`: [B] int, the true (unrounded) number of tokens per sample; padding tokens are
    # skipped, their idx/scales are left untouched. A warp owns a whole token in this kernel,
    # so lengths need no rounding here (unlike higgs_score).
    # Returns codes [..., head_dim // d] uint8 and per-head norms [...] float32 with the same
    # first 3 dims as x -- with heads_first=False that is exactly the landmarks format
    # higgs_score consumes, with heads_first=True the format higgs_dequantize_heads consumes.
    d = grid.shape[1]
    cum_lengths = _cum_lengths(lengths, x.device)

    assert out_idx is not None and out_scales is not None
    # if out_idx is None:
        # out = QuantizedTensor(
        #     idx=torch.empty(*x.shape[:3], x.shape[3] // d, dtype=torch.uint8, device=x.device),
        #     scales=torch.empty(x.shape[:3], dtype=torch.float32, device=x.device),
        # )

    higgs_quantization_cuda.higgs_quantize_heads(
        x,
        cum_lengths,
        grid,
        out_idx, #out.idx,
        out_scales, #out.scales,
        heads_first
    )

    # return out


def higgs_dequantize_heads(
        quantized: QuantizedTensor,
        lengths: torch.Tensor,
        grid: torch.Tensor,
        hadamard_scale: float,
        out: tp.Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Inverse of higgs_quantize_heads(..., heads_first=True) with a [256, 2] (4-bit) grid:
    # codes [B, n_kv_heads, padded_T, head_dim // 2] + scales [B, n_kv_heads, padded_T]
    # -> [B, n_kv_heads, padded_T, head_dim] bf16. hadamard_scale should be 1 / head_dim.
    # Padding tokens (t >= lengths[b]) are skipped, out is left untouched there.
    d = grid.shape[1]
    cum_lengths = _cum_lengths(lengths, quantized.idx.device)

    if out is None:
        out = torch.empty(*quantized.idx.shape[:3], quantized.idx.shape[3] * d,
                          dtype=torch.bfloat16, device=quantized.idx.device)

    higgs_quantization_cuda.higgs_dequantize_heads(
        quantized.idx,
        quantized.scales,
        cum_lengths,
        grid,
        out,
        hadamard_scale
    )

    return out


def higgs_score(
        landmarks: QuantizedTensor,
        lengths: torch.Tensor,
        grid: torch.Tensor,
        query: torch.Tensor,
        hadamard_scale: float,
        block_indices: tp.Optional[torch.Tensor] = None,
        out: tp.Optional[torch.Tensor] = None
) -> torch.Tensor:
    # landmarks hold the whole kv cache ([kv_cache_size, T, n_kv_heads, ...]); the query batch B
    # can be smaller. `block_indices`: [B] int, query i scores the tokens of kv-cache entry
    # block_indices[i] (defaults to the identity mapping). `lengths`: [B] int, the true
    # (unrounded) number of tokens to score for each query, i.e. the current length of its
    # cache entry. The kernel needs each query's flattened token range to be a multiple of
    # tokens-per-warp (so a warp never straddles a query boundary), so we round up here and
    # pass the cumsum.
    if query.ndim == 4:
        assert query.shape[2] == 1
        query = query.squeeze(2)

    batch_size = query.shape[0]
    n_kv_heads = landmarks.idx.shape[2]
    # tokens-per-warp = (32 lanes * 16 bytes) / (n_kv_heads * head_dim/d bytes);  head_dim=128, d=4 -> 16 // n_kv
    tokens_per_warp = 16 // n_kv_heads
    lengths = lengths.to(torch.int64)
    rounded = ((lengths + tokens_per_warp - 1) // tokens_per_warp) * tokens_per_warp
    cum_lengths = torch.zeros(batch_size + 1, dtype=torch.int32, device=landmarks.idx.device)
    cum_lengths[1:] = torch.cumsum(rounded, 0).to(torch.int32)

    if block_indices is None:
        block_indices = torch.arange(batch_size, dtype=torch.int32, device=landmarks.idx.device)
    else:
        block_indices = block_indices.to(torch.int32)

    if out is None:
        out = torch.empty(batch_size, landmarks.idx.shape[1], n_kv_heads,
                          dtype=query.dtype, device=query.device)  # [B, T, n_kv_head] scores

    higgs_quantization_cuda.higgs_score(
        landmarks.idx,
        landmarks.scales,
        cum_lengths,
        block_indices,
        grid,
        query,
        out,
        hadamard_scale
    )

    return out