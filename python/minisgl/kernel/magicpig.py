"""Fused GPU bookkeeping for MagicPIG; attention arithmetic stays in upstream kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(projections, output, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.full((BLOCK,), 0, tl.int32)
    for bit in range(K):
        value = tl.load(projections + rows * K + bit, rows < N, other=0)
        packed |= (value > 0).to(tl.int32) << bit
    tl.store(output + rows, packed, rows < N)


def pack_hashes(projections, bits):
    output = torch.empty(
        (*projections.shape[:-1], projections.shape[-1] // bits),
        dtype=torch.int16,
        device=projections.device,
    )
    _pack[(triton.cdiv(output.numel(), 256),)](projections, output, output.numel(), bits, 256)
    return output


@triton.jit
def _append(
    k,
    v,
    centers,
    cache,
    slots,
    lengths,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    CENTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    position = tl.load(lengths + row) - 1
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HEADS * DIM
    h, d = offsets // DIM, offsets % DIM
    key = tl.load(k + row * KS0 + h * KS1 + d, mask, other=0)
    value = tl.load(v + row * VS0 + h * VS1 + d, mask, other=0)
    if CENTER:
        center = tl.load(centers + slot * HEADS * DIM + offsets, mask, other=0)
        key = (key.to(tl.float32) - center.to(tl.float32)).to(key.dtype)
    address = slot * 2 * HEADS * CAPACITY * DIM + h * CAPACITY * DIM + position * DIM + d
    tl.store(cache + address, key, mask)
    tl.store(cache + address + HEADS * CAPACITY * DIM, value, mask)


def append_kv(k, v, centers, cache, slots, lengths, center):
    heads, dim = k.shape[1:]
    _append[(k.shape[0],)](
        k,
        v,
        centers,
        cache,
        slots,
        lengths,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        heads,
        dim,
        cache.shape[-2],
        center,
        triton.next_power_of_2(heads * dim),
    )
