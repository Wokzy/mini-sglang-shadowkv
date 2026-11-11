"""Slot-indexed BF16 reconstruction for H200 (SM90), prototyped in manual_test.

KernelWiki lang-triton: use tl.dot for the supported tensor-core path and
validate on the target GPU. No architecture-specific scheduling optimization.
"""

import triton
import triton.language as tl


@triton.jit
def reconstruct_keys(
    U,
    SV,
    Rope,
    Positions,
    Counts,
    Slots,
    Starts,
    LandmarkLengths,
    K,
    H: tl.constexpr,
    MAX_T: tl.constexpr,
    R: tl.constexpr,
    CAP: tl.constexpr,
    SELECT: tl.constexpr,
    BM: tl.constexpr = 32,
    BK: tl.constexpr = 32,
):
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    slot = tl.load(Slots + b)
    if tl.load(LandmarkLengths + slot) == 0:
        return
    hits = tl.load(Counts + bh)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = (row >= hits * 8) & (row < SELECT * 8)
    if tl.sum(valid.to(tl.int32), 0) == 0:
        return
    chunk = tl.load(Positions + (slot * H + h) * SELECT + row // 8, row < SELECT * 8, 0)
    token = chunk * 8 + row % 8
    rr = tl.arange(0, BK)
    dd = tl.arange(0, 128)
    acc = tl.full((BM, 128), 0, tl.float32)
    for base in range(triton.cdiv(R, BK)):
        rank = base * BK + rr
        u = tl.load(
            U + (slot.to(tl.int64) * MAX_T + token[:, None]) * R + rank[None, :],
            valid[:, None] & (rank[None, :] < R),
            0,
        )
        sv = tl.load(
            SV + slot * R * H * 128 + rank[:, None] * H * 128 + h * 128 + dd[None, :],
            rank[:, None] < R,
            0,
        )
        acc = tl.dot(u, sv, acc)
    # Match the original materialized BF16 GEMM and BF16 multiply/add RoPE.
    x = acc.to(tl.bfloat16).to(tl.float32)
    other = tl.gather(x, tl.broadcast_to(((dd + 64) % 128)[None, :], (BM, 128)), 1)
    cos = tl.load(Rope + token[:, None] * 128 + (dd % 64)[None, :], valid[:, None], 0).to(
        tl.float32
    )
    sin = tl.load(Rope + token[:, None] * 128 + (dd % 64 + 64)[None, :], valid[:, None], 0).to(
        tl.float32
    )
    a = (x * cos).to(tl.bfloat16).to(tl.float32)
    b = (other * tl.where(dd[None, :] < 64, -sin, sin)).to(tl.bfloat16).to(tl.float32)
    start = tl.load(Starts + slot)
    tl.store(
        K + ((slot * H + h) * CAP + start + row[:, None]) * 128 + dd[None, :], a + b, valid[:, None]
    )


@triton.jit
def append_kv(
    Kin,
    Vin,
    K,
    V,
    Slots,
    Lengths,
    H: tl.constexpr,
    CAP: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(Slots + b)
    pos = tl.load(Lengths + b) - 1
    x = tl.arange(0, BLOCK)
    dst = ((slot * H + x // 128) * CAP + pos) * 128 + x % 128
    tl.store(K + dst, tl.load(Kin + b * KS + x, x < H * 128, 0), x < H * 128)
    tl.store(V + dst, tl.load(Vin + b * VS + x, x < H * 128, 0), x < H * 128)


@triton.jit
def reduce_scores(
    Soft,
    Scores,
    Slots,
    Lengths,
    H: tl.constexpr,
    G: tl.constexpr,
    N,  # Runtime active extent: request turnover must not trigger recompilation.
    STRIDE: tl.constexpr,
    BG: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    bh = tl.program_id(0)
    slot = tl.load(Slots + bh // H)
    length = tl.load(Lengths + slot)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.arange(0, BG)
    x = tl.load(
        Soft + (bh * G + g[:, None]) * STRIDE + n[None, :],
        (g[:, None] < G) & (n[None, :] < length),
        -float("inf"),
    )
    tl.store(Scores + bh * STRIDE + n, tl.max(x, 0), n < N)


@triton.jit
def selected_chunk_ids(
    Indices,
    Landmarks,
    Selected,
    Slots,
    Lengths,
    H: tl.constexpr,
    MAX_C: tl.constexpr,
    SELECT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bh = tl.program_id(0)
    slot = tl.load(Slots + bh // H)
    h = bh % H
    i = tl.arange(0, BLOCK)
    idx = tl.load(Indices + bh * SELECT + i, i < SELECT, 0)
    valid = (i < SELECT) & (tl.load(Lengths + slot) > 0)
    chunk = tl.load(Landmarks + (slot * H + h) * MAX_C + idx, valid, 0)
    tl.store(Selected + bh * SELECT + i, chunk, i < SELECT)
