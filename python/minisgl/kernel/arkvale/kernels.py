"""Independent Triton implementation of ArkVale's cuboid digests and scores.

Behavioral reference: pku-liang/ArkVale, commit
0c3b45ebe7ecc8c81ff4bc0f9c68c04a0866de88, InferState._summarize_keys
and arkvale_cpp/src/estimate.cu. No upstream source is vendored.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _digest(
    K,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BD: tl.constexpr,
    FIRST: tl.constexpr,
):
    page = tl.program_id(0) + FIRST
    head = tl.program_id(1)
    t = page * PAGE + tl.arange(0, PAGE)
    d = tl.arange(0, BD)
    valid = t[:, None] < N
    x = tl.load(K + (t[:, None] * H + head) * D + d[None, :], valid & (d[None, :] < D), other=0).to(
        tl.float32
    )
    hi = tl.max(tl.where(valid, x, float("-inf")), 0)
    lo = tl.min(tl.where(valid, x, float("inf")), 0)
    center = (hi + lo) * 0.5
    count = tl.minimum(PAGE, N - page * PAGE)
    radius = tl.sum(tl.where(valid, tl.abs(x - center[None, :]), 0.0), 0) / count
    offset = ((page * 2) * H + head) * D + d
    tl.store(OUT + offset, center + radius, d < D)
    tl.store(OUT + offset + H * D, center - radius, d < D)


@tr.jit
def _score(
    Q,
    DG,
    OUT,
    H: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
):
    page = tl.program_id(0)
    h = tl.arange(0, BH)
    d = tl.arange(0, BD)
    q = tl.load(Q + h[:, None] * D + d[None, :], (h[:, None] < HQ) & (d[None, :] < D), other=0).to(
        tl.float32
    )
    offset = ((page * 2) * H + h[:, None] // (HQ // H)) * D + d[None, :]
    hi = tl.load(DG + offset, (h[:, None] < HQ) & (d[None, :] < D), other=0).to(tl.float32)
    lo = tl.load(DG + offset + H * D, (h[:, None] < HQ) & (d[None, :] < D), other=0).to(tl.float32)
    # Upstream n_groups=1: mean over all query heads, sum over dimensions.
    scores = tl.sum(tl.maximum(q * hi, q * lo), 1)
    tl.store(OUT + page, tl.sum(scores, 0) / HQ)


@tr.jit
def _gather(
    KV,
    SLOTS,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    kind = tl.program_id(1)
    token = i // (H * D)
    hd = i % (H * D)
    slot = tl.load(SLOTS + token // PAGE, token < N, other=0)
    offset = ((kind * CAP + slot) * PAGE + token % PAGE) * H * D + hd
    val = tl.load(KV + offset, i < N * H * D, other=0)
    tl.store(OUT + kind * N * H * D + i, val, i < N * H * D)


def summarize_keys(keys, page_size, out=None, first_page=0):
    """keys: contiguous [tokens, kv_heads, dim]; out: [pages, 2, heads, dim]."""
    assert keys.is_cuda and keys.is_contiguous() and keys.ndim == 3
    n, h, d = keys.shape
    pages = tr.cdiv(n, page_size)
    if out is None:
        out = torch.empty((pages, 2, h, d), device=keys.device, dtype=keys.dtype)
    if pages > first_page:
        _digest[(pages - first_page, h)](
            keys, out, n, h, d, page_size, tr.next_power_of_2(d), first_page
        )
    return out


def estimate_scores(query, digests):
    """One decode query [query_heads, dim], one shared page selection across heads."""
    assert query.is_cuda and query.is_contiguous() and digests.is_contiguous()
    pages, _, h, d = digests.shape
    assert query.shape[1] == d and query.shape[0] % h == 0
    scores = torch.empty(pages, device=query.device, dtype=torch.float32)
    if pages:
        _score[(pages,)](
            query,
            digests,
            scores,
            h,
            query.shape[0],
            d,
            tr.next_power_of_2(d),
            tr.next_power_of_2(query.shape[0]),
        )
    return scores


def gather_pages(kv, slots, token_count):
    """Pack resident pages in logical order; omit padding in the last page."""
    _, cap, page, h, d = kv.shape
    out = torch.empty((2, token_count, h, d), device=kv.device, dtype=kv.dtype)
    _gather[(tr.cdiv(token_count * h * d, 512), 2)](
        kv, slots, out, token_count, h, d, page, cap, 512
    )
    return out
