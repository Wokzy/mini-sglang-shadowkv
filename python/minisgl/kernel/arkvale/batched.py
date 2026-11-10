"""GPU-resident ArkVale decode bookkeeping and mapped-host transfers.

All grids use active batch -> persistent request slot indirection. The transfer
kernels access pinned, UVA-mapped CPU backing; no miss list returns to the CPU.
"""

import triton as tr
import triton.language as tl


@tr.jit
def append_host(
    K,
    V,
    HOSTS,
    ACTIVE,
    LENGTHS,
    LAYER: tl.constexpr,
    L: tl.constexpr,
    MAX_T: tl.constexpr,
    HD: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(ACTIVE + b).to(tl.int64)
    n = tl.load(LENGTHS + b).to(tl.int64)
    host = tl.load(HOSTS + slot * L + LAYER).to(tl.pointer_type(K.dtype.element_ty))
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(K + b * KS0 + (i // D) * KS1 + i % D, i < HD, other=0)
    v = tl.load(V + b * VS0 + (i // D) * VS1 + i % D, i < HD, other=0)
    tl.store(host + (n - 1) * HD + i, k, i < HD)
    tl.store(host + (MAX_T + n - 1) * HD + i, v, i < HD)


@tr.jit
def score_pages(
    Q,
    DG,
    ACTIVE,
    LENGTHS,
    SCORES,
    LAYER: tl.constexpr,
    L: tl.constexpr,
    MP: tl.constexpr,
    P: tl.constexpr,
    CAP: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
):
    page = tl.program_id(0)
    b = tl.program_id(1)
    slot = tl.load(ACTIVE + b).to(tl.int64)
    pages = tl.cdiv(tl.load(LENGTHS + b), P)
    score = tl.full((), -float("inf"), tl.float32)
    if pages > CAP and page < pages and page >= SINK and page < pages - WINDOW:
        h = tl.arange(0, BH)
        d = tl.arange(0, BD)
        mask = (h[:, None] < HQ) & (d[None, :] < D)
        q = tl.load(Q + b * QS0 + h[:, None] * QS1 + d[None, :], mask, other=0).to(tl.float32)
        offset = (((slot * L + LAYER) * MP + page) * 2 * H + h[:, None] // (HQ // H)) * D + d[
            None, :
        ]
        hi = tl.load(DG + offset, mask, other=0).to(tl.float32)
        lo = tl.load(DG + offset + H * D, mask, other=0).to(tl.float32)
        score = tl.sum(tl.sum(tl.maximum(q * hi, q * lo), 1), 0) / HQ
    tl.store(SCORES + b * MP + page, score)


@tr.jit
def select_pages(
    SCORES,
    ACTIVE,
    LENGTHS,
    MAP,
    AGE,
    CLOCK,
    MISSES,
    ORDER,
    COUNTS,
    OUT_LEN,
    LAYER: tl.constexpr,
    L: tl.constexpr,
    MP: tl.constexpr,
    P: tl.constexpr,
    CAP: tl.constexpr,
    TOPK: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
    INITIAL: tl.constexpr = False,
):
    b = tl.program_id(0)
    slot = tl.load(ACTIVE + b).to(tl.int64)
    state = slot * L + LAYER
    n = tl.load(LENGTHS + b)
    pages = tl.cdiv(n, P)
    p = tl.arange(0, BP)
    c = tl.arange(0, BC)
    protected = (p < pages) & ((p < SINK) | (p >= pages - WINDOW))
    desired = protected
    if pages <= CAP:
        desired = p < pages
    else:
        scores = tl.load(SCORES + b * MP + p, p < MP, other=-float("inf"))
        requested = CAP if INITIAL else TOPK
        remaining = tl.minimum(
            requested - tl.sum(protected.to(tl.int32), 0), pages - tl.sum(protected.to(tl.int32), 0)
        )
        for _ in range(remaining):
            best = tl.max(scores, 0)
            # Explicit deterministic tie rule: lower logical page wins.
            chosen = tl.min(tl.where((scores == best) & (p < pages) & ~desired, p, 2147483647), 0)
            desired = desired | (p == chosen)
            scores = tl.where(p == chosen, -float("inf"), scores)
    mapping = tl.load(MAP + state * CAP + c, c < CAP, other=-1)
    age = tl.load(AGE + state * CAP + c, c < CAP, other=9223372036854775807)
    clock = tl.load(CLOCK + state)
    misses = tl.full((BC,), -1, tl.int32)
    recalled = 0
    evicted = 0
    # Preserve the reference's two ordered ensures, including FIFO insertion
    # order, rather than replacing the resident set with this step's top-k.
    for phase in range(2):
        needed = protected if phase == 0 else desired
        if INITIAL and phase == 0:
            needed = tl.full((BP,), False, tl.int1)
        todo = needed
        for _ in range(tl.sum(needed.to(tl.int32), 0)):
            page = tl.min(tl.where(todo, p, 2147483647), 0)
            hit = tl.sum(((c < CAP) & (mapping == page)).to(tl.int32), 0) > 0
            if not hit:
                free = tl.min(tl.where((c < CAP) & (mapping < 0), c, 2147483647), 0)
                if free == 2147483647:
                    keep = tl.gather(needed, tl.maximum(mapping, 0), 0)
                    oldest = tl.min(tl.where((c < CAP) & ~keep, age, 9223372036854775807), 0)
                    free = tl.min(tl.where((c < CAP) & (age == oldest) & ~keep, c, 2147483647), 0)
                    evicted += 1
                mapping = tl.where(c == free, page, mapping)
                age = tl.where(c == free, clock, age)
                misses = tl.where(c == free, page, misses)
                clock += 1
                recalled += 1
            todo = todo & (p != page)
    tl.store(MAP + state * CAP + c, mapping, c < CAP)
    tl.store(AGE + state * CAP + c, age, c < CAP)
    tl.store(CLOCK + state, clock)
    tl.store(MISSES + b * CAP + c, misses, c < CAP)
    encoded = tl.where((c < CAP) & (mapping >= 0), mapping * BC + c, 2147483647)
    ordered = tl.sort(encoded, descending=False) % BC
    tl.store(ORDER + b * CAP + c, ordered, c < CAP)
    count = tl.sum(((c < CAP) & (mapping >= 0)).to(tl.int32), 0)
    tl.store(OUT_LEN + b, (count - 1) * P + (n - 1) % P + 1)
    old = tl.load(COUNTS + state * 3)
    tl.store(COUNTS + state * 3, old + recalled)
    old = tl.load(COUNTS + state * 3 + 1)
    tl.store(COUNTS + state * 3 + 1, old + evicted)
    old = tl.load(COUNTS + state * 3 + 2)
    tl.store(COUNTS + state * 3 + 2, old + (pages > CAP).to(tl.int32))


@tr.jit
def recall_pages(
    HOSTS,
    RESIDENT,
    ACTIVE,
    MISSES,
    LENGTHS,
    LAYER: tl.constexpr,
    L: tl.constexpr,
    MP: tl.constexpr,
    P: tl.constexpr,
    CAP: tl.constexpr,
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    page = tl.load(MISSES + b * CAP + c).to(tl.int64)
    if page >= 0:
        slot = tl.load(ACTIVE + b).to(tl.int64)
        n = tl.load(LENGTHS + b)
        host = tl.load(HOSTS + slot * L + LAYER).to(tl.pointer_type(RESIDENT.dtype.element_ty))
        x = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        kind = x // (P * HD)
        within = x % (P * HD)
        valid = (x < 2 * P * HD) & (page * P + within // HD < n)
        value = tl.load(host + (kind * MP + page) * P * HD + within, valid, other=0)
        dst = (((slot * L + LAYER) * 2 + kind) * CAP + c) * P * HD + within
        tl.store(RESIDENT + dst, value, x < 2 * P * HD)


@tr.jit
def update_current(
    K,
    V,
    RESIDENT,
    DG,
    ACTIVE,
    LENGTHS,
    MAP,
    LAYER: tl.constexpr,
    L: tl.constexpr,
    MP: tl.constexpr,
    P: tl.constexpr,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    b = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(ACTIVE + b).to(tl.int64)
    state = slot * L + LAYER
    n = tl.load(LENGTHS + b)
    page = (n - 1) // P
    offset = (n - 1) % P
    c = tl.arange(0, BC)
    mapping = tl.load(MAP + state * CAP + c, c < CAP, other=-1)
    resident = tl.min(tl.where((c < CAP) & (mapping == page), c, 2147483647), 0)
    d = tl.arange(0, BD)
    k = tl.load(K + b * KS0 + head * KS1 + d, d < D, other=0)
    v = tl.load(V + b * VS0 + head * VS1 + d, d < D, other=0)
    dst = ((state * 2 * CAP + resident) * P + offset) * H * D + head * D + d
    tl.store(RESIDENT + dst, k, d < D)
    tl.store(RESIDENT + dst + CAP * P * H * D, v, d < D)
    if offset == P - 1:
        t = tl.arange(0, P)
        src = ((state * 2 * CAP + resident) * P + t[:, None]) * H * D + head * D + d[None, :]
        # Supply the just-appended token explicitly: no inter-warp read-after-
        # write dependency is needed for the digest reduction.
        x = tl.load(RESIDENT + src, (d[None, :] < D) & (t[:, None] != offset), other=0).to(
            tl.float32
        )
        x = tl.where(t[:, None] == offset, k[None, :].to(tl.float32), x)
        hi, lo = tl.max(x, 0), tl.min(x, 0)
        center = (hi + lo) * 0.5
        radius = tl.sum(tl.abs(x - center[None, :]), 0) / P
        out = ((state * MP + page) * 2 * H + head) * D + d
        tl.store(DG + out, center + radius, d < D)
        tl.store(DG + out + H * D, center - radius, d < D)


@tr.jit
def gather_resident(
    RESIDENT,
    OUT,
    ACTIVE,
    ORDER,
    LENGTHS,
    LAYER: tl.constexpr,
    L: tl.constexpr,
    SLOTS: tl.constexpr,
    CAP: tl.constexpr,
    P: tl.constexpr,
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    kind = tl.program_id(1)
    slot = tl.load(ACTIVE + b).to(tl.int64)
    n = tl.load(LENGTHS + b)
    i = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    token = i // HD
    c = tl.load(ORDER + b * CAP + token // P, token < n, other=0)
    src = (((slot * L + LAYER) * 2 + kind) * CAP * P + c * P + token % P) * HD + i % HD
    value = tl.load(RESIDENT + src, token < n, other=0)
    dst = (kind * SLOTS + slot) * CAP * P * HD + i
    # FA implementations may multiply masked probabilities by padded V values.
    # Zero the tail: 0 * NaN from uninitialized/reused padding is still NaN.
    tl.store(OUT + dst, value, token < CAP * P)
