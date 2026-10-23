"""Lazily imported Triton implementation; optional, inference-only."""
import torch
import triton
import triton.language as tl

# Cache only compiled launchers, never request tensors or outputs. Shapes,
# strides, dtype, device and pointer alignment are all part of specialization.
_RMS_LAUNCHERS = {}
_ROTARY_LAUNCHERS = {}


def _launch(cache, key, kernel, grid, args):
    launcher = cache.get(key)
    if launcher is None:
        compiled = kernel[grid](*args, enable_fp_fusion=False)
        if len(cache) >= 256:
            cache.clear()
        # Triton 3.2 omits constexprs from the launch ABI; 3.3+ includes them.
        # Derive the argument count from the compiled signature, not a version
        # guess. All constexpr parameters in these kernels follow the tensors.
        cache[key] = (compiled[grid], len(compiled.src.signature))
    else:
        run, count = launcher
        run(*args[:count])


@triton.jit
def _rms_finish(X, V, W, Y, N: tl.constexpr, EPS: tl.constexpr,
                LOW_WEIGHT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    variance = tl.load(V + row)
    normalized = x * tl.rsqrt(variance + EPS)
    if LOW_WEIGHT:
        # Qwen3 casts BEFORE multiplying by its weight. Do not remove this.
        normalized = normalized.to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    tl.store(Y + row * N + col, normalized * w, col < N)


def rms_norm_decode(x, weight, eps, weight_in_fp32):
    variance = x.float().square().mean(dim=-1, keepdim=True)
    out = torch.empty_like(x)
    width = x.shape[-1]
    grid = (x.numel() // width, 1, 1)
    constants = (width, eps, not weight_in_fp32, triton.next_power_of_2(width))
    key = (x.device, x.dtype, grid, constants, x.data_ptr() % 16, weight.data_ptr() % 16)
    _launch(_RMS_LAUNCHERS, key, _rms_finish, grid, (x, variance, weight, out, *constants))
    return out


@triton.jit
def _rotary(Q, K, C, S, OQ, OK, B: tl.constexpr, QH: tl.constexpr,
            KH: tl.constexpr, D: tl.constexpr, QB: tl.constexpr, QHS: tl.constexpr,
            KB: tl.constexpr, KHS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    rotated_col = (col + D // 2) % D
    if row < B * QH:
        offset = (row // QH) * QB + (row % QH) * QHS
        x = tl.load(Q + offset + col, col < D, 0).to(tl.float32)
        r = tl.load(Q + offset + rotated_col, col < D, 0).to(tl.float32)
    else:
        kr = row - B * QH
        offset = (kr // KH) * KB + (kr % KH) * KHS
        x = tl.load(K + offset + col, col < D, 0).to(tl.float32)
        r = tl.load(K + offset + rotated_col, col < D, 0).to(tl.float32)
    r = tl.where(col < D // 2, -r, r)
    c = tl.load(C + col, col < D, 0).to(tl.float32)
    s = tl.load(S + col, col < D, 0).to(tl.float32)
    # Eager PyTorch rounds BOTH products before addition. FMA is disabled.
    a = (x * c).to(Q.dtype.element_ty).to(tl.float32)
    b = (r * s).to(Q.dtype.element_ty).to(tl.float32)
    value = a + b
    if row < B * QH:
        tl.store(OQ + row * D + col, value, col < D)
    else:
        tl.store(OK + (row - B * QH) * D + col, value, col < D)


def rotary_decode(q, k, cos, sin):
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    batch, qheads, _, dim = q.shape
    kheads = k.shape[1]
    grid = (batch * (qheads + kheads), 1, 1)
    constants = (batch, qheads, kheads, dim, q.stride(0), q.stride(1),
                 k.stride(0), k.stride(1), triton.next_power_of_2(dim))
    key = (q.device, q.dtype, constants, q.data_ptr() % 16, k.data_ptr() % 16,
           cos.data_ptr() % 16, sin.data_ptr() % 16)
    _launch(_ROTARY_LAUNCHERS, key, _rotary, grid, (q, k, cos, sin, oq, ok, *constants))
    return oq, ok
