"""Optional Triton W8A16 kernels: no activation quantization or dense weight copy.

Integer weights are converted to the input FP16/BF16 representation inside
the kernel, then accumulated in FP32 and scaled per output channel. All INT8
values in [-127, 127] are exactly representable in FP16 and BF16. Import this
module only for the explicit native CUDA INT8 route, never for default serving.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _matvec(
    X,
    W,
    Scales,
    Bias,
    Y,
    K: tl.constexpr,
    N: tl.constexpr,
    WS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    accum = tl.full((BK, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        cols = block * BK + ks
        x = tl.load(X + row * K + cols, cols < K, 0).to(tl.float32)
        w = tl.load(
            W + cols[:, None] * WS + ns[None, :],
            (cols[:, None] < K) & (ns[None, :] < N),
            0,
        ).to(tl.float32)
        accum += x[:, None] * w
    values = tl.sum(accum, axis=0) * tl.load(Scales + ns, ns < N, 0)
    if HAS_BIAS:
        values += tl.load(Bias + ns, ns < N, 0)
    tl.store(Y + row * N + ns, values, ns < N)


@triton.jit(do_not_specialize=[5])
def _matmul(
    X,
    W,
    Scales,
    Bias,
    Y,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    WS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ms = tl.program_id(0) * BM + tl.arange(0, BM)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        cols = block * BK + ks
        x = tl.load(
            X + ms[:, None] * K + cols[None, :],
            (ms[:, None] < M) & (cols[None, :] < K),
            0,
        )
        w = tl.load(
            W + cols[:, None] * WS + ns[None, :],
            (cols[:, None] < K) & (ns[None, :] < N),
            0,
        ).to(x.dtype)
        accum += tl.dot(x, w)
    values = accum * tl.load(Scales + ns, ns < N, 0)[None, :]
    if HAS_BIAS:
        values += tl.load(Bias + ns, ns < N, 0)[None, :]
    tl.store(
        Y + ms[:, None] * N + ns[None, :], values, (ms[:, None] < M) & (ns[None, :] < N)
    )


def _matmul_tile(rows: int, columns: int, out_features: int) -> tuple[int, int]:
    """Avoid padded work on short, wide projections without tuning at startup.

    Keep the reduction tile unchanged so the W8A16 accumulation contract is
    the same. Other shapes retain the established 32x64 configuration.
    """
    if rows <= 16 and columns == 4096 and out_features >= 6144:
        return 16, 128
    return 32, 64


def weight_only_linear(inputs, weight_t, scales, bias, out_features):
    flat = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    rows, columns = flat.shape
    output = torch.empty((rows, out_features), device=inputs.device, dtype=inputs.dtype)
    bias_argument = scales if bias is None else bias
    # Each decode row gets a true GEMV: no 32-row padding, no activation INT8,
    # no materialized dequantized matrix. Larger prefill uses Tensor Cores.
    # Triton uses the current CUDA device. A model on cuda:1 must not launch
    # kernels on the worker thread's default cuda:0. The guard is thread-local.
    with torch.cuda.device(inputs.device):
        if rows <= 4:
            _matvec[(rows, triton.cdiv(out_features, 32))](
                flat,
                weight_t,
                scales,
                bias_argument,
                output,
                columns,
                out_features,
                weight_t.shape[1],
                bias is not None,
                256,
                32,
                num_warps=4,
            )
        else:
            # M is dynamic and explicitly not specialized: diverse prompt
            # lengths must not compile one kernel per length.
            block_m, block_n = _matmul_tile(rows, columns, out_features)
            _matmul[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
                flat,
                weight_t,
                scales,
                bias_argument,
                output,
                rows,
                columns,
                out_features,
                weight_t.shape[1],
                bias is not None,
                block_m,
                block_n,
                64,
                num_warps=4,
                num_stages=3,
            )
    return output.reshape(*inputs.shape[:-1], out_features)
