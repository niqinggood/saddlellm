"""Experimental, inference-only INT8 weights with optional W8A16 GPU kernels.

Weights are symmetric per-output-channel INT8; activations are dynamically
quantized per token only in the W8A8 experiment. Public W8A16 retains float
activations. Embeddings, output head and normalization stay floating
point. This is lossy quantization, NOT a checkpoint-compatible FP16 fast path.
No external quantization package, compiler extension or remote code is used.
"""

from __future__ import annotations

import functools
import importlib.util
import sys

import torch
from torch import nn
from torch.nn import functional as F


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _quantize_tokens(flat):
    values = flat.float()
    scales = values.abs().amax(dim=-1, keepdim=True) / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = (values / scales).round().clamp(-127, 127).to(torch.int8)
    return quantized, scales


def _dequantize_tokens(accumulators, scales, weight_scales, bias, dtype):
    result = accumulators.float() * scales * weight_scales
    if bias is not None:
        result = result + bias
    return result.to(dtype)


@functools.lru_cache(maxsize=1)
def _compiled_linear():
    if sys.platform != "linux" or importlib.util.find_spec("triton") is None:
        raise RuntimeError("compiled native INT8 requires Linux CUDA and Triton")
    # Compile shape-generic math helpers only. Whole-GEMM compilation hit the
    # Torch 2.6 eight-specialization cache limit on the real 8B comparison.
    # Do not change global Dynamo limits or capture graph/weight pointers.
    return (
        torch.compile(_quantize_tokens, fullgraph=True, dynamic=True),
        torch.compile(_dequantize_tokens, fullgraph=True, dynamic=True),
    )


def _int8_linear(
    inputs, weight_t, weight_scales, bias, out_features: int, *, compiled=False
):
    original_shape = inputs.shape[:-1]
    flat = inputs.reshape(-1, inputs.shape[-1])
    rows, columns = flat.shape
    # CUDA _int_mm requires M > 16 and aligned dimensions in tested Torch 2.6.
    padded_rows = max(32, _round_up(rows, 16))
    quantize, dequantize = (
        _compiled_linear() if compiled else (_quantize_tokens, _dequantize_tokens)
    )
    quantized, scales = quantize(flat)
    quantized = F.pad(
        quantized, (0, weight_t.shape[0] - columns, 0, padded_rows - rows)
    )
    accumulators = torch._int_mm(quantized, weight_t)
    result = dequantize(
        accumulators[:rows, :out_features],
        scales,
        weight_scales[:out_features],
        bias,
        inputs.dtype,
    )
    return result.reshape(*original_shape, out_features)


def cuda_int8_supported(device="cuda", *, activation_bits=16) -> tuple[bool, str]:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return False, "native INT8 requires an available CUDA device"
    if torch.cuda.get_device_capability(target) < (7, 5):
        return False, "native INT8 requires CUDA compute capability >= 7.5"
    try:
        with torch.inference_mode():
            if activation_bits == 16:
                from .Int8WeightKernels import weight_only_linear

                values = torch.ones((1, 32), device=target, dtype=torch.float16)
                weight = torch.ones((32, 48), device=target, dtype=torch.int8)
                scales = torch.ones(48, device=target, dtype=torch.float32)
                result = weight_only_linear(values, weight, scales, None, 48)
                if not bool((result == 32).all()):
                    return False, "W8A16 capability probe returned incorrect values"
                return True, ""
            left = torch.ones((32, 32), device=target, dtype=torch.int8)
            right = torch.ones((32, 48), device=target, dtype=torch.int8)
            if not bool((torch._int_mm(left, right) == 32).all()):
                return False, "CUDA INT8 capability probe returned incorrect values"
    except ImportError:
        return (
            False,
            "native W8A16 INT8 requires Triton; default float serving is unchanged",
        )
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        return False, f"CUDA INT8 capability probe failed: {type(exc).__name__}"
    return True, ""


class CUDAInt8Linear(nn.Module):
    """Frozen INT8 weights; CUDA FP16/BF16 inputs and floating-point output.

    Construct from an exclusively owned CPU Linear BEFORE serving. There is
    deliberately no dense fallback, training path or implicit CPU emulation.
    Only one transposed, contiguous INT8 weight allocation is retained.
    """

    def __init__(
        self,
        source: nn.Linear,
        *,
        compile_kernel: bool = False,
        activation_bits: int = 16,
    ):
        super().__init__()
        if type(source) is not nn.Linear or source.training:
            raise ValueError("native INT8 requires a plain eval-mode Linear")
        weight = source.weight
        if (
            weight.device.type != "cpu"
            or weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or weight.is_meta
            or weight.is_inference()
            or weight.layout != torch.strided
            or weight.requires_grad
            or (source.bias is not None and source.bias.requires_grad)
        ):
            raise ValueError(
                "native INT8 requires frozen ordinary CPU float parameters"
            )
        if source.in_features * 127 * 127 > torch.iinfo(torch.int32).max:
            raise ValueError("INT8 accumulation could overflow int32")
        if (
            any(
                getattr(source, attribute)
                for attribute in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                    "_state_dict_hooks",
                    "_state_dict_pre_hooks",
                    "_load_state_dict_pre_hooks",
                    "_load_state_dict_post_hooks",
                )
            )
            or weight._backward_hooks
        ):
            raise ValueError("native INT8 does not rewrite hooked linears")
        if not isinstance(compile_kernel, bool):
            raise ValueError("compile_kernel must be boolean")
        if type(activation_bits) is not int or activation_bits not in {8, 16}:
            raise ValueError("activation_bits must be 8 or 16")
        if compile_kernel and activation_bits != 8:
            raise ValueError("compiler experiment only applies to W8A8")
        if compile_kernel:
            _compiled_linear()  # Fail before changing a model on unsupported OS.
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.compile_kernel = compile_kernel
        self.activation_bits = activation_bits
        self.kernel_calls = 0
        # Quantize in bounded row chunks: no whole FP32 duplicate of 8B weights.
        with torch.inference_mode(False), torch.no_grad():
            padded_in = _round_up(self.in_features, 16)
            padded_out = _round_up(self.out_features, 16)
            weight_t = torch.zeros((padded_in, padded_out), dtype=torch.int8)
            scales = torch.ones(padded_out, dtype=torch.float32)
            for start in range(0, self.out_features, 256):
                stop = min(start + 256, self.out_features)
                values = weight[start:stop].float()
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("native INT8 weights must be finite")
                scale = values.abs().amax(dim=1) / 127.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                quantized = (
                    (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
                )
                weight_t[: self.in_features, start:stop].copy_(quantized.t())
                scales[start:stop].copy_(scale)
            self.weight_t = nn.Parameter(weight_t, requires_grad=False)
            self.register_buffer("weight_scales", scales)
            self.register_buffer(
                "bias",
                source.bias.detach().float().clone()
                if source.bias is not None
                else None,
            )
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError(
                "native INT8 is inference-only; reload float weights for training"
            )
        return super().train(False)

    def _apply(self, fn, recurse=True):
        # Generic model.half()/bfloat16() must not round quantization scales.
        scales, bias = self.weight_scales, self.bias
        result = super()._apply(fn, recurse=recurse)
        self.weight_scales = scales.to(device=self.weight_t.device, dtype=torch.float32)
        if bias is not None:
            self.bias = bias.to(device=self.weight_t.device, dtype=torch.float32)
        if self.weight_t.dtype != torch.int8:
            raise ValueError("native INT8 weights cannot be cast to another dtype")
        return result

    def forward(self, inputs):
        if self.training or torch.is_grad_enabled():
            raise ValueError("native INT8 requires inference_mode() or no_grad()")
        if (
            inputs.device.type != "cuda"
            or inputs.device != self.weight_t.device
            or inputs.dtype not in {torch.float16, torch.bfloat16}
            or inputs.layout != torch.strided
            or inputs.ndim < 1
            or inputs.shape[-1] != self.in_features
            or inputs.numel() == 0
            or torch.is_autocast_enabled("cuda")
        ):
            raise ValueError(
                "native INT8 requires nonempty CUDA FP16/BF16 inputs without autocast"
            )
        if self.activation_bits == 16:
            from .Int8WeightKernels import weight_only_linear

            result = weight_only_linear(
                inputs, self.weight_t, self.weight_scales, self.bias, self.out_features
            )
        else:
            result = _int8_linear(
                inputs,
                self.weight_t,
                self.weight_scales,
                self.bias,
                self.out_features,
                compiled=self.compile_kernel,
            )
        self.kernel_calls += 1
        return result


def prepare_cuda_int8(
    model: nn.Module, *, compile_kernel: bool = False, activation_bits: int = 16
) -> dict:
    """Lossily replace hidden CPU linears; caller then moves model to CUDA.

    Output heads stay float (including tied embeddings). Structural validation
    precedes replacement; a failed numeric conversion marks the model unusable
    for float export/training and it must be discarded. Original files are never
    modified. Generic
    float checkpoint export is intentionally not supported after quantization.
    """
    if not isinstance(compile_kernel, bool):
        raise ValueError("compile_kernel must be boolean")
    if type(activation_bits) is not int or activation_bits not in {8, 16}:
        raise ValueError("activation_bits must be 8 or 16")
    if compile_kernel and activation_bits != 8:
        raise ValueError("compiler experiment only applies to W8A8")
    if compile_kernel:
        _compiled_linear()
    if getattr(model, "_saddle_cuda_int8", None) is not None:
        raise ValueError("model is already native INT8")
    if any(m.training for m in model.modules()):
        raise ValueError("call model.eval() before preparing native INT8")
    if any(
        p.device.type != "cpu" or p.requires_grad or p.is_meta
        for p in model.parameters()
    ):
        raise ValueError("native INT8 preparation requires a frozen CPU model")
    if (
        nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks
    ):
        raise ValueError("native INT8 does not rewrite globally hooked models")
    # Resolve every target before mutation, including shared/aliased parameters.
    candidates = []
    seen = set()
    for name, module in model.named_modules():
        if name == "lm_head":
            continue
        if isinstance(module, nn.Linear):
            if not name or type(module) is not nn.Linear:
                raise ValueError("native INT8 requires ordinary non-root linears")
            if id(module.weight) in seen:
                raise ValueError(
                    "native INT8 does not rewrite shared hidden linear weights"
                )
            seen.add(id(module.weight))
            if module.weight.layout != torch.strided or module.weight.is_inference():
                raise ValueError("native INT8 requires ordinary strided weights")
            if module.weight.dtype not in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }:
                raise ValueError("native INT8 requires floating-point source weights")
            if module.in_features * 127 * 127 > torch.iinfo(torch.int32).max:
                raise ValueError("INT8 accumulation could overflow int32")
            if (
                any(
                    getattr(module, k)
                    for k in (
                        "_forward_hooks",
                        "_forward_pre_hooks",
                        "_backward_hooks",
                        "_backward_pre_hooks",
                        "_state_dict_hooks",
                        "_state_dict_pre_hooks",
                        "_load_state_dict_pre_hooks",
                        "_load_state_dict_post_hooks",
                    )
                )
                or module.weight._backward_hooks
            ):
                raise ValueError("native INT8 does not rewrite hooked linears")
            candidates.append((name, module))
    if not candidates:
        raise ValueError("native INT8 found no hidden linears")
    # Hidden weights are checked in bounded conversion chunks below. Check
    # retained float parameters too: NaN/Inf in an embedding/norm/head must not
    # be silently hidden when a later activation is cast to integer zero.
    quantized_ids = {id(module.weight) for _, module in candidates}
    for parameter in model.parameters():
        if id(parameter) not in quantized_ids:
            values = parameter.detach().reshape(-1)
            for offset in range(0, values.numel(), 1024**2):
                if not bool(torch.isfinite(values[offset : offset + 1024**2]).all()):
                    raise ValueError(
                        "native INT8 retained float parameters must be finite"
                    )
    report = {
        "method": f"saddle-w8a{activation_bits}",
        "weight_bits": 8,
        "activation_bits": activation_bits,
        "weight_scale": "per_output_channel_symmetric",
        "activation_scale": "per_token_symmetric"
        if activation_bits == 8
        else "none_float_activations",
        "lossy": True,
        "float_output_head": True,
        "compile_kernel": compile_kernel,
        "linear_count": len(candidates),
        "float_weight_bytes_before": 0,
        "int8_weight_bytes": 0,
        "status": "preparing",
    }
    # Do not keep replaced float modules alive: process one target at a time.
    names = [name for name, _ in candidates]
    del candidates, module
    model._saddle_cuda_int8 = report
    try:
        for name in names:
            source = model.get_submodule(name)
            report["float_weight_bytes_before"] += (
                source.weight.numel() * source.weight.element_size()
            )
            replacement = CUDAInt8Linear(
                source, compile_kernel=compile_kernel, activation_bits=activation_bits
            )
            report["int8_weight_bytes"] += replacement.weight_t.numel()
            parent_name, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, replacement)
            del source, replacement
    except Exception:
        report["status"] = "failed_discard_model"
        raise
    report["status"] = "ready"
    return report


def cuda_int8_status(model: nn.Module) -> dict:
    layers = [m for m in model.modules() if type(m) is CUDAInt8Linear]
    return {
        **getattr(model, "_saddle_cuda_int8", {}),
        "linear_count": len(layers),
        "cuda_linears": sum(m.weight_t.is_cuda for m in layers),
        "kernel_calls": sum(m.kernel_calls for m in layers),
        "compiled_linears": sum(m.compile_kernel for m in layers),
        "kernel_backend": "triton_w8a16"
        if layers and all(m.activation_bits == 16 for m in layers)
        else "torch_w8a8",
        "resident_tensor_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        + sum(b.numel() * b.element_size() for b in model.buffers()),
    }
