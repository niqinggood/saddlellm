# ruff: noqa: N999
"""Explicit CUDA execution of portable QAT codes without requantization.

The disk artifact remains the portable W8A32 reference format. This opt-in
runtime transposes its existing INT8 codes and retains the exact FP32 scales,
then uses the existing Triton W8A16 kernels. Norms/embeddings use the requested
activation precision. No dense dequantized matrix is retained by these layers.
Accessing ``weight`` explicitly materializes a temporary matrix for compatible
introspection; the status report counts such accesses separately from kernels.
"""
from __future__ import annotations

import inspect
from types import MethodType

import torch

from .CUDAInt8 import CUDAInt8Linear, cuda_int8_supported


def validate_qat_cuda_request(device, dtype):
    """Fail before artifact/model mutation; never select a fallback engine."""
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("cuda-w8a16 requires an explicit CUDA device") from exc
    if target.type != "cuda":
        raise ValueError("cuda-w8a16 requires an explicit CUDA device")
    if isinstance(dtype, str):
        dtype = {"fp16": torch.float16, "float16": torch.float16,
                 "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(dtype)
    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("cuda-w8a16 requires explicit float16 or bfloat16 activation dtype")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda-w8a16 requires available CUDA; no reference fallback")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability(target) < (8, 0):
        raise RuntimeError("cuda-w8a16 bfloat16 requires compute capability >= 8.0")
    supported, reason = cuda_int8_supported(target, activation_bits=16)
    if not supported:
        raise RuntimeError(f"cuda-w8a16 capability check failed; no fallback: {reason}")
    return target, dtype


class QATW8A16Linear(CUDAInt8Linear):
    """Use preexisting codes/scales with CUDAInt8Linear's guarded execution."""

    def __init__(self, packed):
        # Deliberately bypass CUDAInt8Linear.__init__: that constructor quantizes
        # float weights. Only a transpose/copy occurs here, never re-rounding.
        torch.nn.Module.__init__(self)
        codes, scales, bias = packed.qweight, packed.scales, packed.bias
        if (codes.dtype != torch.int8 or codes.ndim != 2 or codes.device.type != "cpu"
                or bool((codes == -128).any()) or scales.dtype != torch.float32
                or scales.device.type != "cpu" or scales.shape != (codes.shape[0],)
                or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
            raise ValueError("W8A16 requires validated CPU QAT INT8 codes and positive FP32 scales")
        if bias is not None and (bias.dtype != torch.float32 or bias.device.type != "cpu"
                                 or bias.shape != scales.shape or not bool(torch.isfinite(bias).all())):
            raise ValueError("W8A16 QAT bias must be finite CPU FP32 per output channel")
        self.out_features, self.in_features = codes.shape
        self.transposed = packed.transposed
        self.activation_bits, self.compile_kernel = 16, False
        self.kernel_calls = self.weight_materializations = 0
        self.weight_t = torch.nn.Parameter(codes.T.contiguous().clone(), requires_grad=False)
        self.register_buffer("weight_scales", scales.clone())
        self.register_buffer("bias", bias.clone() if bias is not None else None)
        # This empty float buffer tracks activation dtype through model.to().
        # Temporary .weight tensors use it; inherited _apply keeps scales FP32.
        self.register_buffer("_compute_dtype_anchor", torch.empty(0))
        self.eval()

    @property
    def qweight(self):
        return self.weight_t.T

    @property
    def scales(self):
        return self.weight_scales

    @property
    def weight(self):
        # HF model-parallel/head inspection and native APIs may read .weight.
        # Preserve its actual shape/device/dtype/numerics, without pretending an
        # integer parameter is a float matrix or retaining a dense cache.
        self.weight_materializations += 1
        values = (self.qweight.float() * self.weight_scales[:, None]).to(self._compute_dtype_anchor.dtype)
        return values.T if self.transposed else values


def qat_cuda_status(model):
    """Report actual execution and any explicit dense weight materialization."""
    layers = {name: module for name, module in model.named_modules() if isinstance(module, QATW8A16Linear)}
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return {**getattr(model, "_saddle_qat_inference", {}),
            "linear_count": len(layers),
            "cuda_linears": sum(layer.weight_t.is_cuda for layer in layers.values()),
            "kernel_calls": sum(layer.kernel_calls for layer in layers.values()),
            "weight_materializations": sum(layer.weight_materializations for layer in layers.values()),
            "resident_tensor_bytes": sum(storages.values()),
            "int8_weight_bytes": sum(layer.weight_t.numel() for layer in layers.values()),
            "fp32_scale_bytes": sum(layer.weight_scales.numel() * 4 for layer in layers.values()),
            "modules": [{"name": name, "weight_dtype": str(layer.weight_t.dtype),
                         "scale_dtype": str(layer.weight_scales.dtype),
                         "activation_dtype": str(layer._compute_dtype_anchor.dtype),
                         "kernel_calls": layer.kernel_calls,
                         "weight_materializations": layer.weight_materializations}
                        for name, layer in layers.items()]}


def _install_inference_only_guards(model):
    """Block training/export and labels that could bypass quantized forwards."""
    original_train = model.train
    forward_signature = inspect.signature(model.forward)

    def inference_train(self, mode=True):
        if mode:
            raise ValueError("cuda-w8a16 QAT is inference-only; reload the original artifact for training")
        return original_train(False)

    def block_export(self, *args, **kwargs):
        raise ValueError("cuda-w8a16 is a runtime representation; copy/reload the original portable QAT artifact for export")

    def check_inference(module, args, kwargs):
        supplied = forward_signature.bind_partial(*args, **kwargs).arguments
        if module.training or torch.is_grad_enabled() or supplied.get("labels") is not None:
            raise ValueError("cuda-w8a16 requires inference_mode/no_grad, eval, and no training labels")

    model.train = MethodType(inference_train, model)
    for name in ("save_pretrained", "save_pretrained_saddle"):
        if callable(getattr(model, name, None)):
            setattr(model, name, MethodType(block_export, model))
    model.register_forward_pre_hook(check_inference, with_kwargs=True)


def enable_qat_cuda_w8a16(model, *, device, dtype):
    """Validate CPU ownership, replace layers, move, then guard the runtime."""
    from ..compression.QuantizationAwareTraining import _module_types, _set_module

    if getattr(model, "_saddle_qat_inference", None) is not None:
        raise ValueError("QAT CUDA execution was already configured")
    _, packed_type = _module_types()
    candidates = [(name, module) for name, module in model.named_modules() if type(module) is packed_type]
    if not candidates:
        raise ValueError("QAT CUDA execution found no portable INT8 projections")
    if any(value.device.type != "cpu" for value in list(model.parameters()) + list(model.buffers())):
        raise ValueError("QAT CUDA preparation requires an exclusively owned CPU model")
    if (torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks
            or any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())):
        raise ValueError("QAT CUDA loading does not support hooked models")
    for value in list(model.parameters()) + list(model.buffers()):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("QAT CUDA retained model tensors must be finite")
    # Validation is complete before replacing modules. This function is used
    # after validate_qat_cuda_request by the public loader, on a fresh CPU model.
    for name, packed in candidates:
        _set_module(model, name, QATW8A16Linear(packed))
    del candidates
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    model._saddle_qat_inference = {
        "backend": "cuda-w8a16", "storage_backend": "portable-int8-weight-only",
        "kernel_backend": "existing_triton_w8a16", "activation_dtype": str(dtype),
        "scale_dtype": "torch.float32", "codes_requantized": False,
        "resident_dense_weight_cache": False, "inference_only": True,
        "source_artifact_unchanged": True,
    }
    _install_inference_only_guards(model)
    return model


__all__ = ["QATW8A16Linear", "qat_cuda_status"]
