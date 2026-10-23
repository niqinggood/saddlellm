"""Opt-in, checkpoint-compatible projection packing for native inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .SaddleModeling import SaddleAttention, SaddleForCausalLM, SaddleSwiGLU


@dataclass
class NativeInferenceOptimizationReport:
    packed_qkv: int = 0
    packed_gate_up: int = 0
    already_packed: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    cpu_prepack: dict = field(default_factory=dict)
    decode_rotary_packed: int = 0
    fused_decode_norms: int = 0
    fused_decode_rotary: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _can_pack(projections: tuple[nn.Module, ...]) -> bool:
    # Do not flatten adapters, quantized/subclassed Linear implementations,
    # parametrizations or user hooks into a plain matrix.
    if (
        nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks
        or any(type(module) is not nn.Linear for module in projections)
    ):
        return False
    first = projections[0].weight
    for module in projections:
        if (
            module.bias is not None
            or module.weight.ndim != 2
            or module.weight.is_meta
            or not module.weight.is_floating_point()
            or module.weight.layout != torch.strided
            or module.weight.device != first.device
            or module.weight.dtype != first.dtype
            or module.weight.shape[1] != first.shape[1]
            or module.weight.requires_grad != first.requires_grad
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
            or module._backward_pre_hooks
            or module._state_dict_hooks
            or module._load_state_dict_pre_hooks
            or module.weight._backward_hooks
        ):
            return False
    return True


def _pack(projections: tuple[nn.Linear, ...]) -> nn.Linear:
    # The concatenation is the only new weight allocation. Constructing on
    # meta avoids initializing another equally large temporary CUDA matrix.
    # Disable inference_mode so saved/reloaded weights remain ordinary tensors.
    with torch.inference_mode(False), torch.no_grad():
        weight = torch.cat([module.weight.detach() for module in projections])
        packed = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
        packed.weight = nn.Parameter(
            weight, requires_grad=projections[0].weight.requires_grad
        )
    return packed.eval()


def optimize_native_inference(
    model: SaddleForCausalLM,
    *,
    pack_qkv: bool = True,
    pack_decode_rotary: bool = False,
    fused_decode_kernels: bool = False,
    cpu_prepack: bool = False,
    max_cpu_prepack_bytes: int | None = None,
) -> NativeInferenceOptimizationReport:
    """Pack an exclusively owned, eval-mode native model in place, once.

    Q/K/V become one Linear call unless ``pack_qkv=False``; each SwiGLU
    gate/up pair becomes one call. The selective option lets BF16 serving
    retain the gate/up fast path when QKV packing changes greedy output.
    Numerical precision, attention/cache settings and checkpoint configuration
    are unchanged. Existing state-dict hooks retain logical checkpoint keys.
    MLA projections, adapters, quantized layers and hooked linears are skipped.

    ``pack_decode_rotary`` combines Q/K elementwise RoPE calls for CUDA
    single-token inference only. It changes no GEMM, weights or precision;
    CPU, prefill and gradient paths retain the reference implementation.
    Experimental and off by default: initial positive throughput samples did
    not reproduce the required paired-win stability on the shared test machine.

    ``fused_decode_kernels`` explicitly enables optional Triton RMSNorm/RoPE
    kernels for eligible CUDA single-token inference. The RMS reduction and
    low-precision intermediate rounding are retained. Other shapes/devices,
    training and autocast use eager operators. First use compiles kernels;
    serving warms common shapes during model loading. Check actual checkpoint
    output parity before deployment; this is not a universal bitwise guarantee.

    Optional ``cpu_prepack=True`` prepares extra-RAM oneDNN M=1 weight caches
    for safe CPU linears after packing. This does not quantize the model. The
    caller can cap extra cache bytes; unsupported hardware or insufficient
    RAM is reported explicitly. Prefill and gradient paths remain dense.

    Call before serving, compiling, attaching adapters or creating an optimizer,
    never while another request is using this model. Parameter objects change;
    reload a fresh model for subsequent training. The generic model loader does
    not call this function and retains its existing training semantics.
    """
    if not isinstance(model, SaddleForCausalLM):
        raise TypeError("optimize_native_inference requires a native SaddleForCausalLM")
    if any(module.training for module in model.modules()):
        raise ValueError("call model.eval() before optimize_native_inference")
    if not isinstance(pack_qkv, bool):
        raise ValueError("pack_qkv must be boolean")
    if not isinstance(pack_decode_rotary, bool):
        raise ValueError("pack_decode_rotary must be boolean")
    if not isinstance(fused_decode_kernels, bool):
        raise ValueError("fused_decode_kernels must be boolean")
    if not isinstance(cpu_prepack, bool):
        raise ValueError("cpu_prepack must be boolean")
    if max_cpu_prepack_bytes is not None and not cpu_prepack:
        raise ValueError("max_cpu_prepack_bytes requires cpu_prepack=True")
    if max_cpu_prepack_bytes is not None and (
        isinstance(max_cpu_prepack_bytes, bool)
        or not isinstance(max_cpu_prepack_bytes, int)
        or max_cpu_prepack_bytes < 0
    ):
        raise ValueError("max_cpu_prepack_bytes must be a non-negative integer or None")

    report = NativeInferenceOptimizationReport()
    from .NativeFusedKernels import configure_decode_kernels

    for key, value in configure_decode_kernels(model, fused_decode_kernels).items():
        setattr(report, key, value)
    # Pack one module at a time: do not retain all replaced weight objects and
    # temporarily double the entire model's memory footprint.
    for name, module in model.named_modules():
        if type(module) is SaddleAttention:
            module._pack_decode_rotary = bool(
                pack_decode_rotary and module.config.attention_kind != "mla"
                and next(module.parameters()).is_cuda
            )
            report.decode_rotary_packed += int(module._pack_decode_rotary)
            if module.qkv_packed_proj is not None:
                report.already_packed += 1
            elif not pack_qkv:
                report.skipped[name] = "QKV packing disabled for numerical fidelity"
            elif module.config.attention_kind == "mla" and module.config.kv_lora_rank:
                report.skipped[name] = "MLA uses latent projections"
            elif _can_pack((module.q_proj, module.k_proj, module.v_proj)):
                module.qkv_packed_proj = _pack(
                    (module.q_proj, module.k_proj, module.v_proj)
                )
                module.q_proj = module.k_proj = module.v_proj = None
                report.packed_qkv += 1
            else:
                report.skipped[name] = (
                    "Q/K/V are not compatible plain, unhooked linears"
                )
        elif type(module) is SaddleSwiGLU:
            if module.gate_up_packed_proj is not None:
                report.already_packed += 1
            elif _can_pack((module.gate_proj, module.up_proj)):
                module.gate_up_packed_proj = _pack((module.gate_proj, module.up_proj))
                module.gate_proj = module.up_proj = None
                report.packed_gate_up += 1
            else:
                report.skipped[name] = (
                    "gate/up are not compatible plain, unhooked linears"
                )
    if cpu_prepack:
        from .CPULinear import prepare_cpu_linears

        report.cpu_prepack = prepare_cpu_linears(
            model, max_prepack_bytes=max_cpu_prepack_bytes
        )
    return report
