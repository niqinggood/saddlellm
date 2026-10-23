# ruff: noqa: N999
"""Backend-specific INT8 weight-only QAT and its portable inference format.

This backend stores Linear/Conv1D weights as symmetric per-output-channel INT8.
Training and converted inference use the same quantize/dequantize equation.
Activations, biases and other modules remain FP32. Inference dequantizes weights
before floating-point matmul: this is a compact, reloadable reference format,
not a claim of accelerated INT8 kernels or compatibility with AWQ/GPTQ.

Configuration/dry-run imports are deliberately free of torch/transformers.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

BACKEND = "portable-int8-weight-only"
STATE_FILENAME = "qat_model.pt"
REPORT_FILENAME = "qat_report.json"
MANIFEST_FILENAME = "quantization_manifest.json"


@dataclass
class QATConfig:
    enabled: bool = False
    backend: str = BACKEND
    train_data: str | None = None
    validation_data: str | None = None
    validation_split: float = 0.1
    max_steps: int = 100
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 512
    teacher_checkpoint: str | None = None
    temperature: float = 2.0
    distill_weight: float = 0.5
    seed: int = 42
    device: str = "cpu"
    local_files_only: bool = False
    trust_remote_code: bool = False
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    exclude_modules: tuple[str, ...] = ()


@dataclass
class QATResult:
    model: Any
    tokenizer: Any
    report: dict[str, Any]
    output_dir: str | None


def _recovery_config(source, output_dir, config):
    from .RecoveryTrainer import RecoveryConfig

    fields = asdict(config)
    for name in ("enabled", "backend", "exclude_modules"):
        fields.pop(name)
    return RecoveryConfig(student_checkpoint=str(source), output_dir=str(output_dir), **fields)


def validate_qat_config(config: QATConfig, source=None, output_dir=None, *, allow_existing_output=False) -> dict[str, Any]:
    """Validate without loading a model or importing the training stack."""
    if not isinstance(config.enabled, bool):
        raise TypeError("qat.enabled must be a boolean")
    if config.backend != BACKEND:
        raise ValueError(f"QAT backend must be {BACKEND!r}; this format is not dynamic INT8 or AWQ/GPTQ")
    if (not isinstance(config.exclude_modules, (list, tuple))
            or any(not isinstance(name, str) or not name for name in config.exclude_modules)):
        raise ValueError("exclude_modules must contain nonempty module names")
    plan = {"operation": "qat", "enabled": config.enabled, "backend": config.backend,
            "precision": "int8", "weight_scheme": "symmetric_per_output_channel",
            "activation_precision": "float32", "model_loaded": False,
            "config": {name: str(value) if isinstance(value, Path) else value for name, value in asdict(config).items()}}
    if not config.enabled:
        return plan
    if not source or not output_dir:
        raise ValueError("enabled QAT requires source and output_dir")
    from .RecoveryTrainer import validate_recovery_config

    validate_recovery_config(_recovery_config(source, output_dir, config), allow_existing_output=allow_existing_output)
    return plan


@lru_cache(maxsize=1)
def _module_types():
    import torch
    import torch.nn.functional as F

    class FakeQuantLinear(torch.nn.Module):
        """Keep float master weights and use the deployment quantizer forward."""

        def __init__(self, source, transposed=False):
            super().__init__()
            self.weight = source.weight
            self.bias = source.bias
            self.transposed = transposed
            weight = source.weight.T if transposed else source.weight
            self.out_features, self.in_features = weight.shape

        def forward(self, inputs):
            weight = self.weight.T if self.transposed else self.weight
            scales = _scales(weight)
            zeros = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.int32)
            quantized = torch.fake_quantize_per_channel_affine(weight, scales, zeros, 0, -127, 127)
            return F.linear(inputs, quantized, self.bias)

    class PackedInt8Linear(torch.nn.Module):
        """Portable weight-only storage; matmul remains floating point."""

        def __init__(self, in_features, out_features, bias, transposed=False):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.transposed = transposed
            self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scales", torch.ones(out_features, dtype=torch.float32))
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32) if bias else None)

        @property
        def weight(self):
            weight = self.qweight.to(dtype=self.scales.dtype) * self.scales[:, None]
            return weight.T if self.transposed else weight

        def forward(self, inputs):
            weight = self.qweight.to(dtype=self.scales.dtype) * self.scales[:, None]
            return F.linear(inputs, weight, self.bias)

    return FakeQuantLinear, PackedInt8Linear


def _scales(weight):
    import torch

    # Detached qparams make this the usual STE for the float master weights;
    # zero channels receive a finite scale and stay exactly zero.
    return (weight.detach().abs().amax(dim=1) / 127).clamp_min(torch.finfo(torch.float32).eps).float()


def _set_module(model, name, module):
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def prepare_qat_model(model, exclude_modules=()) -> list[dict[str, Any]]:
    """Replace supported projections in-place, preserving parameter identity."""
    import torch

    FakeQuantLinear, _ = _module_types()
    targets = []
    # Avoid importing transformers just to identify an optional Conv1D module.
    for name, module in list(model.named_modules()):
        conv1d = type(module).__name__ == "Conv1D" and type(module).__module__ == "transformers.pytorch_utils"
        if not name or not (type(module) is torch.nn.Linear or conv1d):
            continue
        if any(name == excluded or name.startswith(excluded + ".") for excluded in exclude_modules):
            continue
        if hasattr(module, "weight_orig"):
            raise ValueError("materialize live pruning hooks before prepare_qat_model; run_qat does this while preserving masks")
        replacement = FakeQuantLinear(module, transposed=conv1d)
        _set_module(model, name, replacement)
        targets.append({"name": name, "in_features": replacement.in_features,
                        "out_features": replacement.out_features, "bias": replacement.bias is not None,
                        "transposed": conv1d})
    if not targets:
        raise ValueError("QAT found no supported Linear or transformers Conv1D projections")
    return targets


def convert_qat_model(model) -> list[dict[str, Any]]:
    """Convert master weights to INT8 buffers using the identical QAT equation."""
    import torch

    FakeQuantLinear, PackedInt8Linear = _module_types()
    targets = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, FakeQuantLinear):
            continue
        weight = module.weight.T if module.transposed else module.weight
        scales = _scales(weight)
        packed = PackedInt8Linear(module.in_features, module.out_features,
                                 module.bias is not None, module.transposed).to(weight.device)
        with torch.no_grad():
            zeros = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.int32)
            dequantized = torch.fake_quantize_per_channel_affine(weight, scales, zeros, 0, -127, 127)
            # Recover integer codes from that very same operator. Re-rounding
            # original weights with a different division/multiply order could
            # disagree at floating-point half-step boundaries.
            packed.qweight.copy_(torch.round(dequantized / scales[:, None]).clamp(-127, 127).to(torch.int8))
            packed.scales.copy_(scales)
            if module.bias is not None:
                packed.bias.copy_(module.bias)
        _set_module(model, name, packed)
        targets.append({"name": name, "in_features": packed.in_features, "out_features": packed.out_features,
                        "bias": packed.bias is not None, "transposed": packed.transposed})
    if not targets:
        raise ValueError("model contains no prepared QAT projections")
    return targets


def _save_qat(model, tokenizer, output, source, targets, masks, pruning_manifest):
    import torch

    config = getattr(model, "config", None)
    if config is not None and callable(getattr(config, "save_pretrained", None)):
        config.save_pretrained(str(output))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output))
    torch.save(model.state_dict(), output / STATE_FILENAME)
    manifest = {
        "schema_version": 1, "method": "qat", "backend": BACKEND, "precision": "int8",
        "activation_precision": "float32", "weight_scheme": "symmetric_per_output_channel",
        "source_model": str(source), "state_file": STATE_FILENAME, "modules": targets,
        "self_contained": ((output / "saddle_config.json").is_file() or (output / "config.json").is_file())
        and not bool(getattr(config, "auto_map", None)),
        "inference": "dequantize_weights_then_float_matmul", "accelerated_int8_kernels": False,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if masks:
        from .RecoveryTrainer import MANIFEST_FILENAME as PRUNING_MANIFEST
        from .RecoveryTrainer import MASKS_FILENAME

        torch.save({name: mask.cpu() for name, (_, mask, _) in masks.items()}, output / MASKS_FILENAME)
        pruning = dict(pruning_manifest)
        pruning.update({"masks_file": MASKS_FILENAME, "masked_parameters": sorted(masks), "qat_preserved_masks": True,
                        "note": "Masks describe original float parameters; quantized weights are inference-only."})
        (output / PRUNING_MANIFEST).write_text(json.dumps(pruning, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_portable_qat_model(path, *, model=None, trust_remote_code=False, local_files_only=True):
    """Validate the artifact and restore its portable FP32 model."""
    import torch

    output = Path(path).expanduser()
    manifest = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("method") != "qat" or manifest.get("backend") != BACKEND:
        raise ValueError("unsupported QAT manifest/backend")
    if (manifest.get("schema_version") != 1 or manifest.get("precision") != "int8"
            or manifest.get("activation_precision") != "float32"
            or manifest.get("weight_scheme") != "symmetric_per_output_channel"):
        raise ValueError("unsupported QAT manifest precision/scheme/version")
    if manifest.get("state_file") != STATE_FILENAME:
        raise ValueError("unsupported QAT state file")
    if model is None:
        if (output / "saddle_config.json").is_file():
            from ..models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

            config = SaddleModelConfig(**json.loads((output / "saddle_config.json").read_text(encoding="utf-8")))
            model = SaddleForCausalLM(config)
        elif (output / "config.json").is_file():
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(str(output), trust_remote_code=trust_remote_code,
                                                local_files_only=local_files_only)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        else:
            raise ValueError("QAT artifact has no architecture config; pass model= for the original custom class")
    model = model.float()
    _, PackedInt8Linear = _module_types()
    targets = manifest.get("modules")
    if not isinstance(targets, list) or not targets:
        raise ValueError("QAT manifest must list converted projections")
    seen = set()
    for target in targets:
        if (not isinstance(target, dict) or not isinstance(target.get("name"), str) or not target["name"]
                or target["name"] in seen or not isinstance(target.get("bias"), bool)
                or not isinstance(target.get("transposed"), bool)
                or any(type(target.get(field)) is not int or target[field] < 1 for field in ("in_features", "out_features"))):
            raise ValueError("invalid QAT projection metadata")
        name = target["name"]
        seen.add(name)
        original = model.get_submodule(name)
        weight = getattr(original, "weight", None)
        shape = (target["in_features"], target["out_features"]) if target.get("transposed") else (target["out_features"], target["in_features"])
        if weight is None or tuple(weight.shape) != shape:
            raise ValueError(f"QAT projection architecture mismatch: {name}")
        _set_module(model, name, PackedInt8Linear(target["in_features"], target["out_features"],
                                                 target["bias"], target.get("transposed", False)))
    state = torch.load(output / STATE_FILENAME, map_location="cpu", weights_only=True)
    for target in targets:
        prefix = target["name"] + "."
        codes, scales = state.get(prefix + "qweight"), state.get(prefix + "scales")
        if (not isinstance(codes, torch.Tensor) or codes.dtype != torch.int8
                or tuple(codes.shape) != (target["out_features"], target["in_features"])
                or bool((codes == -128).any()) or not isinstance(scales, torch.Tensor)
                or scales.dtype != torch.float32 or tuple(scales.shape) != (target["out_features"],)
                or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
            raise ValueError(f"invalid QAT INT8 weight state: {target['name']}")
    model.load_state_dict(state, strict=True)
    return model


def load_qat(path, *, model=None, device="cpu", trust_remote_code=False, local_files_only=True,
             inference_backend="reference", dtype=None):
    """Load the portable reference, or explicitly execute its codes on CUDA.

    ``reference`` retains the existing FP32 dequantize/float-matmul contract.
    ``cuda-w8a16`` requires an explicit CUDA device and FP16/BF16 dtype; it
    reuses existing codes/scales with Triton kernels and never falls back.
    The original portable artifact remains unchanged. Runtime CUDA models are
    inference-only and cannot be re-exported as ordinary float checkpoints.
    """
    import torch

    if inference_backend not in {"reference", "cuda-w8a16"}:
        raise ValueError("QAT inference_backend must be reference or cuda-w8a16")
    if inference_backend == "cuda-w8a16":
        from ..models.QATW8A16 import enable_qat_cuda_w8a16, validate_qat_cuda_request

        device, dtype = validate_qat_cuda_request(device, dtype)
    elif dtype not in (None, torch.float32, "float32"):
        raise ValueError("portable QAT reference loading requires float32 dtype")
    model = _load_portable_qat_model(
        path, model=model, trust_remote_code=trust_remote_code, local_files_only=local_files_only,
    )
    if inference_backend == "cuda-w8a16":
        return enable_qat_cuda_w8a16(model, device=device, dtype=dtype)
    model.to(device).eval()
    return model


def run_qat(source, output_dir, config: QATConfig, *, model=None, tokenizer=None,
            teacher_model=None, teacher_tokenizer=None) -> QATResult:
    """Train fake-quantized projections, convert, save and measure the artifact."""
    plan = validate_qat_config(config, source, output_dir)
    if not config.enabled:
        return QATResult(model, tokenizer, {**plan, "skipped": True, "actual_steps": 0}, None)
    recovery_config = _recovery_config(source, output_dir, config)
    from ..data.TrainingSplitAudit import audit_record_splits
    from .RecoveryTrainer import (
        _batch,
        _check_tokenizers,
        _check_trainable,
        _enforce_masks,
        _evaluate,
        _features,
        _logits,
        _pruning_masks,
        _sparsity,
        _split_records,
        recovery_loss,
    )
    train_records, eval_records, data_report = _split_records(recovery_config)
    import torch

    from ..models.ModelLoader import load_causal_lm, load_model_and_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose device='cpu'")
    loaded_model = model is None
    if model is None:
        if tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(source, device=device, dtype=torch.float32,
                local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
        else:
            model = load_causal_lm(source, device=device, dtype=torch.float32,
                local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
    if tokenizer is None:
        raise ValueError("provide tokenizer= with an injected QAT model")
    _check_trainable(model)
    if getattr(model, "peft_config", None):
        raise ValueError("QAT requires a merged floating-point model; merge_and_unload() adapters before this backend")
    model.to(device).float()
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        raise ValueError("QAT tokenizer requires pad_token_id or eos_token_id")
    train = _features(train_records, tokenizer, config.max_seq_length)
    evaluation = _features(eval_records, tokenizer, config.max_seq_length)
    # Validate held-out contexts before replacing projections or optimizing.
    data_report["split_audit"] = audit_record_splits(
        train_records, eval_records,
        train_features=train, validation_features=evaluation,
    )
    if teacher_model is not None and teacher_tokenizer is None:
        raise ValueError("provide teacher_tokenizer with an injected teacher")
    if teacher_model is None and config.teacher_checkpoint and config.distill_weight:
        teacher_model, teacher_tokenizer = load_model_and_tokenizer(config.teacher_checkpoint, device=device, dtype=torch.float32,
            local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
    if teacher_model is not None:
        _check_tokenizers(tokenizer, teacher_tokenizer)
        teacher_model.to(device).float()
        student_storage = {(parameter.device, parameter.untyped_storage().data_ptr()) for parameter in model.parameters() if parameter.numel()}
        if model is teacher_model or any((parameter.device, parameter.untyped_storage().data_ptr()) in student_storage
                                         for parameter in teacher_model.parameters() if parameter.numel()):
            raise ValueError("QAT teacher and student must have independent parameter storage")
    masks, pruning_manifest, mask_report = _pruning_masks(model, str(source), device)
    # Materialize pruning reparameterizations while keeping exact masks bound
    # to the same Parameter objects for gradient and post-update enforcement.
    if any(parameter_name != name for name, (_, _, parameter_name) in masks.items()):
        from torch.nn.utils import prune

        for name, (parameter, mask, parameter_name) in list(masks.items()):
            if parameter_name != name:
                module_name, _, parameter_name = name.rpartition(".")
                prune.remove(model.get_submodule(module_name), parameter_name)
                masks[name] = (parameter, mask, name)
    _enforce_masks(masks)
    before_sparsity = _sparsity(model, masks)
    student_training = model.training
    teacher_training = teacher_model.training if teacher_model is not None else None
    teacher_flags = [parameter.requires_grad for parameter in teacher_model.parameters()] if teacher_model is not None else []
    history = []
    completed = False
    if teacher_model is not None:
        teacher_model.eval().requires_grad_(False)
    try:
        float_validation = _evaluate(model, evaluation, recovery_config, int(pad), device)
        targets = prepare_qat_model(model, config.exclude_modules)
        fake_validation = _evaluate(model, evaluation, recovery_config, int(pad), device)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                     lr=config.learning_rate, weight_decay=config.weight_decay)
        model.train()
        generator = random.Random(config.seed)
        indices = list(range(len(train)))
        cursor = len(indices)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config.seed)
            for step in range(config.max_steps):
                if cursor >= len(indices):
                    generator.shuffle(indices)
                    cursor = 0
                chosen = indices[cursor:cursor + config.batch_size]
                cursor += len(chosen)
                batch = _batch([train[index] for index in chosen], int(pad), device)
                optimizer.zero_grad(set_to_none=True)
                teacher_logits = None
                if teacher_model is not None and config.distill_weight:
                    with torch.no_grad():
                        teacher_logits = _logits(teacher_model, batch)
                loss, metrics = recovery_loss(_logits(model, batch), batch["labels"], attention_mask=batch["attention_mask"],
                    teacher_logits=teacher_logits, temperature=config.temperature, distill_weight=config.distill_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"QAT loss became nonfinite at step {step + 1}")
                loss.backward()
                _enforce_masks(masks, gradients=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                _enforce_masks(masks)
                history.append({"step": step + 1, "loss": float(loss.detach()), "ce_loss": float(metrics["ce_loss"]),
                                "kd_loss": float(metrics["kd_loss"]), "supervised_tokens": metrics["tokens"]})
        optimizer.zero_grad(set_to_none=True)
        trained_validation = _evaluate(model, evaluation, recovery_config, int(pad), device)
        after_sparsity = _sparsity(model, masks)
        if after_sparsity["pruned_weights_nonzero"]:
            raise RuntimeError("pruned weights regrew during QAT")
        targets = convert_qat_model(model)
        model.eval()
        converted_validation = _evaluate(model, evaluation, recovery_config, int(pad), device)
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        manifest = _save_qat(model, tokenizer, output, source, targets, masks, pruning_manifest)
        report = {
            "format_version": 1, "operation": "qat", "enabled": True, "qat": True,
            "backend": BACKEND, "precision": "int8", "activation_precision": "float32",
            "source_model": str(source), "model_injected": not loaded_model,
            "teacher_used": teacher_model is not None and bool(config.distill_weight),
            "device": str(device), "config": plan["config"], "actual_steps": len(history),
            "converted_modules": len(targets), "modules": targets, "data": data_report,
            "train_samples": len(train), "validation_samples": len(evaluation), "loss_history": history,
            "validation_float_before": float_validation, "validation_fake_quant_before": fake_validation,
            "validation_fake_quant_after": trained_validation, "validation_converted": converted_validation,
            "sparsity_before": before_sparsity, "sparsity_after": after_sparsity, "pruning_masks": mask_report,
            "self_contained": manifest["self_contained"], "output_dir": str(output.resolve()),
            "report_path": str((output / REPORT_FILENAME).resolve()),
            "limitations": ["Reference INT8 weight-only format; weights dequantize for float matmul, with no accelerated INT8 kernel claim.",
                            "Activations, biases, embeddings and unsupported modules remain FP32.",
                            "This format is separate from AWQ/GPTQ and PyTorch dynamic INT8; evaluate the final artifact on capability regression data.",
                            "Single-device full-parameter QAT; not a distributed or low-memory training recipe."],
        }
        (output / REPORT_FILENAME).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        completed = True
        return QATResult(model, tokenizer, report, str(output.resolve()))
    finally:
        if not completed:
            model.train(student_training)
        if teacher_model is not None:
            teacher_model.train(teacher_training)
            for parameter, flag in zip(teacher_model.parameters(), teacher_flags):
                parameter.requires_grad_(flag)


__all__ = ["BACKEND", "QATConfig", "QATResult", "convert_qat_model", "load_qat", "prepare_qat_model", "run_qat", "validate_qat_config"]
