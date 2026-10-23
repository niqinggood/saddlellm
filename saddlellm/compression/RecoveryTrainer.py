# ruff: noqa: N999
"""Recovery of an existing floating-point/pruned causal LM.

This is ordinary recovery fine-tuning, optionally with logit distillation. It
does not implement quantization-aware training or train packed PTQ artifacts.
Local datasets are loaded in memory; each JSON/JSONL source record stays on
one side of the validation split, including all turns of a conversation.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MASKS_FILENAME = "pruning_masks.pt"
MANIFEST_FILENAME = "pruning_manifest.json"
REPORT_FILENAME = "recovery_report.json"


@dataclass
class RecoveryConfig:
    student_checkpoint: str
    train_data: str
    output_dir: str
    teacher_checkpoint: str | None = None
    validation_data: str | None = None
    validation_split: float = 0.1
    max_steps: int = 100
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 512
    temperature: float = 2.0
    distill_weight: float = 0.5
    seed: int = 42
    device: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = False
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


def validate_recovery_config(config: RecoveryConfig, *, allow_existing_output: bool = False) -> dict[str, Any]:
    """Validate a dry-run plan without importing torch or loading models."""
    for name in ("student_checkpoint", "train_data", "output_dir"):
        if not isinstance(getattr(config, name), (str, Path)) or not str(getattr(config, name)).strip():
            raise ValueError(f"{name} must be a nonempty path or model reference")
    for name, minimum in (("max_steps", 1), ("batch_size", 1), ("max_seq_length", 2)):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name in ("learning_rate", "temperature", "max_grad_norm"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, upper, inclusive in (("validation_split", 1, False), ("distill_weight", 1, True)):
        value = getattr(config, name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or value > upper or (not inclusive and value == upper)):
            raise ValueError(f"{name} must be in [0, {upper}{']' if inclusive else ')'}")
    if (isinstance(config.weight_decay, bool) or not isinstance(config.weight_decay, (int, float))
            or not math.isfinite(config.weight_decay) or config.weight_decay < 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if not isinstance(config.device, str) or not config.device.strip():
        raise ValueError("device must be 'auto' or a torch device string")
    if config.validation_data and config.validation_split:
        raise ValueError("set validation_split=0 when supplying validation_data")
    for name in ("train_data", "validation_data"):
        value = getattr(config, name)
        if value and not Path(value).expanduser().is_file():
            raise FileNotFoundError(f"{name} must be an existing local .json/.jsonl/.txt file: {value}")
        if value and Path(value).suffix.lower() not in {".json", ".jsonl", ".txt"}:
            raise ValueError(f"{name} supports local .json, .jsonl and .txt files")
    source = Path(config.student_checkpoint).expanduser()
    output = Path(config.output_dir).expanduser()
    for name in ("student_checkpoint", "teacher_checkpoint"):
        reference = getattr(config, name)
        if not reference:
            continue
        checkpoint = Path(reference).expanduser().resolve()
        if checkpoint == output.resolve() or checkpoint in output.resolve().parents:
            raise ValueError(f"output_dir must differ from and remain outside {name}; recovery never writes into source checkpoints")
    if not allow_existing_output and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_dir must be absent or an empty directory")
    if source.is_dir() and (source / "quantization_manifest.json").is_file():
        raise ValueError(_quantization_guidance())
    config_file = source / "config.json"
    if config_file.is_file():
        payload = json.loads(config_file.read_text(encoding="utf-8"))
        if payload.get("quantization_config"):
            raise ValueError(_quantization_guidance())
    return {
        "operation": "compression_recovery",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "objective": "ce_and_teacher_kl" if config.teacher_checkpoint and config.distill_weight else "causal_ce",
        "preserve_pruning_masks": True,
        "qat": False,
        "model_loaded": False,
    }


def _quantization_guidance() -> str:
    return (
        "Recovery requires the floating-point checkpoint before PTQ; packed/quantized "
        "inference artifacts cannot be optimized here. Recover that float/pruned checkpoint "
        "and quantize again. For QAT, use a backend-specific fake-quantization training "
        "recipe matching the deployment format; ordinary recovery SFT is not QAT."
    )


def _records(path: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if source.suffix.lower() == ".txt":
        records = [{"text": line.strip()} for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif source.suffix.lower() == ".jsonl":
        records = []
        for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {source}, line {number}") from exc
    else:
        records = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(records, dict):
            records = records.get("data", records.get("records", [records]))
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{source} must contain a nonempty collection of object records")
    return records


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    if "input_ids" in record:
        ids = record["input_ids"]
        if not isinstance(ids, (list, tuple)) or not ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids
        ):
            raise ValueError("input_ids must be a nonempty one-dimensional list of nonnegative integer token IDs")
        for name in ("labels", "attention_mask"):
            if name not in record:
                continue
            values = record[name]
            if not isinstance(values, (list, tuple)) or len(values) != len(ids):
                raise ValueError(f"{name} must be a one-dimensional list with the same length as input_ids")
            if name == "labels" and any(isinstance(value, bool) or not isinstance(value, int) or (value < 0 and value != -100) for value in values):
                raise ValueError("labels must be integer token IDs or -100")
            if name == "attention_mask" and any(not isinstance(value, (int, bool)) or value not in (0, 1) for value in values):
                raise ValueError("attention_mask must contain only integer 0 and 1 values")
        return {name: record[name] for name in ("input_ids", "labels", "attention_mask") if name in record}
    from ..data.PostTrainingData import PostTrainingDataAdapter

    normalized = PostTrainingDataAdapter.normalize_record(record, task="sft", keep_metadata=False)
    if not normalized:
        raise ValueError("recovery records need text, input_ids, messages, or a prompt/completion pair")
    # Strip redundant schema fields and incidental IDs before leakage checks.
    if "text" in normalized:
        return {"text": normalized["text"]}
    return {"messages": normalized["messages"]}


def _fingerprint(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _active_input_ids(record: dict[str, Any], end: int | None = None) -> list[int]:
    ids = record["input_ids"]
    attention = record.get("attention_mask", [1] * len(ids))
    return [token for token, active in zip(ids[:end], attention[:end]) if active]


def _source_groups(raw: list[dict], normalized: list[dict]) -> list[int]:
    """Connect shared identities transitively, including every chat context."""
    from ..data.ReplayData import _record_identities

    parents = list(range(len(raw)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, (source, record) in enumerate(zip(raw, normalized)):
        identities = _record_identities(source)
        identities.add(("recovery_content", _fingerprint(record)))
        if "input_ids" in record:
            identities.add(("token_content", _fingerprint({"input_ids": _active_input_ids(record)})))
            labels = record.get("labels", [])
            boundary = next((position for position, label in enumerate(labels) if label != -100), 0)
            if boundary:
                identities.add(("token_prompt", _fingerprint({"input_ids": _active_input_ids(record, boundary)})))
        for identity in identities:
            if identity in owners:
                left, right = find(index), find(owners[identity])
                parents[max(left, right)] = min(left, right)
            else:
                owners[identity] = index
    return [find(index) for index in range(len(raw))]


def _split_records(config: RecoveryConfig) -> tuple[list[dict], list[dict], dict]:
    from ..data.GroupSplit import split_group_indices

    raw = _records(config.train_data)
    records = [_normalize_record(record) for record in raw]
    if config.validation_data:
        eval_raw = _records(config.validation_data)
        evaluation = [_normalize_record(record) for record in eval_raw]
        groups = _source_groups(raw + eval_raw, records + evaluation)
        if set(groups[:len(raw)]) & set(groups[len(raw):]):
            raise ValueError("training and validation contain overlapping source records, prompts or conversation groups; use an independent held-out dataset")
        mode = "explicit_file"
    elif config.validation_split:
        groups = _source_groups(raw, records)
        train_indices, eval_indices = split_group_indices(groups, config.validation_split, config.seed)
        evaluation = [records[index] for index in eval_indices]
        records = [records[index] for index in train_indices]
        mode = "source_record_split"
    else:
        evaluation = []
        mode = "disabled"
    return records, evaluation, {"mode": mode, "train_source_records": len(records), "validation_source_records": len(evaluation)}


def _encode(tokenizer: Any, text: str, maximum: int) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=maximum)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)


def _format_messages(tokenizer: Any, messages: list[dict], generation: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
        except (ValueError, TypeError, AttributeError):
            pass
    text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    return text + ("\nassistant:" if generation else "")


def _features(records: list[dict], tokenizer: Any, maximum: int) -> list[dict]:
    result = []
    for record in records:
        if "input_ids" in record:
            ids = list(record["input_ids"])
            labels = list(record.get("labels", ids))
            attention = list(record.get("attention_mask", [1] * len(ids)))
            if len(ids) != len(labels) or len(ids) != len(attention):
                raise ValueError("input_ids, labels, and attention_mask lengths must agree")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
                raise ValueError("input_ids must be nonnegative integer token IDs")
            if any(isinstance(value, bool) or not isinstance(value, int) or (value < 0 and value != -100) for value in labels):
                raise ValueError("labels must be integer token IDs or -100")
            if any(value not in (0, 1) for value in attention):
                raise ValueError("attention_mask must contain only 0 and 1")
            ids, labels, attention = ids[:maximum], labels[:maximum], attention[:maximum]
            candidates = [{"input_ids": ids, "labels": labels, "attention_mask": attention}]
        elif "text" in record:
            ids = _encode(tokenizer, record["text"], maximum)
            candidates = [{"input_ids": ids, "labels": list(ids), "attention_mask": [1] * len(ids)}]
        else:
            candidates = []
            messages = record["messages"]
            for index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                ids = _encode(tokenizer, _format_messages(tokenizer, messages[:index + 1], False), maximum)
                prompt = _encode(tokenizer, _format_messages(tokenizer, messages[:index], True), maximum)
                boundary = 0
                for left, right in zip(ids, prompt):
                    if left != right:
                        break
                    boundary += 1
                # A tokenizer/template with no provable boundary is rejected;
                # supervising the prompt would hide malformed input data.
                if boundary == 0 or boundary >= len(ids):
                    continue
                candidates.append({"input_ids": ids, "labels": [-100] * boundary + ids[boundary:],
                                   "attention_mask": [1] * len(ids)})
        for feature in candidates:
            labels = feature["labels"]
            mask = feature["attention_mask"]
            feature["labels"] = [label if active else -100 for label, active in zip(labels, mask)]
            if len(labels) > 1 and any(feature["labels"][i] != -100 and mask[i - 1] for i in range(1, len(labels))):
                result.append(feature)
    if records and not result:
        raise ValueError("no supervised next-token targets remain after tokenization/truncation; check data and max_seq_length")
    return result


def _batch(features: list[dict], pad: int, device: Any) -> dict[str, Any]:
    import torch

    width = max(len(item["input_ids"]) for item in features)
    ids = torch.full((len(features), width), pad, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, item in enumerate(features):
        count = len(item["input_ids"])
        ids[index, :count] = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
        labels[index, :count] = torch.tensor(item["labels"], dtype=torch.long, device=device)
        attention[index, :count] = torch.tensor(item["attention_mask"], dtype=torch.long, device=device)
    return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def recovery_loss(student_logits, labels, *, attention_mask=None, teacher_logits=None,
                  temperature: float = 2.0, distill_weight: float = 0.5) -> tuple[Any, dict[str, Any]]:
    """Token-mean causal CE and T² KL(teacher || student), with shared masks.

    The KL observes only positions predicting supervised labels. Prompt labels
    of -100, padding, and the final (unpaired) logit never enter either loss.
    """
    import torch.nn.functional as F

    if student_logits.ndim != 3 or student_logits.shape[:2] != labels.shape:
        raise ValueError("student logits must have shape [batch, sequence, vocabulary] matching labels")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= distill_weight <= 1:
        raise ValueError("temperature must be positive and distill_weight must be in [0, 1]")
    target = labels[:, 1:]
    valid = target != -100
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must match labels")
        valid = valid & attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("batch contains no supervised next-token targets")
    selected = student_logits[:, :-1, :][valid].float()
    targets = target[valid]
    if int(targets.min()) < 0 or int(targets.max()) >= selected.shape[-1]:
        raise ValueError("supervised token IDs exceed the student output vocabulary")
    ce = F.cross_entropy(selected, targets)
    kd = selected.new_zeros(())
    effective_weight = distill_weight if teacher_logits is not None else 0.0
    if teacher_logits is not None:
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("teacher/student logit shapes and vocabulary sizes must agree; vocabulary remapping is unsupported")
        teacher = teacher_logits[:, :-1, :][valid].detach().float()
        kd = F.kl_div(F.log_softmax(selected / temperature, dim=-1),
                      F.softmax(teacher / temperature, dim=-1), reduction="batchmean") * temperature ** 2
    loss = (1 - effective_weight) * ce + effective_weight * kd
    return loss, {"ce_loss": ce.detach(), "kd_loss": kd.detach(), "tokens": count}


def _logits(model: Any, batch: dict):
    from ..distillation._model_compat import inputs_requiring_logits

    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**inputs_requiring_logits(model, inputs))
    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("recovery requires full sequence logits from the causal LM")
    return logits


def _check_trainable(model: Any) -> None:
    if (any(getattr(model, name, False) for name in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit", "_saddle_cuda_int8"))
            or getattr(getattr(model, "config", None), "quantization_config", None)):
        raise ValueError(_quantization_guidance())
    for module in model.modules():
        module_type = type(module)
        if ("quantized" in module_type.__module__ or module_type.__name__ in {"Linear4bit", "Linear8bitLt", "QuantLinear", "CUDAInt8Linear"}
                or hasattr(module, "qweight")):
            raise ValueError(_quantization_guidance())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(not parameter.is_floating_point() for parameter in parameters):
        raise ValueError("recovery needs trainable floating-point student parameters; reload the pre-quantization training checkpoint")


def _check_tokenizers(student: Any, teacher: Any) -> None:
    if not hasattr(student, "get_vocab") or not hasattr(teacher, "get_vocab"):
        raise ValueError("logit distillation requires tokenizers exposing get_vocab() to verify token-ID compatibility")
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("teacher/student token-ID vocabularies differ; logit distillation requires identical token mappings")
    for name in ("bos_token_id", "eos_token_id", "unk_token_id"):
        if getattr(student, name, None) != getattr(teacher, name, None):
            raise ValueError(f"teacher/student {name} differs; use matching tokenizers for distillation")


def _pruning_masks(model: Any, checkpoint: str, device: Any) -> tuple[dict, dict, dict]:
    import torch

    directory = Path(checkpoint).expanduser()
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        raise TypeError("pruning_manifest.json must be an object")
    parameters = dict(model.named_parameters())
    masks = {}
    for name, buffer in model.named_buffers():
        if name.endswith("_mask") and name[:-5] + "_orig" in parameters:
            masks[name[:-5]] = buffer.detach().bool()
    source = "live_pruning_hooks" if masks else "none"
    masks_path = directory / MASKS_FILENAME
    declared_file = manifest.get("masks_file")
    if declared_file is not None and declared_file != MASKS_FILENAME:
        raise ValueError(f"unsupported pruning masks_file: {declared_file!r}")
    if declared_file and not masks_path.is_file():
        raise FileNotFoundError(f"pruning manifest declares missing masks: {masks_path}")
    if masks_path.is_file():
        saved = torch.load(masks_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not saved:
            raise ValueError("pruning_masks.pt must contain a nonempty dictionary of parameter masks")
        declared_parameters = manifest.get("masked_parameters")
        if declared_parameters is not None and (
            not isinstance(declared_parameters, list)
            or any(not isinstance(name, str) for name in declared_parameters)
            or len(declared_parameters) != len(saved)
            or set(declared_parameters) != set(saved)
        ):
            raise ValueError("pruning masks do not match the manifest parameter list")
        for name, mask in saved.items():
            if not isinstance(name, str) or not isinstance(mask, torch.Tensor) or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("pruning masks must be named binary tensors")
            if name in masks and not torch.equal(masks[name].cpu(), mask.bool().cpu()):
                raise ValueError(f"saved and live pruning masks disagree for {name}")
            masks[name] = mask.bool()
        source = "saved_exact_masks"
    elif manifest and not masks:
        # The existing pruner materializes masks but does not save them. All
        # zero positions in its declared target weights are conservatively
        # frozen; this also preserves incidental zero weights in those targets.
        from .pruner import DEFAULT_TARGET_MODULES

        targets = manifest.get("target_modules", DEFAULT_TARGET_MODULES)
        if not isinstance(targets, (list, tuple)) or not targets or any(not isinstance(item, str) or not item for item in targets):
            raise ValueError("pruning manifest target_modules must be nonempty module-name patterns")
        for name, parameter in model.named_parameters():
            module_name, _, parameter_name = name.rpartition(".")
            if parameter_name == "weight" and parameter.ndim >= 2 and any(pattern in module_name for pattern in targets):
                masks[name] = parameter.detach().ne(0)
        if not masks:
            raise ValueError("pruning manifest targets match no student weights; refusing recovery without pruning preservation")
        source = "materialized_zero_inference"
    bound = {}
    for name, mask in masks.items():
        parameter_name = name if name in parameters else name + "_orig"
        if parameter_name not in parameters or parameters[parameter_name].shape != mask.shape:
            raise ValueError(f"pruning mask shape/name does not match student parameter {name!r}")
        parameter = parameters[parameter_name]
        mask = mask.to(device=device, dtype=torch.bool)
        if source == "saved_exact_masks" and parameter_name == name and torch.any(parameter.detach()[~mask] != 0):
            raise ValueError(f"pruning mask disagrees with materialized weights for {name!r}")
        bound[name] = (parameter, mask, parameter_name)
    metadata = {"source": source, "masked_parameters": len(bound),
                "inferred_from_materialized_zeros": source == "materialized_zero_inference"}
    return bound, manifest, metadata


def _enforce_masks(masks: dict, *, gradients: bool = False) -> None:
    import torch

    with torch.no_grad():
        for parameter, mask, _ in masks.values():
            tensor = parameter.grad if gradients else parameter
            if tensor is not None:
                tensor.mul_(mask)


def _sparsity(model: Any, masks: dict) -> dict[str, Any]:
    import torch

    total, zeros = 0, 0
    with torch.no_grad():
        for parameter in model.parameters():
            total += parameter.numel()
            zeros += int(parameter.eq(0).sum().item())
        masked = sum(int((~mask).sum().item()) for _, mask, _ in masks.values())
        violations = sum(int(parameter.detach()[~mask].ne(0).sum().item()) for parameter, mask, _ in masks.values())
    return {"total_parameters": total, "zero_parameters": zeros, "zero_fraction": zeros / total if total else 0,
            "preserved_pruned_parameters": masked, "pruned_weights_nonzero": violations}


def _evaluate(model: Any, features: list[dict], config: RecoveryConfig, pad: int, device: Any) -> dict | None:
    if not features:
        return None
    import torch

    training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.no_grad():
            for start in range(0, len(features), config.batch_size):
                batch = _batch(features[start:start + config.batch_size], pad, device)
                _, metrics = recovery_loss(_logits(model, batch), batch["labels"], attention_mask=batch["attention_mask"])
                total_loss += float(metrics["ce_loss"]) * metrics["tokens"]
                total_tokens += metrics["tokens"]
    finally:
        model.train(training)
    loss = total_loss / total_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("validation CE is nonfinite")
    return {"ce_loss": loss, "perplexity": math.exp(loss) if loss < 700 else None, "supervised_tokens": total_tokens,
            "samples": len(features)}


def _save_model(model: Any, tokenizer: Any, output: Path, masks: dict, manifest: dict) -> None:
    import torch

    state = dict(model.state_dict())
    live = False
    for name, (parameter, mask, parameter_name) in masks.items():
        if parameter_name != name:
            live = True
            state.pop(parameter_name, None)
            state.pop(name + "_mask", None)
            state[name] = parameter.detach() * mask
    model.save_pretrained(str(output), **({"state_dict": state} if live else {}))
    tokenizer.save_pretrained(str(output))
    if masks:
        torch.save({name: mask.cpu() for name, (_, mask, _) in masks.items()}, output / MASKS_FILENAME)
        saved_manifest = dict(manifest)
        saved_manifest.update({"format_version": 1, "masks_materialized": True, "masks_file": MASKS_FILENAME,
                               "masked_parameters": sorted(masks),
                               "recovery_preserved_masks": True})
        (output / MANIFEST_FILENAME).write_text(json.dumps(saved_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_recovery(config: RecoveryConfig, *, student_model=None, teacher_model=None,
                 tokenizer=None, teacher_tokenizer=None) -> dict[str, Any]:
    """Train the supplied checkpoint, save a reloadable model and measured report.

    Model/tokenizer injection supports CPU tests and callers with already loaded
    models. An injected teacher must include its tokenizer for compatibility
    verification. Training is single-process/single-device full-parameter (or
    already trainable parameter) recovery; no distributed/QAT conversion occurs.
    """
    plan = validate_recovery_config(config)
    train_records, eval_records, data_report = _split_records(config)
    import torch

    from ..data.TrainingSplitAudit import audit_record_splits
    from ..models.ModelLoader import (
        load_causal_lm,
        load_model_and_tokenizer,
        resolve_model_reference,
    )
    from ..models.TokenizerLoader import load_tokenizer_compatible

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose device='cpu'")
    loaded_student = student_model is None
    if loaded_student:
        if tokenizer is None:
            student_model, tokenizer = load_model_and_tokenizer(config.student_checkpoint, device=device, dtype=torch.float32,
                local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
        else:
            student_model = load_causal_lm(config.student_checkpoint, device=device, dtype=torch.float32,
                local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
    elif tokenizer is None:
        tokenizer = load_tokenizer_compatible(resolve_model_reference(config.student_checkpoint, local_files_only=config.local_files_only),
            local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
    _check_trainable(student_model)
    student_model.to(device)
    if teacher_model is not None and teacher_tokenizer is None:
        raise ValueError("provide teacher_tokenizer with an injected teacher to verify token-ID compatibility")
    if teacher_model is None and config.teacher_checkpoint and config.distill_weight:
        teacher_model, teacher_tokenizer = load_model_and_tokenizer(config.teacher_checkpoint, device=device, dtype=torch.float32,
            local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code)
    if teacher_model is not None:
        if student_model is teacher_model or {id(parameter) for parameter in student_model.parameters()} & {id(parameter) for parameter in teacher_model.parameters()}:
            raise ValueError("teacher and student must be independent model objects with separate parameters")
        _check_tokenizers(tokenizer, teacher_tokenizer)
        teacher_model.to(device)
        # Distinct Parameter wrappers can still alias the same storage. Such
        # an optimizer update would quietly modify the supposedly fixed teacher.
        student_storage = {(parameter.device, parameter.untyped_storage().data_ptr())
                           for parameter in student_model.parameters() if parameter.numel()}
        if any((parameter.device, parameter.untyped_storage().data_ptr()) in student_storage
               for parameter in teacher_model.parameters() if parameter.numel()):
            raise ValueError("teacher and student must have separate parameter storage; clone the teacher checkpoint")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
        if pad is None:
            raise ValueError("tokenizer must define a pad_token_id or eos_token_id; vocabulary resizing is not performed")
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    train = _features(train_records, tokenizer, config.max_seq_length)
    evaluation = _features(eval_records, tokenizer, config.max_seq_length)
    # Audit the actual prepared prompts as well as complete inputs: different
    # answers must not hide a shared held-out context after tokenization.
    data_report["split_audit"] = audit_record_splits(
        train_records, eval_records,
        train_features=train, validation_features=evaluation,
    )
    masks, manifest, mask_report = _pruning_masks(student_model, config.student_checkpoint, device)
    _enforce_masks(masks)
    before_sparsity = _sparsity(student_model, masks)
    student_training = student_model.training
    teacher_training = teacher_model.training if teacher_model is not None else None
    teacher_flags = [parameter.requires_grad for parameter in teacher_model.parameters()] if teacher_model is not None else []
    history = []
    if teacher_model is not None:
        teacher_model.eval().requires_grad_(False)
    try:
        before_validation = _evaluate(student_model, evaluation, config, int(pad), device)
        optimizer = torch.optim.AdamW([parameter for parameter in student_model.parameters() if parameter.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay)
        student_model.train()
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
                loss, metrics = recovery_loss(_logits(student_model, batch), batch["labels"], attention_mask=batch["attention_mask"],
                    teacher_logits=teacher_logits, temperature=config.temperature, distill_weight=config.distill_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"recovery loss became nonfinite at step {step + 1}")
                loss.backward()
                _enforce_masks(masks, gradients=True)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                # Gradient masking alone cannot protect against optimizer state
                # or decoupled weight decay. Reapply after every update as well.
                _enforce_masks(masks)
                history.append({"step": step + 1, "loss": float(loss.detach()), "ce_loss": float(metrics["ce_loss"]),
                                "kd_loss": float(metrics["kd_loss"]), "supervised_tokens": metrics["tokens"]})
                del loss, teacher_logits, batch
        optimizer.zero_grad(set_to_none=True)
        after_validation = _evaluate(student_model, evaluation, config, int(pad), device)
        after_sparsity = _sparsity(student_model, masks)
        if after_sparsity["pruned_weights_nonzero"]:
            raise RuntimeError("pruned weights regrew despite mask enforcement")
        output = Path(config.output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        _save_model(student_model, tokenizer, output, masks, manifest)
        report = {
            "format_version": 1, "operation": "compression_recovery", "qat": False,
            "student_checkpoint": str(config.student_checkpoint), "teacher_checkpoint": str(config.teacher_checkpoint) if config.teacher_checkpoint else None,
            "student_injected": not loaded_student, "teacher_used": teacher_model is not None and bool(config.distill_weight),
            "config": plan["config"], "device": str(device), "actual_steps": len(history),
            "effective_distill_weight": config.distill_weight if teacher_model is not None else 0.0,
            "train_samples": len(train), "validation_samples": len(evaluation), "data": data_report,
            "initial_train_loss": history[0]["loss"], "final_train_loss": history[-1]["loss"],
            "mean_train_loss": sum(item["loss"] for item in history) / len(history), "loss_history": history,
            "validation_before": before_validation, "validation_after": after_validation,
            "sparsity_before": before_sparsity, "sparsity_after": after_sparsity, "pruning_masks": mask_report,
            "output_dir": str(output.resolve()), "report_path": str((output / REPORT_FILENAME).resolve()),
            "limitations": ["Single-device recovery; both models and logits must fit memory.",
                            "Recovery loss does not establish general capability retention; run the regression suite on the saved artifact.",
                            "This is CE/KD recovery, not quantization-aware training."]
        }
        if mask_report["inferred_from_materialized_zeros"]:
            report["limitations"].append("Original pruning masks were absent; all zero positions in declared target weights were frozen and exact masks were saved for future recovery.")
        (output / REPORT_FILENAME).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        student_model.train(student_training)
        if teacher_model is not None:
            teacher_model.train(teacher_training)
            for parameter, flag in zip(teacher_model.parameters(), teacher_flags):
                parameter.requires_grad_(flag)


class RecoveryTrainer:
    """Small object API around :func:`run_recovery`."""

    def __init__(self, config: RecoveryConfig, **models: Any):
        self.config = config
        self.models = models

    def train(self) -> dict[str, Any]:
        return run_recovery(self.config, **self.models)


__all__ = ["RecoveryConfig", "RecoveryTrainer", "recovery_loss", "run_recovery", "validate_recovery_config"]
