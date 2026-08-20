"""Small, reproducible LLM operators used by the visual Saddle workflows.

The older public wrappers in :mod:`saddle_llm` were written around LLaMA-like
modules and do not handle GPT-2's ``Conv1D`` blocks correctly.  This module is
the runtime contract for the real-data examples in ``ai-node``.  It deliberately
keeps models out of workflow state: every function returns JSON-serialisable
paths, lineage and measured metrics only.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SUPPORTED_OPERATORS = {
    "LLMLoad",
    "SFT",
    "Distillation",
    "Pruning",
    "Quantization",
    "LLModelEvaluate",
}


def _value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key]
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: Any, default: Sequence[str] = ()) -> List[str]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
    return str(path.resolve())


def _artifact_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _resolve_path(raw_path: Any, work_dir: Path) -> Path:
    if raw_path in (None, ""):
        raise ValueError("datasetPath is required")
    raw = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, work_dir / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("input data not found: %s" % raw_path)


def _parse_hf_uri(uri: str) -> Tuple[str, str, str]:
    """Parse ``hf://owner/dataset/config/split`` without losing the owner."""

    parts = [part for part in uri[len("hf://") :].split("/") if part]
    if len(parts) < 3:
        raise ValueError(
            "Hugging Face dataset URI must be hf://<dataset>/<config>/<split> "
            "or hf://<owner>/<dataset>/<config>/<split>"
        )
    split = parts[-1]
    config_name = parts[-2]
    dataset_id = "/".join(parts[:-2])
    return dataset_id, config_name, split


def _cached_hf_parquet_files(dataset_id: str, config_name: str, split: str) -> List[Path]:
    cache_root = Path(
        os.getenv("HF_HUB_CACHE")
        or os.path.join(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub")
    )
    cache_names = [
        "datasets--" + dataset_id.replace("/", "--"),
        # Older datasets releases cached Salesforce/wikitext as simply wikitext.
        "datasets--" + dataset_id.split("/")[-1],
    ]
    found: List[Path] = []
    for cache_name in dict.fromkeys(cache_names):
        repository = cache_root / cache_name
        if not repository.exists():
            continue
        revisions: List[str] = []
        main_ref = repository / "refs" / "main"
        if main_ref.exists():
            revisions.append(main_ref.read_text(encoding="utf-8").strip())
        snapshots = repository / "snapshots"
        if snapshots.exists():
            revisions.extend(item.name for item in snapshots.iterdir() if item.is_dir())
        for revision in dict.fromkeys(revisions):
            config_dir = snapshots / revision / config_name
            if config_dir.exists():
                found.extend(sorted(config_dir.glob(split + "-*.parquet")))
    return list(dict.fromkeys(path.resolve() for path in found))


def _record_text(record: Mapping[str, Any], text_column: str) -> str:
    direct = record.get(text_column)
    if direct not in (None, ""):
        return str(direct).strip()
    if record.get("text") not in (None, ""):
        return str(record["text"]).strip()
    instruction = str(record.get("instruction") or "").strip()
    input_text = str(record.get("input") or "").strip()
    output = str(record.get("output") or record.get("response") or "").strip()
    if instruction or output:
        parts = []
        if instruction:
            parts.append("Instruction:\n" + instruction)
        if input_text:
            parts.append("Input:\n" + input_text)
        if output:
            parts.append("Response:\n" + output)
        return "\n\n".join(parts)
    return ""


def _load_texts(path: Path, text_column: str = "text", limit: int = 0) -> List[str]:
    suffix = path.suffix.lower()
    records: Iterable[Mapping[str, Any]]
    if suffix in {".jsonl", ".ndjson"}:
        loaded = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    loaded.append(json.loads(line))
                    if limit and len(loaded) >= limit:
                        break
        records = loaded
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("data") or payload.get("records") or [payload]
        records = payload
    elif suffix in {".csv", ".parquet"}:
        import pandas as pd

        frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        if limit:
            frame = frame.head(limit)
        records = frame.to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8") as handle:
            chunks = [part.strip() for part in handle.read().split("\n\n")]
        texts = [part for part in chunks if part]
        return texts[:limit] if limit else texts

    texts = []
    for record in records:
        if not isinstance(record, Mapping):
            record = {text_column: record}
        text = _record_text(record, text_column)
        if text:
            texts.append(text)
        if limit and len(texts) >= limit:
            break
    if not texts:
        raise ValueError("no non-empty text records found in %s" % path)
    return texts


def _load_text_source(raw_source: Any, work_dir: Path, text_column: str, limit: int) -> Tuple[List[str], str]:
    source = str(raw_source or "").strip()
    if not source.startswith("hf://"):
        path = _resolve_path(raw_source, work_dir)
        return _load_texts(path, text_column=text_column, limit=limit), str(path)

    dataset_id, config_name, split = _parse_hf_uri(source)
    parquet_files = _cached_hf_parquet_files(dataset_id, config_name, split)
    if parquet_files:
        import pandas as pd

        frames = []
        for parquet_file in parquet_files:
            frame = pd.read_parquet(parquet_file, columns=[text_column])
            frames.append(frame)
        records = pd.concat(frames, ignore_index=True).to_dict(orient="records")
        texts = []
        for record in records:
            text = _record_text(record, text_column)
            if text:
                texts.append(text)
            if limit and len(texts) >= limit:
                break
        if texts:
            return texts, source + "#local-snapshot"

    # Cache miss: fall back to the official datasets loader.  This is the only
    # branch that may contact the network.
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config_name, split=split)
    records = dataset.select(range(min(limit, len(dataset)))) if limit else dataset
    texts = [_record_text(record, text_column) for record in records]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError("no non-empty text records found in %s" % source)
    return texts, source


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: Mapping[str, Any]) -> str:
    import torch

    requested = str(_value(config, "device", default="auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _model_ref(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]]) -> str:
    if upstream:
        artifact = upstream.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get("path"):
            return str(artifact["path"])
        for key in ("modelPath", "model_path", "baseModelPath", "base_model_path", "modelId", "model_name"):
            if upstream.get(key):
                return str(upstream[key])
    return str(_value(config, "modelId", "model", "modelPath", "model_name", "teacherModel", default="gpt2"))


def _load_tokenizer(model_ref: str, config: Mapping[str, Any]):
    from transformers import AutoTokenizer

    kwargs: Dict[str, Any] = {
        "revision": str(_value(config, "revision", default="main")),
        "trust_remote_code": _as_bool(_value(config, "trustRemoteCode", default=False)),
        "local_files_only": _as_bool(_value(config, "localFilesOnly", default=False)),
    }
    cache_dir = _value(config, "cacheDir", "cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_ref: str, config: Mapping[str, Any], device: str, for_training: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    dtype_name = str(_value(config, "dtype", default="bfloat16" if device.startswith("cuda") else "float32")).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_name, torch.float32)
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    kwargs: Dict[str, Any] = {
        "revision": str(_value(config, "revision", default="main")),
        "trust_remote_code": _as_bool(_value(config, "trustRemoteCode", default=False)),
        "local_files_only": _as_bool(_value(config, "localFilesOnly", default=False)),
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    cache_dir = _value(config, "cacheDir", "cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    model.to(device)
    if for_training:
        model.config.use_cache = False
    return model


def _token_batch(tokenizer, texts: Sequence[str], max_length: int, device: str):
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    return encoded


def _batches(texts: Sequence[str], batch_size: int, steps: int = 0, seed: int = 42) -> Iterable[List[str]]:
    if not texts:
        return
    indices = list(range(len(texts)))
    rng = random.Random(seed)
    emitted = 0
    while True:
        rng.shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            yield [texts[index] for index in indices[offset : offset + batch_size]]
            emitted += 1
            if steps and emitted >= steps:
                return
        if not steps:
            return


def _autocast(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _evaluate_torch(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int, device: str) -> Dict[str, float]:
    import torch

    model.eval()
    weighted_loss = 0.0
    token_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_texts in _batches(texts, batch_size, seed=0):
            batch = _token_batch(tokenizer, batch_texts, max_length, device)
            with _autocast(device):
                output = model(**batch, use_cache=False)
            valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            weighted_loss += float(output.loss.float().item()) * max(valid_tokens, 1)
            token_count += max(valid_tokens, 1)
    loss = weighted_loss / max(token_count, 1)
    return {
        "eval_loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "eval_tokens": token_count,
        "eval_seconds": time.perf_counter() - started,
    }


def _parameter_metrics(model) -> Dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    bytes_ = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return {
        "parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
        "parameter_memory_mb": float(bytes_ / (1024 * 1024)),
    }


def _normal_result(
    operator: str,
    output_dir: Path,
    message: str,
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    lineage: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "success",
        "message": message,
        "operator": operator,
        "artifact": dict(artifact),
        "metrics": dict(metrics),
        "warnings": list(warnings or []),
        "lineage": dict(lineage or {}),
    }
    artifact_path = artifact.get("path")
    if artifact_path:
        result["modelPath"] = str(artifact_path)
        result["model_path"] = str(artifact_path)
    manifest_path = output_dir / "operator_result.json"
    result["reportPath"] = str(manifest_path.resolve())
    _write_json(manifest_path, result)
    return _json_safe(result)


def _load_operator(config: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    import torch

    started = time.perf_counter()
    device = _device(config)
    model_id = _model_ref(config, None)
    tokenizer = _load_tokenizer(model_id, config)
    model = _load_model(model_id, config, device)
    metrics = _parameter_metrics(model)
    metrics.update({"load_seconds": time.perf_counter() - started, "vocab_size": int(len(tokenizer))})
    revision = str(_value(config, "revision", default="main"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    artifact = {
        "kind": "base_model",
        "format": "huggingface",
        "path": model_id,
        "modelId": model_id,
        "revision": revision,
    }
    return _normal_result(
        "LLMLoad",
        output_dir,
        "Loaded %s (%0.1fM parameters)" % (model_id, metrics["parameter_count"] / 1_000_000),
        artifact,
        metrics,
        lineage={"modelId": model_id, "revision": revision},
    )


def _data_config(config: Mapping[str, Any], output_dir: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    text_column = str(_value(config, "textColumn", "text_column", default="text"))
    train_source = _value(config, "datasetPath", "dataset_path")
    eval_raw = _value(config, "evalDatasetPath", "validationDatasetPath", "eval_dataset_path")
    max_train = int(_value(config, "maxTrainSamples", "max_train_samples", default=256))
    max_eval = int(_value(config, "maxEvalSamples", "max_eval_samples", default=64))
    train_texts, resolved_train = _load_text_source(train_source, output_dir, text_column, max_train)
    if eval_raw:
        eval_texts, resolved_eval = _load_text_source(eval_raw, output_dir, text_column, max_eval)
    else:
        take = min(max_eval, max(1, len(train_texts) // 5))
        eval_texts = train_texts[-take:]
        train_texts = train_texts[:-take] or train_texts
        resolved_eval = resolved_train
    provenance = {
        "datasetPath": resolved_train,
        "evalDatasetPath": resolved_eval,
        "textColumn": text_column,
        "trainSamples": len(train_texts),
        "evalSamples": len(eval_texts),
    }
    return train_texts, eval_texts, provenance


def _sft_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model

    seed = int(_value(config, "seed", default=42))
    _seed_everything(seed)
    device = _device(config)
    model_ref = _model_ref(config, upstream)
    train_texts, eval_texts, provenance = _data_config(config, output_dir)
    max_length = int(_value(config, "maxLength", "maxSeqLength", default=128))
    batch_size = int(_value(config, "batchSize", default=2))
    max_steps = int(_value(config, "maxSteps", default=16))
    grad_accum = max(1, int(_value(config, "gradientAccumulationSteps", default=1)))
    learning_rate = float(_value(config, "learningRate", default=2e-4))
    tokenizer = _load_tokenizer(model_ref, config)
    model = _load_model(model_ref, config, device, for_training=True)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    baseline = _evaluate_torch(model, tokenizer, eval_texts, max_length, batch_size, device)

    targets = _as_list(_value(config, "targetModules"), ("c_attn", "c_proj", "c_fc"))
    lora_config = LoraConfig(
        r=int(_value(config, "loraR", "r", default=8)),
        lora_alpha=int(_value(config, "loraAlpha", "lora_alpha", default=16)),
        lora_dropout=float(_value(config, "loraDropout", "lora_dropout", default=0.05)),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    parameter_metrics = _parameter_metrics(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    started = time.perf_counter()
    for step, batch_texts in enumerate(_batches(train_texts, batch_size, steps=max_steps, seed=seed), start=1):
        batch = _token_batch(tokenizer, batch_texts, max_length, device)
        with _autocast(device):
            loss = model(**batch, use_cache=False).loss / grad_accum
        loss.backward()
        losses.append(float(loss.detach().float().item() * grad_accum))
        if step % grad_accum == 0 or step == max_steps:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    train_seconds = time.perf_counter() - started
    final_eval = _evaluate_torch(model, tokenizer, eval_texts, max_length, batch_size, device)

    adapter_dir = output_dir / "lora_adapter"
    merged_dir = output_dir / "merged_model"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    metrics: Dict[str, Any] = dict(parameter_metrics)
    metrics.update(
        {
            "steps": max_steps,
            "train_loss": sum(losses) / max(len(losses), 1),
            "baseline_eval_loss": baseline["eval_loss"],
            "baseline_perplexity": baseline["perplexity"],
            "eval_loss": final_eval["eval_loss"],
            "perplexity": final_eval["perplexity"],
            "train_seconds": train_seconds,
            "adapter_bytes": _artifact_bytes(adapter_dir),
            "merged_model_bytes": _artifact_bytes(merged_dir),
        }
    )
    artifact = {
        "kind": "fine_tuned_model",
        "format": "huggingface",
        "path": str(merged_dir.resolve()),
        "adapterPath": str(adapter_dir.resolve()),
        "tokenizerPath": str(merged_dir.resolve()),
        "parentPath": model_ref,
    }
    return _normal_result(
        "SFT",
        output_dir,
        "LoRA fine-tuning completed on %d real text samples" % len(train_texts),
        artifact,
        metrics,
        lineage={"parentModel": model_ref, **provenance},
    )


def _copy_sliced_teacher(teacher, student) -> int:
    """Initialise a narrower GPT-2 student with deterministic teacher slices."""

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    teacher_layers = int(getattr(teacher.config, "n_layer", getattr(teacher.config, "num_hidden_layers", 0)))
    student_layers = int(getattr(student.config, "n_layer", getattr(student.config, "num_hidden_layers", 0)))
    layer_map = []
    if teacher_layers and student_layers:
        if student_layers == 1:
            layer_map = [teacher_layers - 1]
        else:
            layer_map = [round(index * (teacher_layers - 1) / (student_layers - 1)) for index in range(student_layers)]
    copied = 0
    with __import__("torch").no_grad():
        for name, target in student_state.items():
            source_name = name
            if name.startswith("transformer.h."):
                parts = name.split(".")
                layer_index = int(parts[2])
                parts[2] = str(layer_map[layer_index])
                source_name = ".".join(parts)
            source = teacher_state.get(source_name)
            if source is None or source.ndim != target.ndim:
                continue
            if any(source.shape[axis] < target.shape[axis] for axis in range(target.ndim)):
                continue
            slices = tuple(slice(0, size) for size in target.shape)
            target.copy_(source[slices].to(dtype=target.dtype, device=target.device))
            copied += target.numel()
    student.load_state_dict(student_state)
    return copied


def _distillation_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM

    seed = int(_value(config, "seed", default=42))
    _seed_everything(seed)
    device = _device(config)
    teacher_ref = _model_ref(config, upstream)
    train_texts, eval_texts, provenance = _data_config(config, output_dir)
    max_length = int(_value(config, "maxLength", default=128))
    batch_size = int(_value(config, "batchSize", default=2))
    max_steps = int(_value(config, "maxSteps", default=24))
    learning_rate = float(_value(config, "learningRate", default=5e-4))
    temperature = float(_value(config, "temperature", default=2.0))
    alpha = float(_value(config, "alpha", default=0.5))
    tokenizer = _load_tokenizer(teacher_ref, config)
    teacher = _load_model(teacher_ref, config, device)
    teacher.eval()
    teacher.config.use_cache = False

    student_config = copy.deepcopy(teacher.config)
    student_layers = int(_value(config, "studentLayers", default=6))
    student_hidden = int(_value(config, "studentHiddenSize", default=256))
    student_heads = int(_value(config, "studentHeads", default=4))
    if student_hidden % student_heads:
        raise ValueError("studentHiddenSize must be divisible by studentHeads")
    for key, value in {
        "n_layer": student_layers,
        "num_hidden_layers": student_layers,
        "n_embd": student_hidden,
        "hidden_size": student_hidden,
        "n_head": student_heads,
        "num_attention_heads": student_heads,
        "n_inner": student_hidden * 4,
        "intermediate_size": student_hidden * 4,
    }.items():
        if hasattr(student_config, key):
            setattr(student_config, key, value)
    student_config.use_cache = False
    student = AutoModelForCausalLM.from_config(student_config).to(device)
    copied_parameters = _copy_sliced_teacher(teacher, student)
    initial = _evaluate_torch(student, tokenizer, eval_texts, max_length, batch_size, device)
    teacher_eval = _evaluate_torch(teacher, tokenizer, eval_texts, max_length, batch_size, device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    losses: List[float] = []
    hard_losses: List[float] = []
    soft_losses: List[float] = []
    started = time.perf_counter()
    student.train()
    for batch_texts in _batches(train_texts, batch_size, steps=max_steps, seed=seed):
        batch = _token_batch(tokenizer, batch_texts, max_length, device)
        with torch.inference_mode(), _autocast(device):
            teacher_logits = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        with _autocast(device):
            student_output = student(**batch, use_cache=False)
        mask = batch["attention_mask"].bool().reshape(-1)
        student_flat = student_output.logits.reshape(-1, student_output.logits.shape[-1])[mask].float()
        teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])[mask].float()
        soft_loss = functional.kl_div(
            functional.log_softmax(student_flat / temperature, dim=-1),
            functional.softmax(teacher_flat / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
        hard_loss = student_output.loss.float()
        loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        hard_losses.append(float(hard_loss.detach().item()))
        soft_losses.append(float(soft_loss.detach().item()))
    train_seconds = time.perf_counter() - started
    final_eval = _evaluate_torch(student, tokenizer, eval_texts, max_length, batch_size, device)
    student.config.use_cache = True
    student_dir = output_dir / "student_model"
    student.save_pretrained(student_dir, safe_serialization=True)
    tokenizer.save_pretrained(student_dir)

    metrics = _parameter_metrics(student)
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    metrics.update(
        {
            "teacher_parameter_count": int(teacher_parameters),
            "parameter_reduction_ratio": 1.0 - metrics["parameter_count"] / teacher_parameters,
            "copied_teacher_parameters": int(copied_parameters),
            "steps": max_steps,
            "distillation_loss": sum(losses) / max(len(losses), 1),
            "hard_loss": sum(hard_losses) / max(len(hard_losses), 1),
            "soft_loss": sum(soft_losses) / max(len(soft_losses), 1),
            "teacher_eval_loss": teacher_eval["eval_loss"],
            "teacher_perplexity": teacher_eval["perplexity"],
            "initial_student_eval_loss": initial["eval_loss"],
            "initial_student_perplexity": initial["perplexity"],
            "eval_loss": final_eval["eval_loss"],
            "perplexity": final_eval["perplexity"],
            "train_seconds": train_seconds,
            "student_model_bytes": _artifact_bytes(student_dir),
        }
    )
    artifact = {
        "kind": "distilled_model",
        "format": "huggingface",
        "path": str(student_dir.resolve()),
        "tokenizerPath": str(student_dir.resolve()),
        "parentPath": teacher_ref,
    }
    return _normal_result(
        "Distillation",
        output_dir,
        "Distilled %0.1fM teacher parameters into a %0.1fM student" % (
            teacher_parameters / 1_000_000,
            metrics["parameter_count"] / 1_000_000,
        ),
        artifact,
        metrics,
        lineage={"teacherModel": teacher_ref, **provenance},
    )


def _prunable_parameters(model, targets: Sequence[str]):
    selected = []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or not parameter.is_floating_point():
            continue
        if not any(target in name for target in targets):
            continue
        selected.append((name, parameter))
    return selected


def _zero_sparsity(parameters: Sequence[Tuple[str, Any]]) -> Tuple[int, int]:
    zeros = sum(int((parameter == 0).sum().item()) for _, parameter in parameters)
    total = sum(parameter.numel() for _, parameter in parameters)
    return zeros, total


def _global_magnitude_threshold(parameters: Sequence[Tuple[str, Any]], sparsity: float, bins: int = 32768) -> float:
    """Find a deterministic global threshold without concatenating huge tensors.

    ``torch.quantile`` rejects tensors above an internal element limit on some
    Windows builds.  A high-resolution accumulated histogram avoids that limit;
    the caller still reports the exact achieved zero count after masking.
    """

    import torch

    max_magnitude = max(float(parameter.detach().abs().max().item()) for _, parameter in parameters)
    if max_magnitude <= 0:
        return 0.0
    histogram = torch.zeros(bins, dtype=torch.float64)
    total = 0
    for _, parameter in parameters:
        values = parameter.detach().abs().float()
        histogram += torch.histc(values, bins=bins, min=0.0, max=max_magnitude).cpu().double()
        total += values.numel()
    target = max(1, int(total * sparsity))
    index = int(torch.searchsorted(histogram.cumsum(0), torch.tensor(float(target))).item())
    index = min(max(index, 0), bins - 1)
    return max_magnitude * float(index + 1) / float(bins)


def _pruning_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    import torch

    seed = int(_value(config, "seed", default=42))
    _seed_everything(seed)
    device = _device(config)
    model_ref = _model_ref(config, upstream)
    _, eval_texts, provenance = _data_config(config, output_dir)
    max_length = int(_value(config, "maxLength", default=128))
    batch_size = int(_value(config, "batchSize", default=2))
    sparsity = float(_value(config, "sparsity", "pruningRatio", default=0.30))
    if not 0.0 < sparsity < 1.0:
        raise ValueError("sparsity must be between 0 and 1")
    tokenizer = _load_tokenizer(model_ref, config)
    model = _load_model(model_ref, config, device)
    baseline = _evaluate_torch(model, tokenizer, eval_texts, max_length, batch_size, device)
    targets = _as_list(_value(config, "targetModules"), ("c_attn", "c_proj", "c_fc"))
    selected = _prunable_parameters(model, targets)
    if not selected:
        raise ValueError("no parameters matched targetModules=%s" % targets)
    zeros_before, selected_total = _zero_sparsity(selected)
    threshold = _global_magnitude_threshold(selected, sparsity)
    started = time.perf_counter()
    with torch.no_grad():
        for _, parameter in selected:
            parameter.masked_fill_(parameter.detach().abs() <= threshold, 0)
    prune_seconds = time.perf_counter() - started
    zeros_after, _ = _zero_sparsity(selected)
    final_eval = _evaluate_torch(model, tokenizer, eval_texts, max_length, batch_size, device)
    pruned_dir = output_dir / "pruned_model"
    model.save_pretrained(pruned_dir, safe_serialization=True)
    tokenizer.save_pretrained(pruned_dir)
    achieved = (zeros_after - zeros_before) / max(selected_total - zeros_before, 1)
    metrics = _parameter_metrics(model)
    metrics.update(
        {
            "requested_sparsity": sparsity,
            "selected_parameter_count": selected_total,
            "selected_zeros_before": zeros_before,
            "selected_zeros_after": zeros_after,
            "achieved_new_sparsity": achieved,
            "baseline_eval_loss": baseline["eval_loss"],
            "baseline_perplexity": baseline["perplexity"],
            "eval_loss": final_eval["eval_loss"],
            "perplexity": final_eval["perplexity"],
            "prune_seconds": prune_seconds,
            "pruned_model_bytes": _artifact_bytes(pruned_dir),
        }
    )
    artifact = {
        "kind": "pruned_model",
        "format": "huggingface",
        "path": str(pruned_dir.resolve()),
        "tokenizerPath": str(pruned_dir.resolve()),
        "parentPath": model_ref,
    }
    return _normal_result(
        "Pruning",
        output_dir,
        "Magnitude pruning produced %0.1f%% sparsity in GPT-style projection weights" % (100 * achieved),
        artifact,
        metrics,
        lineage={"parentModel": model_ref, **provenance},
        warnings=["Unstructured zeros do not by themselves reduce dense Hugging Face file size or guarantee GPU speedup."],
    )


def _export_onnx(model, tokenizer, output_path: Path, max_length: int, opset: int) -> None:
    import torch

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids, attention_mask):
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    model = model.to("cpu").float().eval()
    wrapper = LogitsOnly(model).eval()
    sample = tokenizer(
        "WikiText provides verified encyclopedia passages for language modelling.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            do_constant_folding=True,
            opset_version=opset,
        )


def _evaluate_onnx(
    model_path: Path,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> Dict[str, float]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    encoded = tokenizer(
        list(texts),
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    feeds = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    for _ in range(max(0, warmup_runs)):
        session.run(["logits"], feeds)
    durations = []
    logits = None
    for _ in range(max(1, benchmark_runs)):
        started = time.perf_counter()
        logits = session.run(["logits"], feeds)[0]
        durations.append(time.perf_counter() - started)
    shifted_logits = logits[:, :-1, :].astype(np.float64)
    shifted_labels = feeds["input_ids"][:, 1:]
    shifted_mask = feeds["attention_mask"][:, 1:].astype(bool)
    maxima = shifted_logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted_logits - maxima).sum(axis=-1)) + maxima.squeeze(-1)
    chosen = np.take_along_axis(shifted_logits, shifted_labels[..., None], axis=-1).squeeze(-1)
    losses = (logsumexp - chosen)[shifted_mask]
    loss = float(losses.mean())
    mean_seconds = float(sum(durations) / len(durations))
    token_count = int(shifted_mask.sum())
    return {
        "eval_loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "latency_ms": mean_seconds * 1000.0,
        "tokens_per_second": token_count / max(mean_seconds, 1e-9),
        "eval_tokens": token_count,
    }


def _quantization_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model_ref = _model_ref(config, upstream)
    _, eval_texts, provenance = _data_config(config, output_dir)
    max_length = int(_value(config, "maxLength", default=64))
    max_eval = int(_value(config, "maxEvalSamples", default=8))
    eval_texts = eval_texts[:max_eval]
    opset = int(_value(config, "opset", default=17))
    per_channel = _as_bool(_value(config, "perChannel", default=True), True)
    excluded_nodes = _as_list(
        _value(config, "excludeNodes"),
        ("/wrapped/lm_head/MatMul",),
    )
    warmups = int(_value(config, "warmupRuns", default=2))
    benchmark_runs = int(_value(config, "benchmarkRuns", default=5))
    tokenizer = _load_tokenizer(model_ref, config)
    cpu_config = dict(config)
    cpu_config["dtype"] = "float32"
    model = _load_model(model_ref, cpu_config, "cpu")
    fp32_path = output_dir / "model_fp32.onnx"
    int8_path = output_dir / "model_int8.onnx"
    started = time.perf_counter()
    _export_onnx(model, tokenizer, fp32_path, max_length, opset)
    export_seconds = time.perf_counter() - started
    fp32_bytes = _artifact_bytes(fp32_path)
    started = time.perf_counter()
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
        nodes_to_exclude=excluded_nodes,
    )
    quantize_seconds = time.perf_counter() - started
    int8_bytes = _artifact_bytes(int8_path)
    int8_eval = _evaluate_onnx(int8_path, tokenizer, eval_texts, max_length, warmups, benchmark_runs)
    fp32_eval = _evaluate_onnx(fp32_path, tokenizer, eval_texts, max_length, warmups, benchmark_runs)
    perplexity_ratio = int8_eval["perplexity"] / max(fp32_eval["perplexity"], 1e-12)
    latency_speedup = fp32_eval["latency_ms"] / max(int8_eval["latency_ms"], 1e-12)
    keep_fp32 = _as_bool(_value(config, "keepFp32Onnx", default=True), True)
    if not keep_fp32:
        fp32_path.unlink(missing_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    metrics: Dict[str, Any] = {
        "fp32_onnx_bytes": fp32_bytes,
        "int8_onnx_bytes": int8_bytes,
        "size_reduction_ratio": 1.0 - int8_bytes / max(fp32_bytes, 1),
        "fp32_eval_loss": fp32_eval["eval_loss"],
        "fp32_perplexity": fp32_eval["perplexity"],
        "fp32_latency_ms": fp32_eval["latency_ms"],
        "eval_loss": int8_eval["eval_loss"],
        "perplexity": int8_eval["perplexity"],
        "latency_ms": int8_eval["latency_ms"],
        "tokens_per_second": int8_eval["tokens_per_second"],
        "export_seconds": export_seconds,
        "quantize_seconds": quantize_seconds,
        "excluded_node_count": len(excluded_nodes),
        "perplexity_ratio_vs_fp32": perplexity_ratio,
        "latency_speedup_vs_fp32": latency_speedup,
    }
    artifact = {
        "kind": "quantized_model",
        "format": "onnx_int8",
        "path": str(int8_path.resolve()),
        "tokenizerPath": str(tokenizer_dir.resolve()),
        "parentPath": model_ref,
        "mixedPrecision": bool(excluded_nodes),
        "excludedNodes": excluded_nodes,
        "variants": {
            "fp32": str(fp32_path.resolve()) if keep_fp32 else None,
            "int8": str(int8_path.resolve()),
        },
    }
    warnings = []
    if perplexity_ratio > 1.2:
        warnings.append(
            "INT8 perplexity is %.2fx the FP32 ONNX baseline; tune pruning, excluded nodes or calibration before deployment."
            % perplexity_ratio
        )
    return _normal_result(
        "Quantization",
        output_dir,
        "ONNX Runtime INT8 quantization reduced the deployment artifact by %0.1f%%" % (100 * metrics["size_reduction_ratio"]),
        artifact,
        metrics,
        lineage={"parentModel": model_ref, **provenance},
        warnings=warnings,
    )


def _evaluation_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    if not upstream:
        raise ValueError("LLModelEvaluate requires an upstream model artifact")
    _, eval_texts, provenance = _data_config(config, output_dir)
    max_eval = int(_value(config, "maxEvalSamples", default=32))
    eval_texts = eval_texts[:max_eval]
    max_length = int(_value(config, "maxLength", default=128))
    artifact = upstream.get("artifact") if isinstance(upstream.get("artifact"), Mapping) else {}
    artifact_format = str(artifact.get("format") or "huggingface")
    model_ref = _model_ref(config, upstream)
    tokenizer_ref = str(artifact.get("tokenizerPath") or model_ref)
    tokenizer = _load_tokenizer(tokenizer_ref, config)
    if artifact_format == "onnx_int8" or str(model_ref).lower().endswith(".onnx"):
        metrics = _evaluate_onnx(
            Path(model_ref),
            tokenizer,
            eval_texts,
            max_length,
            int(_value(config, "warmupRuns", default=2)),
            int(_value(config, "benchmarkRuns", default=5)),
        )
    else:
        device = _device(config)
        model = _load_model(model_ref, config, device)
        metrics = _evaluate_torch(
            model,
            tokenizer,
            eval_texts,
            max_length,
            int(_value(config, "batchSize", default=2)),
            device,
        )
        metrics.update(_parameter_metrics(model))
    metrics["artifact_bytes"] = _artifact_bytes(Path(model_ref)) if Path(model_ref).exists() else 0
    report_artifact = {
        "kind": "evaluation_report",
        "format": "json",
        "path": str((output_dir / "operator_result.json").resolve()),
        "parentPath": model_ref,
    }
    return _normal_result(
        "LLModelEvaluate",
        output_dir,
        "Evaluated %s on %d held-out WikiText records (perplexity=%0.3f)" % (
            artifact_format,
            len(eval_texts),
            metrics["perplexity"],
        ),
        report_artifact,
        metrics,
        lineage={"evaluatedArtifact": model_ref, **provenance},
    )


def run_large_model_operator(
    label: str,
    config: Optional[Mapping[str, Any]],
    upstream_result: Optional[Mapping[str, Any]],
    work_dir: str,
) -> Dict[str, Any]:
    """Execute one visual LLM operator and return a JSON-safe result contract."""

    if label not in SUPPORTED_OPERATORS:
        raise ValueError("unsupported real-world LLM operator: %s" % label)
    config = dict(config or {})
    output_dir = Path(work_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = {
        "LLMLoad": lambda: _load_operator(config, output_dir),
        "SFT": lambda: _sft_operator(config, upstream_result, output_dir),
        "Distillation": lambda: _distillation_operator(config, upstream_result, output_dir),
        "Pruning": lambda: _pruning_operator(config, upstream_result, output_dir),
        "Quantization": lambda: _quantization_operator(config, upstream_result, output_dir),
        "LLModelEvaluate": lambda: _evaluation_operator(config, upstream_result, output_dir),
    }
    try:
        return handlers[label]()
    except Exception as exc:
        error_result = {
            "status": "error",
            "message": "%s failed: %s" % (label, exc),
            "operator": label,
            "metrics": {},
            "artifact": {},
            "warnings": [],
        }
        error_result["reportPath"] = _write_json(output_dir / "operator_result.json", error_result)
        return _json_safe(error_result)


__all__ = ["SUPPORTED_OPERATORS", "run_large_model_operator"]
