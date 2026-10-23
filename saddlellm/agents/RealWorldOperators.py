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

from ..distillation._model_compat import inputs_requiring_logits
from .instruction_training import format_training_record, formatted_token_batch, partition_evidence, training_format


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
        total = path.stat().st_size
        if path.suffix.lower() == ".onnx":
            import onnx

            graph = onnx.load(str(path), load_external_data=False)
            locations = {
                entry.value
                for tensor in graph.graph.initializer
                for entry in tensor.external_data
                if entry.key == "location"
            }
            for location in locations:
                external = (path.parent / location).resolve()
                if not external.is_relative_to(path.parent.resolve()):
                    raise ValueError("ONNX external data must stay inside its artifact directory")
                total += external.stat().st_size
        return total
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


def _load_texts(path: Path, text_column: str = "text", limit: int = 0, record_processor=None) -> List[str]:
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
        if record_processor is not None:
            raise ValueError("chat_template training requires JSON/JSONL/CSV/Parquet instruction/messages records")
        with path.open("r", encoding="utf-8") as handle:
            chunks = [part.strip() for part in handle.read().split("\n\n")]
        texts = [part for part in chunks if part]
        return texts[:limit] if limit else texts

    texts = []
    for record in records:
        if not isinstance(record, Mapping):
            record = {text_column: record}
        text = record_processor(record) if record_processor else _record_text(record, text_column)
        if text:
            texts.append(text)
        if limit and len(texts) >= limit:
            break
    if not texts:
        raise ValueError("no non-empty text records found in %s" % path)
    return texts


def _load_text_source(raw_source: Any, work_dir: Path, text_column: str, limit: int, record_processor=None) -> Tuple[List[str], str]:
    source = str(raw_source or "").strip()
    if not source.startswith("hf://"):
        path = _resolve_path(raw_source, work_dir)
        return _load_texts(path, text_column=text_column, limit=limit, record_processor=record_processor), str(path)

    dataset_id, config_name, split = _parse_hf_uri(source)
    parquet_files = _cached_hf_parquet_files(dataset_id, config_name, split)
    if parquet_files:
        import pandas as pd

        frames = []
        for parquet_file in parquet_files:
            frame = pd.read_parquet(parquet_file, columns=None if record_processor else [text_column])
            frames.append(frame)
        records = pd.concat(frames, ignore_index=True).to_dict(orient="records")
        texts = []
        for record in records:
            text = record_processor(record) if record_processor else _record_text(record, text_column)
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
    texts = [record_processor(record) if record_processor else _record_text(record, text_column) for record in records]
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
    tokenizer = AutoTokenizer.from_pretrained(str(_value(config, "tokenizerPath", default=model_ref)), **kwargs)
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
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    cache_dir = _value(config, "cacheDir", "cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    from transformers import AutoConfig
    architecture_config = AutoConfig.from_pretrained(model_ref, **{key: value for key, value in kwargs.items() if key in {"revision", "trust_remote_code", "local_files_only", "cache_dir"}})
    architectures = getattr(architecture_config, "architectures", None) or []
    if architectures and not any("CausalLM" in str(value) or "LMHeadModel" in str(value) for value in architectures):
        raise ValueError("This artifact is not a saved causal language model; embedding/classification models cannot become generators with random LM heads")
    model, loading = AutoModelForCausalLM.from_pretrained(model_ref, output_loading_info=True, **kwargs)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Saved causal model weights are incomplete or mismatched; random initialization cannot qualify training or evaluation")
    model.to(device)
    if for_training:
        model.config.use_cache = False
    return model


def _token_batch(tokenizer, texts: Sequence[str], max_length: int, device: str):
    if texts and isinstance(texts[0], Mapping):
        if not all(isinstance(text, Mapping) and "input_ids" in text and "labels" in text for text in texts):
            raise ValueError("mixed raw and formatted instruction records are not supported")
        return formatted_token_batch(tokenizer, texts, max_length, device)
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


def _data_config(config: Mapping[str, Any], output_dir: Path, tokenizer=None) -> Tuple[List[str], List[str], Dict[str, Any]]:
    text_column = str(_value(config, "textColumn", "text_column", default="text"))
    train_source = _value(config, "datasetPath", "dataset_path")
    eval_raw = _value(config, "evalDatasetPath", "validationDatasetPath", "eval_dataset_path")
    max_train = int(_value(config, "maxTrainSamples", "max_train_samples", default=256))
    max_eval = int(_value(config, "maxEvalSamples", "max_eval_samples", default=64))
    if max_train < 1 or max_eval < 1:
        raise ValueError("maxTrainSamples and maxEvalSamples must be positive")
    mode, response_only = training_format(config)
    if mode == "chat_template" and tokenizer is None:
        raise ValueError("chat_template data loading requires the actual model tokenizer")
    processor = (lambda record: format_training_record(record, tokenizer, response_only)) if mode == "chat_template" else None
    train_texts, resolved_train = _load_text_source(train_source, output_dir, text_column, max_train, processor)
    if eval_raw:
        eval_texts, resolved_eval = _load_text_source(eval_raw, output_dir, text_column, max_eval, processor)
    else:
        take = min(max_eval, max(1, len(train_texts) // 5))
        eval_texts = train_texts[-take:]
        train_texts = train_texts[:-take] or train_texts
        resolved_eval = resolved_train
    if not train_texts or not eval_texts:
        raise ValueError("training and validation sources must contain non-empty usable records")
    provenance = {
        "datasetPath": resolved_train,
        "evalDatasetPath": resolved_eval,
        "textColumn": text_column,
        "trainSamples": len(train_texts),
        "evalSamples": len(eval_texts),
        "trainingTextFormat": mode,
        "responseOnlyLoss": response_only,
        "dataPartitionEvidence": partition_evidence(train_texts, eval_texts, tokenizer, int(_value(config, "maxLength", "maxSeqLength", default=128))),
    }
    return train_texts, eval_texts, provenance


def _sft_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from ..training.PeftDispatch import unquantized_lora_dispatch

    seed = int(_value(config, "seed", default=42))
    _seed_everything(seed)
    device = _device(config)
    model_ref = _model_ref(config, upstream)
    tokenizer = _load_tokenizer(model_ref, config)
    train_texts, eval_texts, provenance = _data_config(config, output_dir, tokenizer)
    max_length = int(_value(config, "maxLength", "maxSeqLength", default=128))
    batch_size = int(_value(config, "batchSize", default=2))
    max_steps = int(_value(config, "maxSteps", default=16))
    grad_accum = max(1, int(_value(config, "gradientAccumulationSteps", default=1)))
    learning_rate = float(_value(config, "learningRate", default=2e-4))
    if max_steps < 1 or batch_size < 1 or max_length < 2 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("SFT needs positive maxSteps, batchSize and learningRate, and maxLength >= 2")
    model = _load_model(model_ref, config, device, for_training=True)
    # Qwen and other causal models pad their vocabulary beyond the tokenizer.
    # Shrinking those pretrained rows changes the base-model contract and makes
    # PEFT save full embedding weights in an otherwise small LoRA adapter.
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
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
    with unquantized_lora_dispatch(model):
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
        if not torch.isfinite(loss).item():
            raise ValueError("SFT produced a non-finite training loss; invalid weights will not be published")
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
    """Initialise a narrower decoder student with deterministic teacher slices."""

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
    copied_storage = set()
    with __import__("torch").no_grad():
        for name, target in student_state.items():
            source_name = name
            for prefix in ("transformer.h.", "model.layers.", "gpt_neox.layers."):
                if name.startswith(prefix) and layer_map:
                    parts = name.split(".")
                    layer_index = int(parts[2])
                    parts[2] = str(layer_map[layer_index])
                    source_name = ".".join(parts)
                    break
            source = teacher_state.get(source_name)
            if source is None or source.ndim != target.ndim:
                continue
            if any(source.shape[axis] < target.shape[axis] for axis in range(target.ndim)):
                continue
            slices = tuple(slice(0, size) for size in target.shape)
            target.copy_(source[slices].to(dtype=target.dtype, device=target.device))
            # Tied embeddings and output heads appear twice in state_dict.
            storage = (target.data_ptr(), target.numel())
            if storage not in copied_storage:
                copied += target.numel()
                copied_storage.add(storage)
    student.load_state_dict(student_state)
    return copied


def _distillation_student_config(teacher_config, config: Mapping[str, Any]):
    """Preserve teacher attention/MLP geometry when reducing depth only."""
    student_config = copy.deepcopy(teacher_config)
    teacher_hidden = int(getattr(teacher_config, "hidden_size", getattr(teacher_config, "n_embd", 256)))
    teacher_heads = int(getattr(teacher_config, "num_attention_heads", getattr(teacher_config, "n_head", 4)))
    student_layers = int(_value(config, "studentLayers", default=6))
    student_hidden = int(_value(config, "studentHiddenSize", default=teacher_hidden))
    student_heads = int(_value(config, "studentHeads", default=teacher_heads))
    if student_layers < 1 or student_hidden < 1 or student_heads < 1:
        raise ValueError("studentLayers, studentHiddenSize and studentHeads must be positive")
    if student_hidden % student_heads:
        raise ValueError("studentHiddenSize must be divisible by studentHeads")
    same_width = student_hidden == teacher_hidden
    same_attention = same_width and student_heads == teacher_heads
    intermediate = int(_value(config, "studentIntermediateSize", default=(
        getattr(teacher_config, "intermediate_size", None) or getattr(teacher_config, "n_inner", None) or teacher_hidden * 4
    ) if same_width else student_hidden * 4))
    head_dim = int(_value(config, "studentHeadDim", default=(
        getattr(teacher_config, "head_dim", None) or teacher_hidden // teacher_heads
    ) if same_attention else student_hidden // student_heads))
    if intermediate < 1 or head_dim < 1:
        raise ValueError("studentIntermediateSize and studentHeadDim must be positive")
    for key, value in {
        "n_layer": student_layers, "num_hidden_layers": student_layers,
        "n_embd": student_hidden, "hidden_size": student_hidden,
        "n_head": student_heads, "num_attention_heads": student_heads,
        "n_inner": intermediate, "intermediate_size": intermediate, "head_dim": head_dim,
    }.items():
        if hasattr(student_config, key):
            setattr(student_config, key, value)
    if hasattr(student_config, "num_key_value_heads"):
        kv_heads = int(_value(config, "studentKeyValueHeads", default=min(int(teacher_config.num_key_value_heads), student_heads)))
        if "studentKeyValueHeads" not in config:
            while kv_heads > 1 and student_heads % kv_heads:
                kv_heads -= 1
        if kv_heads < 1 or student_heads % kv_heads:
            raise ValueError("studentKeyValueHeads must divide studentHeads")
        student_config.num_key_value_heads = kv_heads
    if getattr(student_config, "layer_types", None):
        teacher_layer_types = list(student_config.layer_types)
        student_config.layer_types = [
            teacher_layer_types[round(index * (len(teacher_layer_types) - 1) / max(student_layers - 1, 1))]
            for index in range(student_layers)
        ]
    if hasattr(student_config, "max_window_layers"):
        student_config.max_window_layers = min(student_config.max_window_layers, student_layers)
    student_config.use_cache = False
    return student_config


def _distillation_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM

    seed = int(_value(config, "seed", default=42))
    _seed_everything(seed)
    device = _device(config)
    teacher_ref = _model_ref(config, upstream)
    tokenizer = _load_tokenizer(teacher_ref, config)
    train_texts, eval_texts, provenance = _data_config(config, output_dir, tokenizer)
    max_length = int(_value(config, "maxLength", default=128))
    batch_size = int(_value(config, "batchSize", default=2))
    max_steps = int(_value(config, "maxSteps", default=24))
    learning_rate = float(_value(config, "learningRate", default=5e-4))
    temperature = float(_value(config, "temperature", default=2.0))
    alpha = float(_value(config, "alpha", default=0.5))
    if max_steps < 1 or batch_size < 1 or learning_rate <= 0 or temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Positive steps, batch size, learning rate and temperature, and alpha in [0,1] are required")
    teacher = _load_model(teacher_ref, config, device)
    teacher.eval()
    teacher.config.use_cache = False

    student_config = _distillation_student_config(teacher.config, config)
    # FP32 master parameters/Adam states avoid losing small updates in BF16.
    # CUDA forward still uses mixed precision through _autocast below.
    student = AutoModelForCausalLM.from_config(student_config, dtype=torch.float32).to(device)
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    student_parameters = sum(parameter.numel() for parameter in student.parameters())
    if student_parameters >= teacher_parameters:
        raise ValueError(
            "student architecture is not smaller than the teacher; reduce layers or hidden size"
        )
    copied_parameters = _copy_sliced_teacher(teacher, student)
    initial = _evaluate_torch(student, tokenizer, eval_texts, max_length, batch_size, device)
    teacher_eval = _evaluate_torch(teacher, tokenizer, eval_texts, max_length, batch_size, device)

    optimizer_parameter_dtype = str(next(student.parameters()).dtype)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    losses: List[float] = []
    hard_losses: List[float] = []
    soft_losses: List[float] = []
    started = time.perf_counter()
    student.train()
    for step, batch_texts in enumerate(_batches(train_texts, batch_size, steps=max_steps, seed=seed), 1):
        batch = _token_batch(tokenizer, batch_texts, max_length, device)
        with torch.inference_mode(), _autocast(device):
            teacher_inputs = inputs_requiring_logits(
                teacher,
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "use_cache": False,
                },
            )
            teacher_logits = teacher(**teacher_inputs).logits
        with _autocast(device):
            student_inputs = inputs_requiring_logits(
                student, {**batch, "use_cache": False}
            )
            student_output = student(**student_inputs)
        # Causal logits at position t supervise the token at position t+1.
        # Shift both teachers identically and exclude padded target positions.
        mask = (batch["labels"][:, 1:] != -100).reshape(-1)
        student_flat = student_output.logits[:, :-1, :].reshape(
            -1, student_output.logits.shape[-1]
        )[mask].float()
        teacher_flat = teacher_logits[:, :-1, :].reshape(
            -1, teacher_logits.shape[-1]
        )[mask].float()
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
        if step == 1 or step % 50 == 0 or step == max_steps:
            print("Distillation step %d/%d: loss=%.4f hard=%.4f KL=%.4f" % (
                step, max_steps, losses[-1], hard_losses[-1], soft_losses[-1]), flush=True)
    train_seconds = time.perf_counter() - started
    # Keep FP32 master weights for optimization, but publish the teacher's
    # inference precision so compression is not cancelled by a larger dtype.
    student.to(dtype=next(teacher.parameters()).dtype)
    final_eval = _evaluate_torch(student, tokenizer, eval_texts, max_length, batch_size, device)
    student.config.use_cache = True
    student_dir = output_dir / "student_model"
    student.save_pretrained(student_dir, safe_serialization=True)
    tokenizer.save_pretrained(student_dir)

    metrics = _parameter_metrics(student)
    metrics.update(
        {
            "teacher_parameter_count": int(teacher_parameters),
            "parameter_reduction_ratio": 1.0 - metrics["parameter_count"] / teacher_parameters,
            "copied_teacher_parameters": int(copied_parameters),
            "student_layers": int(getattr(student.config, "num_hidden_layers", getattr(student.config, "n_layer", 0))),
            "student_hidden_size": int(getattr(student.config, "hidden_size", getattr(student.config, "n_embd", 0))),
            "student_head_dim": getattr(student.config, "head_dim", None),
            "optimizer_parameter_dtype": optimizer_parameter_dtype,
            "artifact_parameter_dtype": str(next(student.parameters()).dtype),
            "steps": max_steps,
            "distillation_loss": sum(losses) / max(len(losses), 1),
            "hard_loss": sum(hard_losses) / max(len(hard_losses), 1),
            "soft_loss": sum(soft_losses) / max(len(soft_losses), 1),
            "teacher_eval_loss": teacher_eval["eval_loss"],
            "teacher_perplexity": teacher_eval["perplexity"],
            "student_teacher_perplexity_ratio": final_eval["perplexity"] / teacher_eval["perplexity"],
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
        warnings=(["Student perplexity is %.2fx the teacher on identical held-out data; quality requires further validation" %
                   (final_eval["perplexity"] / teacher_eval["perplexity"])]
                  if final_eval["perplexity"] > teacher_eval["perplexity"] * 2 else []),
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
    tokenizer = _load_tokenizer(model_ref, config)
    _, eval_texts, provenance = _data_config(config, output_dir, tokenizer)
    max_length = int(_value(config, "maxLength", default=128))
    batch_size = int(_value(config, "batchSize", default=2))
    sparsity = float(_value(config, "sparsity", "pruningRatio", default=0.30))
    if not 0.0 < sparsity < 1.0:
        raise ValueError("sparsity must be between 0 and 1")
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
            if getattr(self.wrapped.config, "model_type", "") in {"qwen3", "gpt2"}:
                # Transformers' vmap-based mask construction cannot be traced
                # by the legacy ONNX exporter. A supplied 4D additive mask skips
                # that path while retaining dynamic batch, sequence and padding.
                positions = torch.arange(input_ids.shape[1], device=input_ids.device)
                causal = positions[:, None] >= positions[None, :]
                allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
                dtype = next(self.wrapped.parameters()).dtype
                attention_mask = torch.zeros_like(allowed, dtype=dtype).masked_fill(
                    ~allowed, torch.finfo(dtype).min
                )
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    model = model.to("cpu").float().eval()
    if getattr(model.config, "model_type", "") in {"qwen3", "gpt2"}:
        model.config._attn_implementation = "eager"
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
    batch = _token_batch(tokenizer, texts, max_length, "cpu")
    encoded = {name: value.detach().cpu().numpy() for name, value in batch.items()}
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
    shifted_labels = encoded["labels"][:, 1:]
    shifted_mask = shifted_labels != -100
    shifted_labels = np.where(shifted_mask, shifted_labels, 0)
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
    tokenizer = _load_tokenizer(model_ref, config)
    _, eval_texts, provenance = _data_config({**config, "maxLength": int(_value(config, "maxLength", default=64))}, output_dir, tokenizer)
    max_length = int(_value(config, "maxLength", default=64))
    max_eval = int(_value(config, "maxEvalSamples", default=8))
    eval_texts = eval_texts[:max_eval]
    provenance["evalSamples"] = len(eval_texts)
    provenance["dataPartitionEvidence"]["evalRecordHashes"] = provenance["dataPartitionEvidence"]["evalRecordHashes"][:max_eval]
    provenance["dataPartitionEvidence"]["evalTokenHashes"] = provenance["dataPartitionEvidence"]["evalTokenHashes"][:max_eval]
    evidence = provenance["dataPartitionEvidence"]
    evidence.update(evalSamples=len(eval_texts), evalUniqueSamples=len(set(evidence["evalRecordHashes"])),
                    overlapUniqueSamples=len(set(evidence["trainRecordHashes"]) & set(evidence["evalRecordHashes"])),
                    overlapTokenSequences=len(set(evidence["trainTokenHashes"]) & set(evidence["evalTokenHashes"])))
    opset = int(_value(config, "opset", default=17))
    per_channel = _as_bool(_value(config, "perChannel", default=True), True)
    reduce_range = _as_bool(_value(config, "reduceRange", default=False), False)
    excluded_nodes = _as_list(
        _value(config, "excludeNodes"),
        ("/wrapped/lm_head/MatMul",),
    )
    warmups = int(_value(config, "warmupRuns", default=2))
    benchmark_runs = int(_value(config, "benchmarkRuns", default=5))
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
        reduce_range=reduce_range,
        nodes_to_exclude=excluded_nodes,
        use_external_data_format=True,
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
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(tokenizer_dir)
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
        "reduce_range": reduce_range,
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
    max_eval = int(_value(config, "maxEvalSamples", default=32))
    max_length = int(_value(config, "maxLength", default=128))
    artifact = upstream.get("artifact") if isinstance(upstream.get("artifact"), Mapping) else {}
    artifact_format = str(artifact.get("format") or "huggingface")
    model_ref = _model_ref(config, upstream)
    tokenizer_ref = str(artifact.get("tokenizerPath") or model_ref)
    tokenizer = _load_tokenizer(tokenizer_ref, config)
    _, eval_texts, provenance = _data_config(config, output_dir, tokenizer)
    eval_texts = eval_texts[:max_eval]
    provenance["evalSamples"] = len(eval_texts)
    provenance["dataPartitionEvidence"]["evalRecordHashes"] = provenance["dataPartitionEvidence"]["evalRecordHashes"][:max_eval]
    provenance["dataPartitionEvidence"]["evalTokenHashes"] = provenance["dataPartitionEvidence"]["evalTokenHashes"][:max_eval]
    evidence = provenance["dataPartitionEvidence"]
    evidence.update(evalSamples=len(eval_texts), evalUniqueSamples=len(set(evidence["evalRecordHashes"])),
                    overlapUniqueSamples=len(set(evidence["trainRecordHashes"]) & set(evidence["evalRecordHashes"])),
                    overlapTokenSequences=len(set(evidence["trainTokenHashes"]) & set(evidence["evalTokenHashes"])))
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
