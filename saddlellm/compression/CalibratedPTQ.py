# ruff: noqa: N999
"""Opt-in calibrated AWQ/GPTQ using the maintained LLM Compressor backend.

Algorithms and checkpoint formats are distinct: both algorithms export the
``compressed-tensors`` format, not legacy AutoAWQ/AutoGPTQ kernel layouts.
Importing this module or validating configuration never imports a model stack.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "quantization_manifest.json"
REPORT_FILENAME = "ptq_report.json"
_SINGLE_DEVICE_PIPELINE_REGISTERED = False


@dataclass
class PTQConfig:
    enabled: bool = False
    method: str = "awq"
    bits: int = 4
    group_size: int = 128
    calibration_data: Any = None
    calibration_path: str | None = None
    text_column: str = "text"
    max_samples: int = 128
    max_seq_length: int = 512
    seed: int = 42
    device: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = False
    symmetric: bool = True
    targets: list[str] = field(default_factory=lambda: ["Linear"])
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])
    dampening_frac: float = 0.01
    block_size: int = 128
    actorder: str | None = "static"
    pipeline: str = "basic"


@dataclass
class PTQResult:
    model: Any
    tokenizer: Any
    report: dict[str, Any]
    output_dir: str | None = None


def _config(config: PTQConfig | Mapping[str, Any]) -> PTQConfig:
    if isinstance(config, PTQConfig):
        return config
    if isinstance(config, Mapping):
        try:
            return PTQConfig(**config)
        except TypeError as exc:
            raise ValueError(f"invalid PTQ configuration: {exc}") from exc
    raise TypeError("config must be a PTQConfig or a mapping")


def validate_ptq_config(config: PTQConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and describe a plan without file I/O, imports, or downloads."""
    config = _config(config)
    for name in ("enabled", "local_files_only", "trust_remote_code", "symmetric"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"ptq.{name} must be a boolean")  # noqa: TRY004 - configuration errors share one public exception
    if not isinstance(config.method, str) or config.method not in {"awq", "gptq"}:
        raise ValueError("ptq.method must be 'awq' or 'gptq'")
    if isinstance(config.bits, bool) or not isinstance(config.bits, int) or config.bits not in {4, 8}:
        raise ValueError("ptq.bits must be 4 or 8")
    if (isinstance(config.group_size, bool) or not isinstance(config.group_size, int)
            or config.group_size != -1 and config.group_size <= 0):
        raise ValueError("ptq.group_size must be -1 (per channel) or a positive integer")
    if config.method == "awq" and config.group_size == -1:
        raise ValueError("AWQ requires a positive group_size in the supported LLM Compressor backend")
    for name, minimum in (("max_samples", 1), ("max_seq_length", 2), ("block_size", 1), ("seed", 0)):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"ptq.{name} must be an integer >= {minimum}")
    value = config.dampening_frac
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value < 1):
        raise ValueError("ptq.dampening_frac must be finite and in (0, 1)")
    if config.actorder is not None and (not isinstance(config.actorder, str) or config.actorder not in {"static", "dynamic"}):
        raise ValueError("ptq.actorder must be null, 'static', or 'dynamic'")
    if not isinstance(config.pipeline, str) or config.pipeline not in {"basic", "sequential", "independent"}:
        raise ValueError("ptq.pipeline must be 'basic', 'sequential', or 'independent'")
    if (not isinstance(config.device, str)
            or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", config.device)):
        raise ValueError("ptq.device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    if not isinstance(config.text_column, str) or not config.text_column.strip():
        raise ValueError("ptq.text_column must be a nonempty string")
    for name in ("targets", "ignore"):
        patterns = getattr(config, name)
        if (not isinstance(patterns, (list, tuple)) or name == "targets" and not patterns
                or any(not isinstance(item, str) or not item.strip() for item in patterns)):
            raise ValueError(f"ptq.{name} must be a {'nonempty ' if name == 'targets' else ''}list of strings")
        for pattern in patterns:
            if pattern.startswith("re:"):
                try:
                    re.compile(pattern[3:])
                except re.error as exc:
                    raise ValueError(f"invalid ptq.{name} regular expression: {pattern}") from exc
    if config.calibration_data is not None and config.calibration_path is not None:
        raise ValueError("set only one of ptq.calibration_data and ptq.calibration_path")
    source = config.calibration_path if config.calibration_path is not None else config.calibration_data
    if config.enabled and source is None:
        raise ValueError("enabled PTQ requires explicit local calibration_data or calibration_path")
    if source is not None:
        if isinstance(source, (str, Path)):
            if not str(source).strip() or "://" in str(source):
                raise ValueError("PTQ calibration requires a local file, not a URL or dataset download")
            if Path(source).suffix.lower() not in {".json", ".jsonl", ".txt"}:
                raise ValueError("PTQ calibration files must be .json, .jsonl, or .txt")
        elif not isinstance(source, Sequence) or isinstance(source, (bytes, bytearray)):
            raise ValueError("ptq.calibration_data must be a local path or a sequence of text records")
        elif config.enabled and not source:
            raise ValueError("PTQ calibration_data must not be empty")
    return {
        "operation": "calibrated_ptq",
        "enabled": config.enabled,
        "method": config.method,
        "backend": "llm-compressor",
        "format": "compressed-tensors",
        "bits": config.bits,
        "group_size": config.group_size,
        "symmetric": config.symmetric,
        "targets": list(config.targets),
        "ignore": list(config.ignore),
        "max_samples": config.max_samples,
        "max_seq_length": config.max_seq_length,
        "seed": config.seed,
        "device": config.device,
        "pipeline": config.pipeline,
        "calibration_source": "local_file" if isinstance(source, (str, Path)) else "inline" if source is not None else None,
        "required_extra": "ptq",
        "model_loaded": False,
        "accelerated_inference_verified": False,
    }


plan_calibrated_ptq = validate_ptq_config


def _records(source):
    if not isinstance(source, (str, Path)):
        yield from source
        return
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PTQ calibration file does not exist: {path}")
    if path.suffix.lower() in {".txt", ".jsonl"}:
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if path.suffix.lower() == ".txt":
                    yield line
                else:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid PTQ calibration JSON in {path}, line {number}") from exc
    else:
        with path.open(encoding="utf-8-sig") as stream:
            records = json.load(stream)
        if isinstance(records, dict):
            records = records.get("data", records.get("records", [records]))
        if not isinstance(records, list):
            raise ValueError("PTQ calibration JSON must contain a list of records")
        yield from records


def load_calibration_texts(config: PTQConfig | Mapping[str, Any]) -> list[str]:
    """Deduplicate and reservoir-sample explicit data using a private seeded RNG."""
    config = _config(config)
    validate_ptq_config(config)
    source = config.calibration_path if config.calibration_path is not None else config.calibration_data
    if source is None:
        raise ValueError("PTQ requires explicit calibration data")
    rng = random.Random(config.seed)
    fingerprints: set[bytes] = set()
    selected: list[str] = []
    count = 0
    for index, record in enumerate(_records(source), 1):
        text = record.get(config.text_column) if isinstance(record, Mapping) else record
        if not isinstance(text, str):
            raise ValueError(f"PTQ calibration record {index} must be text or contain string field {config.text_column!r}")  # noqa: TRY004
        text = text.strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if digest in fingerprints:
            continue
        fingerprints.add(digest)
        count += 1
        if len(selected) < config.max_samples:
            selected.append(text)
        else:
            replacement = rng.randrange(count)
            if replacement < config.max_samples:
                selected[replacement] = text
    if not selected:
        raise ValueError("PTQ calibration has no nonempty unique text samples")
    return selected


def _load_backend(method: str):
    from ..utils.OptionalDependencies import OptionalDependencyError

    try:
        oneshot = importlib.import_module("llmcompressor").oneshot
        dataset_type = importlib.import_module("datasets").Dataset
        if method == "awq":
            try:
                awq = importlib.import_module("llmcompressor.modifiers.transform.awq").AWQModifier
            except ModuleNotFoundError as exc:
                if not exc.name or not "llmcompressor.modifiers.transform.awq".startswith(exc.name):
                    raise
                awq = importlib.import_module("llmcompressor.modifiers.awq").AWQModifier
                # In the compatible 0.6 backend AWQModifier includes the
                # quantization mixin; newer releases split out the transform.
                return oneshot, dataset_type, awq, None
            quantization = importlib.import_module("llmcompressor.modifiers.quantization").QuantizationModifier
            return oneshot, dataset_type, awq, quantization
        try:
            gptq = importlib.import_module("llmcompressor.modifiers.gptq").GPTQModifier
        except ModuleNotFoundError as exc:
            if not exc.name or not "llmcompressor.modifiers.gptq".startswith(exc.name):
                raise
            gptq = importlib.import_module("llmcompressor.modifiers.quantization").GPTQModifier
        return oneshot, dataset_type, gptq, None
    except (ImportError, OSError, AttributeError) as exc:
        raise OptionalDependencyError(
            f"{method.upper()} requires a compatible LLM Compressor installation; "
            "install `saddlellm[ptq]` in a compatible environment. "
            f"Backend import failed: {exc}"
        ) from exc


def _matches(pattern: str, name: str, module) -> bool:
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return pattern == name or pattern == type(module).__name__


def _target_names(model, config: PTQConfig, torch) -> list[str]:
    selected = []
    for name, module in model.named_modules():
        if not any(_matches(pattern, name, module) for pattern in config.targets):
            continue
        if any(_matches(pattern, name, module) for pattern in config.ignore):
            continue
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"PTQ target {name!r} is not a supported torch.nn.Linear layer")  # noqa: TRY004
        if config.group_size > 0 and module.in_features % config.group_size:
            raise ValueError(f"ptq.group_size={config.group_size} must divide input dimension {module.in_features} of {name}")
        selected.append(name)
    if not selected:
        raise ValueError("PTQ found no supported Linear targets; check targets/ignore and model architecture")
    return selected


def _device(config: PTQConfig, torch) -> str:
    device = "cuda:0" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("PTQ requested CUDA, but CUDA is unavailable; use device='cpu' or a CUDA environment")
        index = int(device.split(":")[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"PTQ CUDA device index {index} does not exist")
    return device


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _calibration_pipeline(config: PTQConfig, oneshot) -> str:
    """Register a public basic-pipeline extension that honors the caller device.

    The pinned backend's built-in basic pipeline automatically redispatches to
    every visible accelerator. Its documented registry/lifecycle extension points
    let us preserve single-device placement without altering quantization maths.
    """
    if config.pipeline != "basic" or not getattr(oneshot, "__module__", "").startswith("llmcompressor"):
        return config.pipeline
    global _SINGLE_DEVICE_PIPELINE_REGISTERED
    name = "saddlellm_single_device_basic"
    if not _SINGLE_DEVICE_PIPELINE_REGISTERED:
        pipeline_type = importlib.import_module("llmcompressor.pipelines").CalibrationPipeline
        callbacks = importlib.import_module("llmcompressor.core").LifecycleCallbacks
        forward_context = importlib.import_module("llmcompressor.utils").calibration_forward_context
        pad_mask = importlib.import_module("llmcompressor.modifiers.utils.pytorch_helpers").apply_pad_mask_to_batch
        tensors_to_device = importlib.import_module("llmcompressor.pytorch.utils.helpers").tensors_to_device

        class SingleDeviceBasicPipeline(pipeline_type):
            @staticmethod
            def __call__(model, dataloader, dataset_args):
                execution_device = next(model.parameters()).device
                callbacks.calibration_epoch_start()
                with forward_context(model):
                    for batch in dataloader:
                        model(**tensors_to_device(pad_mask(batch), execution_device))
                callbacks.calibration_epoch_end()

        pipeline_type.register(name)(SingleDeviceBasicPipeline)
        _SINGLE_DEVICE_PIPELINE_REGISTERED = True
    return name


def _verify_export(stage: Path, names: list[str]) -> None:
    saved_config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
    quantization = saved_config.get("quantization_config") or {}
    if (quantization.get("quant_method") != "compressed-tensors"
            or quantization.get("format") not in {"pack-quantized", "int-quantized"}
            or quantization.get("quantization_status") != "compressed"
            or not quantization.get("config_groups")):
        raise RuntimeError("PTQ backend did not export a compressed-tensors integer quantization_config; refusing to report success")
    files = list(stage.glob("*.safetensors"))
    if not files:
        raise RuntimeError("PTQ backend did not export safetensors model weights; refusing to report success")
    # Inspect tensor metadata lazily: validating a many-GB checkpoint must not
    # materialize a second copy of every packed tensor on CPU.
    safe_open = importlib.import_module("safetensors").safe_open
    tensor_types = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for key in stream.keys():  # noqa: SIM118 - safe_open exposes keys(), not dictionary iteration
                tensor_types[key] = stream.get_slice(key).get_dtype()
    for name in names:
        weight = f"{name}.weight_packed" if f"{name}.weight_packed" in tensor_types else f"{name}.weight"
        if tensor_types.get(weight) not in {"I32", "I8", "U8"} or f"{name}.weight_scale" not in tensor_types:
            raise RuntimeError(f"PTQ export lacks integer weights/scales for target {name}; refusing to report success")


def run_calibrated_ptq(
    model_name_or_path,
    output_dir,
    config: PTQConfig | Mapping[str, Any],
    *,
    model=None,
    tokenizer=None,
) -> PTQResult:
    """Calibrate and export AWQ/GPTQ; disabled calls are strict no-ops.

    Enabled calls mutate the supplied floating-point model. They require an
    absent or empty output directory and publish a manifest only after checking
    the backend actually wrote a compressed, self-contained HF checkpoint.
    """
    config = _config(config)
    plan = validate_ptq_config(config)
    if model is None and model_name_or_path is not None and not isinstance(model_name_or_path, (str, Path)):
        model, model_name_or_path = model_name_or_path, None
    if not config.enabled:
        return PTQResult(model, tokenizer, {**plan, "status": "disabled"})
    if output_dir is None or not str(output_dir).strip():
        raise ValueError("enabled PTQ requires output_dir")
    target = Path(output_dir).expanduser().resolve()
    if model_name_or_path is not None:
        source = Path(model_name_or_path).expanduser().resolve()
        if target == source or source.is_dir() and source in target.parents:
            raise ValueError("PTQ output_dir must remain outside the source checkpoint")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("PTQ output_dir must be absent or an empty directory")
    texts = load_calibration_texts(config)
    oneshot, dataset_type, modifier_type, quantization_type = _load_backend(config.method)
    torch = importlib.import_module("torch")
    device = _device(config, torch)
    if model is None or tokenizer is None:
        if model_name_or_path is None:
            raise ValueError("PTQ requires an in-memory model and tokenizer or model_name_or_path")
        transformers = importlib.import_module("transformers")
        load_kwargs = {"local_files_only": config.local_files_only, "trust_remote_code": config.trust_remote_code}
        if model is None:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(model_name_or_path), dtype=torch.float32 if device == "cpu" else torch.float16, **load_kwargs
            )
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_name_or_path), **load_kwargs)
    if not callable(getattr(model, "save_pretrained", None)) or not callable(getattr(getattr(model, "config", None), "save_pretrained", None)):
        raise ValueError("PTQ requires a Hugging Face causal LM with save_pretrained and a serializable config")  # noqa: TRY004
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("PTQ requires floating-point source weights; do not quantize an already quantized model")
    if len(set(map(str, (getattr(model, "hf_device_map", None) or {}).values()))) > 1:
        raise ValueError("PTQ entry point requires a single-device model; load without a sharded/offloaded hf_device_map")
    names = _target_names(model, config, torch)
    encoded = []
    for text in texts:
        row = tokenizer(text, truncation=True, max_length=config.max_seq_length, padding=False, add_special_tokens=True)
        ids = row.get("input_ids")
        if not isinstance(ids, (list, tuple)) or len(ids) < 2:
            raise ValueError("every PTQ calibration sample must tokenize to at least two tokens")
        encoded.append({"input_ids": list(ids), "attention_mask": list(row.get("attention_mask", [1] * len(ids)))})
    dataset = dataset_type.from_list(encoded)
    weights = {"num_bits": config.bits, "type": "int", "symmetric": config.symmetric,
               "strategy": "channel" if config.group_size == -1 else "group"}
    if config.group_size > 0:
        weights["group_size"] = config.group_size
    groups = {"group_0": {"targets": list(config.targets), "weights": weights}}
    if config.method == "awq":
        if quantization_type is None:
            recipe = modifier_type(config_groups=groups, ignore=list(config.ignore))
        else:
            recipe = [modifier_type(), quantization_type(config_groups=groups, ignore=list(config.ignore))]
    else:
        recipe = modifier_type(config_groups=groups, ignore=list(config.ignore),
                               dampening_frac=config.dampening_frac, block_size=config.block_size, actorder=config.actorder)
    model.to(device)
    model.eval()
    calibration_pipeline = _calibration_pipeline(config, oneshot)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.ptq-", dir=target.parent) as temporary:
        stage = Path(temporary) / "checkpoint"
        stage.mkdir()
        metadata_snapshot = Path(temporary) / "source"
        metadata_snapshot.mkdir()
        # Some backend versions inspect config.name_or_path internally. Point it
        # at a local config snapshot so even in-memory/offline models stay local.
        original_name = getattr(model.config, "_name_or_path", "")
        original_config = model.config
        model.config.save_pretrained(metadata_snapshot)
        model.config._name_or_path = str(metadata_snapshot)
        try:
            calibrated = oneshot(model=model, tokenizer=tokenizer, dataset=dataset, recipe=recipe,
                                 max_seq_length=config.max_seq_length, num_calibration_samples=len(encoded),
                                 shuffle_calibration_samples=False, log_dir=None,
                                 pipeline=calibration_pipeline,
                                 trust_remote_code_model=config.trust_remote_code)
            if calibrated is not None:
                model = calibrated
            model.save_pretrained(str(stage), save_compressed=True, safe_serialization=True)
            tokenizer.save_pretrained(str(stage))
            _verify_export(stage, names)
            # LLM Compressor 0.6 packs the supplied model in place while
            # saving. Restore runnable floating-point views of those quantized
            # values with its public decompressor, without a second model load.
            if any(not getattr(module, "weight", torch.empty(0)).is_floating_point()
                   or hasattr(module, "weight_packed") for name, module in model.named_modules() if name in names):
                compressor_type = importlib.import_module("compressed_tensors.compressors").ModelCompressor
                saved_config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
                compressor = compressor_type.from_compression_config(saved_config["quantization_config"])
                compressor.decompress_model(model)
            model.to(device)
            model._saddle_quantization_load_mode = "compressed-tensors-decompressed"
            exported_config_path = stage / "config.json"
            exported_config = json.loads(exported_config_path.read_text(encoding="utf-8"))
            exported_config.pop("_name_or_path", None)
            exported_config_path.write_text(json.dumps(exported_config, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            original_config._name_or_path = original_name
            model.config._name_or_path = original_name
        report = {**plan, "status": "completed", "model_loaded": True, "device": device,
                  "calibration_samples": len(encoded), "calibration_tokens": sum(len(row["input_ids"]) for row in encoded),
                  "calibration_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest(),
                  "quantized_modules": names, "output_dir": str(target),
                  "returned_model_mode": "compressed-tensors-decompressed",
                  "calibration_pipeline": calibration_pipeline,
                  "versions": {name: _version(name) for name in ("llmcompressor", "compressed-tensors", "torch", "transformers")},
                  "inference": {"format": "compressed-tensors", "requires": "compatible Transformers/compressed-tensors or vLLM",
                                "accelerated_inference_verified": False}}
        manifest = {"method": config.method, "backend": "llm-compressor", "format": "compressed-tensors",
                    "state_file": None, "self_contained": True, "bits": config.bits, "group_size": config.group_size,
                    "precision": f"int{config.bits}", "source_model": str(model_name_or_path) if model_name_or_path else None,
                    "report_file": REPORT_FILENAME}
        (stage / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / REPORT_FILENAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if target.exists():
            target.rmdir()  # Only an empty, prevalidated directory is accepted.
        stage.replace(target)
    return PTQResult(model, tokenizer, report, str(target))


def load_calibrated_ptq(
    path,
    *,
    device="cpu",
    dtype=None,
    trust_remote_code: bool = False,
    local_files_only: bool = True,
):
    """Load a local packed checkpoint into a runnable floating-point model.

    LLM Compressor 0.6's compressed-tensors release cannot preprocess the meta
    modules created by Transformers 4.57's quantized ``from_pretrained``. This
    compatibility path uses the public model constructor and compressor instead.
    It allocates one decoded model on CPU and streams untouched tensors and
    decompressed layers from safetensors; it does not promise packed execution.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"PTQ loading requires an existing local checkpoint directory: {source}")
    from ..utils.OptionalDependencies import OptionalDependencyError

    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        compressor_type = importlib.import_module("compressed_tensors.compressors").ModelCompressor
        safe_open = importlib.import_module("safetensors").safe_open
    except (ImportError, OSError, AttributeError) as exc:
        raise OptionalDependencyError(f"PTQ checkpoint loading requires `saddlellm[ptq]`; backend import failed: {exc}") from exc
    _verify_export(source, [])
    config = transformers.AutoConfig.from_pretrained(
        str(source), trust_remote_code=trust_remote_code, local_files_only=local_files_only,
    )
    compression_config = getattr(config, "quantization_config", None)
    if not isinstance(compression_config, dict):
        raise ValueError("PTQ checkpoint is missing its compressed-tensors quantization configuration")  # noqa: TRY004
    delattr(config, "quantization_config")
    selected_dtype = dtype
    if selected_dtype is None or selected_dtype == "auto":
        selected_dtype = getattr(config, "dtype", None) or torch.float32
    if isinstance(selected_dtype, str):
        aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        selected_dtype = getattr(torch, aliases.get(selected_dtype, selected_dtype.removeprefix("torch.")), None)
    if not isinstance(selected_dtype, torch.dtype) or not selected_dtype.is_floating_point:
        raise ValueError("PTQ decoded loading requires a floating-point dtype")
    target_device = str(device) if device is not None else "cpu"
    if target_device == "auto":
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(target_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PTQ loading requested CUDA, but CUDA is unavailable")
    with torch.device("cpu"):
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code, dtype=selected_dtype,
        )
    compressor = compressor_type.from_compression_config(compression_config)
    if compressor is None:
        raise ValueError("PTQ checkpoint does not define a usable decompressor")
    files = sorted(source.glob("*.safetensors"))
    all_keys = set()
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as stream:
            all_keys.update(stream.keys())
    quantized_names = {key.removesuffix(".weight_scale") for key in all_keys if key.endswith(".weight_scale")}
    _verify_export(source, sorted(quantized_names))
    state_keys = set(model.state_dict())
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    tensor_identities = {name: id(tensor) for name, tensor in tensors.items()}
    assigned = set()
    with torch.no_grad():
        for file in files:
            with safe_open(str(file), framework="pt", device="cpu") as stream:
                for key in stream.keys():  # noqa: SIM118
                    if key not in state_keys or key.endswith(".weight") and key.removesuffix(".weight") in quantized_names:
                        continue
                    value = stream.get_tensor(key)
                    if value.shape != tensors[key].shape:
                        raise ValueError(f"PTQ checkpoint tensor has an incompatible shape: {key}")
                    tensors[key].copy_(value)
                    assigned.add(key)
        # Decompression replaces parameters one layer at a time. Do not retain
        # references to the old initialized weights while new ones accumulate.
        del tensors
        compressor.decompress(str(source), model)
    assigned.update(f"{name}.weight" for name in quantized_names)
    assigned_identities = {tensor_identities[name] for name in assigned if name in tensor_identities}
    missing = [name for name in state_keys if name not in assigned and tensor_identities[name] not in assigned_identities]
    if missing:
        raise ValueError(f"PTQ checkpoint is missing required model tensors: {', '.join(sorted(missing)[:8])}")
    model.tie_weights()
    if (source / "generation_config.json").is_file():
        model.generation_config = transformers.GenerationConfig.from_pretrained(str(source), local_files_only=True)
    model.to(device=resolved_device, dtype=selected_dtype)
    model.eval()
    model._saddle_quantization_load_mode = "compressed-tensors-decompressed"
    model._saddle_source_quantization_config = compression_config
    model.config._name_or_path = str(source)
    return model


__all__ = [
    "PTQConfig",
    "PTQResult",
    "load_calibrated_ptq",
    "load_calibration_texts",
    "plan_calibrated_ptq",
    "run_calibrated_ptq",
    "validate_ptq_config",
]
