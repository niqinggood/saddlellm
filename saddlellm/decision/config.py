"""Lightweight configuration shared by CLI, validation and the decision stage."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DecisionStageConfig:
    enabled: bool = True
    task_type: str = "choice"
    method: str = "supervised"
    freeze_backbone: bool = True
    dataset_dir: str = ""
    backbone_name_or_path: str | None = None
    output_dir: str = "decision"
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    projection_rank: int = 64
    cpu_threads: int = 4
    encoder_batch_size: int = 16
    max_length: int = 512
    encoder_device: str = "auto"
    encoder_dtype: str = "auto"
    typed_backbone: str = "native"
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_candidates: int = 100
    brier_weight: float = .1
    ordinal_weight: float = .1
    rl_weight: float = .1
    rl_samples: int = 8
    rl_std: float = .5
    supervised_weight: float = 1.
    initial_model: str | None = None
    optimizer_device: str = "model"
    query_pooling: str = "mean"
    instance_question_in_query: bool = False
    backbone_learning_rate: float | None = None
    head_learning_rate: float | None = None
    head_warmup_epochs: int = 0
    early_stopping_patience: int = 0
    evaluate_test: bool = True

    @classmethod
    def from_dict(cls, values, *, backbone_name_or_path=None):
        if not isinstance(values, dict):
            raise TypeError("decision must be a mapping")
        settings = dict(values)
        if settings.get("task_type") == "typed":
            hf = settings.get("typed_backbone", "native") == "hf"
            defaults = {"freeze_backbone": False, "batch_size": 1 if hf else 16,
                        "learning_rate": .00002 if hf else .001, "epochs": 3 if hf else 20,
                        "max_length": 256, "encoder_dtype": "float32"}
            settings = {**defaults, **settings}
            if hf and "hidden_size" not in settings:
                import json
                source = settings.get("backbone_name_or_path") or backbone_name_or_path
                if source and (Path(source).expanduser() / "config.json").is_file():
                    settings["hidden_size"] = json.loads((Path(source).expanduser() / "config.json").read_text(encoding="utf-8"))["hidden_size"]
        return cls(**settings)


def decision_plan(config, *, inspect_data: bool = False) -> dict:
    """Validate without importing Torch, loading weights or creating files."""
    cfg = config.decision
    if cfg.task_type == "typed":
        return typed_decision_plan(config, inspect_data=inspect_data)
    if type(cfg.enabled) is not bool or type(cfg.freeze_backbone) is not bool:
        raise ValueError("decision.enabled and freeze_backbone must be booleans")
    if cfg.task_type != "choice":
        raise ValueError("decision supports Choice only; Score and Noul training are not implemented")
    if cfg.method != "supervised":
        raise ValueError("decision.method must be supervised; RLCD is not implemented")
    if (cfg.query_pooling != "mean" or cfg.instance_question_in_query is not False or cfg.backbone_learning_rate is not None
            or cfg.head_learning_rate is not None or cfg.head_warmup_epochs != 0
            or cfg.early_stopping_patience != 0 or cfg.evaluate_test is not True):
        raise ValueError("typed training controls require decision.task_type: typed")
    if not cfg.freeze_backbone:
        raise ValueError("decision currently trains the head only; freeze_backbone must be true")
    if config.stages != ["decision"]:
        raise ValueError("decision currently requires stages: [decision]; it includes its own evaluation and model save")
    if config.distributed.strategy != "single" or config.distributed.num_gpus != 1 or config.distributed.num_nodes != 1:
        raise ValueError("decision supports single-process training only")
    if config.backends.training != "default" or config.model_backend != "hf" or config.model_blueprint:
        raise ValueError("decision requires the default runtime and an HF local text backbone")
    if cfg.encoder_device not in {"auto", "cpu", "cuda"} or cfg.encoder_dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("invalid decision.encoder_device or encoder_dtype")
    for name in ("epochs", "batch_size", "projection_rank", "cpu_threads", "encoder_batch_size", "max_length"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f"decision.{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or (value <= 0 if name == "learning_rate" else value < 0):
            raise ValueError(f"invalid decision.{name}")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("decision seed must be an integer in [0, 2**32)")
    backbone = cfg.backbone_name_or_path or config.model_name_or_path
    if cfg.enabled and (not isinstance(backbone, str) or not backbone.strip() or not isinstance(cfg.dataset_dir, str) or not cfg.dataset_dir.strip()):
        raise ValueError("decision requires dataset_dir and backbone_name_or_path (or model.name_or_path)")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
        raise ValueError("decision.output_dir must be a nonempty directory path")
    plan = {"status": "planned" if cfg.enabled else "skipped", "type": "Choice", "method": "supervised", "backbone": backbone, "dataset_dir": cfg.dataset_dir,
            "settings": asdict(cfg), "supported_types": ["Choice"], "rlcd_implemented": False,
            "split_roles": {"train": "head optimization", "validation": "checkpoint selection by NLL", "calibration": "temperature fitting", "test": "final evaluation only"},
            "backbone_bundled": False, "head_device": "cpu"}
    if inspect_data and cfg.enabled:
        import json

        from .dataset import read_decision_dataset

        if not (Path(backbone).expanduser() / "config.json").is_file():
            raise ValueError("decision backbone must be a local model directory with config.json")
        backbone_config = json.loads((Path(backbone).expanduser() / "config.json").read_text(encoding="utf-8"))
        width = backbone_config.get("hidden_size")
        if type(width) is not int or width < cfg.projection_rank:
            raise ValueError("decision.projection_rank exceeds backbone hidden_size or hidden_size is invalid")
        _, records, hashes = read_decision_dataset(cfg.dataset_dir)
        plan["counts"] = {name: len(rows) for name, rows in records.items()}
        plan["dataset_hashes"] = hashes
    return plan


def typed_decision_plan(config, *, inspect_data=False):
    from .TypedSchema import TypedDecisionConfig, read_typed_dataset

    cfg = config.decision
    if type(cfg.enabled) is not bool:
        raise ValueError("decision.enabled must be boolean")
    if config.stages != ["decision"]:
        raise ValueError("typed decision requires stages: [decision]")
    if config.distributed.strategy != "single" or config.distributed.num_gpus != 1 or config.distributed.num_nodes != 1:
        raise ValueError("typed decision supports single-process training only")
    if config.backends.training != "default" or config.model_blueprint:
        raise ValueError("typed decision requires the default training runtime and no LM blueprint")
    if not isinstance(cfg.dataset_dir, str) or not cfg.dataset_dir.strip() or not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
        raise ValueError("typed decision requires dataset_dir and output_dir")
    if cfg.encoder_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("invalid typed decision encoder_device")
    if cfg.encoder_dtype not in {"auto", "float32"}:
        raise ValueError("typed training currently uses float32 weights; encoder_dtype must be auto or float32")
    backbone = cfg.backbone_name_or_path or config.model_name_or_path
    model_config = TypedDecisionConfig(backbone=cfg.typed_backbone,
        backbone_name_or_path=backbone if cfg.typed_backbone == "hf" else None,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
        max_length=cfg.max_length, max_candidates=cfg.max_candidates, freeze_backbone=cfg.freeze_backbone,
        query_pooling=cfg.query_pooling, instance_question_in_query=cfg.instance_question_in_query)
    # Validate scalar settings without importing tensor dependencies.
    for name in ("epochs", "batch_size", "cpu_threads", "rl_samples"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f"decision.{name} must be positive integer")
    for name in ("learning_rate", "weight_decay", "brier_weight", "ordinal_weight", "rl_weight", "rl_std", "supervised_weight"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid decision.{name}")
    for name in ("backbone_learning_rate", "head_learning_rate"):
        value = getattr(cfg, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(f"decision.{name} must be positive and finite, or null")
    for name in ("head_warmup_epochs", "early_stopping_patience"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 0:
            raise ValueError(f"decision.{name} must be a nonnegative integer")
    if type(cfg.evaluate_test) is not bool:
        raise ValueError("decision.evaluate_test must be boolean")
    if cfg.method not in {"supervised", "reinforce_brier"} or cfg.learning_rate == 0 or cfg.rl_std == 0 or cfg.rl_samples < 2:
        raise ValueError("invalid typed training objective or sampling settings")
    if cfg.optimizer_device not in {"model", "cpu"}:
        raise ValueError("optimizer_device must be model or cpu")
    if cfg.method == "supervised" and cfg.supervised_weight == 0 or cfg.method == "reinforce_brier" and cfg.rl_weight == 0:
        raise ValueError("selected training objective must have a positive weight")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("decision seed must be integer in [0,2**32)")
    plan = {"status": "planned" if cfg.enabled else "skipped", "type": "Typed", "supported_types": ["Choice", "Score", "Noul"],
            "method": cfg.method, "backbone": backbone, "backbone_bundled": True,
            "rlcd_implemented": False, "model_config": model_config.to_dict(), "settings": asdict(cfg),
            "split_roles": {"train": "joint optimization", "validation": "checkpoint selection by NLL", "calibration": "scope temperatures", "test": "final evaluation only"}}
    if inspect_data and cfg.enabled:
        if cfg.typed_backbone == "hf":
            import json
            path = Path(backbone).expanduser()
            values = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if values.get("hidden_size") != cfg.hidden_size:
                raise ValueError("HF hidden_size differs from decision.hidden_size")
        if cfg.initial_model and not (Path(cfg.initial_model) / "decision_config.json").is_file():
            raise ValueError("initial_model must be a saved typed model")
        _, records, hashes = read_typed_dataset(cfg.dataset_dir)
        plan["counts"], plan["dataset_hashes"] = {k: len(v) for k, v in records.items()}, hashes
    return plan
