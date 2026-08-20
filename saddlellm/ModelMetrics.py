"""Model-structure metrics for SaddleLLM experiments."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelMetricSnapshot:
    step: int
    architecture: Dict
    router: Dict = field(default_factory=dict)
    loss_items: Dict = field(default_factory=dict)
    cache: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelMetricsExtractor:
    """Extract architecture and latest runtime metrics from modular models."""

    @staticmethod
    def architecture(model) -> Dict:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return {"model_type": type(model).__name__}
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(getattr(cfg, "__dict__", {}))
        keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "attention_kind",
            "attention_backend",
            "mla_cache_mode",
            "ffn_kind",
            "num_experts",
            "num_experts_per_tok",
            "shared_expert",
            "aux_loss_free",
            "multi_token_prediction",
            "mtp_extra_tokens",
            "residual_topology",
            "residual_initialization",
        ]
        result = {key: data.get(key) for key in keys if key in data}
        layer_configs = data.get("layer_configs") or []
        if layer_configs:
            result["layers"] = [
                {
                    "name": layer.get("name", f"layer-{index}"),
                    "attention_kind": (layer.get("attention") or {}).get("kind"),
                    "attention_backend": (layer.get("attention") or {}).get("backend"),
                    "ffn_kind": (layer.get("ffn") or {}).get("kind"),
                    "residual_topology": (layer.get("residual") or {}).get("topology"),
                    "residual_initialization": (layer.get("residual") or {}).get("initialization"),
                }
                for index, layer in enumerate(layer_configs)
            ]
        return result

    @staticmethod
    def runtime(model) -> Dict:
        metrics = {}
        if hasattr(model, "router_metrics"):
            metrics["router"] = model.router_metrics()
        metrics["cache"] = ModelMetricsExtractor.cache_profile(model)
        return metrics

    @staticmethod
    def cache_profile(model) -> Dict:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return {}
        dtype_bytes = 2
        layer_configs = getattr(cfg, "layer_configs", None) or []
        resolved = (
            [cfg.for_layer(index) for index in range(len(layer_configs))]
            if layer_configs and hasattr(cfg, "for_layer")
            else [cfg] * getattr(cfg, "num_hidden_layers", 0)
        )
        per_layer = []
        for layer_cfg in resolved:
            hidden = getattr(layer_cfg, "hidden_size", 0)
            heads = getattr(layer_cfg, "num_attention_heads", 1)
            kv_heads = getattr(layer_cfg, "num_key_value_heads", heads)
            head_dim = getattr(layer_cfg, "head_dim", None) or hidden // max(1, heads)
            is_latent = (
                getattr(layer_cfg, "attention_kind", "gqa") == "mla"
                and getattr(layer_cfg, "mla_cache_mode", "kv") == "latent"
                and getattr(layer_cfg, "kv_lora_rank", 0) > 0
            )
            per_layer.append(
                getattr(layer_cfg, "kv_lora_rank", 0) * dtype_bytes
                if is_latent
                else 2 * kv_heads * head_dim * dtype_bytes
            )
        modes = {
            "mla_latent"
            if getattr(layer_cfg, "attention_kind", "gqa") == "mla"
            and getattr(layer_cfg, "mla_cache_mode", "kv") == "latent"
            else "kv"
            for layer_cfg in resolved
        }
        total = sum(per_layer)
        return {
            "mode": next(iter(modes)) if len(modes) == 1 else "mixed",
            "bytes_per_token_per_layer": (
                per_layer[0] if len(set(per_layer)) == 1 and per_layer else per_layer
            ),
            "bytes_per_token_by_layer": per_layer,
            "bytes_per_token_total": total,
        }

    @staticmethod
    def snapshot(model, step: int, logs: Optional[Dict] = None) -> ModelMetricSnapshot:
        runtime = ModelMetricsExtractor.runtime(model)
        loss_items = {}
        if logs:
            loss_items = {
                key: value for key, value in logs.items()
                if key.startswith("loss") or key.endswith("_loss") or key.startswith("mtp_")
            }
        return ModelMetricSnapshot(
            step=step,
            architecture=ModelMetricsExtractor.architecture(model),
            router=runtime.get("router", {}),
            cache=runtime.get("cache", {}),
            loss_items=loss_items,
        )


try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    TrainerCallback = object


class ModelMetricsCallback(TrainerCallback):
    """Trainer callback that writes model architecture/runtime metrics."""

    def __init__(self, output_dir: str = "./model_metrics", log_every: int = 1):
        self.output_dir = output_dir
        self.log_every = max(1, int(log_every))
        os.makedirs(output_dir, exist_ok=True)
        self.metrics_path = os.path.join(output_dir, "model_metrics.jsonl")
        self.summary_path = os.path.join(output_dir, "model_metrics_summary.json")
        self.latest: Optional[ModelMetricSnapshot] = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and getattr(state, "is_world_process_zero", True):
            self._record(ModelMetricsExtractor.snapshot(model, int(getattr(state, "global_step", 0) or 0)))
        return control

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if (
            model is None
            or not getattr(state, "is_world_process_zero", True)
            or step % self.log_every != 0
        ):
            return control
        self._record(ModelMetricsExtractor.snapshot(model, step, logs=logs or {}))
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None and getattr(state, "is_world_process_zero", True):
            self._record(ModelMetricsExtractor.snapshot(model, int(getattr(state, "global_step", 0) or 0)))
        return control

    def _record(self, snapshot: ModelMetricSnapshot):
        self.latest = snapshot
        data = snapshot.to_dict()
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
