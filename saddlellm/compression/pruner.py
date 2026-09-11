"""Safe, in-memory model-pruning helpers.

The wrapper intentionally implements only pruning operations that PyTorch can
apply generically to ``torch.nn.Linear`` modules. Attention-head and whole-layer
pruning depend on each model architecture and therefore fail explicitly instead
of pretending to produce a valid checkpoint.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.utils.prune as torch_prune
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from datasets import Dataset


PruningMethod = Literal[
    "l1_unstructured",
    "random_unstructured",
    "ln_structured",
    "head_structured",
    "layer_structured",
    "unstructured",
    "structured",
]

PRUNING_MANIFEST_FILENAME = "pruning_manifest.json"

__all__ = ["PRUNING_MANIFEST_FILENAME", "ModelPruner", "PruningMethod"]
SUPPORTED_PRUNING_METHODS = {
    "l1_unstructured",
    "random_unstructured",
    "ln_structured",
    "head_structured",
    "layer_structured",
}
PRUNING_METHOD_ALIASES = {
    "unstructured": "l1_unstructured",
    "structured": "ln_structured",
}
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "dense",
    "fc",
    "c_attn",
    "c_proj",
)


class ModelPruner:
    """Apply structured or unstructured pruning to a loaded model.

    Pass an in-memory model for composition with :class:`Lightweight`, or pass
    ``model_name`` to load a Hugging Face model explicitly. Constructing the
    class without either value never triggers an implicit model download.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pruning_method: PruningMethod = "l1_unstructured",
        pruning_ratio: float = 0.3,
        global_pruning: bool = True,
        device_map: str = "auto",
        *,
        model=None,
        tokenizer=None,
        target_modules: Optional[Sequence[str]] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.pruning_method = self._normalize_method(pruning_method)
        self.pruning_ratio = self._validate_ratio(pruning_ratio)
        self.global_pruning = bool(global_pruning)
        self.device_map = device_map
        self.trust_remote_code = bool(trust_remote_code)
        self.model = model
        self.tokenizer = tokenizer
        self.target_modules = tuple(
            DEFAULT_TARGET_MODULES if target_modules is None else target_modules
        )
        if not self.target_modules:
            raise ValueError(
                "target_modules must contain at least one module-name pattern"
            )

        self.pruning_masks: Dict[str, torch.Tensor] = {}
        self.parameter_importance: Dict[str, float] = defaultdict(float)

        if self.model is None and self.model_name:
            self._load_base_model()

    @staticmethod
    def _normalize_method(method: str) -> str:
        resolved = PRUNING_METHOD_ALIASES.get(method, method)
        if resolved not in SUPPORTED_PRUNING_METHODS:
            choices = ", ".join(
                sorted(SUPPORTED_PRUNING_METHODS | set(PRUNING_METHOD_ALIASES))
            )
            raise ValueError(
                f"unsupported pruning method {method!r}; choose one of: {choices}"
            )
        return resolved

    @staticmethod
    def _validate_ratio(ratio: float) -> float:
        try:
            resolved = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pruning_ratio must be a number in the range [0, 1)"
            ) from exc
        if not 0 <= resolved < 1:
            raise ValueError("pruning_ratio must be in the range [0, 1)")
        return resolved

    def _load_base_model(self):
        if not self.model_name:
            raise ValueError("model_name or an in-memory model is required")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model

    def _get_pruning_parameters(self) -> List[Tuple[str, torch.nn.Module, str]]:
        """Return uniquely named linear weights selected for pruning."""
        if self.model is None:
            raise ValueError(
                "model_name or an in-memory model is required before pruning"
            )

        parameters = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                pattern in name for pattern in self.target_modules
            ):
                parameters.append((name, module, "weight"))
        return parameters

    def fit(
        self,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        epochs: int = 1,
        batch_size: int = 2,
    ):
        """Collect mean absolute gradients as optional importance metadata."""
        if train_dataset is None:
            return self
        if self.model is None or self.tokenizer is None:
            raise ValueError("fit requires both a model and tokenizer")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Retained for API compatibility. Importance collection itself does not
        # evaluate a model and therefore does not consume eval_dataset.
        _ = eval_dataset

        def collate_fn(examples):
            try:
                texts = [example["text"] for example in examples]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "each pruning sample must contain a 'text' field"
                ) from exc
            return self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        try:
            input_device = self.model.device
        except AttributeError:
            input_device = next(self.model.parameters()).device

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        for _ in range(epochs):
            for inputs in loader:
                if hasattr(inputs, "to"):
                    inputs = inputs.to(input_device)
                else:
                    inputs = {
                        key: value.to(input_device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                outputs.loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None:
                        self.parameter_importance[name] = (
                            parameter.grad.abs().mean().item()
                        )
        return self

    def prune(
        self,
        model=None,
        *,
        sparsity: Optional[float] = None,
        method: Optional[PruningMethod] = None,
    ):
        """Apply pruning and return the resulting in-memory model."""
        if model is not None:
            self.model = model
        if sparsity is not None:
            self.pruning_ratio = self._validate_ratio(sparsity)
        if method is not None:
            self.pruning_method = self._normalize_method(method)
        if self.model is None:
            raise ValueError(
                "model_name or an in-memory model is required before pruning"
            )

        if self.pruning_method in {"head_structured", "layer_structured"}:
            raise NotImplementedError(
                f"{self.pruning_method} pruning is architecture-specific; "
                "use l1_unstructured, random_unstructured, or ln_structured"
            )

        parameters = self._get_pruning_parameters()
        if not parameters:
            patterns = ", ".join(self.target_modules)
            raise ValueError(
                f"no torch.nn.Linear modules matched target_modules: {patterns}"
            )

        module_parameters = [
            (module, parameter_name) for _, module, parameter_name in parameters
        ]
        if self.pruning_method == "ln_structured":
            for _, module, parameter_name in parameters:
                torch_prune.ln_structured(
                    module,
                    name=parameter_name,
                    amount=self.pruning_ratio,
                    n=2,
                    dim=0,
                )
        elif self.global_pruning:
            pruning_method = (
                torch_prune.L1Unstructured
                if self.pruning_method == "l1_unstructured"
                else torch_prune.RandomUnstructured
            )
            torch_prune.global_unstructured(
                module_parameters,
                pruning_method=pruning_method,
                amount=self.pruning_ratio,
            )
        else:
            pruning_function = (
                torch_prune.l1_unstructured
                if self.pruning_method == "l1_unstructured"
                else torch_prune.random_unstructured
            )
            for _, module, parameter_name in parameters:
                pruning_function(module, name=parameter_name, amount=self.pruning_ratio)

        self.pruning_masks = {
            f"{name}.{parameter_name}": getattr(module, f"{parameter_name}_mask")
            .detach()
            .clone()
            for name, module, parameter_name in parameters
        }
        return self.model

    def remove_masks(self):
        """Permanently materialize current masks into the model weights."""
        if self.model is None:
            raise ValueError("no pruned model is available")
        for _, module, parameter_name in self._get_pruning_parameters():
            if hasattr(module, f"{parameter_name}_orig"):
                torch_prune.remove(module, parameter_name)
        return self.model

    def _materialized_state_dict(self) -> Dict[str, torch.Tensor]:
        """Build a portable state dict without mutating active pruning hooks."""
        state_dict = dict(self.model.state_dict())
        for name, module, parameter_name in self._get_pruning_parameters():
            original_key = (
                f"{name}.{parameter_name}_orig" if name else f"{parameter_name}_orig"
            )
            mask_key = (
                f"{name}.{parameter_name}_mask" if name else f"{parameter_name}_mask"
            )
            weight_key = f"{name}.{parameter_name}" if name else parameter_name
            if original_key in state_dict:
                state_dict.pop(original_key, None)
                state_dict.pop(mask_key, None)
                state_dict[weight_key] = (
                    getattr(module, parameter_name).detach().clone()
                )
        return state_dict

    def save(self, path: str, remove_masks: bool = False):
        """Save a loadable checkpoint while optionally retaining live masks."""
        if self.model is None:
            raise ValueError("no pruned model is available to save")
        output_dir = Path(path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("pruned model path must be a directory")
        output_dir.mkdir(parents=True, exist_ok=True)

        if remove_masks:
            self.remove_masks()
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(
                output_dir, state_dict=self._materialized_state_dict()
            )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        manifest = {
            "format_version": 1,
            "method": self.pruning_method,
            "pruning_ratio": self.pruning_ratio,
            "global_pruning": self.global_pruning,
            "target_modules": list(self.target_modules),
            "masks_materialized": True,
        }
        (output_dir / PRUNING_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a checkpoint previously saved by :meth:`save`."""
        manifest_path = Path(path) / PRUNING_MANIFEST_FILENAME
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            kwargs.setdefault(
                "pruning_method", manifest.get("method", "l1_unstructured")
            )
            kwargs.setdefault("pruning_ratio", manifest.get("pruning_ratio", 0.3))
            kwargs.setdefault("global_pruning", manifest.get("global_pruning", True))
            kwargs.setdefault("target_modules", manifest.get("target_modules"))
        return cls(model_name=path, **kwargs)
