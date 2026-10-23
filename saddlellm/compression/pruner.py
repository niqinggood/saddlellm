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
from transformers.pytorch_utils import Conv1D

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
_PRUNING_MASKS_FILENAME = "pruning_masks.pt"

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
    "qkv_packed_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "gate_up_packed_proj",
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
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.pruning_method = self._normalize_method(pruning_method)
        self.pruning_ratio = self._validate_ratio(pruning_ratio)
        self.global_pruning = bool(global_pruning)
        self.device_map = device_map
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
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
        if self.model is not None and self.model_name:
            self._load_saved_masks(Path(self.model_name))

    def _mask_parameter(self, name: str):
        """Resolve a canonical mask name to its module and current parameter."""
        module_name, _, parameter_name = name.rpartition(".")
        if not parameter_name or parameter_name.endswith(("_orig", "_mask")):
            raise ValueError(f"pruning mask has a noncanonical parameter name: {name!r}")
        try:
            module = self.model.get_submodule(module_name) if module_name else self.model
        except AttributeError as exc:
            raise ValueError(f"pruning mask names an unknown parameter: {name!r}") from exc
        original_name = parameter_name + "_orig"
        parameter = module._parameters.get(original_name)
        if parameter is None:
            parameter = module._parameters.get(parameter_name)
        if parameter is None or not parameter.is_floating_point():
            raise ValueError(f"pruning mask needs a floating-point parameter: {name!r}")
        return module, parameter_name, parameter

    def _validated_masks(self, masks):
        """Validate every binding before applying any pruning hooks."""
        if not isinstance(masks, dict) or not masks:
            raise ValueError("pruning_masks.pt must contain a nonempty dictionary of masks")
        validated = {}
        for name, mask in masks.items():
            if (not isinstance(name, str) or not isinstance(mask, torch.Tensor)
                    or mask.layout != torch.strided or not torch.all((mask == 0) | (mask == 1))):
                raise ValueError("pruning masks must be named binary dense tensors")
            module, parameter_name, parameter = self._mask_parameter(name)
            if parameter.shape != mask.shape:
                raise ValueError(f"pruning mask shape does not match parameter {name!r}")
            validated[name] = mask.detach().to(device=parameter.device, dtype=torch.bool).clone()
            # Materialized checkpoints must already contain the advertised zeros.
            # Refusing inconsistent artifacts avoids silently deleting weights.
            if not hasattr(module, parameter_name + "_orig") and torch.any(parameter.detach()[~validated[name]] != 0):
                raise ValueError(f"pruning mask disagrees with materialized weights for {name!r}")
        return validated

    def _collect_masks(self):
        """Keep masks from earlier targets and every active pruning hook."""
        collected = dict(self.pruning_masks)
        for module_name, module in self.model.named_modules():
            for original_name, _ in module.named_parameters(recurse=False):
                if not original_name.endswith("_orig"):
                    continue
                parameter_name = original_name[:-5]
                mask = getattr(module, parameter_name + "_mask", None)
                if mask is None:
                    continue
                if mask.layout != torch.strided or not torch.all((mask == 0) | (mask == 1)):
                    raise ValueError("live pruning masks must be binary dense tensors")
                name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                if name in collected:
                    prior = collected[name].to(mask.device)
                    if prior.shape != mask.shape:
                        raise ValueError(f"existing pruning mask shape changed for {name!r}")
                    mask = mask.bool() & prior.bool()
                collected[name] = mask
        self.pruning_masks = self._validated_masks(collected) if collected else {}
        return self.pruning_masks

    def _restore_masks(self, masks):
        """Reattach removed masks and intersect them with existing live masks."""
        validated = self._validated_masks(masks)
        for name, mask in validated.items():
            module, parameter_name, _ = self._mask_parameter(name)
            live = getattr(module, parameter_name + "_mask", None)
            if live is not None:
                mask = mask & live.bool()
            if live is None or not torch.equal(live.bool(), mask):
                torch_prune.custom_from_mask(module, parameter_name, mask)
            validated[name] = mask.detach().clone()
        self.pruning_masks = validated

    def _load_saved_masks(self, directory: Path):
        manifest_path = directory / PRUNING_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if not isinstance(manifest, dict):
            raise TypeError("pruning manifest must be an object")
        declared_file = manifest.get("masks_file")
        if declared_file is not None and declared_file != _PRUNING_MASKS_FILENAME:
            raise ValueError(f"unsupported pruning masks_file: {declared_file!r}")
        masks_path = directory / _PRUNING_MASKS_FILENAME
        if declared_file and not masks_path.is_file():
            raise FileNotFoundError(f"pruning manifest declares missing masks: {masks_path}")
        if masks_path.is_file():
            saved = torch.load(masks_path, map_location="cpu", weights_only=True)
            expected = manifest.get("masked_parameters")
            if expected is not None:
                if (not isinstance(expected, list) or not expected
                        or any(not isinstance(name, str) for name in expected)
                        or len(expected) != len(set(expected))):
                    raise ValueError("pruning manifest masked_parameters must be distinct parameter names")
                if not isinstance(saved, dict) or set(saved) != set(expected):
                    raise ValueError("pruning sidecar does not match manifest masked_parameters")
            self._restore_masks(saved)

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
        cpu_target = str(self.device_map).lower() == "cpu"
        dtype = torch.float32 if cpu_target or not torch.cuda.is_available() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
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
            if isinstance(module, (torch.nn.Linear, Conv1D)) and any(
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
        if model is not None and model is not self.model:
            self.model = model
            self.pruning_masks = {}
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
                "no torch.nn.Linear modules matched target_modules "
                f"(or compatible Conv1D projections): {patterns}"
            )

        # A previous remove_masks() only materializes weights. Reattach its
        # exact mask before another pruning pass so random/structured pruning
        # cannot forget old targets or previously removed positions.
        self._collect_masks()
        if self.pruning_masks:
            self._restore_masks(self.pruning_masks)

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

        self._collect_masks()
        return self.model

    def remove_masks(self):
        """Permanently materialize current masks into the model weights."""
        if self.model is None:
            raise ValueError("no pruned model is available")
        self._collect_masks()
        for name in self.pruning_masks:
            module, parameter_name, _ = self._mask_parameter(name)
            if hasattr(module, f"{parameter_name}_orig"):
                torch_prune.remove(module, parameter_name)
        return self.model

    def _materialized_state_dict(self) -> Dict[str, torch.Tensor]:
        """Build a portable state dict without mutating active pruning hooks."""
        self._collect_masks()
        state_dict = dict(self.model.state_dict())
        for name, mask in self.pruning_masks.items():
            _, _, parameter = self._mask_parameter(name)
            state_dict.pop(name + "_orig", None)
            state_dict.pop(name + "_mask", None)
            # module.weight is refreshed only by the pruning forward hook and
            # can be stale immediately after optimizer.step(). Use current orig.
            state_dict[name] = parameter.detach() * mask
        return state_dict

    def save(self, path: str, remove_masks: bool = False):
        """Save a loadable checkpoint while optionally retaining live masks."""
        if self.model is None:
            raise ValueError("no pruned model is available to save")
        self._collect_masks()
        state_dict = self._materialized_state_dict()
        output_dir = Path(path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("pruned model path must be a directory")
        output_dir.mkdir(parents=True, exist_ok=True)

        if remove_masks:
            self.remove_masks()
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict
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
        masks_path = output_dir / _PRUNING_MASKS_FILENAME
        if self.pruning_masks:
            torch.save({name: mask.cpu() for name, mask in self.pruning_masks.items()}, masks_path)
            manifest["masks_file"] = _PRUNING_MASKS_FILENAME
            manifest["masked_parameters"] = sorted(self.pruning_masks)
        elif masks_path.is_file():
            # Reusing an output directory must not retain an unrelated sidecar.
            masks_path.unlink()
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
            if not isinstance(manifest, dict):
                raise TypeError("pruning manifest must be an object")
            kwargs.setdefault(
                "pruning_method", manifest.get("method", "l1_unstructured")
            )
            kwargs.setdefault("pruning_ratio", manifest.get("pruning_ratio", 0.3))
            kwargs.setdefault("global_pruning", manifest.get("global_pruning", True))
            kwargs.setdefault("target_modules", manifest.get("target_modules"))
        return cls(model_name=path, **kwargs)
