"""Frozen text embeddings plus a trainable, dynamic-candidate Choice head.

No generation or vocabulary projection is used. The low-rank residual starts
at zero, so an untrained head preserves the embedding cosine ranking. This is
a decision research baseline, not an implementation of Jev or RLCD.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .Schema import DecisionModelConfig, json_object


_CONFIG_FILE = "decision_config.json"
_HEAD_FILE = "decision_head.safetensors"
_CALIBRATION_FILE = "calibration.json"
_MODEL_TYPE = "saddle_choice_decision"


def last_valid_token_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last unmasked token, including left/right/interior padding."""
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden_states and attention_mask must have shapes [B,L,D] and [B,L]")
    if hidden_states.shape[:2] != attention_mask.shape or hidden_states.shape[1] == 0:
        raise ValueError("hidden_states and attention_mask lengths must match and be non-empty")
    if hidden_states.device != attention_mask.device:
        raise ValueError("hidden_states and attention_mask must be on the same device")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask must contain only zeros and ones")
    valid = attention_mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every encoded row must contain at least one valid token")
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    last_positions = positions.masked_fill(~valid, -1).amax(dim=1)
    return hidden_states[torch.arange(valid.shape[0], device=valid.device), last_positions]


class FrozenTextEncoder:
    """Local-only HF AutoModel adapter; intentionally separate from head state.

    A caller may inject a tokenizer and an AutoModel-like module for offline
    tests. Production loading never enables remote code or silently truncates.
    """

    def __init__(
        self,
        config: DecisionModelConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = "auto",
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if (model is None) != (tokenizer is None):
            raise ValueError("inject both model and tokenizer, or neither")
        if model is None:
            from transformers import AutoModel, AutoTokenizer

            kwargs = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if config.model_revision is not None:
                kwargs["revision"] = config.model_revision
            tokenizer = AutoTokenizer.from_pretrained(
                config.backbone_name_or_path, padding_side="left", **kwargs
            )
            if isinstance(dtype, str) and dtype != "auto":
                aliases = {"fp32": torch.float32, "float32": torch.float32,
                           "fp16": torch.float16, "float16": torch.float16,
                           "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
                if dtype not in aliases:
                    raise ValueError("unsupported encoder dtype")
                dtype = aliases[dtype]
            model = AutoModel.from_pretrained(
                config.backbone_name_or_path, torch_dtype=dtype, **kwargs
            )
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        actual_dim = getattr(getattr(model, "config", None), "hidden_size", None)
        if actual_dim is not None and int(actual_dim) != config.embedding_dim:
            raise ValueError("backbone hidden_size differs from configured embedding_dim")

    def encode(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            raise ValueError("texts must be non-empty")
        self.model.eval()
        # Validate the entire call before running any partial batches. In
        # particular, do not silently drop the end of state or criteria text.
        tokenized = self.tokenizer(
            list(texts), padding=False, truncation=False, add_special_tokens=True
        )
        ids = tokenized["input_ids"]
        if any(len(row) == 0 or len(row) > self.config.max_length for row in ids):
            raise ValueError(f"encoded input exceeds max_length={self.config.max_length} or is empty; no truncation was performed")
        masks = tokenized.get("attention_mask", [[1] * len(row) for row in ids])
        vectors = []
        for start in range(0, len(ids), self.config.encoder_batch_size):
            rows = {"input_ids": ids[start : start + self.config.encoder_batch_size],
                    "attention_mask": masks[start : start + self.config.encoder_batch_size]}
            batch = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
            batch = {name: value.to(self.device) for name, value in batch.items()
                     if name in ("input_ids", "attention_mask")}
            with torch.no_grad():
                result = self.model(**batch, use_cache=False, return_dict=True)
                pooled = last_valid_token_pool(result.last_hidden_state, batch["attention_mask"])
                vectors.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(vectors, dim=0)


class _ChoiceHead(nn.Module):
    def __init__(self, config: DecisionModelConfig) -> None:
        super().__init__()
        self.down = nn.Linear(config.embedding_dim, config.projection_rank, bias=False)
        self.up = nn.Linear(config.projection_rank, config.embedding_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.log_scale = nn.Parameter(torch.tensor(math.log(config.initial_scale), dtype=torch.float32))


class ChoiceDecisionModel(nn.Module):
    """A trainable head over frozen state/question and candidate embeddings.

    ``score`` and ``forward`` accept cached CPU/GPU tensors and do not load the
    encoder. ``encode_*`` return detached normalized float32 CPU features by
    default. Encoder placement is independent of ``model.to(head_device)``.
    """

    def __init__(
        self,
        config: DecisionModelConfig,
        *,
        encoder: Any = None,
        load_encoder: bool = False,
        encoder_device: str | torch.device = "cpu",
        encoder_dtype: str | torch.dtype = "auto",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, DecisionModelConfig):
            raise TypeError("config must be DecisionModelConfig")
        self.config = config
        self.head = _ChoiceHead(config)
        self.metadata = json_object(metadata, name="metadata")
        self._calibration = {"temperature": 1.0, "calibrated": False, "metadata": {}}
        self.encoder_device = torch.device(encoder_device)
        self.encoder_dtype = encoder_dtype
        # Bypass Module registration: moving/saving/optimizing the head must
        # never move or serialize a possibly much larger frozen encoder.
        object.__setattr__(self, "_encoder", encoder)
        if isinstance(encoder, nn.Module):
            encoder.requires_grad_(False)
            encoder.eval()
        if load_encoder:
            self._get_encoder()

    @property
    def encoder_loaded(self) -> bool:
        return self._encoder is not None

    def _get_encoder(self):
        if self._encoder is None:
            encoder = FrozenTextEncoder(
                self.config, device=self.encoder_device, dtype=self.encoder_dtype
            )
            object.__setattr__(self, "_encoder", encoder)
        return self._encoder

    def head_parameters(self):
        return self.head.parameters()

    @property
    def calibration_temperature(self) -> float:
        return self._calibration["temperature"]

    @property
    def calibration(self) -> dict:
        return json_object(self._calibration, name="calibration")

    def set_calibration(self, temperature: float, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._validate_temperature(temperature)
        self._calibration = {
            "temperature": float(temperature),
            "calibrated": True,
            "metadata": json_object(metadata, name="calibration metadata"),
        }

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)):
            raise ValueError("calibration temperature must be numeric")
        if not math.isfinite(temperature) or not 1e-6 <= temperature <= 1e6:
            raise ValueError("calibration temperature must be finite and between 1e-6 and 1e6")

    def _texts(self, values: str | Sequence[str], *, name: str) -> list[str]:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a non-empty string or sequence of strings")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{name} must contain non-blank strings")
        if any(len(value) > self.config.max_text_chars for value in values):
            raise ValueError(f"{name} exceeds max_text_chars={self.config.max_text_chars}")
        return list(values)

    def _encode(self, texts: Sequence[str], *, device: str | torch.device) -> Tensor:
        if any(len(text) > self.config.max_text_chars for text in texts):
            raise ValueError("formatted decision input exceeds max_text_chars")
        encoder = self._get_encoder()
        if isinstance(encoder, nn.Module):
            encoder.eval()
        with torch.no_grad():
            result = torch.as_tensor(encoder.encode(list(texts))).detach().to(device=device, dtype=torch.float32)
        if result.shape != (len(texts), self.config.embedding_dim):
            raise ValueError("encoder features do not match [rows, embedding_dim]")
        self._check_vectors(result, name="encoder features")
        return F.normalize(result, dim=-1)

    def encode_states(
        self, states: str | Sequence[str], questions: str | Sequence[str],
        *, device: str | torch.device = "cpu",
    ) -> Tensor:
        states = self._texts(states, name="states")
        if isinstance(questions, str):
            questions = [questions] * len(states)
        questions = self._texts(questions, name="questions")
        if len(states) != len(questions):
            raise ValueError("states and questions must have equal lengths")
        texts = [f"Instruct: {question}\nQuery: {state}" for state, question in zip(states, questions)]
        return self._encode(texts, device=device)

    def encode_choices(
        self, descriptions: Sequence[str], question: str,
        *, device: str | torch.device = "cpu",
    ) -> Tensor:
        if isinstance(descriptions, str):
            raise ValueError("descriptions must be a sequence of candidate descriptions")
        descriptions = self._texts(descriptions, name="descriptions")
        if not 2 <= len(descriptions) <= self.config.max_candidates:
            raise ValueError(f"Choice requires 2..{self.config.max_candidates} candidates")
        question = self._texts(question, name="question")
        if len(question) != 1:
            raise ValueError("encode_choices accepts one question per candidate set")
        # Include the question in the descriptor representation so identical
        # words can have different meanings for different decision criteria.
        texts = [f"Question: {question[0]}\nOption: {description}" for description in descriptions]
        return self._encode(texts, device=device)

    @staticmethod
    def _check_vectors(features: Tensor, *, name: str, valid: Optional[Tensor] = None) -> None:
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"{name} must contain finite values")
        norms = torch.linalg.vector_norm(features.float(), dim=-1)
        if valid is not None:
            norms = norms[valid]
        if bool((norms <= 1e-12).any()):
            raise ValueError(f"{name} must contain non-zero valid vectors")

    def score(
        self, state_features: Tensor, candidate_features: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return uncalibrated float32 logits [B,K], with invalid slots -inf."""
        if not isinstance(state_features, Tensor) or not isinstance(candidate_features, Tensor):
            raise TypeError("state_features and candidate_features must be torch tensors")
        if state_features.ndim != 2 or state_features.shape[0] == 0:
            raise ValueError("state_features must have non-empty shape [B,D]")
        if candidate_features.ndim not in (2, 3):
            raise ValueError("candidate_features must have shape [K,D] or [B,K,D]")
        if state_features.shape[-1] != self.config.embedding_dim or candidate_features.shape[-1] != self.config.embedding_dim:
            raise ValueError("feature dimension differs from configured embedding_dim")
        batch, count = state_features.shape[0], candidate_features.shape[-2]
        if not 2 <= count <= self.config.max_candidates:
            raise ValueError(f"Choice requires 2..{self.config.max_candidates} candidate slots")
        if candidate_features.ndim == 3 and candidate_features.shape[0] != batch:
            raise ValueError("per-row candidates must have the same batch size as states")
        device, dtype = self.head.down.weight.device, self.head.down.weight.dtype
        states = state_features.to(device=device, dtype=dtype)
        candidates = candidate_features.to(device=device, dtype=dtype)
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(batch, -1, -1)
        if candidate_mask is None:
            mask = torch.ones((batch, count), dtype=torch.bool, device=device)
        else:
            if not isinstance(candidate_mask, Tensor) or candidate_mask.dtype != torch.bool:
                raise ValueError("candidate_mask must be a boolean tensor")
            if candidate_mask.shape == (count,):
                candidate_mask = candidate_mask.unsqueeze(0).expand(batch, -1)
            if candidate_mask.shape != (batch, count):
                raise ValueError("candidate_mask must have shape [K] or [B,K]")
            mask = candidate_mask.to(device)
        if bool((mask.sum(dim=-1) < 2).any()):
            raise ValueError("every Choice row must have at least two valid candidates")
        self._check_vectors(states, name="state_features")
        self._check_vectors(candidates, name="candidate_features", valid=mask)
        states = F.normalize(states, dim=-1)
        candidates = F.normalize(candidates, dim=-1)
        projected = F.normalize(states + self.head.up(self.head.down(states)), dim=-1)
        similarity = torch.einsum("bd,bkd->bk", projected, candidates).float()
        scale = self.head.log_scale.float().clamp(math.log(1e-3), math.log(1e3)).exp()
        logits = similarity * scale
        if not bool(torch.isfinite(logits[mask]).all()):
            raise ValueError("decision head produced non-finite logits")
        return logits.masked_fill(~mask, -torch.inf)

    def forward(self, state_features: Tensor, candidate_features: Tensor,
                candidate_mask: Optional[Tensor] = None) -> Tensor:
        return self.score(state_features, candidate_features, candidate_mask)

    def predict_proba(self, state_features: Tensor, candidate_features: Tensor,
                      candidate_mask: Optional[Tensor] = None) -> Tensor:
        return torch.softmax(self.score(state_features, candidate_features, candidate_mask) / self.calibration_temperature, dim=-1)

    def save_pretrained(self, directory: str | os.PathLike, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Persist only the head and JSON metadata; never pickle executable objects."""
        from safetensors.torch import save_file

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        saved_metadata = self.metadata if metadata is None else json_object(metadata, name="metadata")
        state = {name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()}
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("cannot save non-finite head weights")
        with tempfile.TemporaryDirectory(prefix=".decision-save-", dir=destination) as work:
            work = Path(work)
            save_file(state, str(work / _HEAD_FILE))
            self._write_json(work / _CALIBRATION_FILE, self.calibration)
            payload = {
                "schema_version": 1, "model_type": _MODEL_TYPE,
                "config": self.config.to_dict(), "metadata": saved_metadata,
                "backbone_bundled": False,
                "head_file": _HEAD_FILE, "head_sha256": self._sha256(work / _HEAD_FILE),
                "calibration_file": _CALIBRATION_FILE,
                "calibration_sha256": self._sha256(work / _CALIBRATION_FILE),
            }
            self._write_json(work / _CONFIG_FILE, payload)
            # The config is the final commit record. A partial replacement
            # cannot silently load weights/calibration from a different save.
            for name in (_HEAD_FILE, _CALIBRATION_FILE, _CONFIG_FILE):
                os.replace(work / name, destination / name)

    @classmethod
    def from_pretrained(
        cls, directory: str | os.PathLike, *, encoder: Any = None,
        load_encoder: bool = False, device: str | torch.device = "cpu",
        encoder_device: str | torch.device = "cpu", encoder_dtype: str | torch.dtype = "auto",
        expected_backbone_identity: Optional[Mapping[str, Any]] = None,
        backbone_name_or_path: Optional[str] = None,
    ) -> "ChoiceDecisionModel":
        from safetensors.torch import load_file

        source = Path(directory)
        payload = cls._read_json(source / _CONFIG_FILE)
        if payload.get("schema_version") != 1 or payload.get("model_type") != _MODEL_TYPE:
            raise ValueError("unsupported decision artifact schema/model_type")
        if payload.get("backbone_bundled") is not False:
            raise ValueError("this decision loader expects an external frozen backbone")
        for name, hash_key, file_key in ((_HEAD_FILE, "head_sha256", "head_file"),
                                         (_CALIBRATION_FILE, "calibration_sha256", "calibration_file")):
            if payload.get(file_key) != name or payload.get(hash_key) != cls._sha256(source / name):
                raise ValueError(f"decision artifact integrity mismatch: {name}")
        config = DecisionModelConfig.from_dict(payload["config"])
        if expected_backbone_identity is not None:
            expected = json_object(expected_backbone_identity, name="expected_backbone_identity")
            if config.backbone_identity != expected:
                raise ValueError("backbone identity does not match the requested identity")
        if backbone_name_or_path is not None:
            config = replace(config, backbone_name_or_path=backbone_name_or_path)
        model = cls(config, encoder=encoder, load_encoder=False,
                    encoder_device=encoder_device, encoder_dtype=encoder_dtype,
                    metadata=payload.get("metadata"))
        weights = load_file(str(source / _HEAD_FILE), device="cpu")
        if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
            raise ValueError("decision artifact contains non-finite head weights")
        model.head.load_state_dict(weights, strict=True)
        calibration = cls._read_json(source / _CALIBRATION_FILE)
        cls._validate_temperature(calibration.get("temperature"))
        if not isinstance(calibration.get("calibrated"), bool):
            raise ValueError("calibration metadata must explicitly state calibrated true/false")
        if not calibration["calibrated"] and calibration["temperature"] != 1.0:
            raise ValueError("uncalibrated artifacts must use temperature 1.0")
        model._calibration = {
            "temperature": float(calibration["temperature"]),
            "calibrated": calibration["calibrated"],
            "metadata": json_object(calibration.get("metadata"), name="calibration metadata"),
        }
        model.to(device).eval()
        if load_encoder:
            model._get_encoder()
        return model

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"),
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")))
        return json_object(payload, name=path.name)


__all__ = ["ChoiceDecisionModel", "DecisionModelConfig", "FrozenTextEncoder", "last_valid_token_pool"]
