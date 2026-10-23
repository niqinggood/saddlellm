"""Discriminative typed decisions with shared state encoding and isolated queries.

The small native byte Transformer is an offline research backbone. Local HF
backbones can be jointly trained. This is our architecture, not Jev's network.
"""

# ruff: noqa: N999
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import torch
from torch import nn

from .dataset import sha256
from .TypedSchema import (
    TYPES,
    TypedDecisionConfig,
    candidate_query_text,
    candidates,
    scope_key,
    validate_record,
)


class _NativeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(258, config.hidden_size, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_heads,
            config.hidden_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )

    def forward(self, ids, mask):
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.layers(
            self.embedding(ids) + self.position(positions), src_key_padding_mask=~mask
        )


class TypedDecisionModel(nn.Module):
    def __init__(self, config: TypedDecisionConfig, *, restore_backbone=None):
        super().__init__()
        self.config = config
        self.tokenizer = None
        if config.backbone == "native":
            self.backbone = _NativeEncoder(config)
        else:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            source = restore_backbone or config.backbone_name_or_path
            kwargs = {"local_files_only": True, "trust_remote_code": False}
            tokenizer_kwargs = dict(kwargs)
            if restore_backbone:
                saved_config = json.loads(
                    (Path(source) / "config.json").read_text(encoding="utf-8")
                )
                if saved_config.get("model_type") in {"qwen2", "qwen3"}:
                    # Some 4.57.x builds misidentify locally saved non-Mistral
                    # configs as Mistral. Preserve Qwen's trained regex rather
                    # than replacing it with a different model's tokenization.
                    tokenizer_kwargs["fix_mistral_regex"] = False
            self.tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("HF tokenizer requires a padding or EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backbone = (
                AutoModel.from_config(
                    AutoConfig.from_pretrained(source, **kwargs),
                    trust_remote_code=False,
                )
                if restore_backbone
                else AutoModel.from_pretrained(source, **kwargs)
            )
            if getattr(self.backbone.config, "hidden_size", None) != config.hidden_size:
                raise ValueError("HF hidden_size differs from typed model hidden_size")
        self.backbone.requires_grad_(not config.freeze_backbone)
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, batch_first=True, dropout=0.0
        )
        self.heads = nn.ModuleDict(
            {
                kind: nn.Sequential(
                    nn.Linear(config.hidden_size * 3, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, 1),
                )
                for kind in TYPES
            }
        )
        self.calibration = {}
        self.metadata = {}

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def _tokens(self, texts):
        device = next(self.parameters()).device
        if self.tokenizer is None:
            rows = [
                [257] + [byte + 1 for byte in text.encode("utf-8")] for text in texts
            ]
        else:
            rows = self.tokenizer(
                texts, truncation=False, padding=False, add_special_tokens=True
            )["input_ids"]
        if any(not row or len(row) > self.config.max_length for row in rows):
            raise ValueError(
                f"input exceeds max_length={self.config.max_length}; no truncation performed"
            )
        pad = 0 if self.tokenizer is None else self.tokenizer.pad_token_id
        ids = torch.full(
            (len(rows), max(map(len, rows))), pad, dtype=torch.long, device=device
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, device=device)
            mask[i, : len(row)] = True
        return ids, mask

    def _encode(self, tokens):
        ids, mask = tokens
        if self.tokenizer is None:
            hidden = self.backbone(ids, mask)
        else:
            hidden = self.backbone(
                input_ids=ids,
                attention_mask=mask.long(),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        return hidden, mask

    def _pool_query(self, hidden, mask):
        """Summarize query tokens; state tokens remain a full attention sequence."""
        if self.config.query_pooling == "mean":
            return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        # Token counts minus one assume right padding. Select the last true mask
        # position instead so left padding and mixed padding work as well.
        positions = torch.arange(mask.shape[1], device=mask.device)
        last = positions.expand_as(mask).masked_fill(~mask, -1).amax(1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last]

    def forward(self, records):
        """Return a list of logits, preserving each request's candidate order.

        Encode each distinct state once in this batch. Questions have separate
        query encodings and never attend to other questions. No persistent
        cache is used while the backbone weights are being updated.
        """
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("records must be nonempty")
        rows = [
            validate_record(
                {k: v for k, v in row.items() if k != "target"},
                require_target=False,
                max_candidates=self.config.max_candidates,
            )
            for row in records
        ]
        states, state_indices, query_texts, owners, kinds, counts = (
            [],
            {},
            [],
            [],
            [],
            [],
        )
        for row in rows:
            if row["state"] not in state_indices:
                state_indices[row["state"]] = len(states)
                states.append(row["state"])
            options = candidates(row)
            counts.append(len(options))
            for item in options:
                # Candidate IDs are output handles, never semantic input.
                query_texts.append(
                    candidate_query_text(
                        row,
                        item,
                        instance_question_in_query=self.config.instance_question_in_query,
                    )
                )
                owners.append(state_indices[row["state"]])
                kinds.append(row["type"])
        # Tokenize both groups before any forward pass, failing without partial computation.
        state_tokens, query_tokens = self._tokens(states), self._tokens(query_texts)
        state_hidden, state_mask = self._encode(state_tokens)
        query_hidden, query_mask = self._encode(query_tokens)
        pooled = self._pool_query(query_hidden, query_mask)
        indices = torch.tensor(owners, device=pooled.device)
        attended, _ = self.cross_attention(
            pooled.unsqueeze(1),
            state_hidden[indices],
            state_hidden[indices],
            key_padding_mask=~state_mask[indices],
            need_weights=False,
        )
        attended = attended[:, 0]
        fused = torch.cat((pooled, attended, pooled * attended), dim=-1)
        # No index embedding or cross-candidate mixing: Choice permutation equivariance.
        logits = torch.stack(
            [self.heads[kind](fused[i]).squeeze(-1) for i, kind in enumerate(kinds)]
        ).float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("typed decision produced nonfinite logits")
        return list(logits.split(counts))

    @torch.inference_mode()
    def predict(self, records):
        if self.training:
            raise ValueError("call model.eval() before prediction")
        results = []
        for row, logits in zip(records, self(records)):
            question = validate_record(
                {k: v for k, v in row.items() if k != "target"},
                require_target=False,
                max_candidates=self.config.max_candidates,
            )
            entry = self.calibration.get(scope_key(question))
            status = (
                "calibrated"
                if entry
                else "out_of_scope"
                if self.calibration
                else "uncalibrated"
            )
            temperature = entry["temperature"] if entry else 1.0
            # Match offline metrics precision: tiny logit margins must not be
            # rounded into an artificial tie before selecting a candidate.
            probs = (logits.double() / temperature).softmax(-1).cpu().tolist()
            options = candidates(question)
            winner = min(range(len(probs)), key=lambda i: (-probs[i], options[i]["id"]))
            result = {
                "type": question["type"].capitalize(),
                "calibration_status": status,
                "temperature_used": temperature,
                "max_probability": max(probs),
                "probabilities": [
                    {**option, "probability": p} for option, p in zip(options, probs)
                ],
            }
            if question["type"] == "choice":
                result["choice_id"] = options[winner]["id"]
            elif question["type"] == "noul":
                result["probability_true"] = probs[1]
            else:
                result["level_id"] = options[winner]["id"]
                result["expected_score"] = sum(
                    option["value"] * p for option, p in zip(options, probs)
                )
            if "id" in row:
                result["id"] = row["id"]
            results.append(result)
        return results

    def save_pretrained(self, directory):
        """Publish a complete immutable artifact, including the trained backbone."""
        from safetensors.torch import save_file

        destination = Path(directory).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(
            prefix=".typed-save-", dir=destination.parent
        ) as work:
            staging = Path(work) / "model"
            staging.mkdir()
            weights = {
                k: v.detach().cpu().contiguous().clone()
                for k, v in self.state_dict().items()
            }
            if any(not bool(torch.isfinite(v).all()) for v in weights.values()):
                raise ValueError("cannot save nonfinite weights")
            save_file(weights, str(staging / "typed_decision.safetensors"))
            if self.tokenizer is not None:
                self.backbone.config.save_pretrained(staging / "backbone")
                self.tokenizer.save_pretrained(staging / "backbone")
            payload = {
                "schema_version": 2,
                "model_type": "saddle_typed_decision",
                "config": self.config.to_dict(),
                "metadata": self.metadata,
                "backbone_bundled": True,
            }
            for filename, value in (
                ("decision_config.json", payload),
                ("calibration.json", self.calibration),
            ):
                (staging / filename).write_text(
                    json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
            manifest = {
                str(p.relative_to(staging)).replace("\\", "/"): sha256(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            staging.rename(destination)

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu"):
        from safetensors.torch import load_file

        root = Path(directory).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not {
            "decision_config.json",
            "calibration.json",
            "typed_decision.safetensors",
        } <= set(manifest):
            raise ValueError("incomplete typed model manifest")
        for relative, expected in manifest.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256(path) != expected:
                raise ValueError("typed model artifact hash mismatch")
        payload = json.loads(
            (root / "decision_config.json").read_text(encoding="utf-8")
        )
        if (
            payload.get("schema_version") != 2
            or payload.get("model_type") != "saddle_typed_decision"
        ):
            raise ValueError("unsupported typed decision artifact")
        config = TypedDecisionConfig(**payload["config"])
        model = cls(
            config,
            restore_backbone=str(root / "backbone")
            if config.backbone == "hf"
            else None,
        )
        model.load_state_dict(
            load_file(str(root / "typed_decision.safetensors")), strict=True
        )
        model.calibration = json.loads(
            (root / "calibration.json").read_text(encoding="utf-8")
        )
        for entry in model.calibration.values():
            t = entry.get("temperature")
            if (
                isinstance(t, bool)
                or not isinstance(t, (int, float))
                or not math.isfinite(t)
                or not 0.01 <= t <= 100
            ):
                raise ValueError("invalid saved calibration temperature")
        model.metadata = payload["metadata"]
        return model.to(device).eval()
