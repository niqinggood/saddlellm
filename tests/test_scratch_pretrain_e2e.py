from __future__ import annotations

import json

import pytest
import torch

from saddlellm.models.ModelLoader import is_saddle_checkpoint, load_causal_lm
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def _save_tiny_tokenizer(path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = tokenizers.Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
                "world": 5,
                "tiny": 6,
                "model": 7,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def _scratch_config(tmp_path, *, max_steps: int, resume: bool, preset=None):
    tokenizer_path = tmp_path / "tokenizer"
    if not tokenizer_path.exists():
        _save_tiny_tokenizer(tokenizer_path)

    corpus = tmp_path / "corpus.jsonl"
    if not corpus.exists():
        rows = [
            {"text": "hello world tiny model"},
            {"text": "tiny model hello world"},
            {"text": "hello tiny world model"},
            {"text": "model world tiny hello"},
        ]
        corpus.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    config = {
        "project": "scratch-e2e",
        "experiment": "tiny-native-pretrain",
        "stages": ["pretrain"],
        "pretrain_mode": "scratch",
        "model": {
            "backend": "saddle",
            "config": "tiny-e2e-blueprint",
            "tokenizer": str(tokenizer_path),
            "local_files_only": True,
            "blueprint": {
                "name": "tiny-e2e-blueprint",
                "hidden_size": 16,
                "vocab_size": 8,
                "num_layers": 1,
                "max_position_embeddings": 16,
                "attention": {
                    "kind": "gqa",
                    "backend": "eager",
                    "num_heads": 4,
                    "num_kv_heads": 2,
                },
                "ffn": {"kind": "swiglu", "intermediate_size": 32},
            },
        },
        "data": {
            "sources": [
                {
                    "type": "local",
                    "path": str(corpus),
                    "text_column": "text",
                    "streaming": False,
                }
            ],
            "max_seq_length": 8,
            "min_text_length": 1,
            "dedup_method": "exact",
            "quality_min_score": 0.0,
            "num_proc": 1,
            "pack_sequences": True,
            "append_eos": True,
            "validation_split": 0.0,
        },
        "training": {
            "max_steps": max_steps,
            "resume_from_checkpoint": resume,
            "per_device_batch_size": 1,
            "global_batch_size": 2,
            "learning_rate": 1e-3,
            "min_lr": 0.0,
            "warmup_steps": 0,
            "save_every_steps": 1,
            "eval_every_steps": 1,
            "log_every_steps": 1,
            "keep_last_n_checkpoints": 2,
            "load_best_model_at_end": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "stability_monitor": False,
            "loss_chunk_size": 2,
        },
        "distributed": {
            "strategy": "single",
            "num_gpus": 1,
            "num_nodes": 1,
            "gradient_accumulation_steps": 2,
            "bf16": False,
            "fp16": False,
        },
        "logging": {
            "output_dir": str(tmp_path / "run"),
            "backend": "local",
        },
    }
    if preset == "deepseek_style":
        from saddlellm.models.ModelBlueprint import ModelBlueprintLab

        blueprint = ModelBlueprintLab.deepseek_style_moe(
            hidden_size=16,
            layers=1,
            heads=4,
            kv_heads=2,
            experts=4,
            experts_per_token=2,
            seq_length=16,
        )
        blueprint.vocab_size = 8
        blueprint.attention.backend = "eager"
        config["model"]["blueprint"] = blueprint.to_dict()
    elif preset == "hybrid_attnres":
        from saddlellm.models.ModelBlueprint import ModelBlueprintLab

        blueprint = ModelBlueprintLab.hybrid_kda(
            hidden_size=16,
            layers=4,
            heads=4,
            kv_heads=2,
            vocab_size=8,
            seq_length=16,
            attention_residuals="block",
        )
        config["model"]["blueprint"] = blueprint.to_dict()
        config["training"]["gradient_checkpointing"] = True
    return config


@pytest.mark.parametrize("preset", [None, "deepseek_style", "hybrid_attnres"])
def test_scratch_pretrain_checkpoint_resume_and_generation(tmp_path, preset):
    first = TrainingOrchestrator.from_dict(
        _scratch_config(tmp_path, max_steps=1, resume=False, preset=preset)
    ).run()
    first_result = first["pretrain"]
    assert first_result["global_step"] == 1
    assert first_result["gradient_accumulation_steps"] == 2
    assert first_result["actual_global_batch_size"] == 2
    assert first_result["train_sequences"] > 0
    assert first_result["resumed_from_checkpoint"] is None
    assert is_saddle_checkpoint(tmp_path / "run" / "checkpoints" / "checkpoint-1")

    resumed = TrainingOrchestrator.from_dict(
        _scratch_config(tmp_path, max_steps=2, resume=True, preset=preset)
    ).run()
    resumed_result = resumed["pretrain"]
    assert resumed_result["global_step"] == 2
    assert resumed_result["resumed_from_checkpoint"].endswith("checkpoint-1")

    final_model = tmp_path / "run" / "final_model"
    assert is_saddle_checkpoint(final_model)
    model = load_causal_lm(final_model, device="cpu").eval()
    if preset == "deepseek_style":
        assert model.config.attention_kind == "mla"
        assert model.config.ffn_kind == "moe"
        assert model.config.aux_loss_free
        assert model.config.router_aux_loss_coef == 0.0
        assert model.config.multi_token_prediction
        assert len(model.mtp_heads) == 3
    elif preset == "hybrid_attnres":
        assert model.config.residual_attention_mode == "block"
        assert [layer.config.attention_kind for layer in model.layers] == [
            "kda",
            "kda",
            "kda",
            "gqa",
        ]
    prompt = torch.tensor([[1, 4, 5]], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=prompt).logits
        generated = model.generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
        )
    assert torch.isfinite(logits).all()
    assert generated.shape == (1, 5)
