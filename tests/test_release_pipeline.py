from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from saddlellm.ModelExporter import ModelExportRequest, ModelExporter
from saddlellm.PostTrainingCompatibility import post_training_runtime_report
from saddlellm.ReleaseGate import EvaluationReleaseGate
from saddlellm.SimpleFlow import SimpleFlowCompiler
from saddlellm.TrainingConfigValidator import TrainingConfigValidator


def test_release_gate_accepts_nested_scores_and_rejects_missing_required_metric():
    accepted = EvaluationReleaseGate.evaluate(
        {"perplexity": {"score": 12.5}, "quality": {"exact_match": 0.82}},
        {
            "perplexity": {"max": 20},
            "quality.exact_match": {"min": 0.8},
        },
    )
    assert accepted.accepted
    assert accepted.status == "accepted"

    rejected = EvaluationReleaseGate.evaluate(
        {"perplexity": {"score": 25}},
        {
            "perplexity": {"max": 20},
            "exact_match": {"min": 0.8},
        },
    )
    assert not rejected.accepted
    assert rejected.status == "rejected"
    assert len(rejected.failures) == 2


def test_release_gate_rejects_invalid_rules():
    result = EvaluationReleaseGate.evaluate(
        {"score": 1.0},
        {"score": {"min": 2, "max": 1}},
    )
    assert result.status == "invalid"
    assert not result.accepted


def test_full_simple_flow_adds_gate_then_export():
    compiled = SimpleFlowCompiler.compile(
        {
            "model": "local/tiny-model",
            "flow": "full",
            "data": {"sft": "sft.jsonl", "preference": "preference.jsonl"},
            "evaluation": {
                "dataset": "eval.jsonl",
                "gate": {
                    "enabled": True,
                    "rules": {"perplexity": {"max": 30}},
                },
            },
            "output": "outputs/full",
        }
    )
    assert compiled["stages"] == ["sft", "preference", "eval", "export"]
    assert compiled["eval"]["gate"]["rules"]["perplexity"]["max"] == 30
    assert compiled["export"]["enabled"] is True
    assert compiled["export"]["require_gate"] is True


def test_export_only_simple_flow_does_not_require_training_data():
    compiled = SimpleFlowCompiler.compile(
        {
            "model": "local/tiny-model",
            "flow": "export",
            "evaluation": {"enabled": False},
            "output": "outputs/export-only",
        }
    )
    assert compiled["stages"] == ["export"]
    assert compiled["export"]["output_dir"].endswith("release")


def test_validator_enforces_gate_order_and_rules():
    report = TrainingConfigValidator.validate(
        {
            "model": {"name_or_path": "local/tiny-model"},
            "stages": ["export", "eval"],
            "eval": {
                "enabled": True,
                "gate": {
                    "enabled": True,
                    "rules": {"perplexity": {"max": 20}},
                },
            },
            "export": {"enabled": True, "require_gate": True},
            "distributed": {"bf16": False, "fp16": False},
        },
        inspect_data=False,
    )
    assert not report.valid
    assert any("eval before export" in issue for issue in report.issues)


def test_local_checkpoint_export_writes_verified_manifest(tmp_path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"gpt2"}\n', encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"tiny-weights")
    (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (source / "training_args.bin").write_bytes(b"training-only")
    (source / "checkpoint-1").mkdir()
    (source / "checkpoint-1" / "model.safetensors").write_bytes(b"old")

    gate = EvaluationReleaseGate.evaluate(
        {"perplexity": {"score": 10}},
        {"perplexity": {"max": 20}},
    ).to_dict()
    destination = tmp_path / "release"
    result = ModelExporter.export(
        ModelExportRequest(
            model_path=str(source),
            output_dir=str(destination),
        ),
        gate=gate,
    )

    assert result.status == "completed"
    assert Path(result.manifest_path).is_file()
    assert not (destination / "training_args.bin").exists()
    assert not (destination / "checkpoint-1").exists()
    manifest = json.loads((destination / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"]["accepted"] is True
    assert manifest["source_type"] == "local_checkpoint"
    assert manifest["source_model"] == "checkpoint"
    assert manifest["release_id"].startswith("saddlellm-")


def test_generic_tokenizers_backend_has_a_fast_tokenizer_fallback(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from saddlellm.TokenizerLoader import load_tokenizer_compatible

    tokenizer = tokenizers.Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_tokenizer_compatible(str(tmp_path), local_files_only=True)
    assert loaded.encode("hello") == [1]


@pytest.mark.skipif(
    not post_training_runtime_report()["compatible"],
    reason="requires requirements-posttrain.txt",
)
def test_lora_adapter_export_merges_to_a_loadable_full_model(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    from saddlellm.PostTrainingCompatibility import stabilize_peft_optional_backends

    base = tmp_path / "base"
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=32,
            n_ctx=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).save_pretrained(base, safe_serialization=True)
    backend = tokenizers.Tokenizer(
        WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4},
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
    tokenizer.save_pretrained(base)

    stabilize_peft_optional_backends()
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
        ),
    )
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)

    destination = tmp_path / "merged"
    result = ModelExporter.export(
        ModelExportRequest(
            model_path=str(adapter),
            output_dir=str(destination),
            local_files_only=True,
            device="cpu",
            dtype="float32",
        )
    )
    assert result.adapter_merged is True
    assert not (destination / "adapter_config.json").exists()
    AutoModelForCausalLM.from_pretrained(destination, local_files_only=True)
    AutoTokenizer.from_pretrained(destination, local_files_only=True)


def test_orchestrator_gate_failure_writes_summary(monkeypatch, tmp_path):
    from saddlellm.LLModelEvalute import Evaluator
    from saddlellm.TrainingOrchestrator import TrainingOrchestrator

    monkeypatch.setattr(
        Evaluator,
        "evaluate",
        lambda self, **kwargs: {
            "model": kwargs["model_path"],
            "metrics": {"perplexity": {"score": 99.0}},
            "error": None,
        },
    )
    orchestrator = TrainingOrchestrator.from_dict(
        {
            "model": {"name_or_path": "local/tiny-model"},
            "stages": ["eval"],
            "distributed": {"bf16": False, "fp16": False},
            "logging": {"output_dir": str(tmp_path)},
            "eval": {
                "enabled": True,
                "dataset": "unused",
                "gate": {
                    "enabled": True,
                    "rules": {"perplexity": {"max": 20}},
                },
            },
        }
    )
    with pytest.raises(RuntimeError, match="release gate rejected"):
        orchestrator.run()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["results"]["eval"]["status"] == "rejected"
    assert summary["results"]["pipeline"]["failed_stage"] == "eval"
    assert (tmp_path / "release_gate.json").is_file()


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=4096):
        import torch

        length = min(max(1, len(text.split())), max_length)
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "route one<STOP>ignored"


class _FakeModel:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=512)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        import torch

        generated = torch.tensor([[2, 3, 4]], dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


def test_openai_compatible_api_health_auth_and_token_slicing():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from saddlellm.InferenceServer import InferenceServerSettings, create_inference_app

    app = create_inference_app(
        InferenceServerSettings(
            model_path="fake/release",
            model_name="tiny-release",
            api_key="secret",
            default_max_new_tokens=3,
            max_new_tokens=8,
        ),
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
    )
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").status_code == 401
    headers = {"Authorization": "Bearer secret"}
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "tiny-release",
            "messages": [{"role": "user", "content": "plan a route"}],
            "temperature": 0,
            "max_tokens": 3,
            "stop": "<STOP>",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "route one"
    assert payload["usage"]["completion_tokens"] == 3
    assert payload["choices"][0]["finish_reason"] == "stop"

    stream_response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert stream_response.status_code == 400
