from __future__ import annotations

import json

import pytest
import torch

from saddlellm.models.ModelBlueprint import (
    AttentionBlueprint,
    FFNBlueprint,
    ModelBlueprint,
)
from saddlellm.runtime.ModelExporter import ModelExportRequest, ModelExporter
from saddlellm.models.ModelLoader import (
    is_saddle_checkpoint,
    load_causal_lm,
    load_model_and_tokenizer,
)
from saddlellm.models.SaddleModeling import SaddleForCausalLM
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def _tiny_saddle_model() -> SaddleForCausalLM:
    blueprint = ModelBlueprint(
        name="tiny-native-lifecycle",
        hidden_size=16,
        vocab_size=8,
        num_layers=1,
        max_position_embeddings=32,
        attention=AttentionBlueprint(
            kind="gqa",
            backend="eager",
            num_heads=4,
            num_kv_heads=2,
        ),
        ffn=FFNBlueprint(kind="swiglu", intermediate_size=32),
    )
    return blueprint.build_model()


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


@pytest.fixture
def native_checkpoint(tmp_path):
    torch.manual_seed(101)
    path = tmp_path / "native-checkpoint"
    model = _tiny_saddle_model().eval()
    model.save_pretrained(path)
    tokenizer = _save_tiny_tokenizer(path)
    return path, model, tokenizer


@pytest.fixture
def hf_checkpoint(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    path = tmp_path / "hf-checkpoint"
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
    ).save_pretrained(path, safe_serialization=True)
    _save_tiny_tokenizer(path)
    return path


def test_unified_loader_detects_and_loads_native_checkpoint(native_checkpoint):
    path, original, _ = native_checkpoint
    input_ids = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)

    assert is_saddle_checkpoint(path)
    loaded = load_causal_lm(path, device="cpu", dtype="float32").eval()

    assert isinstance(loaded, SaddleForCausalLM)
    with torch.inference_mode():
        expected = original(input_ids=input_ids).logits
        actual = loaded(input_ids=input_ids).logits
    assert torch.equal(expected, actual)


def test_unified_loader_routes_local_hf_checkpoint(hf_checkpoint):
    from transformers import GPT2LMHeadModel

    assert not is_saddle_checkpoint(hf_checkpoint)
    model, tokenizer = load_model_and_tokenizer(
        hf_checkpoint,
        device="cpu",
        dtype="float32",
        local_files_only=True,
    )

    assert isinstance(model, GPT2LMHeadModel)
    assert tokenizer.encode("hello") == [4]
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[1, 4, 2]])).logits
    assert logits.shape == (1, 3, 8)


def test_unified_loader_rejects_incomplete_native_checkpoint(tmp_path):
    path = tmp_path / "incomplete-native"
    path.mkdir()
    (path / "saddle_config.json").write_text("{}\n", encoding="utf-8")

    assert not is_saddle_checkpoint(path)
    with pytest.raises(FileNotFoundError, match=r"incomplete SaddleLLM checkpoint.*pytorch_model\.bin"):
        load_causal_lm(path, device="cpu")


def test_native_generate_accepts_standard_generation_arguments():
    model = _tiny_saddle_model().eval()
    input_ids = torch.tensor([[1, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=0,
        eos_token_id=None,
        use_cache=True,
    )

    assert generated.shape == (1, 4)
    assert torch.equal(generated[:, :2], input_ids)


def test_evaluator_loads_native_checkpoint_and_computes_perplexity(
    native_checkpoint, tmp_path, monkeypatch
):
    from saddlellm.evaluation.LLModelEvalute import Evaluator
    import saddlellm.models.ModelLoader as model_loader

    path, _, _ = native_checkpoint
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"text":"hello world"}\n{"text":"tiny model"}\n',
        encoding="utf-8",
    )
    calls = []
    original = model_loader.load_model_and_tokenizer

    def recording_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(model_loader, "load_model_and_tokenizer", recording_loader)
    result = Evaluator().evaluate(
        model_path=str(path),
        dataset=str(dataset),
        metrics=["perplexity"],
        max_samples=2,
        max_length=8,
        batch_size=2,
        cpu=True,
    )

    assert result["error"] is None, result["error"]
    assert calls and calls[0][0][0] == str(path)
    score = result["metrics"]["perplexity"]["score"]
    assert score > 0 and torch.isfinite(torch.tensor(score))


def test_inference_loader_and_engine_run_native_checkpoint(native_checkpoint, monkeypatch):
    from saddlellm.runtime.InferenceServer import (
        CompletionRequest,
        InferenceServerSettings,
        LocalGenerationEngine,
        load_inference_model,
    )
    import saddlellm.models.ModelLoader as model_loader

    path, _, _ = native_checkpoint
    calls = []
    original = model_loader.load_causal_lm

    def recording_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(model_loader, "load_causal_lm", recording_loader)
    settings = InferenceServerSettings(
        model_path=str(path),
        device="cpu",
        dtype="float32",
        local_files_only=True,
        default_max_new_tokens=2,
        max_new_tokens=4,
    )
    model, tokenizer = load_inference_model(settings)
    payload = LocalGenerationEngine(model, tokenizer, settings).complete(
        CompletionRequest(prompt="hello", temperature=0.0, max_tokens=2)
    )

    assert calls and calls[0][0][0] == str(path)
    assert payload["object"] == "text_completion"
    assert payload["usage"]["completion_tokens"] <= 2
    assert isinstance(payload["choices"][0]["text"], str)


def test_native_export_is_reloadable_and_rejects_hf_conversion(native_checkpoint, tmp_path):
    source, model, _ = native_checkpoint
    destination = tmp_path / "native-release"

    result = ModelExporter.export(
        ModelExportRequest(
            model_path=str(source),
            output_dir=str(destination),
            format="saddle",
            merge_lora=False,
            safe_serialization=False,
            local_files_only=True,
            device="cpu",
            dtype="float32",
        )
    )

    assert result.status == "completed"
    assert result.format == "saddle"
    assert result.source_type == "native_checkpoint"
    assert (destination / "saddle_config.json").is_file()
    assert (destination / "model_blueprint.json").is_file()
    assert (destination / "pytorch_model.bin").is_file()
    manifest = json.loads((destination / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "saddle"
    assert manifest["source_type"] == "native_checkpoint"

    input_ids = torch.tensor([[1, 4, 5]], dtype=torch.long)
    reloaded = load_causal_lm(destination, device="cpu").eval()
    with torch.inference_mode():
        assert torch.equal(model(input_ids=input_ids).logits, reloaded(input_ids=input_ids).logits)

    with pytest.raises(ValueError, match=r"cannot be exported as format='hf'.*format='saddle'"):
        ModelExporter.export(
            ModelExportRequest(
                model_path=str(source),
                output_dir=str(tmp_path / "invalid-hf-release"),
                format="hf",
                merge_lora=False,
            )
        )


def test_native_export_rejects_checkpoint_without_tokenizer(tmp_path):
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    source = tmp_path / "weights-only-native"
    SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).save_pretrained(source)
    with pytest.raises(RuntimeError, match=r"tokenizer files are missing"):
        ModelExporter.export(
            ModelExportRequest(
                model_path=str(source),
                output_dir=str(tmp_path / "invalid-release"),
                format="saddle",
                merge_lora=False,
            )
        )

    # Metadata alone is not a usable tokenizer vocabulary/model.
    (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"tokenizer files are missing"):
        ModelExporter.export(
            ModelExportRequest(
                model_path=str(source),
                output_dir=str(tmp_path / "metadata-only-release"),
                format="saddle",
                merge_lora=False,
            )
        )


def _training_config(tmp_path, *, stages, model_path=None, pretrain_mode="scratch"):
    data = {
        "model": {
            "config": "qwen-tiny-160m",
            "backend": "saddle",
            "name_or_path": str(model_path) if model_path is not None else None,
        },
        "pretrain_mode": pretrain_mode,
        "stages": list(stages),
        "distributed": {"bf16": False, "fp16": False},
        "logging": {"output_dir": str(tmp_path / ("run-" + "-".join(stages)))},
        "eval": {"enabled": "eval" in stages, "dataset": "unused.jsonl"},
        "export": {"enabled": "export" in stages, "format": "saddle"},
    }
    if "pretrain" in stages:
        data["data"] = {"sources": [{"type": "local", "path": "unused.txt"}]}
    if "sft" in stages:
        data["sft"] = {
            "enabled": True,
            "data_path": "unused.jsonl",
            "use_lora": False,
            "use_qlora": False,
        }
    if "preference" in stages:
        data["preference"] = {
            "enabled": True,
            "method": "dpo",
            "data_path": "unused.jsonl",
            "use_lora": False,
            "use_qlora": False,
        }
    return data


def test_saddle_stage_guard_allows_full_parameter_sft_and_dpo(tmp_path):
    allowed = _training_config(
        tmp_path,
        stages=["tokenizer", "pretrain", "sft", "preference", "eval", "export"],
    )
    orchestrator = TrainingOrchestrator.from_dict(allowed)
    orchestrator._close_logging_handlers()

    lora = _training_config(tmp_path, stages=["sft"])
    lora["sft"]["use_lora"] = True
    with pytest.raises(ValueError, match=r"does not support LoRA for sft"):
        TrainingOrchestrator.from_dict(lora)

    orpo = _training_config(tmp_path, stages=["preference"])
    orpo["preference"]["method"] = "orpo"
    with pytest.raises(ValueError, match=r"supported methods: dpo"):
        TrainingOrchestrator.from_dict(orpo)


def test_native_sft_dpo_pipeline_trains_reloadable_checkpoint(native_checkpoint, tmp_path):
    checkpoint, original, _ = native_checkpoint
    sft_data = tmp_path / "native-sft.jsonl"
    sft_data.write_text(
        '\n'.join([
            json.dumps({"instruction": "hello", "output": "world"}),
            json.dumps({"instruction": "tiny", "output": "model"}),
        ]) + '\n',
        encoding="utf-8",
    )
    preference_data = tmp_path / "native-dpo.jsonl"
    preference_data.write_text(
        json.dumps({"prompt": "hello", "chosen": "world", "rejected": "tiny"}) + '\n',
        encoding="utf-8",
    )
    config = {
        "model": {
            "config": "qwen-tiny-160m",
            "backend": "saddle",
            "name_or_path": str(checkpoint),
        },
        "stages": ["sft", "preference"],
        "training": {
            "max_steps": 1,
            "dataloader_num_workers": 0,
            "stability_monitor": False,
        },
        "distributed": {"bf16": False, "fp16": False},
        "logging": {"output_dir": str(tmp_path / "native-posttrain"), "backend": "local"},
        "sft": {
            "enabled": True,
            "data_path": str(sft_data),
            "use_lora": False,
            "use_qlora": False,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "max_seq_length": 16,
            "warmup_steps": 0,
            "save_steps": 0,
            "logging_steps": 1,
        },
        "preference": {
            "enabled": True,
            "method": "dpo",
            "data_path": str(preference_data),
            "use_lora": False,
            "use_qlora": False,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "max_length": 16,
            "max_prompt_length": 8,
            "warmup_steps": 0,
            "save_steps": 0,
            "logging_steps": 1,
        },
        "eval": {"enabled": False},
    }

    stages = TrainingOrchestrator.from_dict(config).run()

    assert stages["sft"]["status"] == "completed"
    assert stages["sft"]["result"]["backend"] == "saddle"
    assert stages["preference"]["status"] == "completed"
    assert stages["preference"]["result"]["objective"] == "dpo"
    final_path = tmp_path / "native-posttrain" / "preference_checkpoints"
    assert is_saddle_checkpoint(final_path)
    reloaded = load_causal_lm(final_path, device="cpu").eval()
    input_ids = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    with torch.inference_mode():
        logits = reloaded(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()
    assert any(
        not torch.equal(original.state_dict()[name], reloaded.state_dict()[name])
        for name in original.state_dict()
    )


def test_saddle_continue_preflight_requires_complete_local_checkpoint(
    native_checkpoint, tmp_path
):
    checkpoint, _, _ = native_checkpoint
    no_path = _training_config(
        tmp_path,
        stages=["pretrain"],
        pretrain_mode="continue",
    )
    with pytest.raises(ValueError, match=r"continue.*requires model\.name_or_path"):
        TrainingOrchestrator.from_dict(no_path)

    valid = _training_config(
        tmp_path,
        stages=["pretrain"],
        model_path=checkpoint,
        pretrain_mode="continue",
    )
    orchestrator = TrainingOrchestrator.from_dict(valid)
    orchestrator._close_logging_handlers()

    missing = _training_config(
        tmp_path,
        stages=["pretrain"],
        model_path=tmp_path / "does-not-exist",
        pretrain_mode="continue",
    )
    parsed_missing = TrainingOrchestrator._parse_config(missing)
    assert parsed_missing.model_name_or_path == str(tmp_path / "does-not-exist")
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError), match=r"continue|checkpoint|model|load"):
        load_causal_lm(parsed_missing.model_name_or_path, device="cpu", local_files_only=True)

    incomplete = tmp_path / "incomplete-continue"
    incomplete.mkdir()
    (incomplete / "saddle_config.json").write_text("{}\n", encoding="utf-8")
    invalid = _training_config(
        tmp_path,
        stages=["pretrain"],
        model_path=incomplete,
        pretrain_mode="continue",
    )
    parsed_incomplete = TrainingOrchestrator._parse_config(invalid)
    with pytest.raises((FileNotFoundError, ValueError), match=r"incomplete|pytorch_model\.bin|checkpoint"):
        load_causal_lm(parsed_incomplete.model_name_or_path, device="cpu")


def test_saddle_trainer_checkpoint_is_native_and_base_resume_can_restore(tmp_path):
    from transformers import TrainingArguments

    from saddlellm.training.NativeTrainer import SaddleTrainer
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    model = SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    trainer = SaddleTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            report_to="none",
        ),
    )
    trainer.state.global_step = 7
    checkpoint = tmp_path / "checkpoint-7"
    trainer._save(str(checkpoint))

    assert is_saddle_checkpoint(checkpoint)
    assert (checkpoint / "training_args.bin").is_file()
    manifest = json.loads((checkpoint / "saddle_checkpoint.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["global_step"] == 7

    restored = SaddleForCausalLM(model.config)
    resume_trainer = SaddleTrainer(
        model=restored,
        args=TrainingArguments(
            output_dir=str(tmp_path / "resume"),
            use_cpu=True,
            report_to="none",
        ),
    )
    resume_trainer._load_from_checkpoint(str(checkpoint))
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_resume_config_is_normalized_and_native_export_defaults_are_safe(tmp_path):
    base = _training_config(tmp_path, stages=["pretrain"])
    base["training"] = {"resume_from_checkpoint": "false"}
    parsed = TrainingOrchestrator._parse_config(base)
    assert parsed.training.resume_from_checkpoint is False

    base["training"] = {"resume_from_checkpoint": "checkpoint-12"}
    parsed = TrainingOrchestrator._parse_config(base)
    assert parsed.training.resume_from_checkpoint == "checkpoint-12"

    base["training"] = {"resume_from_checkpoint": 12}
    with pytest.raises(TypeError, match="boolean or checkpoint path"):
        TrainingOrchestrator._parse_config(base)

    export = _training_config(tmp_path, stages=["export"])
    export["export"].pop("format")
    parsed = TrainingOrchestrator._parse_config(export)
    assert parsed.export.format == "saddle"
    assert parsed.export.merge_lora is None
