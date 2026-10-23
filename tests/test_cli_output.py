import json
import importlib

from saddlellm import cli


def test_windows_cli_configures_utf8_output(monkeypatch) -> None:
    calls = []

    class Stream:
        @staticmethod
        def reconfigure(**kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli.sys, "stdout", Stream())
    monkeypatch.setattr(cli.sys, "stderr", Stream())

    cli._configure_windows_stdio()

    assert calls == [
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
    ]


def test_inspect_data_can_audit_multiple_splits(tmp_path, capsys) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        '{"case_id":"shared","instruction":"train prompt","output":"answer"}\n',
        encoding="utf-8",
    )
    validation.write_text(
        '{"case_id":"shared","instruction":"validation prompt","output":"answer"}\n',
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "inspect-data",
            str(train),
            "--validation-data",
            str(validation),
            "--group-field",
            "case_id",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["split_leakage"]["pairs"]["train_validation"][
        "group_overlap"
    ] == 1


def test_lightweight_commands_have_safe_dry_run_contracts(tmp_path, capsys) -> None:
    quantized = tmp_path / "quantized"
    assert (
        cli.main(["lightweight", "model", str(quantized), "--dry-run"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "quantize"
    assert payload["method"] == "dynamic"
    assert payload["precision"] == "int8"
    assert payload["serve_compatible"] is True
    assert not quantized.exists()

    pruned = tmp_path / "pruned"
    assert cli.main(["prune", "model", str(pruned), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "prune"
    assert payload["requested_sparsity"] == 0.15
    assert "no speedup guarantee" not in payload["warnings"][0]
    assert len(payload["warnings"]) == 2
    assert not pruned.exists()


def test_direct_logit_distill_dry_run_is_explicit(tmp_path, capsys) -> None:
    data = tmp_path / "corpus.jsonl"
    data.write_text('{"text":"nursing"}\n', encoding="utf-8")
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text(
        '{"instruction":"question","output":"answer"}\n', encoding="utf-8"
    )
    output = tmp_path / "student"

    assert (
        cli.main(
            [
                "distill",
                "--mode",
                "logits",
                "--teacher",
                "teacher-model",
                "--data",
                str(data),
                "--eval-data",
                str(eval_data),
                "--student-layers",
                "8",
                "--student-hidden-size",
                "512",
                "--student-heads",
                "8",
                "--output-dir",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "logits"
    assert payload["status"] == "planned"
    assert payload["resolved_config"]["studentLayers"] == 8
    assert payload["resolved_config"]["datasetPath"] == str(data)
    assert payload["evaluation"]["status"] == "planned"
    assert payload["evaluation"]["data"] == str(eval_data.resolve())
    assert not output.exists()


def test_logit_distill_runs_requested_paired_evaluation(
    tmp_path, capsys, monkeypatch
) -> None:
    operators = importlib.import_module("saddlellm.agents.RealWorldOperators")
    data = tmp_path / "corpus.jsonl"
    data.write_text('{"text":"nursing"}\n', encoding="utf-8")
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text(
        '{"instruction":"question","output":"answer"}\n', encoding="utf-8"
    )
    output = tmp_path / "student"
    model_path = output / "student_model"

    monkeypatch.setattr(
        operators,
        "run_large_model_operator",
        lambda *_args: {
            "status": "success",
            "model_path": str(model_path),
            "metrics": {
                "teacher_perplexity": 3.0,
                "initial_student_perplexity": 5.0,
                "perplexity": 4.0,
            },
            "lineage": {"evalSamples": 2},
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_distillation_evaluation",
        lambda *_args, **_kwargs: {"status": "completed", "task": "completion"},
    )

    assert (
        cli.main(
            [
                "distill",
                "--mode",
                "logits",
                "--teacher",
                "teacher-model",
                "--data",
                str(data),
                "--eval-data",
                str(eval_data),
                "--student-layers",
                "1",
                "--student-hidden-size",
                "8",
                "--student-heads",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluation"]["internal"]["outcome"] == "improved"
    assert payload["evaluation"]["external"]["status"] == "completed"
    assert "evaluate" not in payload["next"]


def test_direct_data_distill_builds_mopd_then_sft(tmp_path, capsys) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"instruction":"question"}\n', encoding="utf-8")

    assert (
        cli.main(
            [
                "distill",
                "--mode",
                "data",
                "--student",
                "student-model",
                "--teacher",
                "teacher-model",
                "--teacher-url",
                "http://127.0.0.1:8001",
                "--data",
                str(prompts),
                "--domain",
                "nursing",
                "--concurrency",
                "6",
                "--eval-data",
                str(prompts),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    config = payload["resolved_config"]
    assert config["stages"] == ["mopd"]
    assert config["mopd"]["teacher_concurrency"] == 6
    assert config["mopd"]["teachers"][0]["domain"] == "nursing"
    assert config["mopd"]["requires_review"] is True
    assert "sft" not in config
    assert payload["evaluation"]["status"] == "deferred_until_reviewed_training"


def test_direct_sft_train_supports_positional_stage(tmp_path, capsys) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"instruction":"question","output":"answer"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    exit_code = cli.main(
        [
            "train",
            "sft",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--data",
            str(data),
            "--epochs",
            "1",
            "--learning-rate",
            "0.00005",
            "--output-dir",
            str(output_dir),
            "--local-files-only",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    config = payload["resolved_config"]
    assert payload["config_source"] == "command-line"
    assert payload["stages"] == ["sft"]
    assert config["model"]["name_or_path"] == "Qwen/Qwen3-0.6B"
    assert config["sft"]["data_path"] == str(data)
    assert config["sft"]["epochs"] == 1
    assert config["sft"]["learning_rate"] == 0.00005
    assert config["sft"]["use_lora"] is True
    assert config["sft"]["use_qlora"] is False
    assert config["sft"]["local_files_only"] is True
    assert config["logging"]["output_dir"] == str(output_dir)
    assert not (output_dir / "command_config.json").exists()


def test_direct_sft_train_supports_stage_option_and_qlora(tmp_path, capsys) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "--stage",
                "sft",
                "--model",
                "local-model",
                "--data",
                str(data),
                "--method",
                "qlora",
                "--precision",
                "fp16",
                "--dry-run",
            ]
        )
        == 0
    )

    config = json.loads(capsys.readouterr().out)["resolved_config"]
    assert config["sft"]["use_lora"] is True
    assert config["sft"]["use_qlora"] is True
    assert config["sft"]["precision"] == "fp16"
    assert config["distributed"]["bf16"] is False
    assert config["distributed"]["fp16"] is True


def test_direct_sft_accepts_explicit_validation_and_all_linear(tmp_path, capsys) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"instruction":"q1","output":"a1"}\n', encoding="utf-8")
    validation.write_text(
        '{"instruction":"q2","output":"a2"}\n', encoding="utf-8"
    )

    assert (
        cli.main(
            [
                "train",
                "sft",
                "--model",
                "local-model",
                "--data",
                str(train),
                "--validation-data",
                str(validation),
                "--method",
                "qlora",
                "--lora-target-modules",
                "all-linear",
                "--dry-run",
            ]
        )
        == 0
    )

    config = json.loads(capsys.readouterr().out)["resolved_config"]
    assert config["sft"]["validation_data_path"] == str(validation)
    assert config["sft"]["validation_split"] == 0.0
    assert config["sft"]["load_best_model_at_end"] is True
    assert config["sft"]["lora"]["target_modules"] == "all-linear"


def test_direct_validation_data_rejects_ambiguous_use(tmp_path, capsys) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{}\n', encoding="utf-8")
    validation.write_text('{}\n', encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "sft",
                "--model",
                "local-model",
                "--data",
                str(train),
                "--validation-data",
                str(validation),
                "--validation-split",
                "0.1",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "either --validation-data or --validation-split" in json.loads(
        capsys.readouterr().out
    )["error"]


def test_direct_preference_passes_explicit_validation_data(tmp_path, capsys) -> None:
    train = tmp_path / "preference.jsonl"
    validation = tmp_path / "preference-validation.jsonl"
    row = '{"prompt":"q","chosen":"good","rejected":"bad"}\n'
    train.write_text(row, encoding="utf-8")
    validation.write_text(row, encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "dpo",
                "--model",
                "local-model",
                "--data",
                str(train),
                "--validation-data",
                str(validation),
                "--lora-target-modules",
                "all-linear",
                "--dry-run",
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)["resolved_config"]
    assert config["preference"]["validation_data_path"] == str(validation)
    assert config["preference"]["validation_split"] == 0.0
    assert config["preference"]["lora"]["target_modules"] == "all-linear"


def test_direct_sft_allows_explicit_saddle_backend(tmp_path, capsys) -> None:
    data = tmp_path / "sft.jsonl"
    data.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "sft",
                "--backend",
                "saddle",
                "--model",
                "native-model",
                "--data",
                str(data),
                "--method",
                "full",
                "--dry-run",
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)["resolved_config"]
    assert config["model"]["backend"] == "saddle"


def test_direct_single_stage_exposes_exact_resume(tmp_path, capsys) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-1"

    assert (
        cli.main(
            [
                "train",
                "sft",
                "--model",
                "local-model",
                "--data",
                str(data),
                "--resume-from-checkpoint",
                str(checkpoint),
                "--dry-run",
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)["resolved_config"]
    assert config["training"]["resume_from_checkpoint"] == str(checkpoint)


def test_direct_multistage_resume_requires_config_file(tmp_path, capsys) -> None:
    sft = tmp_path / "sft.jsonl"
    preference = tmp_path / "preference.jsonl"
    sft.write_text("{}\n", encoding="utf-8")
    preference.write_text("{}\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "sft,dpo",
                "--model",
                "local-model",
                "--data",
                str(sft),
                "--dpo-data",
                str(preference),
                "--resume-from-checkpoint",
                "checkpoint-1",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "only for one direct stage" in json.loads(capsys.readouterr().out)["error"]


def test_direct_continue_pretrain_builds_quality_controlled_config(
    tmp_path, capsys
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"nursing knowledge"}\n', encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "pretrain",
                "--model",
                "local-model",
                "--data",
                str(corpus),
                "--max-steps",
                "2",
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "4",
                "--quality-min-score",
                "0.4",
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    config = payload["resolved_config"]
    assert payload["stages"] == ["pretrain"]
    assert config["pretrain_mode"] == "continue"
    assert config["model"]["name_or_path"] == "local-model"
    assert config["model"]["backend"] == "hf"
    assert config["data"]["sources"][0]["path"] == str(corpus)
    assert config["data"]["append_eos"] is True
    assert config["data"]["pack_sequences"] is True
    assert config["data"]["validation_split"] == 0.01
    assert config["data"]["quality_min_score"] == 0.4
    assert config["training"]["global_batch_size"] == 8
    assert config["training"]["max_steps"] == 2
    assert config["distributed"]["gradient_accumulation_steps"] == 4


def test_direct_scratch_pretrain_requires_architecture_and_tokenizer(
    tmp_path, capsys
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"nursing knowledge"}\n', encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "pretrain",
                "--pretrain-mode",
                "scratch",
                "--data",
                str(corpus),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "--model-config" in json.loads(capsys.readouterr().out)["error"]

    assert (
        cli.main(
            [
                "train",
                "pretrain",
                "--pretrain-mode",
                "scratch",
                "--model-config",
                "qwen-tiny-160m",
                "--tokenizer",
                "local-tokenizer",
                "--data",
                str(corpus),
                "--dry-run",
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)["resolved_config"]
    assert config["pretrain_mode"] == "scratch"
    assert config["model"]["name_or_path"] is None
    assert config["model"]["config"] == "qwen-tiny-160m"
    assert config["model"]["tokenizer"] == "local-tokenizer"
    assert config["model"]["backend"] == "saddle"


def test_direct_pretrain_must_be_first_and_can_chain_sft(tmp_path, capsys) -> None:
    corpus = tmp_path / "corpus.jsonl"
    sft = tmp_path / "sft.jsonl"
    corpus.write_text('{"text":"nursing knowledge"}\n', encoding="utf-8")
    sft.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "sft,pretrain",
                "--model",
                "local-model",
                "--data",
                str(sft),
                "--pretrain-data",
                str(corpus),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "must be the first" in json.loads(capsys.readouterr().out)["error"]

    assert (
        cli.main(
            [
                "train",
                "pretrain,sft",
                "--model",
                "local-model",
                "--data",
                str(corpus),
                "--sft-data",
                str(sft),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["pretrain", "sft"]
    assert payload["orchestrator_stages"] == ["pretrain", "sft"]
    assert payload["pipeline_plan"]["nodes"][1]["needs"] == ["pretrain"]
    assert payload["resolved_config"]["sft"]["data_path"] == str(sft)


def test_direct_multi_stage_preserves_requested_left_to_right_order(
    tmp_path, capsys
) -> None:
    paths = {}
    for stage in ("sft", "ppo", "kto"):
        path = tmp_path / f"{stage}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        paths[stage] = path

    assert (
        cli.main(
            [
                "train",
                "--stage",
                "sft,ppo,kto",
                "--model",
                "local-model",
                "--data",
                str(paths["sft"]),
                "--ppo-data",
                str(paths["ppo"]),
                "--kto-data",
                str(paths["kto"]),
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["sft", "ppo", "kto"]
    assert payload["orchestrator_stages"] == ["sft", "rlhf", "preference"]
    assert payload["stage_mapping"] == [
        {"requested": "sft", "orchestrator": "sft"},
        {"requested": "ppo", "orchestrator": "rlhf"},
        {"requested": "kto", "orchestrator": "preference"},
    ]
    nodes = payload["pipeline_plan"]["nodes"]
    assert [(node["name"], node["needs"]) for node in nodes] == [
        ("sft", []),
        ("rlhf", ["sft"]),
        ("preference", ["rlhf"]),
    ]
    config = payload["resolved_config"]
    assert config["direct_pipeline"]["execution_semantics"] == "strict_left_to_right"
    assert config["sft"]["data_path"] == str(paths["sft"])
    assert config["rlhf"]["data_path"] == str(paths["ppo"])
    assert config["preference"]["data_path"] == str(paths["kto"])
    assert config["preference"]["method"] == "kto"
    assert config["preference"]["per_device_batch_size"] == 2


def test_direct_single_preference_stage_accepts_data_shorthand(tmp_path, capsys) -> None:
    data = tmp_path / "dpo.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "dpo",
                "--model",
                "local-model",
                "--data",
                str(data),
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["dpo"]
    assert payload["orchestrator_stages"] == ["preference"]
    assert payload["resolved_config"]["preference"]["data_path"] == str(data)
    assert payload["resolved_config"]["preference"]["method"] == "dpo"


def test_auto_posttrain_selects_sft_and_explains_decision(tmp_path, capsys) -> None:
    data = tmp_path / "sft.jsonl"
    data.write_text(
        '{"instruction":"question","output":"answer"}\n',
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "train",
                "--stage",
                "auto-posttrain",
                "--model",
                "local-model",
                "--data",
                str(data),
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["auto-posttrain"]
    assert payload["selected_stages"] == ["sft"]
    assert payload["orchestrator_stages"] == ["sft"]
    assert payload["auto_decision"]["selected_stage"] == "sft"
    assert payload["auto_decision"]["confidence"] == 1.0
    assert payload["resolved_config"]["sft"]["data_path"] == str(data)


def test_auto_posttrain_selects_dpo_or_balanced_kto(tmp_path, capsys) -> None:
    dpo = tmp_path / "dpo.jsonl"
    dpo.write_text(
        '{"prompt":"q","chosen":"good","rejected":"bad"}\n',
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "train",
                "auto-posttrain",
                "--model",
                "local-model",
                "--data",
                str(dpo),
                "--dry-run",
            ]
        )
        == 0
    )
    dpo_payload = json.loads(capsys.readouterr().out)
    assert dpo_payload["selected_stages"] == ["dpo"]
    assert dpo_payload["resolved_config"]["preference"]["method"] == "dpo"

    kto = tmp_path / "kto.jsonl"
    kto.write_text(
        "\n".join(
            [
                '{"prompt":"q1","completion":"good","label":true}',
                '{"prompt":"q2","completion":"bad","label":false}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "train",
                "auto-posttrain",
                "--model",
                "local-model",
                "--data",
                str(kto),
                "--dry-run",
            ]
        )
        == 0
    )
    kto_payload = json.loads(capsys.readouterr().out)
    assert kto_payload["selected_stages"] == ["kto"]
    assert kto_payload["resolved_config"]["preference"]["method"] == "kto"
    assert kto_payload["resolved_config"]["preference"]["per_device_batch_size"] == 2


def test_auto_posttrain_rejects_mixed_or_one_sided_kto_data(tmp_path, capsys) -> None:
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        "\n".join(
            [
                '{"instruction":"q","output":"a"}',
                '{"prompt":"q","chosen":"good","rejected":"bad"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "train",
                "auto-posttrain",
                "--model",
                "local-model",
                "--data",
                str(mixed),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "one unambiguous data contract" in json.loads(
        capsys.readouterr().out
    )["error"]

    one_sided = tmp_path / "one-sided-kto.jsonl"
    one_sided.write_text(
        '{"prompt":"q","completion":"good","label":true}\n',
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "train",
                "auto-posttrain",
                "--model",
                "local-model",
                "--data",
                str(one_sided),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "both desirable" in json.loads(capsys.readouterr().out)["error"]


def test_direct_multi_stage_requires_data_for_every_stage(tmp_path, capsys) -> None:
    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text("{}\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "train",
                "--stage",
                "sft,kto",
                "--model",
                "local-model",
                "--data",
                str(sft_data),
                "--dry-run",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert "Missing data" in payload["error"]
    assert "--kto-data" in payload["error"]


def test_direct_ppo_pipeline_fails_before_any_training_stage(tmp_path, capsys) -> None:
    sft_data = tmp_path / "sft.jsonl"
    ppo_data = tmp_path / "ppo.jsonl"
    kto_data = tmp_path / "kto.jsonl"
    for path in (sft_data, ppo_data, kto_data):
        path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "run"

    assert (
        cli.main(
            [
                "train",
                "--stage",
                "sft,ppo,kto",
                "--model",
                "local-model",
                "--data",
                str(sft_data),
                "--ppo-data",
                str(ppo_data),
                "--kto-data",
                str(kto_data),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "NotImplementedError"
    assert "No earlier or later stage was run" in payload["error"]
    assert not (output_dir / "sft_checkpoints").exists()


def test_direct_train_persists_equivalent_config(
    tmp_path, capsys, monkeypatch
) -> None:
    orchestrator_module = importlib.import_module(
        "saddlellm.training.TrainingOrchestrator"
    )

    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    captured = {}

    class FakeOrchestrator:
        @classmethod
        def from_dict(cls, config):
            captured["config"] = config
            return cls()

        @staticmethod
        def run():
            return {"sft": {"status": "completed"}}

    monkeypatch.setattr(
        orchestrator_module,
        "TrainingOrchestrator",
        FakeOrchestrator,
    )

    assert (
        cli.main(
            [
                "train",
                "sft",
                "--model",
                "local-model",
                "--data",
                str(data),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    saved = output_dir / "command_config.json"
    payload = json.loads(capsys.readouterr().out)
    assert saved.is_file()
    assert json.loads(saved.read_text(encoding="utf-8")) == captured["config"]
    assert payload["config"] == str(saved.resolve())
    assert payload["next"]["dashboard"].endswith(f'"{output_dir}" --open')


def test_train_rejects_mixing_config_and_direct_options(tmp_path, capsys) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"stages":["sft"]}\n', encoding="utf-8")

    assert cli.main(["train", str(config), "--model", "other", "--dry-run"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "either a YAML/JSON config or direct" in payload["error"]


def test_short_commands_and_long_compatibility_aliases(monkeypatch) -> None:
    seen = []

    monkeypatch.setattr(
        cli,
        "_cmd_evaluate_model",
        lambda args: seen.append((args.command, args.model, args.data)) or 0,
    )
    monkeypatch.setattr(
        cli,
        "_cmd_serve_model",
        lambda args: seen.append((args.command, args.model_path)) or 0,
    )
    monkeypatch.setattr(
        cli,
        "_cmd_compare_evaluations",
        lambda args: seen.append(
            (args.command, args.base_validation, args.candidate_validation)
        )
        or 0,
    )
    monkeypatch.setattr(
        cli,
        "_cmd_export_model",
        lambda args: seen.append((args.command, args.model_path, args.output_dir))
        or 0,
    )

    assert cli.main(["evaluate", "model", "eval.jsonl"]) == 0
    assert (
        cli.main(
            [
                "compare",
                "--base-validation",
                "base.json",
                "--candidate-validation",
                "candidate.json",
            ]
        )
        == 0
    )
    assert cli.main(["export", "checkpoint", "release-dir"]) == 0
    assert cli.main(["serve", "release"]) == 0
    assert cli.main(["evaluate-model", "old-model", "old-eval.jsonl"]) == 0
    assert cli.main(["serve-model", "old-release"]) == 0
    assert seen == [
        ("evaluate", "model", "eval.jsonl"),
        ("compare", "base.json", "candidate.json"),
        ("export", "checkpoint", "release-dir"),
        ("serve", "release"),
        ("evaluate-model", "old-model", "old-eval.jsonl"),
        ("serve-model", "old-release"),
    ]


def test_evaluate_accepts_generic_prompt_robustness_contract(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "_cmd_evaluate_model",
        lambda args: captured.update(vars(args)) or 0,
    )

    assert (
        cli.main(
            [
                "evaluate",
                "model",
                "eval.jsonl",
                "--case-field",
                "source_id",
                "--variant-field",
                "prompt_style",
            ]
        )
        == 0
    )
    assert captured["case_field"] == "source_id"
    assert captured["variant_field"] == "prompt_style"


def test_completion_evaluate_rejects_multiple_choice_prompt_variants(capsys) -> None:
    assert (
        cli.main(
            [
                "evaluate",
                "model",
                "eval.jsonl",
                "--task",
                "completion",
                "--prompt-variants",
                "variants.json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert "only supported with --task multiple-choice" in payload["error"]


def test_serve_accepts_runtime_quantization(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "_cmd_serve_model",
        lambda args: captured.update(vars(args)) or 0,
    )

    assert cli.main(["serve", "adapter", "--quantization", "4bit"]) == 0
    assert captured["quantization"] == "4bit"
