"""Regression gates exercise real provenance with deterministic CPU evaluators."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from saddlellm.evaluation.ModelIdentity import compute_model_identity, verify_gate_model
from saddlellm.evaluation.RegressionSuite import (
    compare_regression_suite,
    load_regression_suite,
    run_regression_suite,
    validate_regression_output,
)
from saddlellm.evaluation.ReleaseGate import release_gate_accepted


@pytest.fixture
def setup(tmp_path):
    models = []
    for name in ("baseline", "candidate", "other"):
        model = tmp_path / name
        model.mkdir()
        (model / "model.safetensors").write_bytes(name.encode())
        (model / "tokenizer_config.json").write_text(
            '{"chat_template":"same"}', encoding="utf-8"
        )
        models.append(str(model))
    data = tmp_path / "records.jsonl"
    data.write_text(
        '{"id":"one","instruction":"question","answer":"A","output":"answer"}\n',
        encoding="utf-8",
    )
    suite = {
        "name": "retention",
        "tasks": [
            {
                "name": "general",
                "task": "multiple-choice",
                "data": str(data),
                "split": "test",
                "options": {"max_length": 128, "local_files_only": True},
                "metrics": {"accuracy": {"max_drop": 0.02, "higher_is_better": True}},
            },
            {
                "name": "domain",
                "task": "completion",
                "data": str(data),
                "options": {"max_length": 128},
                "metrics": {
                    "completion_perplexity": {
                        "max_drop": 0.5,
                        "higher_is_better": False,
                        "max": 5,
                    }
                },
            },
        ],
    }
    calls = []

    def evaluator(model, source, *, task, split=None, **options):
        calls.append(
            {"model": model, "data": source, "task": task, "split": split, **options}
        )
        baseline = model == models[0]
        data_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        return {
            "task_type": task.replace("-", "_"),
            "model": model,
            "model_identity": compute_model_identity(model, options.get("adapter")),
            "data_sha256": data_hash,
            "samples": 1,
            "accuracy": 0.8 if baseline else 0.79,
            "completion_perplexity": 4.0 if baseline else 4.5,
            "by_group": {"reasoning": {"accuracy": 0.9 if baseline else 0.91}},
            "evaluation_identity": {
                "task": task,
                "data_sha256": data_hash,
                "split": split,
                "limit": 0,
                "samples": 1,
                "records_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            json.loads(line)
                            for line in Path(source)
                            .read_text(encoding="utf-8")
                            .splitlines()
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "source_records": 1,
            },
            "details": [{"record_id": "one", "correct": baseline}],
        }

    return models, data, suite, calls, evaluator


def run(setup, **kwargs):
    models, _, suite, _, evaluator = setup
    return run_regression_suite(
        models[0], models[1], suite, evaluator=evaluator, **kwargs
    )


def reports(result):
    return (
        {
            task["name"]: copy.deepcopy(task["baseline_report"])
            for task in result["tasks"]
        },
        {
            task["name"]: copy.deepcopy(task["candidate_report"])
            for task in result["tasks"]
        },
    )


def test_suite_runs_same_data_and_options_and_exports_bound_gate(setup, tmp_path):
    output = tmp_path / "report.json"
    result = run(setup, output=output)
    assert result["status"] == "accepted"
    assert release_gate_accepted(result)
    assert result["output"] == str(output.resolve())
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["tasks"][0]["metrics"]["accuracy"]["delta"] == pytest.approx(-0.01)
    assert result["tasks"][1]["metrics"]["completion_perplexity"]["drop"] == 0.5
    for baseline, candidate in zip(setup[3][::2], setup[3][1::2]):
        assert {key: value for key, value in baseline.items() if key != "model"} == {
            key: value for key, value in candidate.items() if key != "model"
        }
    assert verify_gate_model(
        result, compute_model_identity(setup[0][1]), required=True
    )["accepted"]


def test_both_metric_directions_and_candidate_absolute_bounds(setup):
    setup[2]["tasks"][0]["metrics"]["accuracy"]["max_drop"] = 0.005
    result = run(setup)
    assert result["status"] == "failed"
    assert not result["accepted"]
    assert result["tasks"][0]["gate"]["checks"][0]["status"] == "failed"
    assert result["tasks"][1]["status"] == "accepted"
    setup[2]["tasks"][0]["metrics"]["accuracy"]["max_drop"] = 0.02
    setup[2]["tasks"][1]["metrics"]["completion_perplexity"]["max"] = 4.4
    assert run(setup)["status"] == "failed"


def test_nested_metric_and_exact_drop_boundary(setup):
    setup[2]["tasks"][0]["metrics"] = {
        "by_group.reasoning.accuracy": {
            "max_drop": 0,
            "higher_is_better": True,
            "min": 0.91,
        }
    }
    result = run(setup)
    assert result["accepted"]
    assert (
        result["tasks"][0]["metrics"]["by_group.reasoning.accuracy"]["candidate"]
        == 0.91
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_load_paths_relative_to_suite_and_no_model_loading(tmp_path, suffix):
    config = {
        "tasks": [
            {
                "name": "task",
                "data": "../records.jsonl",
                "options": {"prompt_variants": "variants.json"},
                "metrics": {"accuracy": {"max_drop": 0, "higher_is_better": True}},
            }
        ]
    }
    path = tmp_path / f"suite{suffix}"
    if suffix == ".yaml":
        import yaml

        path.write_text(yaml.safe_dump(config), encoding="utf-8")
    else:
        path.write_text(json.dumps(config), encoding="utf-8")
    result = load_regression_suite(path)
    assert result["tasks"][0]["data"] == str(
        (tmp_path.parent / "records.jsonl").resolve()
    )
    assert result["tasks"][0]["options"]["prompt_variants"] == str(
        (tmp_path / "variants.json").resolve()
    )
    assert load_regression_suite(result) == result


@pytest.mark.parametrize(
    "rule",
    [
        {"max_drop": -1, "higher_is_better": True},
        {"max_drop": float("nan"), "higher_is_better": True},
        {"max_drop": True, "higher_is_better": True},
        {"max_drop": 0, "higher_is_better": "false"},
        {"max_drop": 0, "higher_is_better": True, "min": 2, "max": 1},
        {"max_drop": 0, "higher_is_better": True, "max_dorp": 0.1},
    ],
)
def test_reject_invalid_metric_rules(setup, rule):
    setup[2]["tasks"][0]["metrics"] = {"accuracy": rule}
    with pytest.raises(ValueError):
        load_regression_suite(setup[2])


def test_duplicate_tasks_and_reserved_options_fail_before_inference(setup):
    setup[2]["tasks"][1]["name"] = "general"
    with pytest.raises(ValueError, match="duplicate"):
        run(setup)
    assert not setup[3]
    setup[2]["tasks"][1]["name"] = "domain"
    setup[2]["tasks"][0]["options"]["adapter"] = "hidden-adapter"
    with pytest.raises(ValueError, match="cannot override"):
        run(setup)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_nonfinite_or_boolean_metrics_fail_closed_and_json_is_portable(
    setup, tmp_path, value
):
    result = run(setup)
    baseline, candidate = reports(result)
    candidate["general"]["accuracy"] = value
    target = tmp_path / "invalid.json"
    result = compare_regression_suite(setup[2], baseline, candidate, output=target)
    assert result["status"] == "invalid"
    assert not result["accepted"]
    assert "finite number" in result["tasks"][0]["errors"][0]
    json.dumps(result, allow_nan=False)
    assert "NaN" not in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda report: report.pop("accuracy"), "missing"),
        (
            lambda report: report["regression_provenance"]["settings"][
                "options"
            ].update(max_length=999),
            "provenance",
        ),
        (lambda report: report.update(data_sha256="different"), "data hash"),
        (lambda report: report.update(task_type="rubric"), "task type"),
        (lambda report: report.update(samples=2), "identity samples"),
        (
            lambda report: report["evaluation_identity"].update(
                records_sha256="b" * 64
            ),
            "identity records_sha256",
        ),
        (
            lambda report: report.update(generation_config={"do_sample": True}),
            "generation_config",
        ),
        (
            lambda report: report["model_identity"].update(status="unresolved"),
            "unbound",
        ),
    ],
)
def test_report_mismatches_are_invalid(setup, mutation, expected):
    baseline, candidate = reports(run(setup))
    mutation(candidate["general"])
    result = compare_regression_suite(setup[2], baseline, candidate)
    assert result["status"] == "invalid"
    assert expected in result["tasks"][0]["errors"][0]


def test_model_identity_must_be_consistent_across_domains(setup):
    baseline, candidate = reports(run(setup))
    candidate["domain"]["model_identity"] = compute_model_identity(setup[0][2])
    result = compare_regression_suite(setup[2], baseline, candidate)
    assert result["status"] == "invalid"
    assert "identity differs" in result["tasks"][1]["errors"][0]


def test_completion_tokenizer_changes_invalidate_comparison(setup):
    (Path(setup[0][1]) / "tokenizer_config.json").write_text(
        '{"chat_template":"changed"}', encoding="utf-8"
    )
    result = run(setup)
    assert result["status"] == "invalid"
    assert "tokenizers/templates" in result["tasks"][1]["errors"][0]


def test_comparison_loads_saved_reports_and_requires_exact_task_set(setup, tmp_path):
    baseline, candidate = reports(run(setup))
    for mapping, role in ((baseline, "base"), (candidate, "candidate")):
        for name, report in mapping.items():
            path = tmp_path / f"{role}-{name}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            mapping[name] = path
    assert compare_regression_suite(setup[2], baseline, candidate)["accepted"]
    candidate["extra"] = candidate["general"]
    result = compare_regression_suite(setup[2], baseline, candidate)
    assert result["status"] == "invalid"
    assert "exactly match" in result["errors"][0]


def test_changed_data_or_model_and_wrong_evaluator_artifact_fail_closed(setup):
    original = setup[4]

    def wrong_model(*args, **kwargs):
        report = original(*args, **kwargs)
        report["model_identity"] = compute_model_identity(setup[0][2])
        return report

    result = run_regression_suite(
        setup[0][0], setup[0][1], setup[2], evaluator=wrong_model
    )
    assert result["status"] == "invalid"
    assert "requested artifact" in result["tasks"][0]["errors"][0]
    baseline, candidate = reports(run(setup))
    setup[1].write_text('{"instruction":"different"}', encoding="utf-8")
    assert (
        compare_regression_suite(setup[2], baseline, candidate)["status"] == "invalid"
    )


def test_evaluator_failure_does_not_skip_other_tasks(setup):
    original = setup[4]

    def broken(model, *args, **kwargs):
        if kwargs["task"] == "multiple-choice" and model == setup[0][1]:
            raise RuntimeError("runtime unavailable")
        return original(model, *args, **kwargs)

    result = run_regression_suite(setup[0][0], setup[0][1], setup[2], evaluator=broken)
    assert result["status"] == "invalid"
    assert "runtime unavailable" in result["tasks"][0]["errors"][0]
    assert result["tasks"][1]["accepted"]


def test_default_shared_runner_preserves_record_and_setting_provenance(
    setup, monkeypatch
):
    from saddlellm.evaluation import MultipleChoiceEvaluator

    seen = []

    def backend(request):
        seen.append(request)
        return {
            "task_type": "multiple_choice",
            "model": request.model,
            "data_sha256": hashlib.sha256(Path(request.data).read_bytes()).hexdigest(),
            "samples": 1,
            "accuracy": 0.8,
            "details": [{"record_id": "one", "correct": True}],
        }

    monkeypatch.setattr(MultipleChoiceEvaluator, "evaluate_multiple_choice", backend)
    setup[2]["tasks"] = setup[2]["tasks"][:1]
    result = run_regression_suite(setup[0][0], setup[0][1], setup[2])
    assert result["accepted"]
    assert [request.max_length for request in seen] == [128, 128]
    assert [request.split for request in seen] == ["test", "test"]
    assert all(request.local_files_only for request in seen)
    assert (
        result["tasks"][0]["baseline_report"]["evaluation_identity"]["source_records"]
        == 1
    )


def test_separate_adapter_is_in_identity_and_only_changes_artifact_options(
    setup, tmp_path
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"trained-adapter")
    result = run(setup, candidate_adapter=str(adapter))
    assert result["accepted"]
    assert result["candidate_model_identity"]["adapter"]["status"] == "verified"
    assert "adapter" not in setup[3][0]
    assert setup[3][1]["adapter"] == str(adapter)


def test_shared_wrong_split_or_limit_is_invalid_even_when_both_reports_agree(setup):
    baseline, candidate = reports(run(setup))
    for mapping in (baseline, candidate):
        mapping["general"]["evaluation_identity"]["split"] = "validation"
    result = compare_regression_suite(setup[2], baseline, candidate)
    assert result["status"] == "invalid"
    assert "identity split mismatch" in result["tasks"][0]["errors"][0]


def test_data_changing_during_evaluation_fails_closed(setup):
    original = setup[4]

    def changing(*args, **kwargs):
        report = original(*args, **kwargs)
        setup[1].write_text(
            setup[1].read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        return report

    result = run_regression_suite(
        setup[0][0], setup[0][1], setup[2], evaluator=changing
    )
    assert result["status"] == "invalid"
    assert "changed while evaluating" in result["tasks"][0]["errors"][0]


def test_accuracy_tolerance_boundary_does_not_fail_on_float_subtraction(setup):
    original = setup[4]

    def boundary(model, *args, **kwargs):
        report = original(model, *args, **kwargs)
        if model == setup[0][1]:
            report["accuracy"] = 0.78
        return report

    assert run_regression_suite(setup[0][0], setup[0][1], setup[2], evaluator=boundary)[
        "accepted"
    ]


@pytest.mark.parametrize("source", ["data", "weights", "tokenizer"])
def test_report_output_cannot_overwrite_evaluation_inputs(setup, source):
    model = Path(setup[0][1])
    target = {
        "data": setup[1],
        "weights": model / "model.safetensors",
        "tokenizer": model / "tokenizer_config.json",
    }[source]
    original = target.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        run(setup, output=target)
    assert target.read_bytes() == original
    assert not setup[3]


def test_report_output_cannot_overwrite_suite_or_saved_evaluation_report(
    setup, tmp_path
):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(setup[2]), encoding="utf-8")
    original = suite_path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        run_regression_suite(
            setup[0][0], setup[0][1], suite_path, evaluator=setup[4], output=suite_path
        )
    assert suite_path.read_bytes() == original
    baseline, candidate = reports(run(setup))
    report_path = tmp_path / "base-evaluation.json"
    report_path.write_text(json.dumps(baseline["general"]), encoding="utf-8")
    baseline["general"] = report_path
    with pytest.raises(ValueError, match="must not overwrite"):
        compare_regression_suite(setup[2], baseline, candidate, output=report_path)


def test_same_baseline_and_candidate_artifact_is_valid_and_zero_delta(setup):
    result = run_regression_suite(
        setup[0][0], setup[0][0], setup[2], evaluator=setup[4]
    )
    assert result["accepted"]
    assert (
        result["baseline_model_identity"]["fingerprint"]
        == result["candidate_model_identity"]["fingerprint"]
    )
    assert all(
        metric["delta"] == 0
        for task in result["tasks"]
        for metric in task["metrics"].values()
    )


def tokenizer_pair(setup):
    payload = {
        "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1}},
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "added_tokens": [],
        "truncation": None,
        "padding": None,
    }
    for model in setup[0][:2]:
        (Path(model) / "tokenizer.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return payload


def test_tokenizer_runtime_padding_truncation_and_json_layout_are_comparable(setup):
    payload = tokenizer_pair(setup)
    payload["truncation"] = {
        "direction": "Right",
        "max_length": 32,
        "strategy": "LongestFirst",
        "stride": 0,
    }
    payload["padding"] = {"strategy": "BatchLongest", "pad_id": 0}
    (Path(setup[0][1]) / "tokenizer.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = run(setup)
    assert result["accepted"]
    baseline, candidate = reports(result)
    left = baseline["domain"]["tokenizer_semantic_identity"]
    right = candidate["domain"]["tokenizer_semantic_identity"]
    assert left["fingerprint"] == right["fingerprint"]
    assert left["files"][0]["source_sha256"] != right["files"][0]["source_sha256"]
    # Persisted signatures remain usable when model artifacts are offline.
    for model in setup[0][:2]:
        (Path(model) / "tokenizer.json").unlink()
    assert compare_regression_suite(setup[2], baseline, candidate)["accepted"]


@pytest.mark.parametrize(
    "field, changed",
    [
        ("model", {"type": "WordLevel", "vocab": {"a": 1, "b": 0}}),
        ("normalizer", {"type": "NFKC"}),
        ("pre_tokenizer", {"type": "WhitespaceSplit"}),
        ("post_processor", {"type": "TemplateProcessing"}),
        ("decoder", {"type": "WordPiece"}),
        ("added_tokens", [{"id": 2, "content": "special", "special": True}]),
    ],
)
def test_tokenizer_actual_semantic_changes_are_rejected(setup, field, changed):
    payload = tokenizer_pair(setup)
    payload[field] = changed
    (Path(setup[0][1]) / "tokenizer.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = run(setup)
    assert result["status"] == "invalid"
    assert "tokenizers/templates" in result["tasks"][1]["errors"][0]


def test_tokenizer_only_top_level_runtime_keys_are_ignored(setup):
    payload = tokenizer_pair(setup)
    payload["model"]["padding"] = "meaningful-nested-field"
    (Path(setup[0][1]) / "tokenizer.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(model_fingerprint="0" * 64),
        lambda value: value.update(fingerprint="0" * 64),
        lambda value: value["files"][0].update(source_sha256="0" * 64),
        lambda value: value["files"][0].update(semantic_sha256="0" * 64),
        lambda value: value.update(files=[]),
    ],
)
def test_invalid_or_unbound_tokenizer_signatures_fail_closed(setup, mutation):
    tokenizer_pair(setup)
    baseline, candidate = reports(run(setup))
    mutation(candidate["domain"]["tokenizer_semantic_identity"])
    result = compare_regression_suite(setup[2], baseline, candidate)
    assert result["status"] == "invalid"
    assert "tokenizer semantic signature" in result["tasks"][1]["errors"][0]


def test_output_planning_guard_never_fingerprints_models_or_reads_dataset(
    setup, monkeypatch, tmp_path
):
    import saddlellm.evaluation.RegressionSuite as regression

    monkeypatch.setattr(
        regression,
        "compute_model_identity",
        lambda *a, **k: pytest.fail("planning fingerprinted weights"),
    )
    monkeypatch.setattr(
        regression, "_provenance", lambda *a, **k: pytest.fail("planning read dataset")
    )
    validate_regression_output(
        setup[2],
        tmp_path / "report.json",
        baseline_model=setup[0][0],
        candidate_model=setup[0][1],
    )
    with pytest.raises(ValueError, match="outside input checkpoint"):
        validate_regression_output(
            setup[2], Path(setup[0][0]) / "config.json", baseline_model=setup[0][0]
        )
    with pytest.raises(ValueError, match="must not overwrite"):
        validate_regression_output(setup[2], setup[1])


def test_tokenizer_unknown_binary_formats_require_exact_raw_bytes(setup):
    for model in setup[0][:2]:
        (Path(model) / "vocab.model").write_bytes(b"binary-token-mapping\x00\xff")
    assert run(setup)["accepted"]
    (Path(setup[0][1]) / "vocab.model").write_bytes(b"binary-token-mapping\x00\xfe")
    assert run(setup)["status"] == "invalid"


def test_runtime_named_keys_in_tokenizer_config_are_preserved(setup):
    (Path(setup[0][1]) / "tokenizer_config.json").write_text(
        '{"chat_template":"same","padding":{"meaningful":"preserve"}}', encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


def mirrored_tokenizer_runtime(setup):
    payload = tokenizer_pair(setup)
    payload["truncation"] = {
        "direction": "Right",
        "max_length": 32,
        "strategy": "LongestFirst",
        "stride": 0,
    }
    (Path(setup[0][1]) / "tokenizer.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    config = {
        "chat_template": "same",
        "max_length": 32,
        "truncation_side": "right",
        "stride": 0,
        "truncation_strategy": "longest_first",
    }
    (Path(setup[0][1]) / "tokenizer_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return payload, config


def test_hf_load_save_mirrored_default_runtime_metadata_is_equivalent(setup):
    mirrored_tokenizer_runtime(setup)
    # The evaluator passes max_length=128, overriding the cached prior value 32.
    assert setup[2]["tasks"][1]["options"]["max_length"] == 128
    assert run(setup)["accepted"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("truncation_side", "left"),
        ("stride", 1),
        ("truncation_strategy", "only_first"),
        ("max_length", 33),
    ],
)
def test_nondefault_or_unmatched_runtime_config_is_preserved(setup, key, value):
    _, config = mirrored_tokenizer_runtime(setup)
    config[key] = value
    (Path(setup[0][1]) / "tokenizer_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


def test_nondefault_backend_truncation_side_is_preserved_without_mirrored_config(setup):
    payload = tokenizer_pair(setup)
    payload["truncation"] = {
        "direction": "Left",
        "max_length": 32,
        "strategy": "LongestFirst",
        "stride": 0,
    }
    (Path(setup[0][1]) / "tokenizer.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


def test_tokenizer_config_max_length_without_matching_backend_cache_is_preserved(setup):
    tokenizer_pair(setup)
    (Path(setup[0][1]) / "tokenizer_config.json").write_text(
        '{"chat_template":"same","max_length":32}', encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


def special_token_pair(setup):
    source = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "additional_special_tokens": ["<extra-a>", "<extra-b>"],
    }
    defaults = {
        "lstrip": False,
        "rstrip": False,
        "single_word": False,
        "normalized": False,
    }
    candidate = {
        key: (
            [{"content": item, **defaults} for item in value]
            if isinstance(value, list)
            else {"content": value, **defaults}
        )
        for key, value in source.items()
    }
    for model, payload in zip(setup[0][:2], (source, candidate)):
        (Path(model) / "special_tokens_map.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return source, candidate


def test_special_token_strings_and_default_added_token_maps_are_equivalent(setup):
    special_token_pair(setup)
    assert run(setup)["accepted"]


def test_special_tokens_explicit_default_special_flag_and_added_token_type_are_equivalent(
    setup,
):
    _, candidate = special_token_pair(setup)
    candidate["bos_token"].update(special=True, __type="AddedToken")
    (Path(setup[0][1]) / "special_tokens_map.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    assert run(setup)["accepted"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("lstrip", True),
        ("rstrip", True),
        ("single_word", True),
        ("normalized", True),
        ("special", False),
        ("content", "<different>"),
        ("id", 999),
        ("unknown_attribute", False),
    ],
)
def test_special_token_true_flags_content_ids_or_unknown_fields_are_preserved(
    setup, field, value
):
    _, candidate = special_token_pair(setup)
    candidate["bos_token"][field] = value
    (Path(setup[0][1]) / "special_tokens_map.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


def test_additional_special_token_order_and_semantics_are_preserved(setup):
    _, candidate = special_token_pair(setup)
    candidate["additional_special_tokens"].reverse()
    (Path(setup[0][1]) / "special_tokens_map.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"


def test_special_token_normalization_does_not_weaken_template_or_vocab_checks(setup):
    special_token_pair(setup)
    tokenizer_pair(setup)
    payload = json.loads(
        (Path(setup[0][1]) / "tokenizer.json").read_text(encoding="utf-8")
    )
    payload["model"]["vocab"]["a"] = 99
    (Path(setup[0][1]) / "tokenizer.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert run(setup)["status"] == "invalid"
