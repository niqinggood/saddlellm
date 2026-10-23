"""Evidence identity, pairing and read-only boundaries of the workbench."""

import json

import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for study_id in ("study-one", "study-two"):
        study = root / "outputs" / "studies" / study_id
        write_json(
            study / "resolved_config.json",
            {
                "model": {"name": "fixture-model"},
                "data": {"train_samples": 4},
                "evaluation": {"k": 2},
                "matrix": {"seeds": [42]},
            },
        )
        write_json(
            study / "experiment_summary.json",
            {
                "status": "completed",
                "completed": 2,
                "run_ids": ["base_direct_seed_42", "sft_direct_from_a_seed_42"],
            },
        )
        write_json(
            study / "doctor.json",
            {
                "status": "ok",
                "checks": [{"name": "python", "ok": True, "value": "fixture"}],
                "errors": [],
                "warnings": [],
            },
        )
        write_json(
            study / "launch_command.json",
            [
                "python",
                "-m",
                "saddlellm.cli",
                "experiment",
                "--project",
                str(root),
                "all",
            ],
        )
        write_json(
            study / "data_manifest_seed_42.json",
            {"seed": 42, "audit": {}, "selected": {}},
        )
        comparison = []
        for run_id, first_predictions in (
            ("base_direct_seed_42", ["disagree", "agree", "agree", "disagree"]),
            ("sft_direct_from_a_seed_42", ["agree", "disagree", "agree", "disagree"]),
        ):
            records = [
                {
                    "example_id": f"A:Q{i}::org",
                    "case_key": f"Q{i}::org",
                    "task": "A",
                    "question_id": f"Q{i}",
                    "organization": "org",
                    "expected": ["agree"],
                    "user_prompt": f"{study_id}: question {i} <img src=x onerror=alert(1)>",
                    # Draw two deliberately disagrees, so pairing must use draw one.
                    "responses": [
                        f"<answer>{label}</answer>",
                        "<answer>agree</answer>",
                    ],
                    "predictions": [[label], ["agree"]],
                    "format_valid": [True, True],
                }
                for i, label in enumerate(first_predictions)
            ]
            metrics = {"accuracy": 0.5, "macro_f1": 0.4, "num_examples": 4, "k": 2}
            method = "base" if run_id.startswith("base") else "sft"
            write_json(
                study / "runs" / run_id / "run_result.json",
                {
                    "status": "completed",
                    "run_id": run_id,
                    "method": method,
                    "source": "-" if method == "base" else "A",
                    "seed": 42,
                    "post": {"benchmark_a": metrics},
                    "pre": {"benchmark_a": metrics},
                    "training": {},
                    "duration_s": 12.5,
                },
            )
            write_json(
                study / "runs" / run_id / "final_benchmark_a.json",
                {"metrics": metrics, "records": records},
            )
            comparison.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "source": "A",
                    "target": "A",
                    "post_accuracy": 0.5,
                }
            )
        write_json(study / "comparison.json", {"num_runs": 2, "rows": comparison})
        write_json(
            study / "issue_log.json",
            [
                {
                    "id": "fixture",
                    "status": "recorded",
                    "description_zh": "Fixture issue.",
                }
            ],
        )
        (study / "analysis").mkdir()
        (study / "analysis" / "REPORT.html").write_text(
            "<h1>Fixture report</h1>", encoding="utf-8"
        )
    return root


@pytest.fixture
def client(project):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from saddlellm.runtime.WorkbenchAPI import create_workbench_app

    return TestClient(create_workbench_app(str(project), study="study-one"))


def test_api_has_explicit_studies_and_initial_selection(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "saddlellm-workbench",
        "read_only": True,
    }
    payload = client.get("/api/workbench/studies").json()
    assert {row["id"] for row in payload["studies"]} == {"study-one", "study-two"}
    assert payload["initial_study"] == "study-one"
    assert payload["read_only"] is True
    assert client.get("/").status_code == 200
    for asset in ("workbench.css", "workbench.js"):
        assert client.get(f"/assets/{asset}").status_code == 200


def test_pairing_uses_same_study_and_first_draw(client):
    url = "/api/workbench/studies/study-one/predictions"
    params = {"run_id": "sft_direct_from_a_seed_42", "task": "A", "limit": 10}
    payload = client.get(url, params=params).json()
    assert payload["total"] == 4
    assert len(payload["rows"]) == 4
    assert {row["outcome"] for row in payload["rows"]} == {
        "improved",
        "regressed",
        "both_correct",
        "both_wrong",
    }
    assert all("study-one" in row["input"] for row in payload["rows"])
    for outcome in ("improved", "regressed", "both_correct", "both_wrong"):
        page = client.get(url, params={**params, "outcome": outcome}).json()
        assert page["total"] == 1
        assert len(page["rows"]) == 1
        assert page["rows"][0]["outcome"] == outcome
    page = client.get(url, params={**params, "offset": 2, "limit": 1}).json()
    assert page["total"] == 4
    assert len(page["rows"]) == 1


def test_missing_baseline_does_not_fabricate_regressions(client, project):
    path = (
        project
        / "outputs/studies/study-one/runs/base_direct_seed_42/final_benchmark_a.json"
    )
    path.unlink()
    response = client.get(
        "/api/workbench/studies/study-one/predictions",
        params={
            "run_id": "sft_direct_from_a_seed_42",
            "task": "A",
            "outcome": "regressed",
        },
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_mismatched_gold_is_not_compared(client, project):
    path = (
        project
        / "outputs/studies/study-one/runs/base_direct_seed_42/final_benchmark_a.json"
    )
    content = json.loads(path.read_text(encoding="utf-8"))
    content["records"][0]["expected"] = ["disagree"]
    write_json(path, content)
    page = client.get(
        "/api/workbench/studies/study-one/predictions",
        params={
            "run_id": "sft_direct_from_a_seed_42",
            "task": "A",
            "outcome": "improved",
        },
    ).json()
    assert page["total"] == 0


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 101},
        {"limit": 0},
        {"offset": -1},
        {"task": "C"},
        {"outcome": "bogus"},
    ],
)
def test_api_rejects_invalid_prediction_filters(client, params):
    response = client.get(
        "/api/workbench/studies/study-one/predictions",
        params={
            "run_id": "sft_direct_from_a_seed_42",
            **params,
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        "/api/workbench/studies/no-such-study",
        "/api/workbench/studies/study-one/runs/no-such-run",
        "/api/workbench/studies/study-one/artifacts/no-such-file",
        "/assets/chat.js",
        "/assets/..%5Ccli.py",
    ],
)
def test_unknown_ids_and_assets_are_not_arbitrary_files(client, path):
    assert client.get(path).status_code == 404


def test_workbench_has_no_mutating_or_training_route(client, project):
    before = {
        str(path.relative_to(project)): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert client.post("/api/workbench/studies").status_code == 405
    assert client.post("/api/workbench/train").status_code == 404
    client.get("/api/workbench/studies/study-one")
    after = {
        str(path.relative_to(project)): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_cli_launch_uses_workbench_factory_and_clean_interrupt(monkeypatch, project):
    import uvicorn

    from saddlellm import cli

    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs))
    )
    assert (
        cli.main(
            [
                "workbench",
                "--project",
                str(project),
                "--study",
                "study-one",
                "--port",
                "7999",
            ]
        )
        == 0
    )
    assert calls[0][0].state.initial_study == "study-one"
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 7999

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn, "run", interrupted)
    assert cli.main(["workbench", "--project", str(project)]) == 0


def test_baseline_with_different_seed_is_not_paired(client, project):
    path = (
        project / "outputs/studies/study-one/runs/base_direct_seed_42/run_result.json"
    )
    content = json.loads(path.read_text(encoding="utf-8"))
    content["seed"] = 7
    write_json(path, content)
    page = client.get(
        "/api/workbench/studies/study-one/predictions",
        params={
            "run_id": "sft_direct_from_a_seed_42",
            "task": "A",
            "outcome": "improved",
        },
    ).json()
    assert page["total"] == 0


def test_baseline_with_different_prompt_is_not_paired(client, project):
    path = (
        project
        / "outputs/studies/study-one/runs/base_direct_seed_42/final_benchmark_a.json"
    )
    content = json.loads(path.read_text(encoding="utf-8"))
    content["records"][0]["user_prompt"] = "A different task on the same ID"
    write_json(path, content)
    page = client.get(
        "/api/workbench/studies/study-one/predictions",
        params={
            "run_id": "sft_direct_from_a_seed_42",
            "task": "A",
            "outcome": "improved",
        },
    ).json()
    assert page["total"] == 0


def test_raw_sha_and_cache_prevent_changed_data_from_entering_history(client, project):
    import hashlib

    path = project / "data" / "topic" / "train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "problem": json.dumps(
                    {
                        "question_id": "Q0",
                        "question": "Raw consultation",
                        "organization": "org",
                        "response": "Real response",
                    }
                ),
                "answer": ["A"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = project / "outputs/studies/study-one/data_manifest_seed_42.json"
    write_json(
        manifest,
        {
            "seed": 42,
            "selected": {},
            "audit": {
                "dataset_sources": {
                    "B": {
                        "files": [
                            {
                                "path": str(path),
                                "split": "train",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                }
            },
        },
    )
    url = "/api/workbench/studies/study-one/data"
    params = {"source": "B", "split": "original_train"}
    original = client.get(url, params=params).json()
    assert original["total"] == 1
    assert original["rows"][0]["raw"]["answer"] == ["A"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = client.get(url, params=params).json()
    assert changed["total"] == 0
    assert any("SHA256" in note for note in changed["notes"])


def test_manifest_cannot_read_files_outside_project(client, project, tmp_path):
    path = tmp_path / "private.jsonl"
    path.write_text('{"problem":"{}","answer":["secret"]}\n', encoding="utf-8")
    manifest = project / "outputs/studies/study-one/data_manifest_seed_42.json"
    write_json(
        manifest,
        {
            "seed": 42,
            "selected": {},
            "audit": {
                "dataset_sources": {
                    "B": {
                        "files": [{"path": str(path), "split": "train"}],
                    }
                }
            },
        },
    )
    page = client.get(
        "/api/workbench/studies/study-one/data",
        params={
            "source": "B",
            "split": "original_train",
        },
    ).json()
    assert page["total"] == 0
    assert "secret" not in json.dumps(page)


def test_artifact_is_downloaded_from_fixed_allowlist(client):
    response = client.get("/api/workbench/studies/study-one/artifacts/report-html")
    assert response.status_code == 200
    assert response.text == "<h1>Fixture report</h1>"
    assert response.headers["content-disposition"].startswith("attachment;")


@pytest.mark.parametrize("missing", ["expected", "predictions", "user_prompt"])
def test_incomplete_record_is_not_a_claimed_improvement(client, project, missing):
    path = (
        project
        / "outputs/studies/study-one/runs/sft_direct_from_a_seed_42/final_benchmark_a.json"
    )
    content = json.loads(path.read_text(encoding="utf-8"))
    content["records"][0].pop(missing)
    write_json(path, content)
    response = client.get(
        "/api/workbench/studies/study-one/predictions",
        params={
            "run_id": "sft_direct_from_a_seed_42",
            "task": "A",
            "outcome": "improved",
        },
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_credentials_are_not_in_public_config_or_command(client, project):
    study = project / "outputs/studies/study-one"
    config = json.loads((study / "resolved_config.json").read_text(encoding="utf-8"))
    config["model"]["api_key"] = "private-api-value"
    config["model"]["hf_token"] = "private-hf-value"
    write_json(study / "resolved_config.json", config)
    write_json(
        study / "launch_command.json",
        ["python", "--api-key", "private-cli-value", "--password=private-password"],
    )
    response = client.get("/api/workbench/studies/study-one")
    assert response.status_code == 200
    assert all(
        value not in response.text
        for value in (
            "private-api-value",
            "private-hf-value",
            "private-cli-value",
            "private-password",
        )
    )
