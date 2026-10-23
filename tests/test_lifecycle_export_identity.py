from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from saddlellm.evaluation.EvaluationComparison import build_evaluation_comparison
from saddlellm.evaluation.ModelIdentity import compute_model_identity
from saddlellm.runtime.ModelExporter import ModelExporter, ModelExportRequest


def model(root, weights=b"AAAA"):
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(weights)
    (root / "tokenizer.json").write_text('{"vocab":{"red":1}}', encoding="utf-8")
    return root


def gate(source):
    return {
        "accepted": True,
        "candidate_model_identity": compute_model_identity(source),
    }


def manifest(result):
    return json.loads(open(result.manifest_path, encoding="utf-8").read())


@pytest.mark.parametrize("export_format", ["saddle", "native"])
def test_cli_exposes_native_export_without_requiring_gate(
    tmp_path, export_format, capsys
):
    from test_native_model_lifecycle import _save_tiny_tokenizer, _tiny_saddle_model

    from saddlellm import cli

    source = tmp_path / "native-source"
    _tiny_saddle_model().eval().save_pretrained(source)
    _save_tiny_tokenizer(source)
    destination = tmp_path / "native-release"
    assert (
        cli.main(
            [
                "export",
                str(source),
                str(destination),
                "--format",
                export_format,
                "--no-merge-lora",
                "--local-files-only",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    payload = json.loads((destination / "release_manifest.json").read_text("utf-8"))
    assert payload["format"] == "saddle"
    assert payload["source_type"] == "native_checkpoint"
    assert payload["gate"] is None


def test_ordinary_export_has_no_gate_and_hashes_only_finished_model(
    tmp_path, monkeypatch
):
    from saddlellm.evaluation import ModelIdentity

    source = model(tmp_path / "model")
    original = ModelIdentity._digest_file
    observed = []

    def record(path):
        observed.append(path)
        return original(path)

    monkeypatch.setattr(ModelIdentity, "_digest_file", record)
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(tmp_path / "release"))
    )
    payload = manifest(result)
    assert payload["gate"] is None
    assert payload["model_identity"]["status"] == "verified"
    assert payload["source_model_identity"] is None
    assert "model.safetensors" not in payload["hashes"]
    assert all(source not in path.parents for path in observed)


def test_equal_size_different_weights_have_different_release_ids(tmp_path):
    first = model(tmp_path / "a", b"AAAA")
    second = model(tmp_path / "b", b"BBBB")
    a = ModelExporter.export(
        ModelExportRequest(str(first), str(tmp_path / "release-a"))
    )
    b = ModelExporter.export(
        ModelExportRequest(str(second), str(tmp_path / "release-b"))
    )
    assert a.release_id != b.release_id
    assert manifest(a)["inventory"] == manifest(b)["inventory"]
    assert manifest(a)["hashes"] == manifest(b)["hashes"]


@pytest.mark.parametrize("change_at_same_path", [False, True])
def test_gate_cannot_export_other_model_or_replaced_weights(
    tmp_path, change_at_same_path
):
    first = model(tmp_path / "a", b"AAAA")
    evidence = gate(first)
    if change_at_same_path:
        (first / "model.safetensors").write_bytes(b"BBBB")
        source = first
    else:
        source = model(tmp_path / "b", b"BBBB")
    destination = tmp_path / "release"
    with pytest.raises(ValueError, match="does not match"):
        ModelExporter.export(
            ModelExportRequest(str(source), str(destination), require_gate=True),
            gate=evidence,
        )
    assert not destination.exists()


def test_gate_failure_does_not_overwrite_existing_destination(tmp_path):
    first = model(tmp_path / "a")
    second = model(tmp_path / "b", b"BBBB")
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ModelExporter.export(
            ModelExportRequest(
                str(second), str(destination), require_gate=True, overwrite=True
            ),
            gate=gate(first),
        )
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_strict_export_allows_relocated_identical_artifact(tmp_path):
    first = model(tmp_path / "a")
    shutil.copytree(first, tmp_path / "copy")
    result = ModelExporter.export(
        ModelExportRequest(
            str(tmp_path / "copy"), str(tmp_path / "release"), require_gate=True
        ),
        gate=gate(first),
    )
    assert manifest(result)["gate"]["accepted"]
    assert manifest(result)["gate_binding"]["status"] == "matched"


def test_optional_wrong_gate_exports_without_claiming_current_model_passed(tmp_path):
    first = model(tmp_path / "a")
    second = model(tmp_path / "b", b"BBBB")
    result = ModelExporter.export(
        ModelExportRequest(str(second), str(tmp_path / "release")), gate=gate(first)
    )
    payload = manifest(result)
    assert payload["gate"]["accepted"] is False
    assert payload["gate"]["reported_accepted"] is True
    assert payload["gate_binding"]["status"] == "mismatch"
    assert result.warnings


def test_old_gate_is_optional_but_explicit_strict_export_rejects_it(tmp_path):
    source = model(tmp_path / "model")
    old = {"accepted": True, "model": str(source)}
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(tmp_path / "ordinary")), gate=old
    )
    assert not manifest(result)["gate"]["accepted"]
    assert manifest(result)["gate_binding"]["status"] == "unbound"
    with pytest.raises(ValueError, match="Re-run evaluate/compare"):
        ModelExporter.export(
            ModelExportRequest(
                str(source), str(tmp_path / "strict"), require_gate=True
            ),
            gate=old,
        )
    assert not (tmp_path / "strict").exists()


def test_strict_remote_reference_does_not_download_or_prepare_destination(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("strict unresolved export must not materialize or download a model")

    monkeypatch.setattr(ModelExporter, "_materialize_model", forbidden)
    destination = tmp_path / "remote"
    with pytest.raises(ValueError, match="local model"):
        ModelExporter.export(
            ModelExportRequest(
                "unavailable-org/unavailable-model", str(destination), require_gate=True
            ),
            gate={"accepted": True},
        )
    assert not destination.exists()


def test_materialized_release_identity_describes_output_not_adapter_inputs(
    tmp_path, monkeypatch
):
    base = model(tmp_path / "base")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"ADPT")

    def materialize(source, destination, request):
        shutil.copy2(base / "config.json", destination / "config.json")
        shutil.copy2(base / "tokenizer.json", destination / "tokenizer.json")
        (destination / "model.safetensors").write_bytes(b"MERG")
        return []

    monkeypatch.setattr(ModelExporter, "_merge_adapter", materialize)
    destination = tmp_path / "merged"
    evidence = gate(adapter)
    result = ModelExporter.export(
        ModelExportRequest(str(adapter), str(destination), require_gate=True),
        gate=evidence,
    )
    payload = manifest(result)
    assert (
        payload["source_model_identity"]["fingerprint"]
        == evidence["candidate_model_identity"]["fingerprint"]
    )
    assert (
        payload["model_identity"]["fingerprint"]
        == compute_model_identity(destination)["fingerprint"]
    )
    assert (
        payload["model_identity"]["fingerprint"]
        != payload["source_model_identity"]["fingerprint"]
    )
    assert payload["gate"]["binding_target"] == "source_model_identity"


def test_public_weight_hash_option_reuses_finished_identity_digest(
    tmp_path, monkeypatch
):
    source = model(tmp_path / "model")
    original = ModelExporter._sha256

    def non_weight_hash(path):
        assert path.name != "model.safetensors", (
            "weight digest was already computed by identity"
        )
        return original(path)

    monkeypatch.setattr(ModelExporter, "_sha256", non_weight_hash)
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(tmp_path / "release"), hash_weights=True)
    )
    assert len(manifest(result)["hashes"]["model.safetensors"]) == 64


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["before-copy", "after-copy", "copied-weights"])
def test_export_rejects_mid_copy_changes_without_publishing_stale_gate(
    tmp_path, monkeypatch, existing, mutation,
):
    source = model(tmp_path / "source")
    (source / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"TokenizersBackend"}', encoding="utf-8",
    )
    evidence = gate(source)
    destination = tmp_path / "release"
    if existing:
        destination.mkdir()
        (destination / "keep.txt").write_bytes(b"previous-release")
        (destination / "release_manifest.json").write_text('{"gate":{"accepted":true}}', encoding="utf-8")
    previous = {path.name: path.read_bytes() for path in destination.iterdir()} if existing else {}
    original_copy = ModelExporter._copy_local_release

    def copy(current_source, staged):
        if mutation == "before-copy":
            (current_source / "model.safetensors").write_bytes(b"NEW!")
        original_copy(current_source, staged)
        if mutation == "after-copy":
            (current_source / "model.safetensors").write_bytes(b"NEW!")
        elif mutation == "copied-weights":
            (staged / "model.safetensors").write_bytes(b"BAD!")

    monkeypatch.setattr(ModelExporter, "_copy_local_release", copy)
    with pytest.raises(ValueError, match="content|weights"):
        ModelExporter.export(
            ModelExportRequest(str(source), str(destination), require_gate=True, overwrite=True),
            gate=evidence,
        )
    if existing:
        assert {path.name: path.read_bytes() for path in destination.iterdir()} == previous
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".release.export-*"))


def test_export_rechecks_source_after_final_inventory(tmp_path, monkeypatch):
    source = model(tmp_path / "source")
    evidence = gate(source)
    original_inventory = ModelExporter._inventory

    def inventory(staged, **kwargs):
        result = original_inventory(staged, **kwargs)
        (source / "model.safetensors").write_bytes(b"late-save")
        return result

    monkeypatch.setattr(ModelExporter, "_inventory", inventory)
    destination = tmp_path / "release"
    with pytest.raises(ValueError, match="changed during export"):
        ModelExporter.export(ModelExportRequest(str(source), str(destination), require_gate=True), gate=evidence)
    assert not destination.exists()


@pytest.mark.parametrize("changed_component", ["base", "adapter"])
def test_merge_rejects_changes_to_either_input_without_overwriting_target(
    tmp_path, monkeypatch, changed_component,
):
    base = model(tmp_path / "base")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"ADPT")
    evidence = gate(adapter)
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "keep.txt").write_bytes(b"previous-release")

    def merge(source, staged, request):
        shutil.copy2(base / "config.json", staged / "config.json")
        (staged / "model.safetensors").write_bytes(b"MERG")
        if changed_component == "base":
            (base / "model.safetensors").write_bytes(b"NEW!")
        else:
            (adapter / "adapter_model.safetensors").write_bytes(b"NEW!")
        return []

    monkeypatch.setattr(ModelExporter, "_merge_adapter", merge)
    with pytest.raises(ValueError, match="source or adapter content changed"):
        ModelExporter.export(
            ModelExportRequest(str(adapter), str(destination), require_gate=True, overwrite=True), gate=evidence,
        )
    assert list(destination.iterdir()) == [destination / "keep.txt"]
    assert (destination / "keep.txt").read_bytes() == b"previous-release"


def test_unchanged_local_serialization_conversion_retains_a_valid_source_gate(tmp_path, monkeypatch):
    source = model(tmp_path / "source")
    (source / "model.safetensors").rename(source / "pytorch_model.bin")
    evidence = gate(source)

    def convert(current_source, staged, request):
        shutil.copy2(Path(current_source) / "config.json", staged / "config.json")
        (staged / "model.safetensors").write_bytes(b"new-serialization")

    monkeypatch.setattr(ModelExporter, "_materialize_model", convert)
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(tmp_path / "release"), require_gate=True), gate=evidence,
    )
    payload = manifest(result)
    assert payload["gate"]["accepted"] is True
    assert payload["source_model_identity"]["fingerprint"] != payload["model_identity"]["fingerprint"]
    assert payload["model_identity"]["model"]["reference"] == result.model_path
    assert any("transformed the evaluated source" in warning for warning in result.warnings)


def test_optional_gate_with_unresolved_remote_source_still_allows_materialization(tmp_path, monkeypatch):
    local = model(tmp_path / "local")
    evidence = gate(local)
    monkeypatch.setattr(
        ModelExporter, "_materialize_model",
        lambda reference, staged, request: ModelExporter._copy_local_release(local, staged),
    )
    result = ModelExporter.export(
        ModelExportRequest("unavailable-org/optional-reference", str(tmp_path / "release")), gate=evidence,
    )
    payload = manifest(result)
    assert result.status == "completed"
    assert payload["gate"]["accepted"] is False
    assert payload["gate_binding"]["status"] == "unbound"
    assert payload["source_model_identity"]["status"] == "unresolved"


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_publish_failure_restores_or_preserves_the_previous_release(
    tmp_path, monkeypatch, rollback_fails,
):
    source = model(tmp_path / "source")
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "keep.txt").write_bytes(b"previous-release")
    (destination / "release_manifest.json").write_bytes(b"previous-manifest")
    original_rename = Path.rename

    def rename(path, target):
        if path.parent.name.startswith(".release.export-"):
            raise OSError("publication failed")
        if rollback_fails and path.parent.name.startswith(".release.previous-"):
            raise OSError("rollback failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    error_type = RuntimeError if rollback_fails else OSError
    with pytest.raises(error_type, match="preserved at|publication failed") as error:
        ModelExporter.export(
            ModelExportRequest(str(source), str(destination), require_gate=True, overwrite=True), gate=gate(source),
        )
    if rollback_fails:
        assert not destination.exists()
        backups = list(tmp_path.glob(".release.previous-*/release"))
        assert len(backups) == 1
        previous = backups[0]
        assert str(previous) in str(error.value)
    else:
        previous = destination
        assert not list(tmp_path.glob(".release.previous-*"))
    assert (previous / "keep.txt").read_bytes() == b"previous-release"
    assert (previous / "release_manifest.json").read_bytes() == b"previous-manifest"
    assert not list(tmp_path.glob(".release.export-*"))


def test_successful_verified_overwrite_commits_release_and_removes_backup(tmp_path):
    source = model(tmp_path / "source")
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "old.txt").write_bytes(b"old")
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(destination), overwrite=True, require_gate=True), gate=gate(source),
    )
    assert manifest(result)["gate"]["accepted"] is True
    assert not (destination / "old.txt").exists()
    assert (destination / "model.safetensors").read_bytes() == b"AAAA"
    assert not list(tmp_path.glob(".release.previous-*"))


def evaluation(identity, *, correct):
    return {
        "model": "same/display/path",
        "task_type": "classification",
        "model_identity": identity,
        "details": [
            {
                "record_id": str(i),
                "question": f"q{i}",
                "answer": "yes",
                "correct": correct,
            }
            for i in range(8)
        ],
    }


def test_comparison_propagates_matching_content_identities(tmp_path):
    base = compute_model_identity(model(tmp_path / "base"))
    candidate = compute_model_identity(model(tmp_path / "candidate", b"BBBB"))
    result = build_evaluation_comparison(
        base_validation=evaluation(base, correct=False),
        candidate_validation=evaluation(candidate, correct=True),
        base_test=evaluation(base, correct=False),
        candidate_test=evaluation(candidate, correct=True),
    )
    assert result["accepted"] and result["candidate_model_identity"] == candidate
    assert result["base_model_identity"] == base
    assert result["model_identity_binding"]["status"] == "bound"


@pytest.mark.parametrize("side", ["base", "candidate"])
def test_comparison_rejects_same_path_but_different_validation_test_contents(
    tmp_path, side
):
    first = compute_model_identity(model(tmp_path / "a"))
    second = compute_model_identity(model(tmp_path / "b", b"BBBB"))
    args = {
        "base_validation": evaluation(first, correct=False),
        "candidate_validation": evaluation(first, correct=True),
        "base_test": evaluation(first, correct=False),
        "candidate_test": evaluation(first, correct=True),
    }
    args[f"{side}_test"]["model_identity"] = second
    with pytest.raises(ValueError, match="model identities differ"):
        build_evaluation_comparison(**args)


def test_partial_legacy_test_cannot_inherit_new_validation_identity(tmp_path):
    identity = compute_model_identity(model(tmp_path / "model"))
    old_test = evaluation(None, correct=True)
    with pytest.raises(ValueError, match="only bound in one report"):
        build_evaluation_comparison(
            base_validation=evaluation(identity, correct=False),
            candidate_validation=evaluation(identity, correct=True),
            base_test=evaluation(identity, correct=False),
            candidate_test=old_test,
        )
