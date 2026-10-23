from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from saddlellm.evaluation.ModelIdentity import (
    compute_model_identity,
    model_identities_match,
    verify_gate_model,
)


def model(root, weights=b"AAAA"):
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(weights)
    (root / "tokenizer.json").write_text('{"vocab":{"red":1}}', encoding="utf-8")
    return root


def test_content_identity_matches_relocated_artifact(tmp_path):
    first = model(tmp_path / "first")
    shutil.copytree(first, tmp_path / "copy")
    left = compute_model_identity(first)
    right = compute_model_identity(tmp_path / "copy")
    assert left["model"]["reference"] != right["model"]["reference"]
    assert model_identities_match(left, right)


@pytest.mark.parametrize("filename,value", [
    ("model.safetensors", b"BBBB"),
    ("config.json", b'{"model_type":"llama"}'),
    ("tokenizer.json", b'{"vocab":{"red":2}}'),
])
def test_inference_contents_change_identity_even_at_same_path(tmp_path, filename, value):
    root = model(tmp_path / "model")
    previous = compute_model_identity(root)
    (root / filename).write_bytes(value)
    assert not model_identities_match(previous, compute_model_identity(root))


def test_training_state_and_previous_checkpoints_do_not_change_model_identity(tmp_path):
    root = model(tmp_path / "model")
    previous = compute_model_identity(root)
    (root / "optimizer.pt").write_bytes(b"optimizer")
    (root / "training_summary.json").write_text('{"loss":9}', encoding="utf-8")
    model(root / "checkpoint-1", b"older weights")
    assert model_identities_match(previous, compute_model_identity(root))


def test_adapter_directory_and_explicit_base_adapter_are_identical(tmp_path):
    base = model(tmp_path / "base")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base), "r": 2}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"AAAA")
    explicit = compute_model_identity(base, adapter)
    assert model_identities_match(explicit, compute_model_identity(adapter))
    (adapter / "adapter_model.safetensors").write_bytes(b"BBBB")
    assert not model_identities_match(explicit, compute_model_identity(adapter))


def test_remote_base_identity_is_unresolved_without_download(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"unavailable-org/unavailable-model"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"AAAA")
    identity = compute_model_identity(adapter)
    assert identity["status"] == "unresolved" and identity["fingerprint"] is None
    assert identity["adapter"]["status"] == "verified"
    assert identity["model"]["files"] == []
    assert "no remote files" in identity["warnings"][0]


def test_identity_import_and_remote_reference_have_no_model_imports():
    command = "from saddlellm.evaluation.ModelIdentity import compute_model_identity; import sys; r=compute_model_identity('unavailable-org/unavailable-model'); assert r['status']=='unresolved'; assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'huggingface_hub' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)


def test_gate_accepts_same_contents_in_a_different_directory(tmp_path):
    root = model(tmp_path / "original")
    shutil.copytree(root, tmp_path / "copy")
    gate = {"accepted": True, "candidate_model_identity": compute_model_identity(root)}
    result = verify_gate_model(gate, compute_model_identity(tmp_path / "copy"), required=True)
    assert result["status"] == "matched" and result["accepted"]


def test_nested_gate_identity_and_legacy_warning(tmp_path):
    identity = compute_model_identity(model(tmp_path / "model"))
    gate = {"gate": {"accepted": True, "model_identity": identity}}
    assert verify_gate_model(gate, identity, required=True)["accepted"]
    legacy = verify_gate_model({"accepted": True, "model": "old/path"}, identity)
    assert legacy["status"] == "unbound" and not legacy["accepted"]
    assert legacy["reported_accepted"] and legacy["warnings"]
    with pytest.raises(ValueError, match="Re-run evaluate/compare"):
        verify_gate_model({"accepted": True}, identity, required=True)


def test_a_rejected_gate_stays_rejected_for_matching_contents(tmp_path):
    identity = compute_model_identity(model(tmp_path / "model"))
    with pytest.raises(ValueError, match="not accepted"):
        verify_gate_model({"accepted": False, "model_identity": identity}, identity, required=True)
