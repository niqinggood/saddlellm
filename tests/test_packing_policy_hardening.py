import copy

import pytest

from saddlellm.framework.pipeline import normalized_run_fingerprint
from saddlellm.training.PackingPolicy import (
    apply_packing_attention,
    resolve_packing_attention,
)


def test_unsupported_isolation_is_rejected_and_defaults_are_explicit():
    assert resolve_packing_attention("auto", "saddle") == "isolated"
    assert resolve_packing_attention("auto", "hf") == "concat"
    assert resolve_packing_attention("isolated", "hf", False) == "unpacked"
    for backend in ("hf", "megatron"):
        with pytest.raises(ValueError, match="does not enforce document_ids"):
            resolve_packing_attention("isolated", backend)


def test_concat_removes_isolation_and_position_resets_but_preserves_loss():
    from datasets import Dataset

    source = Dataset.from_list([{"input_ids": [1, 2, 3, 4], "labels": [1, 2, -100, 4],
                                "document_ids": [0, 0, 1, 1], "position_ids": [0, 1, 0, 1]}])
    concat = apply_packing_attention(source, "concat")
    assert concat[0] == {"input_ids": [1, 2, 3, 4], "labels": [1, 2, -100, 4]}
    assert apply_packing_attention(source, "isolated") is source
    assert "document_ids" in source.column_names


def test_auto_policy_keeps_old_run_fingerprints_while_explicit_policy_changes_it():
    config = {"stages": ["pretrain"], "data": {"pack_sequences": True}}
    legacy = normalized_run_fingerprint(config)
    current = copy.deepcopy(config)
    current["data"]["packing_attention"] = "auto"
    assert normalized_run_fingerprint(current) == legacy
    current["data"]["packing_attention"] = "concat"
    assert normalized_run_fingerprint(current) != legacy


def test_streaming_concat_removes_boundaries_without_materializing():
    from datasets import IterableDataset

    def rows():
        yield {"input_ids": [1, 2], "document_ids": [0, 1], "position_ids": [0, 0]}
    source = IterableDataset.from_generator(rows)
    assert source.column_names is None
    assert next(iter(apply_packing_attention(source, "concat"))) == {"input_ids": [1, 2]}


def test_unused_checkpoint_selection_keeps_resume_fingerprint():
    legacy = {"stages": ["eval"], "eval": {"enabled": True}}
    current = copy.deepcopy(legacy)
    current["eval"].update(checkpoint_candidates=[], selection_task=None,
                           selection_metric=None, selection_higher_is_better=True)
    assert normalized_run_fingerprint(current) == normalized_run_fingerprint(legacy)
