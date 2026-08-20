import json

import pytest

from saddlellm.ModalityCodec import (
    CallableModalityCodec,
    ModalityCodecRegistry,
    ModalityCodecSpec,
)
from saddlellm.MultimodalData import MultimodalDataAdapter, MultimodalSegment


def test_legacy_image_record_normalizes_to_v2_segments(tmp_path):
    sample = MultimodalDataAdapter.normalize_record(
        {
            "id": "legacy-1",
            "image": "frame.png",
            "question": "what is here?",
            "answer": "a room",
        },
        image_root=str(tmp_path),
    )

    assert sample is not None
    assert sample.schema_version == 2
    assert sample.sample_id == "legacy-1"
    assert len(sample.images) == 1
    assert len(sample.segments) == 1
    assert sample.segments[0].modality == "image"
    assert sample.segments[0].uri == str(tmp_path / "frame.png")
    assert sample.to_dict()["images"][0]["path"] == str(tmp_path / "frame.png")


def test_unified_media_episode_preserves_timing_targets_and_dynamics(tmp_path):
    sample = MultimodalDataAdapter.normalize_record(
        {
            "sample_id": "episode-7",
            "prompt": "continue the scene with matching music",
            "segments": [
                {
                    "modality": "video",
                    "role": "observation",
                    "uri": "input.mp4",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "codec": "causal-video-vae-v1",
                },
                {
                    "modality": "audio",
                    "role": "target",
                    "uri": "target.wav",
                    "start_time": 2.0,
                    "end_time": 4.0,
                },
            ],
            "actions": [[0.0, 1.0], [1.0, 0.0]],
            "rewards": [0.0, 1.0],
            "dones": [False, True],
            "timestamps": [0.0, 1.0, 2.0],
            "provenance": {"dataset": "unit-test"},
            "license": "test-only",
        },
        image_root=str(tmp_path),
        task="world_dynamics",
    )

    assert sample is not None
    assert [segment.modality for segment in sample.segments] == ["video", "audio"]
    assert sample.segments[0].codec == "causal-video-vae-v1"
    assert sample.actions[-1] == [1.0, 0.0]
    assert sample.dones[-1] is True
    assert "<video>" in sample.messages[0]["content"]
    assert "<audio>" not in sample.messages[0]["content"]
    payload = sample.to_dict()
    assert payload["schema_version"] == 2
    assert payload["provenance"]["dataset"] == "unit-test"


def test_episode_alignment_and_segment_time_are_validated():
    with pytest.raises(ValueError, match="same length as actions"):
        MultimodalDataAdapter.normalize_record(
            {
                "segments": [{"modality": "video", "uri": "x.mp4"}],
                "actions": [1, 2],
                "rewards": [1.0],
            },
            task="world_dynamics",
        )
    with pytest.raises(ValueError, match="earlier than start_time"):
        MultimodalSegment(
            modality="audio", uri="x.wav", start_time=2.0, end_time=1.0
        )


def test_normalization_report_counts_modalities(tmp_path):
    input_path = tmp_path / "media.jsonl"
    output_path = tmp_path / "normalized.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "prompt": "draw and score",
                "image": {"path": "target.png", "role": "target"},
                "audio": {"path": "target.wav", "role": "target"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = MultimodalDataAdapter.normalize_file(
        str(input_path), str(output_path), task="conditional_generation"
    )

    assert report.kept_records == 1
    assert report.modality_counts == {"image": 1, "audio": 1}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert {segment["role"] for segment in payload["segments"]} == {"target"}


def test_modality_codec_registry_is_lazy_and_typed():
    name = "unit-text-codec"
    spec = ModalityCodecSpec(
        name=name,
        modality="text",
        latent_kind="discrete",
        trainable=False,
    )
    ModalityCodecRegistry.register(
        spec,
        lambda: CallableModalityCodec(
            spec,
            encoder=lambda value: list(value),
            decoder=lambda tokens: "".join(tokens),
        ),
        overwrite=True,
    )
    try:
        codec = ModalityCodecRegistry.build(name)
        assert codec.decode(codec.encode("abc")) == "abc"
        assert ModalityCodecRegistry.list_specs("text")[0].modality == "text"
        assert codec.fingerprint()["latent_kind"] == "discrete"
    finally:
        ModalityCodecRegistry.unregister(name)


def test_world_episode_normalizes_to_world_model_loader_contract():
    sample = MultimodalDataAdapter.normalize_record(
        {
            "id": "trajectory-1",
            "observations": [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]],
            "actions": [[0.1], [0.1]],
            "rewards": [0.0, 1.0],
            "dones": [False, True],
        },
        task="world_dynamics",
    )
    assert sample is not None
    payload = sample.to_dict()
    assert len(payload["observations"]) == len(payload["actions"]) + 1


def test_world_episode_rejects_misaligned_observations():
    with pytest.raises(ValueError, match=r"actions length \+ 1"):
        MultimodalDataAdapter.normalize_record(
            {
                "observations": [[0.0], [0.1]],
                "actions": [[0.1], [0.1]],
            },
            task="world_dynamics",
        )
