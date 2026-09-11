import json
import wave

import numpy as np
from PIL import Image

from saddlellm.multimodal.BuiltinMediaCodecs import (
    FrameVideoCodec,
    HashTextConditionEncoder,
    RGBImageCodec,
    WAVResidualCodec,
)
from saddlellm.multimodal.LatentFlowTrainer import CachedLatentDataset
from saddlellm.multimodal.MediaCache import MediaCacheBuildConfig, ShardedNpzStore, build_media_cache
from saddlellm.multimodal.MusicCodeTrainer import CachedMusicCodeDataset
from saddlellm.multimodal.VideoLatentFlowTrainer import CachedVideoLatentDataset
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator
from saddlellm.cli import main as cli_main


def _write_wav(path, samples, sample_rate=8000):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def test_builtin_codecs_round_trip_shapes(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (9, 7), (30, 120, 210)).save(image_path)
    image_codec = RGBImageCodec(height=4, width=6)
    image_latent = image_codec.encode(image_path)
    assert image_latent.shape == (3, 4, 6)
    assert image_codec.decode(image_latent).size == (6, 4)

    wav_path = tmp_path / "tone.wav"
    time = np.arange(800, dtype=np.float32) / 8000
    _write_wav(wav_path, np.sin(2 * np.pi * 220 * time) * 0.5)
    audio_codec = WAVResidualCodec(
        sample_rate=8000,
        frame_size=80,
        max_frames=8,
        num_codebooks=2,
        codebook_size=16,
    )
    codes = audio_codec.encode(wav_path)
    assert codes.shape == (2, 8)
    _, attention_mask = audio_codec.encode_with_attention_mask(wav_path)
    assert attention_mask.tolist() == [1] * 8
    assert audio_codec.decode(codes).shape == (640,)

    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (8, 8), "red").save(frames / "000.png")
    Image.new("RGB", (8, 8), "blue").save(frames / "001.png")
    video_codec = FrameVideoCodec(frames=3, height=4, width=4)
    video_latents = video_codec.encode(frames)
    assert video_latents.shape == (3, 3, 4, 4)
    assert len(video_codec.decode(video_latents)) == 3


def test_hash_encoder_is_deterministic_and_normalized():
    encoder = HashTextConditionEncoder(dimension=16, ngram=2, seed=7)
    first = encoder.encode(["相同文本", "另一段"])
    second = encoder.encode(["相同文本", "另一段"])
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_image_cache_builds_shards_and_is_directly_trainable(tmp_path):
    rows = []
    for index, color in enumerate(("red", "green", "blue")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        rows.append({"prompt": f"a {color} square", "image": path.name})
    manifest_source = tmp_path / "images.jsonl"
    manifest_source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "image_cache"
    config = MediaCacheBuildConfig(
        input_path=str(manifest_source),
        output_dir=str(output),
        modality="image",
        codec_config={"height": 4, "width": 4},
        text_encoder_config={"dimension": 6},
        shard_size=2,
    )
    manifest = build_media_cache(config)
    assert manifest["status"] == "completed"
    assert manifest["samples"] == 3
    assert len(manifest["shards"]) == 2
    assert manifest["arrays"]["latents"]["shape"] == [3, 4, 4]
    assert build_media_cache(config)["shards"] == manifest["shards"]

    store = ShardedNpzStore(str(output), ("latents", "conditions"))
    assert len(store) == 3
    assert store.get(-1)["latents"].shape == (3, 4, 4)
    dataset = CachedLatentDataset(str(output))
    assert dataset.infer_model_config(
        patch_size=2, hidden_size=8, num_layers=1, num_heads=2
    ).condition_dim == 6
    assert dataset[1]["latents"].shape == (3, 4, 4)


def test_audio_and_video_cache_match_generation_datasets(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, np.linspace(-0.5, 0.5, 400, dtype=np.float32))
    audio_manifest = tmp_path / "audio.json"
    audio_manifest.write_text(
        json.dumps([{"caption": "rising tone", "audio": wav_path.name}]),
        encoding="utf-8",
    )
    audio_output = tmp_path / "audio_cache"
    build_media_cache(
        MediaCacheBuildConfig(
            input_path=str(audio_manifest),
            output_dir=str(audio_output),
            modality="audio",
            codec_config={
                "sample_rate": 8000,
                "frame_size": 40,
                "max_frames": 5,
                "num_codebooks": 2,
                "codebook_size": 8,
            },
            text_encoder_config={"dimension": 4},
            shard_size=1,
        )
    )
    music = CachedMusicCodeDataset(str(audio_output))
    music_config = music.infer_model_config(
        hidden_size=8, num_layers=1, num_heads=2
    )
    assert music_config.num_codebooks == 2
    assert music_config.max_sequence_length == 5
    assert music[0]["attention_mask"].shape == (5,)

    frame_dir = tmp_path / "video_frames"
    frame_dir.mkdir()
    Image.new("RGB", (8, 8), "black").save(frame_dir / "0.png")
    Image.new("RGB", (8, 8), "white").save(frame_dir / "1.png")
    video_manifest = tmp_path / "video.jsonl"
    video_manifest.write_text(
        json.dumps({"text": "fade", "video": frame_dir.name}) + "\n",
        encoding="utf-8",
    )
    video_output = tmp_path / "video_cache"
    build_media_cache(
        MediaCacheBuildConfig(
            input_path=str(video_manifest),
            output_dir=str(video_output),
            modality="video",
            codec_config={"frames": 2, "height": 4, "width": 4},
            text_encoder_config={"dimension": 4},
            shard_size=1,
        )
    )
    video = CachedVideoLatentDataset(str(video_output))
    video_config = video.infer_model_config(
        temporal_patch_size=1,
        patch_size=2,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
    )
    assert video_config.latent_frames == 2
    assert video_config.latent_channels == 3


def test_non_strict_cache_records_failures_without_losing_valid_rows(tmp_path):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (4, 4), "white").save(valid)
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        json.dumps({"prompt": "missing", "image": "missing.png"})
        + "\n"
        + json.dumps({"prompt": "valid", "image": valid.name})
        + "\n",
        encoding="utf-8",
    )
    result = build_media_cache(
        MediaCacheBuildConfig(
            input_path=str(source),
            output_dir=str(tmp_path / "mixed_cache"),
            modality="image",
            codec_config={"height": 4, "width": 4},
            text_encoder_config={"dimension": 4},
            strict=False,
        )
    )
    assert result["samples"] == 1
    assert result["failures"][0]["record_index"] == 0


def test_raw_image_manifest_to_training_checkpoint_in_one_pipeline(tmp_path):
    rows = []
    for index, color in enumerate(("red", "blue")):
        image_path = tmp_path / f"raw-{index}.png"
        Image.new("RGB", (8, 8), color).save(image_path)
        rows.append({"prompt": f"a {color} tile", "image": image_path.name})
    source = tmp_path / "raw-images.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output_root = tmp_path / "pipeline"
    results = TrainingOrchestrator.from_dict(
        {
            "stages": ["media_cache", "image_generation"],
            "media_cache": {
                "enabled": True,
                "jobs": [
                    {
                        "input_path": str(source),
                        "output_dir": "raw_cache",
                        "modality": "image",
                        "codec_config": {"height": 4, "width": 4},
                        "text_encoder_config": {"dimension": 6},
                        "shard_size": 1,
                    }
                ],
            },
            "image_generation": {
                "enabled": True,
                "output_dir": "flow",
                "patch_size": 2,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "batch_size": 1,
                "max_steps": 1,
                "checkpoint_steps": 0,
                "device": "cpu",
            },
            "logging": {"output_dir": str(output_root), "backend": "local"},
            "eval": {"enabled": False},
        }
    ).run()
    assert results["media_cache"]["status"] == "completed"
    assert results["image_generation"]["status"] == "completed"
    assert results["image_generation"]["global_step"] == 1
    assert (output_root / "raw_cache" / "manifest.json").is_file()
    assert (output_root / "flow" / "latent_flow_model.bin").is_file()


def test_pipeline_dry_run_does_not_materialize_cache(tmp_path):
    source = tmp_path / "dry.jsonl"
    source.write_text(
        json.dumps({"prompt": "x", "image": "not-read-in-dry-run.png"}) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "dry-pipeline"
    results = TrainingOrchestrator.from_dict(
        {
            "stages": ["media_cache", "image_generation"],
            "training": {"dry_run": True},
            "media_cache": {
                "enabled": True,
                "jobs": [
                    {
                        "input_path": str(source),
                        "output_dir": "cache",
                        "modality": "image",
                    }
                ],
            },
            "image_generation": {"enabled": True},
            "logging": {"output_dir": str(output_root), "backend": "local"},
            "eval": {"enabled": False},
        }
    ).run()
    assert results["media_cache"]["status"] == "planned"
    assert results["image_generation"]["status"] == "planned"
    assert not (output_root / "cache").exists()


def test_media_cache_cli_build_and_list(tmp_path, capsys):
    image_path = tmp_path / "cli.png"
    Image.new("RGB", (4, 4), "purple").save(image_path)
    source = tmp_path / "cli.jsonl"
    source.write_text(
        json.dumps({"prompt": "purple", "image": image_path.name}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "cache.yaml"
    config.write_text(
        "cache:\n"
        "  input_path: cli.jsonl\n"
        "  output_dir: cli-cache\n"
        "  modality: image\n"
        "  codec_config: {height: 4, width: 4}\n"
        "  text_encoder_config: {dimension: 4}\n",
        encoding="utf-8",
    )
    assert cli_main(["build-media-cache", str(config)]) == 0
    build_payload = json.loads(capsys.readouterr().out)
    assert build_payload["status"] == "completed"
    assert (tmp_path / "cli-cache" / "manifest.json").is_file()
    assert cli_main(["media-codecs", "--modality", "image"]) == 0
    codecs_payload = json.loads(capsys.readouterr().out)
    assert codecs_payload["codecs"][0]["name"] == "baseline-rgb-image"


def test_resume_rejects_changed_codec_identity(tmp_path):
    image_path = tmp_path / "resume.png"
    Image.new("RGB", (4, 4), "orange").save(image_path)
    source = tmp_path / "resume.jsonl"
    source.write_text(
        json.dumps({"prompt": "orange", "image": image_path.name}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "resume-cache"
    build_media_cache(
        MediaCacheBuildConfig(
            input_path=str(source),
            output_dir=str(output),
            modality="image",
            codec_config={"height": 4, "width": 4},
            text_encoder_config={"dimension": 4},
        )
    )
    try:
        build_media_cache(
            MediaCacheBuildConfig(
                input_path=str(source),
                output_dir=str(output),
                modality="image",
                codec_config={"height": 8, "width": 8},
                text_encoder_config={"dimension": 4},
            )
        )
    except ValueError as exc:
        assert "Cannot resume media cache" in str(exc)
    else:
        raise AssertionError("changed codec identity must be rejected")
