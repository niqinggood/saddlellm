import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from saddlellm.runtime.InferenceDevice import (
    resolve_inference_device,
    resolve_inference_dtype,
)
from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
)


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (False, False, "cpu"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (True, True, "cuda"),
    ],
)
def test_auto_device_order_and_conservative_dtype(cuda, mps, expected):
    assert resolve_inference_device("auto", fake_torch(cuda, mps)) == expected
    assert resolve_inference_dtype("auto", expected) == (
        "float16" if expected == "cuda" else "float32"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_explicit_devices_are_not_silently_fallen_back(device):
    assert resolve_inference_device(device, fake_torch(True, True)) == device
    if device != "cpu":
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_inference_device(device, fake_torch())


def test_missing_mps_backend_and_invalid_device_or_dtype():
    instance = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert resolve_inference_device("auto", instance) == "cpu"
    with pytest.raises(ValueError):
        resolve_inference_device("unknown", instance)
    with pytest.raises(ValueError):
        resolve_inference_dtype("float64", "cpu")
    assert resolve_inference_dtype("float16", "mps") == "float16"


def test_cli_accepts_explicit_mps_without_loading_a_model(monkeypatch):
    from saddlellm import cli

    seen = []
    monkeypatch.setattr(
        cli, "_cmd_serve_model", lambda args: seen.append(args.device) or 0
    )
    assert cli.main(["serve", "MODEL", "--device", "mps"]) == 0
    assert seen == ["mps"]


@pytest.mark.parametrize("kind", ["native", "adapter"])
def test_mps_loader_forwards_and_moves_native_or_adapter_without_real_mps(
    monkeypatch, tmp_path, kind
):
    import torch
    import transformers

    from saddlellm.models import ModelLoader, TokenizerLoader
    from saddlellm.runtime.InferenceServer import load_inference_model

    seen = {}

    class Model:
        def to(self, device):
            seen["moved"] = device
            return self

        def eval(self):
            return self

    def load(path, **kwargs):
        seen.update(kwargs)
        return Model()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(ModelLoader, "load_causal_lm", load)
    if kind == "adapter":
        import peft

        from saddlellm.training import PostTrainingCompatibility

        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "offline-base"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            PostTrainingCompatibility, "stabilize_peft_optional_backends", lambda: []
        )
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
        monkeypatch.setattr(
            peft.PeftModel, "from_pretrained", lambda model, *args, **kwargs: model
        )
    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(
        TokenizerLoader, "load_tokenizer_compatible", lambda *args, **kwargs: tokenizer
    )
    model, loaded_tokenizer = load_inference_model(
        InferenceServerSettings(
            model_path=str(tmp_path),
            device="auto",
            fuse_projections=False,
            merge_adapter=False,
        )
    )
    assert seen["moved"] == "mps"
    if kind == "native":
        assert seen["device"] == "mps"
    else:
        assert "device_map" not in seen
    assert seen["dtype"] == torch.float32 and loaded_tokenizer is tokenizer
    # These routing stubs exercise no Apple kernel.
    assert model is not None


def test_readiness_tracks_lifespan_without_requiring_an_empty_queue():
    async def run():
        model = SimpleNamespace(device="cpu")
        settings = InferenceServerSettings(batch_size=2, web_ui=False)
        app = create_inference_app(settings, model=model, tokenizer=object())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/readyz")).status_code == 503
            async with app.router.lifespan_context(app):
                assert (await client.get("/readyz")).json() == {"status": "ready"}
                health = (await client.get("/health")).json()
                assert health["status"] == "ok" and health["ready"]
                assert health["device"] == "cpu"
                app.state.batcher.closed = True
                assert (await client.get("/readyz")).status_code == 503
                assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/readyz")).status_code == 503

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_real_loopback_http_benchmark_smoke(stream):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "tools/benchmark_native_serving.py"),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--lengths",
        "3,4",
        "--requests",
        "2",
        "--concurrency",
        "2",
        "--decode-tokens",
        "2",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--batch-size",
        "2",
        "--bucket-size",
        "8",
    ]
    if stream:
        command.append("--stream")
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=30
    )
    report = json.loads(result.stdout)
    assert report["all_token_counts_equal"] and report["all_output_text_equal"]
    assert report["shared_model_targets_measured_sequentially"]
    assert not report["microbatch_enabled_in_streaming"]
    if stream:
        assert report["targets"]["individual"]["ttft_samples"] == 2
        assert report["targets"]["individual"]["p99_ttft_ms"] > 0
    else:
        health = report["targets"]["length_bucket"]["batching_health"]
        # Real socket arrival can exceed a 5 ms coalescing deadline. Batch size
        # is an upper bound, not a promise; deterministic queue tests cover B=2.
        assert 1 <= health["largest_batch"] <= 2
        assert health["recent_padding_tokens"] == health["largest_batch"] - 1


def test_container_and_platform_ci_describe_a_minimal_serving_image():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["saddlellm", "serve"]' in dockerfile
    assert "USER 10001:10001" in dockerfile and "/readyz" in dockerfile
    assert "--local-files-only" in dockerfile and "--no-web-ui" in dockerfile
    assert "COPY . " not in dockerfile
    assert "**" in (root / ".dockerignore").read_text(encoding="utf-8")
    workflow = yaml.safe_load(
        (root / ".github/workflows/inference-platforms.yml").read_text(encoding="utf-8")
    )
    assert set(workflow["jobs"]["native-cpu"]["strategy"]["matrix"]["os"]) == {
        "ubuntu-latest",
        "windows-latest",
        "macos-latest",
    }
    assert "container-cpu" in workflow["jobs"]
