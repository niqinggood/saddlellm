"""Deployment selection tests: no real vLLM model or throughput assertion."""

import json
import socket
import subprocess
import sys

import pytest

from saddlellm import cli
from saddlellm.runtime import ServingPlan
from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
)
from saddlellm.runtime.VLLMConfig import VLLMServingConfig


@pytest.fixture
def environment(monkeypatch):
    values = {"platform": "Linux", "torch_version": "tested", "cuda_available": True,
              "vllm_version": "0.30.0", "vllm_supported": True}
    monkeypatch.setattr(ServingPlan, "probe_serving_environment", lambda: dict(values))
    return values


def test_auto_selects_continuous_backend_without_mutating_settings(environment):
    original = InferenceServerSettings(model_path="offline-model", backend="auto")
    resolved = ServingPlan.resolve_serving_settings(original)
    assert resolved.backend == "vllm" and resolved.backend_selection["continuous_batching"]
    assert original.backend == "auto" and original.backend_selection is None
    assert resolved.quantization == "none" and resolved.dtype == "auto"
    assert resolved.enable_thinking is None and not resolved.truncate_input
    assert not resolved.backend_selection["startup_fallback"]


@pytest.mark.parametrize("change", [
    {"platform": "Windows"}, {"platform": "Darwin"}, {"cuda_available": False},
    {"vllm_supported": False, "vllm_version": "0.29.0"},
    {"vllm_supported": False, "vllm_version": None},
])
def test_auto_local_fallback_is_reported_and_high_throughput_can_be_required(environment, change):
    environment.update(change)
    settings = InferenceServerSettings(model_path="MODEL", backend="auto")
    resolved = ServingPlan.resolve_serving_settings(settings)
    assert resolved.backend == "local" and resolved.backend_selection["warnings"]
    settings.require_high_throughput = True
    with pytest.raises(RuntimeError, match="High-throughput"):
        ServingPlan.resolve_serving_settings(settings)


@pytest.mark.parametrize("options,expected", [
    ({"device": "cpu"}, "local"), ({"device": "mps"}, "local"),
    ({"quantization": "int8", "device": "cuda"}, "native"),
    ({"quantization": "4bit"}, "local"), ({"batch_size": 4}, "local"),
    ({"prefix_cache_mb": 64}, "local"), ({"truncate_input": True}, "local"),
    ({"native_attention_backend": "eager"}, "native"),
    ({"native_fused_kernels": True, "device": "cuda"}, "native"),
    ({"cpu_prepack_linear": True, "device": "cpu"}, "native"),
])
def test_explicit_local_native_controls_survive_auto_selection(environment, options, expected):
    resolved = ServingPlan.resolve_serving_settings(InferenceServerSettings(backend="auto", **options))
    assert resolved.backend == expected


@pytest.mark.parametrize("marker", ["saddle_config.json", "adapter_config.json", "quantization_manifest.json"])
def test_native_or_adapter_directory_is_not_misrouted_to_vllm(environment, tmp_path, marker):
    (tmp_path / marker).write_text("{}", encoding="utf-8")
    resolved = ServingPlan.resolve_serving_settings(InferenceServerSettings(model_path=str(tmp_path), backend="auto"))
    assert resolved.backend == "local"


def test_conflicting_controls_and_unsupported_explicit_engine_knobs_fail(environment):
    config = VLLMServingConfig(max_num_seqs=8)
    with pytest.raises(ValueError, match="conflict"):
        ServingPlan.resolve_serving_settings(InferenceServerSettings(backend="auto", batch_size=4, vllm=config))
    environment["platform"] = "Windows"
    with pytest.raises(RuntimeError, match="vLLM options"):
        ServingPlan.resolve_serving_settings(InferenceServerSettings(backend="auto", vllm=config))


def test_auto_factory_dispatch_and_startup_failures_never_load_hf(environment, monkeypatch):
    from saddlellm.runtime import InferenceServer, VLLMServing

    monkeypatch.setattr(InferenceServer, "load_inference_model", lambda *_: pytest.fail("Unexpected local fallback"))
    marker = object()
    monkeypatch.setattr(VLLMServing, "create_vllm_inference_app", lambda settings: marker)
    assert create_inference_app(InferenceServerSettings(backend="auto", model_path="MODEL")) is marker

    def fail(settings):
        raise RuntimeError("engine initialization failed")

    monkeypatch.setattr(VLLMServing, "create_vllm_inference_app", fail)
    with pytest.raises(RuntimeError, match="initialization failed"):
        create_inference_app(InferenceServerSettings(backend="auto", model_path="MODEL"))


def test_cli_default_dry_run_never_loads_model_or_prints_credentials(environment, monkeypatch, capsys):
    from saddlellm.runtime import InferenceServer

    monkeypatch.setattr(InferenceServer, "create_inference_app", lambda *_: pytest.fail("dry-run loaded model"))
    monkeypatch.setenv("SADDLE_TEST_KEY", "dummy-private-test-value")
    assert cli.main(["serve", "MODEL", "--dry-run", "--api-key-env", "SADDLE_TEST_KEY"]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["backend_selection"]["requested"] == "auto"
    assert report["backend_selection"]["selected"] == "vllm"
    assert report["thinking_policy"] == "model_default"
    assert report["input_policy"] == "reject_overlong"
    assert "dummy-private-test-value" not in output


def test_missing_high_throughput_backend_is_a_cli_error(environment, capsys):
    environment["vllm_supported"] = False
    assert cli.main(["serve", "MODEL", "--require-high-throughput", "--dry-run"]) == 1
    assert "High-throughput" in json.loads(capsys.readouterr().out)["error"]


def test_native_stream_batch_cli_is_explicit_and_visible(environment, capsys):
    args = ["serve", "MODEL", "--native-stream-batching", "--batch-size", "4",
            "--max-concurrency", "4", "--batch-wait-ms", "2", "--dry-run"]
    assert cli.main(args) == 1
    assert "explicit backend=native" in json.loads(capsys.readouterr().out)["error"]
    assert cli.main(args + ["--backend", "native"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["native_stream_batching"] is True
    assert result["batch_wait_ms"] == 2
    assert result["backend_selection"]["selected"] == "native"


def test_native_decode_mode_cli_roundtrip_and_missing_batching_error(environment, capsys):
    args = ["serve", "MODEL", "--backend", "native", "--native-decode-mode", "throughput", "--dry-run"]
    assert cli.main(args) == 1
    assert "requires native_stream_batching" in json.loads(capsys.readouterr().out)["error"]
    assert cli.main(args + ["--native-stream-batching", "--batch-size", "4", "--max-concurrency", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["native_decode_mode"] == "throughput"


def test_busy_port_fails_before_any_model_is_loaded(monkeypatch, capsys):
    from saddlellm.runtime import InferenceServer

    monkeypatch.setattr(InferenceServer, "create_inference_app", lambda *_: pytest.fail("model loaded before port check"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert cli.main(["serve", "MODEL", "--backend", "local", "--port", str(port)]) == 1
    assert "Cannot bind" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("operating_system,expected_reuse", [("Linux", True), ("Windows", False)])
def test_port_preflight_reuses_posix_time_wait_but_not_windows_live_ports(monkeypatch, operating_system, expected_reuse):
    options = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            options.append(args)

        def bind(self, address):
            assert address == ("127.0.0.1", 8112)
            assert bool(options) is expected_reuse

    monkeypatch.setattr(ServingPlan.platform, "system", lambda: operating_system)
    monkeypatch.setattr(ServingPlan.socket, "socket", lambda *args: Probe())
    ServingPlan.check_serving_address("127.0.0.1", 8112)
    assert options == ([(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)] if expected_reuse else [])


@pytest.mark.parametrize("port", [-1, 65536, True])
def test_invalid_port_has_no_model_side_effect(port):
    with pytest.raises(ValueError, match="port"):
        ServingPlan.check_serving_address("127.0.0.1", port)


def test_public_python_serve_entry_stays_lazy_and_deploys_one_app(environment, monkeypatch):
    subprocess.run([sys.executable, "-c", "from saddlellm import serve; import sys; assert callable(serve); assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"], check=True, timeout=20)
    import uvicorn

    from saddlellm import serve
    from saddlellm.runtime import InferenceServer

    calls = []
    monkeypatch.setattr(ServingPlan, "check_serving_address", lambda *args: None)
    monkeypatch.setattr(InferenceServer, "create_inference_app", lambda settings: calls.append(settings) or "APP")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    serve("MODEL", require_high_throughput=True, port=8112)
    assert calls[0].backend == "vllm"
    assert calls[1][0] == "APP" and calls[1][1]["port"] == 8112
