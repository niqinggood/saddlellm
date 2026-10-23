import json
from importlib import metadata
from types import SimpleNamespace

import pytest

from saddlellm.utils import EnvironmentDiagnostics as diagnostics


@pytest.fixture
def environment(monkeypatch):
    versions = dict(torch="2.6.0", transformers="4.57.6", fastapi="0.115.0", uvicorn="0.30.0",
                    pydantic="2.10.0", safetensors="0.4.3", accelerate="1.12.0", peft="0.18.1",
                    trl="0.26.2", datasets="3.6.0", vllm="0.30.0")
    calls = []
    cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0,
                           get_device_name=lambda i: f"GPU-{i}",
                           mem_get_info=lambda i: ((i + 1) * 1024**3, 16 * 1024**3))
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.4"),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    def importer(name):
        calls.append(name)
        if name == "torch":
            return torch
        if name == "psutil":
            return SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=64 * 1024**3, available=32 * 1024**3))
        return SimpleNamespace()

    monkeypatch.setattr(diagnostics.metadata, "version", version)
    monkeypatch.setattr(diagnostics.importlib, "import_module", importer)
    monkeypatch.setattr(diagnostics.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Linux")
    return SimpleNamespace(versions=versions, calls=calls, cuda=cuda, torch=torch)


@pytest.mark.parametrize("profile", ["serve", "native"])
def test_cpu_serving_does_not_require_training_stack(environment, profile):
    for name in ("datasets", "trl"):
        environment.versions.pop(name)
    report = diagnostics.environment_report(profile, "cpu")
    assert report["ready"] and report["selected_device"] == "cpu"
    assert report["post_training"] is None
    assert not {"datasets", "trl"}.intersection(environment.calls)
    assert report["ready_scope"] == "dependency_device_preflight_not_model_capacity"


def test_training_still_reports_missing_training_dependencies(environment):
    environment.versions.pop("trl")
    report = diagnostics.environment_report("train")
    assert not report["ready"]
    assert any("trl" in item for item in report["issues"])


def test_each_gpu_uses_driver_free_memory_not_process_allocation(environment):
    environment.cuda.is_available = lambda: True
    environment.cuda.device_count = lambda: 2
    environment.cuda.memory_allocated = lambda *_: pytest.fail("process allocation is not free GPU memory")
    report = diagnostics.environment_report("native", "cuda")
    assert report["ready"]
    assert [gpu["free_gb"] for gpu in report["gpus"]] == [1, 2]
    assert report["gpu_memory_gb"] == 1
    assert len([warning for warning in report["warnings"] if "GiB free" in warning]) == 2


def test_unreadable_memory_is_unknown_not_optimistic(environment):
    environment.cuda.is_available = lambda: True
    environment.cuda.device_count = lambda: 1

    def fail(_):
        raise RuntimeError("driver error")

    environment.cuda.mem_get_info = fail
    report = diagnostics.environment_report("native")
    assert report["gpu_memory_gb"] is None
    assert any("unknown" in warning for warning in report["warnings"])


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_device_fails(environment, device):
    report = diagnostics.environment_report("serve", device)
    assert not report["ready"]
    assert any("unavailable" in issue for issue in report["issues"])


def test_bad_version_or_broken_binary_produces_report(environment, monkeypatch):
    environment.versions["transformers"] = "5.0.0"
    original = diagnostics.importlib.import_module

    def importer(name):
        if name == "torch":
            raise OSError("private binary path")
        return original(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", importer)
    report = diagnostics.environment_report("serve")
    assert not report["ready"] and not report["packages_ok"]
    assert any("OSError" in issue for issue in report["issues"])
    assert any("outside supported" in issue for issue in report["issues"])
    assert "private binary path" not in json.dumps(report)


def test_vllm_requires_supported_platform_but_does_not_import_engine(environment, monkeypatch):
    environment.cuda.is_available = lambda: True
    environment.cuda.device_count = lambda: 1
    report = diagnostics.environment_report("vllm")
    assert report["ready"] and report["import_checks"]["vllm"] == "metadata_only"
    assert "vllm" not in environment.calls and "trl" not in environment.calls
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    assert not diagnostics.environment_report("vllm")["ready"]


def test_cli_profile_and_exit_code(environment, capsys):
    from saddlellm.cli import main

    assert main(["doctor", "--profile", "native", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == "native"
    environment.versions.pop("torch")
    assert main(["doctor", "--profile", "serve"]) == 1
    assert not json.loads(capsys.readouterr().out)["ready"]
