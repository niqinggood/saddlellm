from dataclasses import asdict

import pytest

from saddlellm.utils.TrainerUtils import EnvCheckResult


def test_environment_result_serializes_correct_gpu_total_memory_key() -> None:
    result = EnvCheckResult(
        python_version="3.11.0",
        cuda_available=True,
        cuda_version="12.4",
        gpu_count=1,
        gpu_name="test",
        gpu_memory_gb=12.0,
        gpu_total_memory_gb=16.0,
        ram_total_gb=64.0,
        ram_available_gb=32.0,
        disk_free_gb=100.0,
        pytorch_version="2.6.0",
        transformers_version="4.57.6",
        packages_ok=True,
        issues=[],
        recommendations=[],
        ready=True,
    )

    payload = asdict(result)

    assert payload["gpu_total_memory_gb"] == 16.0
    assert "gpu_total_memoryory_gb" not in payload


def test_environment_result_keeps_deprecated_attribute_alias() -> None:
    result = EnvCheckResult(
        python_version="3.11.0",
        cuda_available=False,
        cuda_version="N/A",
        gpu_count=0,
        gpu_name="",
        gpu_memory_gb=0.0,
        gpu_total_memory_gb=0.0,
        ram_total_gb=0.0,
        ram_available_gb=0.0,
        disk_free_gb=0.0,
        pytorch_version="2.6.0",
        transformers_version="4.57.6",
        packages_ok=True,
        issues=[],
        recommendations=[],
        ready=False,
    )

    with pytest.warns(DeprecationWarning, match="gpu_total_memory_gb"):
        assert result.gpu_total_memoryory_gb == 0.0
