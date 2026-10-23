import pytest

from tools.benchmark_native_single_row import acceptance_gate


def summary():
    return {"base": {"median_ttft_s": 0.1, "max_peak_allocated_bytes": 100},
            "new": {"median_ttft_s": 0.1, "max_peak_allocated_bytes": 100},
            "speed_ratio": 1.1, "faster_pairs": 7}


def test_gate_accepts_qualifying_sample():
    assert acceptance_gate(summary(), ("base", "new"), True, 7)["passed"]


@pytest.mark.parametrize("failure", ["tokens", "repeats", "speed", "pairs", "ttft", "memory", "empty_text"])
def test_gate_rejects_individual_failure(failure):
    data = summary()
    if failure == "speed":
        data["speed_ratio"] = 1.01
    if failure == "pairs":
        data["faster_pairs"] = 5
    if failure == "ttft":
        data["new"]["median_ttft_s"] = 0.106
    if failure == "memory":
        data["new"]["max_peak_allocated_bytes"] = 106
    if failure == "empty_text":
        data["new"]["median_ttft_s"] = None
    assert not acceptance_gate(data, ("base", "new"), failure != "tokens", 3 if failure == "repeats" else 7)["passed"]
