"""Verify saved 2.40 serving evidence; never executes a model or changes services.

Recovers each benchmark JSON from its preserved multi-run stdout log. Reports
HTTP protocol/throughput, runtime integrity and original weight provenance as
separate facts; tiny vLLM fixture results are never called an 8B benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=root / "artifacts/simple_serving_20260927")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    artifact, server = args.artifacts, args.artifacts / "server"

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    # systemd expands ${...} in shell arguments; an initial filename reused
    # http_c.json. Raw stdout preserved ALL three reports without overwriting.
    # Recover exact JSON objects, not recalculated or estimated measurements.
    remaining = (server / "logs/http-bench-v2.log").read_text(encoding="utf-8")
    decoder, stream_reports = json.JSONDecoder(), {}
    while remaining.strip():
        remaining = remaining.lstrip()
        report, consumed = decoder.raw_decode(remaining)
        remaining = remaining[consumed:]
        concurrency = report["concurrency"]
        if concurrency in stream_reports:
            raise ValueError("Repeated concurrency in preserved benchmark log")
        stream_reports[concurrency] = report
        recovered = server / f"http_c{concurrency}.json"
        if recovered.exists():
            assert read(recovered) == report, "Recovered report conflicts with existing evidence"
        else:
            with recovered.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
    assert set(stream_reports) == {1, 4, 8}
    final = read(server / "server_final_state.json")
    provenance = read(root / "artifacts/qwen3_serving_20260927/prepared/hf_provenance.json")
    weights_match = (final["original_weight_sha256"] == provenance["official_lfs_sha256"])
    wheel = artifact / "wheel_verified/saddlellm-2.40-py3-none-any.whl"
    candidate = artifact / "wheel_candidate/saddlellm-2.40-py3-none-any.whl"
    release = artifact / "wheel_release/saddlellm-2.40-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive, zipfile.ZipFile(candidate) as earlier, zipfile.ZipFile(release) as intermediate:
        runtime = [name for name in archive.namelist()
                   if name.endswith(".py") and name.startswith(("saddlellm/", "saddle_llm/"))]
        changed_source = [name for name in runtime if archive.read(name) != (root / name).read_bytes()]
        changed_candidate = [name for name in runtime if archive.read(name) != earlier.read(name)]
        changed_release = [name for name in runtime if archive.read(name) != intermediate.read(name)]
    junit = ET.parse(artifact / "related_tests_restartfix.xml")
    tests = {key: sum(int(suite.get(key, "0")) for suite in junit.iter("testsuite"))
             for key in ("tests", "failures", "errors", "skipped")}
    tool_matches = {name: (server / name).read_bytes() == (root / "tools" / name).read_bytes()
                    for name in ("benchmark_serving.py", "smoke_openai_service.py", "capture_serving_evidence.py")}
    sdk = read(server / "sdk_smoke_8b_restartfix.json")
    port_restart = read(server / "port_restart_probe.json")
    port_busy = read(server / "port_busy_probe.json")
    tiny = read(server / "sdk_smoke_vllm_fixture.json")
    probe = read(server / "vllm_fixture_probe/probe.json")
    unbatched = read(server / "http_nonstream_c8_unbatched.json")
    batched = read(server / "http_nonstream_c8_batch8.json")
    left, right = unbatched["targets"]["saddle"], batched["targets"]["saddle"]
    workloads_equal = all(unbatched[key] == batched[key] for key in (
        "model_alias", "protocol", "scope", "prompt_sha256", "requests_per_round",
        "concurrency", "max_tokens", "warmup_rounds", "repeats",
    ))
    response_equal = [
        (row["prompt_tokens"], row["completion_tokens"], row["text_sha256"], row["finish_reason"])
        for row in left["samples"]
    ] == [
        (row["prompt_tokens"], row["completion_tokens"], row["text_sha256"], row["finish_reason"])
        for row in right["samples"]
    ]
    result = {
        "scope": "2.40 deployment/SDK integrity + full Qwen3-8B W8A16 HTTP measurements; NOT vLLM 8B superiority",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "runtime_files_checked": len(runtime), "source_differences": changed_source,
        "candidate_runtime_differences": changed_candidate, "intermediate_runtime_differences": changed_release,
        "baseline_difference_scope": "ServingPlan.py only: POSIX port preflight TIME_WAIT restart fix; measured model execution code unchanged",
        "server_wheel_matches_final": hashlib.sha256(wheel.read_bytes()).hexdigest() == final["wheel_sha256"],
        "server_installation_accepted": final["installation_accepted"],
        "original_weight_hashes_match_official_provenance": weights_match,
        "original_revision": provenance["revision"], "original_parameters": 8190735360,
        "junit": tests, "sdk_8b_accepted": sdk["accepted"],
        "real_linux_port_restart_accepted": port_restart["accepted"],
        "real_linux_live_port_rejected": port_busy["accepted"],
        "executed_tools_match_local": tool_matches,
        "native_int8_compute_dtype": sdk["health_before"]["native_int8"]["compute_dtype"],
        "streaming_native_serial": {str(c): {key: report["targets"]["saddle"][key] for key in (
            "generated_tokens_per_second", "p50_response_ms", "p95_response_ms", "p95_ttft_ms",
            "latency_samples", "fixed_budget_response_fraction",
        )} for c, report in stream_reports.items()},
        "native_nonstream_batch_comparison": {
            "workloads_equal": workloads_equal, "usage_text_and_finish_reasons_equal": response_equal,
            "same_prompt_output_equivalence_accepted": workloads_equal and response_equal,
            "business_quality_verified": False,
            "unbatched_tokens_per_second": left["generated_tokens_per_second"],
            "batch8_tokens_per_second": right["generated_tokens_per_second"],
            "batch8_relative_gain": right["generated_tokens_per_second"] / left["generated_tokens_per_second"] - 1,
            "unbatched_p95_response_ms": left["p95_response_ms"],
            "batch8_p95_response_ms": right["p95_response_ms"],
            "warning": "One repeated public raw prompt, non-streaming greedy, fixed microbatches, NOT continuous batching or broad quality proof",
        },
        "vllm_real_gpu_text_protocol_accepted": (
            tiny["accepted"] and len(probe["parallel_completions"]) == 4
            and probe["chat_stream"]["done"] and probe["completion_stream"]["done"]
        ),
        "vllm_fixture_scope": "Untrained 78,835,712-parameter fixture; NOT full Qwen3-8B throughput/quality",
        "business_health_accepted": final["business_health_accepted"],
        "own_jobs_stopped": final["owned_jobs_stopped"],
        "gpu_memory_observation": final["gpu_memory_observation"],
    }
    result["accepted"] = (
        not changed_source
        and changed_candidate == changed_release == ["saddlellm/runtime/ServingPlan.py"]
        and result["server_wheel_matches_final"] and final["installation_accepted"]
        and weights_match and tests["tests"] == 618 and port_restart["accepted"] and port_busy["accepted"]
        and all(tool_matches.values())
        and tests["failures"] == tests["errors"] == tests["skipped"] == 0
        and sdk["accepted"] and result["vllm_real_gpu_text_protocol_accepted"]
        and final["business_health_accepted"] and final["owned_jobs_stopped"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)
    if not result["accepted"]:
        raise SystemExit("Evidence integrity check failed")


if __name__ == "__main__":
    main()
