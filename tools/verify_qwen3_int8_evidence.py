"""Verify a full-8B experiment against current source, wheel and original weights.

Does not execute a model or imply general quality/vLLM superiority. Reports
are read-only inputs; the output is a new verification artifact.
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
    parser.add_argument("--artifacts", type=Path, default=root / "artifacts/qwen3_8b_optimization_20260927")
    parser.add_argument("--provenance", type=Path, default=root / "artifacts/qwen3_serving_20260927/prepared/hf_provenance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    artifact_root = args.artifacts.resolve()
    server = artifact_root / "server/w8a16-release"
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    benchmark = read(server / "gpu_w8a16_b1_release.json")
    smoke = read(server / "api_smoke_verified.json")
    health = read(server / "server_final_state.json")
    provenance = read(args.provenance)
    wheel_path = artifact_root / "wheel_w8a16_release/saddlellm-2.39-py3-none-any.whl"
    wheel_sha = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    source_checks = {}
    for name, expected in benchmark["implementation_sha256"].items():
        directory = root / ("tools" if name == "benchmark_qwen3_int8.py" else "saddlellm/models")
        source_checks[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    weight_checks = {
        name: digest == provenance["official_lfs_sha256"].get(name)
        for name, digest in benchmark["checkpoint_sha256"].items()
    }
    with zipfile.ZipFile(wheel_path) as wheel:
        names = [
            name for name in wheel.namelist()
            if name.endswith(".py") and name.startswith(("saddlellm/", "saddle_llm/"))
        ]
        different = [name for name in names if wheel.read(name) != (root / name).read_bytes()]
    suites = ET.parse(artifact_root / "w8a16_related_tests_release.xml").getroot()
    junit = {
        key: sum(int(suite.get(key, "0")) for suite in suites.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    result = {
        "scope": "Recorded full Qwen3-8B W8A16 evidence integrity; NOT a general performance/quality guarantee",
        "wheel_sha256": wheel_sha,
        "wheel_matches_tested_server_wheel": wheel_sha == health["tested_wheel_sha256"],
        "runtime_files_checked": len(names),
        "wheel_runtime_different_from_workspace": different,
        "source_matches_executed_benchmark": source_checks,
        "original_weight_hashes_match_official_provenance": weight_checks,
        "original_revision": provenance["revision"],
        "original_parameters": benchmark["original_parameters"],
        "same_quantization_equivalence_accepted": benchmark["same_quantization_equivalence_accepted"],
        "all_generated_tokens_equal": benchmark["all_generated_tokens_equal"],
        "api_smoke_accepted": smoke["accepted"],
        "business_health_accepted": health["business_health_accepted"],
        "own_jobs_inactive": all(value == "inactive" for value in health["owned_jobs"].values()),
        "junit": junit,
    }
    result["accepted"] = (
        result["wheel_matches_tested_server_wheel"] and not different and len(names) > 100
        and len(source_checks) == 7 and all(source_checks.values())
        and len(weight_checks) == 5 and all(weight_checks.values())
        and benchmark["original_parameters"] == 8_190_735_360
        and benchmark["conditions"]["activation_bits"] == 16
        and benchmark["precision_equivalent_to_float"] is False
        and benchmark["same_quantization_equivalence_accepted"]
        and benchmark["all_generated_tokens_equal"]
        and smoke["accepted"] and health["business_health_accepted"]
        and result["own_jobs_inactive"]
        and junit == {"tests": 570, "failures": 0, "errors": 0, "skipped": 0}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
