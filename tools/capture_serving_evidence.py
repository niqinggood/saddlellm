"""Capture read-only Linux deployment evidence after an owned serving experiment.

Does not import/load a model, change services, or record service environments.
Writes one NEW generated JSON report. Wheel/install bytes, original weight
hashes, resource observations and selected health statuses are separate facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--model")
    parser.add_argument("--business-unit", action="append", default=[])
    parser.add_argument("--owned-unit", action="append", default=[])
    parser.add_argument("--check-url", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        parser.error("output exists; keep previous evidence")
    for unit in args.business_unit + args.owned_unit:
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", unit):
            parser.error("unit must be an explicit systemd .service name")
    for url in args.check_url:
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            parser.error("health checks must be credential-free loopback HTTP URLs")
    wheel, site = Path(args.wheel), Path(args.site)
    with zipfile.ZipFile(wheel) as archive:
        files = [name for name in archive.namelist()
                 if name.endswith(".py") and name.startswith(("saddlellm/", "saddle_llm/"))]
        different = [name for name in files
                     if not (site / name).is_file() or archive.read(name) != (site / name).read_bytes()]
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only deployment evidence; GPU memory is an observation, not a peak",
        "python": sys.version, "platform": platform.platform(), "wheel": str(wheel),
        "wheel_sha256": digest(wheel), "installed_site": str(site),
        "runtime_files_checked": len(files), "installed_runtime_differences": different,
        "versions": {}, "business_units": {}, "owned_units": {}, "http_status": {},
    }
    for package in ("saddlellm", "torch", "transformers", "triton", "vllm"):
        try:
            report["versions"][package] = version(package)
        except PackageNotFoundError:
            report["versions"][package] = None
    for category, units in (("business_units", args.business_unit), ("owned_units", args.owned_unit)):
        for unit in units:
            result = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=10)
            report[category][unit] = result.stdout.strip() or "unknown"
    for url in args.check_url:
        try:
            with urlopen(url, timeout=10) as response:
                report["http_status"][url] = response.status
        except HTTPError as error:
            report["http_status"][url] = error.code
        except (URLError, TimeoutError):
            report["http_status"][url] = "unavailable"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    report["gpu_memory_observation"] = gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
    if args.model:
        model = Path(args.model)
        shards = sorted(model.glob("*.safetensors"))
        report["model_directory"] = str(model.resolve())
        report["original_weight_sha256"] = {shard.name: digest(shard) for shard in shards}
        report["original_weight_bytes"] = sum(shard.stat().st_size for shard in shards)
    report["installation_accepted"] = bool(files) and not different
    report["business_health_accepted"] = (
        bool(report["business_units"]) and bool(report["http_status"])
        and all(value == "active" for value in report["business_units"].values())
        and all(value == 200 for value in report["http_status"].values())
    )
    report["owned_jobs_stopped"] = bool(report["owned_units"]) and all(
        value in {"inactive", "failed"} for value in report["owned_units"].values()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"installation_accepted": report["installation_accepted"],
                      "business_health_accepted": report["business_health_accepted"],
                      "owned_jobs_stopped": report["owned_jobs_stopped"],
                      "output": str(destination)}), flush=True)
    if not report["installation_accepted"]:
        raise SystemExit("Wheel/install runtime mismatch")


if __name__ == "__main__":
    main()
