"""Isolated Linux GPU verification; writes a completion result even on failure."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def command(args, *, timeout=180, env=None):
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {args[0:3]}"
        )
    return result.stdout


def http_test(model_path):
    import uvicorn

    from saddlellm.decision.Server import DecisionServerSettings, create_decision_app
    from saddlellm.decision.TypedSchema import read_typed_dataset

    # Bind loopback only, choosing an unused port; shut down after verification.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_decision_app(
        DecisionServerSettings(
            model_path=str(model_path),
            device="cuda",
            api_key="verification-only",
            web_ui=False,
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if time.monotonic() >= deadline:
                raise TimeoutError("API failed to start")
            time.sleep(0.1)
        _, splits, _ = read_typed_dataset(ROOT / "data/decision_typed_example")
        examples = splits["test"][:3]
        payload = {
            "state": examples[0]["state"],
            "questions": [
                {
                    k: row[k]
                    for k in ("type", "question", "choices", "levels")
                    if k in row
                }
                for row in examples
            ],
        }
        url = f"http://127.0.0.1:{port}/v1/decisions"
        raw = json.dumps(payload).encode()
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    url, data=raw, headers={"Content-Type": "application/json"}
                ),
                timeout=10,
            )
            raise AssertionError("API failed to enforce authentication")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        durations, responses = [], []
        for _ in range(6):
            start = time.perf_counter()
            request = urllib.request.Request(
                url,
                data=raw,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer verification-only",
                },
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.load(response)
            durations.append((time.perf_counter() - start) * 1000)
            responses.append(answer)
            assert [r["type"] for r in answer["data"]] == ["Choice", "Noul", "Score"]
            for row in answer["data"]:
                assert (
                    abs(sum(c["probability"] for c in row["probabilities"]) - 1) < 1e-5
                )
        changed = json.loads(json.dumps(payload))
        changed["questions"][0]["question"] = "An unseen question"
        request = urllib.request.Request(
            url,
            data=json.dumps(changed).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer verification-only",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert (
                json.load(response)["data"][0]["calibration_status"] == "out_of_scope"
            )
        return {
            "real_loopback_http": True,
            "authenticated": True,
            "unauthorized_status": 401,
            "first_request_ms": durations[0],
            "warm_request_ms": durations[1:],
            "response": responses[-1],
        }
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def main():
    import torch

    from saddlellm.decision.TypedSchema import TypedDecisionConfig
    from saddlellm.decision.TypedTraining import TypedTrainingConfig, train_typed
    from tools.prepare_decision_typed import prepare

    result = {"status": "running", "remote_root": str(ROOT), "steps": {}}
    try:
        inventory = command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
        result["gpu_inventory"] = inventory.strip()
        # Install test tools under this run only; leave the shared venv untouched.
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(ROOT / "testdeps"),
                "pytest==8.3.5",
                "fastapi==0.115.14",
                "httpx==0.28.1",
            ],
            timeout=180,
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT) + ":" + str(ROOT / "testdeps"),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        )
        sys.path.insert(0, str(ROOT / "testdeps"))
        os.environ["PYTHONPATH"] = env["PYTHONPATH"]
        tests = command(
            [
                sys.executable,
                "-m",
                "pytest",
                *[str(p) for p in sorted((ROOT / "tests").glob("test*decision*.py"))],
                "-q",
            ],
            env=env,
        )
        result["steps"]["regression"] = tests.strip()
        command(
            [
                sys.executable,
                "-m",
                "saddlellm.cli",
                "validate-config",
                "configs/decision_typed.yaml",
            ]
        )
        # Exercise dataset inference and the framework's canonical stage.
        command(
            [
                sys.executable,
                "-m",
                "saddlellm.cli",
                "train",
                "decision",
                "--data",
                "data/decision_typed_example",
                "--epochs",
                "12",
                "--output-dir",
                str(ROOT / "native_supervised"),
            ],
            timeout=300,
        )
        supervised_model = ROOT / "native_supervised/decision/attempt-001/model"
        baseline = json.loads((supervised_model.parent / "report.json").read_text())
        result["steps"]["native_supervised"] = baseline["test"]["trained_calibrated"]
        result["steps"]["http"] = http_test(supervised_model)
        # Same dataset, initialization seed, epochs and batch size for comparison.
        rl = train_typed(
            ROOT / "data/decision_typed_example",
            ROOT / "native_reinforce",
            device="cuda",
            model_config=TypedDecisionConfig(),
            config=TypedTrainingConfig(
                epochs=12, batch_size=16, learning_rate=0.001, method="reinforce_brier"
            ),
        )
        result["steps"]["native_reinforce"] = rl["test"]["trained_calibrated"]
        # HF backbone test is kept separate from native fixture quality results.
        prepare(ROOT / "data/hf_joint_fixture", train_states=6, heldout_states=3)
        backbone = Path("/home/medmodel/models/Qwen3-Embedding-0.6B")
        hidden_size = json.loads((backbone / "config.json").read_text())["hidden_size"]
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        result["hf_free_memory_before_bytes"] = free
        if free < 6 * 1024**3:
            raise RuntimeError(
                "HF joint test with CPU Adam states requires at least 6 GiB free GPU memory"
            )
        # Limit our allocator to the observed free-memory budget, preserving 1 GiB.
        torch.cuda.set_per_process_memory_fraction((free - 1024**3) / total)
        torch.cuda.reset_peak_memory_stats()
        hf = train_typed(
            ROOT / "data/hf_joint_fixture",
            ROOT / "hf_joint",
            device="cuda",
            model_config=TypedDecisionConfig(
                backbone="hf",
                backbone_name_or_path=str(backbone),
                hidden_size=hidden_size,
                num_heads=8,
                max_length=256,
            ),
            config=TypedTrainingConfig(
                epochs=1,
                batch_size=1,
                learning_rate=0.00002,
                cpu_threads=4,
                optimizer_device="cpu",
            ),
        )
        result["steps"]["hf_joint"] = {
            "backbone_trainable_parameters": hf["backbone_trainable_parameters"],
            "backbone_update_probe": hf["backbone_update_probe"],
            "artifact_reload_verified": hf["artifact_reload_verified"],
            "best_epoch": hf["selection"]["best_epoch"],
            "test": hf["test"]["trained_calibrated"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_device": "cpu",
            "scope": "One epoch on 18 synthetic training rows: integration only",
        }
        result["status"] = "passed"
    except Exception:  # noqa: BLE001 -- Always preserve a result for the remote verification harness.
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        save(ROOT / "verification_result.json", result)
        print(json.dumps({"verification_status": result["status"]}), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
