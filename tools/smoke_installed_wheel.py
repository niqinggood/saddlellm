"""Run with python -I, outside the checkout, after installing a built wheel.

No downloads: creates a random tiny CPU model, starts the installed CLI on
loopback, checks SDK/HTTP/assets/history policy, and stops only its own child.
Not a quality, GPU, training or performance benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path


def create_fixture(directory: Path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    torch.manual_seed(7)
    model = SaddleForCausalLM(SaddleModelConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=[],
    )).eval()
    model.save_pretrained(str(directory))
    vocab = {"<pad>": 0, "<unk>": 1, **{f"word{i}": i for i in range(2, 16)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}word3 "
    tokenizer.save_pretrained(str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-stream-batching", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; preserve earlier validation reports")
    import saddlellm
    import httpx
    from openai import OpenAI

    source_root = args.source_root.resolve()
    installed_path = Path(saddlellm.__file__).resolve()
    if installed_path.is_relative_to(source_root) or Path.cwd().resolve().is_relative_to(source_root):
        raise RuntimeError("Run outside the checkout with python -I and an independently installed wheel")
    report = {"version": saddlellm.__version__, "installed_package": str(installed_path),
              "scope": "isolated wheel; offline tiny CPU model; real CLI and loopback SDK", "accepted": False}
    report["runtime_versions"] = {name: metadata.version(name) for name in (
        "torch", "transformers", "accelerate", "peft", "fastapi", "starlette", "uvicorn",
        "pydantic", "openai", "httpx", "tokenizers", "huggingface-hub",
    )}
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="saddle-installed-smoke-") as temporary, tempfile.TemporaryFile() as log:
            work = Path(temporary)
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            env.pop("PYTHONPATH", None)
            report["pip_check"] = subprocess.run([sys.executable, "-m", "pip", "check"],
                                                  capture_output=True, text=True, check=True, timeout=60).stdout.strip()
            doctor = subprocess.run([sys.executable, "-I", "-c", "from saddlellm.cli import main; raise SystemExit(main())",
                                     "doctor", "--profile", "native", "--device", "cpu"],
                                    cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", check=True, timeout=90)
            report["doctor"] = json.loads(doctor.stdout)
            create_fixture(work / "model")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            command = [sys.executable, "-I", "-c", "from saddlellm.cli import main; raise SystemExit(main())",
                       "serve", str(work / "model"), "--backend", "native", "--device", "cpu", "--dtype", "float32",
                       "--host", "127.0.0.1", "--port", str(port), "--local-files-only", "--no-chat-history",
                       "--max-input-tokens", "64", "--max-new-tokens", "8", "--default-max-new-tokens", "4"]
            if args.native_stream_batching:
                command += ["--native-stream-batching", "--batch-size", "4", "--max-concurrency", "4",
                            "--batch-token-budget", "512"]
            process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                with httpx.Client(base_url=base, trust_env=False, timeout=20) as http:
                    deadline = time.monotonic() + 90
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("Installed CLI exited before readiness")
                        try:
                            health = http.get("/health").json()
                            if health.get("ready"):
                                break
                        except (httpx.HTTPError, ValueError):
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Installed service readiness timeout")
                        time.sleep(0.2)
                    assert health["chat_history_enabled"] is False
                    assert 'data-history-enabled="false"' in http.get("/").text
                    for asset in ("chat.js", "chat.css", "marked.min.js", "katex/katex.min.js", "katex/fonts/KaTeX_Main-Regular.woff2"):
                        assert http.get("/assets/" + asset).status_code == 200, asset
                    assert http.get("/api/chat/sessions").status_code == 404
                    with OpenAI(api_key="local-smoke-only", base_url=base + "/v1",
                                http_client=httpx.Client(trust_env=False, timeout=30), max_retries=0) as sdk:
                        alias = sdk.models.list().data[0].id
                        response = sdk.chat.completions.create(model=alias, messages=[{"role": "user", "content": "word4 word5"}],
                                                              max_tokens=4, temperature=0)
                        assert response.usage.completion_tokens == 4
                        with sdk.chat.completions.create(model=alias, messages=[{"role": "user", "content": "word4 word5"}],
                                                        max_tokens=4, temperature=0, stream=True,
                                                        stream_options={"include_usage": True}) as stream:
                            chunks = list(stream)
                        assert chunks[-1].usage.completion_tokens == 4
                        assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == response.choices[0].message.content
                        completion = sdk.completions.create(model=alias, prompt="word4", max_tokens=2, temperature=0)
                        assert completion.usage.completion_tokens == 2
                        if args.native_stream_batching:
                            def chat_stream(_):
                                with sdk.chat.completions.create(
                                    model=alias, messages=[{"role": "user", "content": "word4 word5"}],
                                    max_tokens=4, temperature=0, stream=True, stream_options={"include_usage": True},
                                ) as parts:
                                    rows = list(parts)
                                assert rows[-1].usage.completion_tokens == 4
                                return "".join(choice.delta.content or "" for row in rows for choice in row.choices)

                            with ThreadPoolExecutor(4) as pool:
                                texts = list(pool.map(chat_stream, range(4)))
                            assert all(text == response.choices[0].message.content for text in texts)
                    assert not (work / ".saddlellm").exists()
                    final_health = http.get("/health").json()
                    assert final_health["admission"]["active"] == 0
                    if args.native_stream_batching:
                        batch_status = final_health["native_stream_batching"]
                        assert batch_status["batched_forwards"] > 0
                        assert all(batch_status[key] == 0 for key in ("registered", "pending", "inflight", "retained_cohorts"))
                        report["concurrent_stream_texts_equal"] = True
                    report.update(accepted=True, health=final_health,
                                  checks=["pip_check", "doctor", "CLI", "CPU native generation", "SDK chat/completion/SSE",
                                          "packaged assets", "history disabled", "admission released"])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
                log.seek(0)
                report["service_log"] = log.read().decode("utf-8", errors="replace")[-6000:]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"accepted": report["accepted"], "report": str(args.output), "error": report.get("error")}))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
