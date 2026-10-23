import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_script("serve_transformers_baseline")
preparation = load_script("prepare_qwen3_experiment")


def test_raw_baseline_real_tiny_qwen3_cpu_and_request_validation():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            eos_token_id=None,
            pad_token_id=0,
            bos_token_id=1,
        )
    ).eval()
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        decode=lambda ids, **kw: " ".join(map(str, ids)),
    )

    class Tokenizer:
        pad_token_id = 0
        decode = staticmethod(tokenizer.decode)

        def __call__(self, prompt, **kwargs):
            return {
                "input_ids": torch.tensor([[2, 3, 4]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    async def run():
        app = baseline.create_app(model, Tokenizer(), context_limit=16)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            assert (await client.get("/readyz")).json()[
                "max_concurrent_generations"
            ] == 1
            body = (
                await client.post(
                    "/v1/completions", json={"prompt": "x", "max_tokens": 2}
                )
            ).json()
            assert body["usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
            assert body["choices"][0]["finish_reason"] == "length"
            for payload in [{"stream": True}, {"temperature": 0.7}, {"max_tokens": 15}]:
                response = await client.post(
                    "/v1/completions", json={"prompt": "x", **payload}
                )
                assert response.status_code == 400
            assert (
                await client.post(
                    "/v1/completions", json={"prompt": "x", "model": "wrong"}
                )
            ).status_code == 404

    asyncio.run(run())


def test_prepare_tokenizer_only_verified_hashes_fresh_output(monkeypatch, tmp_path):
    import transformers

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"test weight identity only")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"] + "\n<think>\n\n</think>\n\n"

        def encode(self, text, **kwargs):
            return [1, 2, 3]

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    output = tmp_path / "prepared"
    manifest = preparation.prepare(model, output, hash_weights=True)
    assert manifest["weight_hashes_computed"]
    assert not manifest["official_repository_verified"]
    assert (
        len(json.loads((output / "performance.json").read_text(encoding="utf-8"))) == 16
    )
    assert len(manifest["quality_cases"]) == 8
    assert manifest["files"]["model.safetensors"]["sha256"] == preparation.file_hash(
        model / "model.safetensors"
    )
    with pytest.raises(FileExistsError):
        preparation.prepare(model, output)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter"):
        preparation.prepare(model, tmp_path / "not-written")


def test_matrix_runs_identical_phases_and_captures_samples(monkeypatch, tmp_path):
    benchmark = load_script("benchmark_serving")
    monkeypatch.setitem(sys.modules, "benchmark_serving", benchmark)
    matrix = load_script("run_qwen3_serving_experiment")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps({"files_manifest_sha256": "identity"}), encoding="utf-8"
    )
    phases = []

    async def fake_compare(args):
        phases.append(args)
        return {
            "targets": {
                "saddle": {
                    "generated_tokens_per_second": 10,
                    "p50_response_ms": 1,
                    "p95_response_ms": 2,
                }
            }
        }

    monkeypatch.setattr(benchmark, "compare", fake_compare)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        matrix.httpx,
        "AsyncClient",
        lambda **kw: original_client(
            **kw,
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"model": "qwen"})
            ),
        ),
    )
    monkeypatch.setattr(
        matrix.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="test GPU")
    )
    args = SimpleNamespace(
        url="http://127.0.0.1:8211",
        label="saddle",
        model="qwen3-8b",
        prepared_dir=str(prepared),
        output_dir=str(tmp_path / "result"),
        precision="bf16",
        protocol="openai",
        cache_policy="disabled",
        stream=True,
    )
    asyncio.run(matrix.run(args))
    assert [p.concurrency for p in phases] == [1, 4, 8, 1, 1, 4]
    assert all(p.capture_text for p in phases)
    assert all(p.checkpoint_sha256 == "identity" for p in phases)
    assert len(list((tmp_path / "result").glob("*.json"))) == 7
    args.url = "http://example.com:8211"
    with pytest.raises(ValueError, match="loopback"):
        asyncio.run(matrix.run(args))


def test_summary_never_calls_ollama_quantization_equal_precision(tmp_path):
    summary = load_script("summarize_qwen3_experiment")
    for label, protocol, precision, output in [
        ("raw", "openai", "bf16", "same"),
        ("saddle", "openai", "bf16", "same"),
        ("ollama", "ollama", "Q4_K_M", "changed"),
    ]:
        folder = tmp_path / label
        folder.mkdir()
        report = {
            "phase": "performance_c1",
            "protocol": protocol,
            "precision_asserted_by_operator": precision,
            "checkpoint_sha256_asserted_by_operator": "manifest"
            if protocol == "openai"
            else None,
            "prompt_sha256": "prompt",
            "max_tokens": 64,
            "concurrency": 1,
            "requests_per_round": 1,
            "repeats": 1,
            "targets": {
                label: {
                    "generated_tokens_per_second": 40,
                    "p50_response_ms": 100,
                    "p95_response_ms": 100,
                    "p99_response_ms": 100,
                    "samples": [
                        {
                            "text_sha256": output,
                            "prompt_tokens": 3,
                            "completion_tokens": 64,
                        }
                    ],
                }
            },
        }
        (folder / "performance_c1.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    rows = {r["backend"]: r for r in summary.summarize(tmp_path)["rows"]}
    assert rows["saddle"]["same_asserted_model_precision_as_raw"]
    assert not rows["ollama"]["same_asserted_model_precision_as_raw"]
    assert rows["ollama"]["same_workload_as_raw"]
    assert rows["ollama"]["text_mismatches_to_raw"] == 1


def test_preflight_does_not_benchmark_before_backend_is_ready(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "benchmark_serving", load_script("benchmark_serving")
    )
    matrix = load_script("run_qwen3_serving_experiment")
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(200, json={"ready": False}),
            httpx.Response(200, json={"ready": True}),
        ]
    )

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(matrix.asyncio, "sleep", no_sleep)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://local",
            transport=httpx.MockTransport(lambda r: next(responses)),
        ) as client:
            await matrix.wait_ready(client, "openai")
        async with httpx.AsyncClient(
            base_url="http://local",
            transport=httpx.MockTransport(lambda r: httpx.Response(503)),
        ) as client:
            with pytest.raises(RuntimeError, match="no measurements"):
                await matrix.wait_ready(client, "openai", timeout=0)

    asyncio.run(run())
