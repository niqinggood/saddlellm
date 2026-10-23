"""Create an untrained tiny fixture / probe a loopback vLLM Saddle API.

This validates real GPU integration, not Qwen3-8B speed or model capability.
All prompts are non-sensitive. Each output directory must be new.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit


def prepare_fixture(output, tokenizer_source):
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    output = Path(output)
    if output.exists():
        raise FileExistsError("Fixture output must be new")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    torch.manual_seed(173)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    fixture = {
        "warning": "Random untrained tiny Qwen3 fixture; NOT Qwen3-8B and NOT a capability/speed benchmark",
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": 2,
        "hidden_size": 256,
        "context_limit": 128,
        "tokenizer_source": str(tokenizer_source),
        "seed": 173,
    }
    (output / "integration_fixture.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )
    print(json.dumps(fixture), flush=True)


async def probe(url, output):
    import httpx

    if urlsplit(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Integration probe only accepts loopback endpoints")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Probe output must be new")
    report = {
        "scope": "Real GPU/HTTP integration using an untrained tiny fixture, not an 8B performance or quality benchmark",
        "started_unix": time.time(),
    }
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=120) as client:
        deadline = time.perf_counter() + 120
        while True:
            try:
                response = await client.get("/readyz", timeout=3)
                if response.is_success:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() >= deadline:
                raise RuntimeError("Saddle vLLM adapter did not become ready")
            await asyncio.sleep(0.5)
        health = (await client.get("/health")).json()
        assert (
            health["backend"] == "vllm" and health["streaming_backend"] == "vllm_async"
        )
        assert health["batching"]["mode"] == "vllm_continuous" and health["ready"]
        report["health_before"] = health
        model = health["model"]

        async def request(i):
            response = await client.post(
                "/v1/completions",
                json={
                    "model": model,
                    "prompt": f"请输出数字。编号{i}。",
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 8,
                },
            )
            response.raise_for_status()
            body = response.json()
            usage = body["usage"]
            assert 0 < usage["completion_tokens"] <= 8
            assert (
                usage["total_tokens"]
                == usage["prompt_tokens"] + usage["completion_tokens"]
            )
            return body

        report["parallel_completions"] = await asyncio.gather(
            *(request(i) for i in range(4))
        )
        for chat in [False, True]:
            body = {
                "model": model,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 8,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            body.update(
                {"messages": [{"role": "user", "content": "你好"}]}
                if chat
                else {"prompt": "你好"}
            )
            events, text, done = [], "", False
            async with client.stream(
                "POST", "/v1/chat/completions" if chat else "/v1/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line == "data: [DONE]":
                        done = True
                    elif line.startswith("data: "):
                        event = json.loads(line[6:])
                        assert "error" not in event
                        events.append(event)
                        if event["choices"]:
                            choice = event["choices"][0]
                            text += (
                                choice["delta"].get("content", "")
                                if chat
                                else choice["text"]
                            )
            assert done and events[-1]["choices"] == [] and "usage" in events[-1]
            assert events[-2]["choices"][0]["finish_reason"] in {"stop", "length"}
            report["chat_stream" if chat else "completion_stream"] = {
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "usage": events[-1]["usage"],
                "done": done,
            }
        report["health_after"] = (await client.get("/health")).json()
        assert report["health_after"]["admission"]["in_flight"] == 0
        assert report["health_after"]["admission"]["completed"] == 6
    output.mkdir(parents=True)
    (output / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"status": "passed", "requests": 6, "output": str(output)}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "probe"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-source")
    parser.add_argument("--url", default="http://127.0.0.1:8214")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.tokenizer_source:
            parser.error("prepare requires --tokenizer-source")
        prepare_fixture(args.output_dir, args.tokenizer_source)
    else:
        asyncio.run(probe(args.url, args.output_dir))


if __name__ == "__main__":
    main()
