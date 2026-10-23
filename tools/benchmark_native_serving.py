"""Real loopback HTTP benchmark on a shared random, untrained native model.

Compare individual, equal-length microbatch and opt-in length buckets. No
download, checkpoint write or external service. --stream measures first visible
text on the individual path; streaming does NOT use microbatch scheduling.
All local servers share one immutable model and are measured sequentially.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import socket
import threading
import time
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import httpx
import torch
import uvicorn

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
)

try:
    from tools.benchmark_serving import compare
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    # Executing a file puts tools/, rather than the repository root, on sys.path.
    from benchmark_serving import compare


class NumericTokenizer:
    """Synthetic numeric tokens, not a real language tokenizer/quality benchmark."""

    pad_token_id = 0
    eos_token_id = None

    def __call__(self, text, *, max_length, **kwargs):
        values = [int(item) for item in text.split()][:max_length]
        ids = torch.tensor([values], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, **kwargs):
        return "".join(
            f"{int(item)} " for item in ids if int(item) != self.pad_token_id
        )


@contextmanager
def loopback_server(app):
    """Bind only an OS-assigned loopback port; stop and join on every exit."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error", access_log=False
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("local benchmark server failed to start")
            if time.monotonic() > deadline:
                raise RuntimeError("local benchmark server startup timed out")
            # This helper runs outside the async request loop.
            time.sleep(0.005)
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("local benchmark server did not stop")


def run(args):
    lengths = [int(value) for value in args.lengths.split(",")]
    if (
        not lengths
        or min(
            lengths
            + [
                args.requests,
                args.concurrency,
                args.decode_tokens,
                args.repeats,
                args.batch_size,
                args.bucket_size,
            ]
        )
        < 1
    ):
        raise ValueError("lengths, counts and bucket size must be positive")
    if args.requests > 65536 or args.warmup < 0:
        raise ValueError("requests must be <= 65536 and warmup non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(173)
    context_limit = max(lengths) + args.decode_tokens
    model = (
        SaddleForCausalLM(
            SaddleModelConfig(
                vocab_size=1024,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                max_position_embeddings=context_limit,
                attention_backend="sdpa",
                eos_token_id=[],
            )
        )
        .to(device=args.device, dtype=getattr(torch, args.dtype))
        .eval()
    )
    model.prewarm_rope_cache(context_limit, freeze=True)
    tokenizer = NumericTokenizer()
    prompts = [
        " ".join(
            str(3 + (index * 37 + position * 13) % 1021) for position in range(length)
        )
        for index, length in enumerate(lengths)
    ]
    modes = {"individual": (1, 0)}
    if not args.stream:
        modes.update(
            {
                "exact_microbatch": (args.batch_size, 0),
                "length_bucket": (args.batch_size, args.bucket_size),
            }
        )
    with ExitStack() as stack:
        targets = []
        for label, (batch_size, bucket_size) in modes.items():
            settings = InferenceServerSettings(
                model_name="synthetic-native",
                web_ui=False,
                max_input_tokens=max(lengths),
                default_max_new_tokens=args.decode_tokens,
                max_new_tokens=args.decode_tokens,
                batch_size=batch_size,
                batch_length_bucket_size=bucket_size,
                batch_token_budget=args.batch_size * context_limit,
                max_pending_requests=max(64, args.concurrency),
                batch_wait_ms=5,
            )
            # Explicit budget enables the individual queue too, making work-aware
            # admission/order comparable. B=1 coalescing never adds a delay.
            app = create_inference_app(settings, model=model, tokenizer=tokenizer)
            url = stack.enter_context(loopback_server(app))
            targets.append(f"{label}={url}")
        compare_args = SimpleNamespace(
            target=targets,
            model="synthetic-native",
            prompts_file=None,
            prompt=None,
            requests=args.requests,
            concurrency=args.concurrency,
            max_tokens=args.decode_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            timeout=120,
            api_key_env=None,
            checkpoint_sha256=None,
            cache_policy="prefix cache disabled; same synthetic prompts",
            stream=args.stream,
        )
        report = asyncio.run(compare(compare_args, prompts_override=prompts))
        for target, record in zip(targets, report["targets"].values()):
            _, url = target.split("=", 1)
            with httpx.Client(trust_env=False, timeout=10) as client:
                record["batching_health"] = client.get(url[:-3] + "/health").json()[
                    "batching"
                ]
    report.update(
        {
            "scope": "real loopback HTTP, same random untrained 4-layer model; synthetic numeric tokens",
            "hardware": torch.cuda.get_device_name()
            if args.device == "cuda"
            else platform.processor(),
            "torch_version": torch.__version__,
            "serving_dependencies": {
                name: importlib.metadata.version(name)
                for name in (
                    "fastapi",
                    "uvicorn",
                    "pydantic",
                    "transformers",
                    "accelerate",
                    "peft",
                )
            },
            "python_version": platform.python_version(),
            "device": args.device,
            "dtype": args.dtype,
            "prompt_lengths": lengths,
            "batch_size": args.batch_size,
            "bucket_size": args.bucket_size,
            "seed": 173,
            "shared_model_targets_measured_sequentially": True,
            "microbatch_enabled_in_streaming": False,
        }
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--lengths", default="33,48,57,62,129,144,153,158")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["all_token_counts_equal"] or not report["all_output_text_equal"]:
        # Keep stdout JSON intact in collectors merging stdout and stderr.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
