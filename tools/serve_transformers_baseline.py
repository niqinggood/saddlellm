"""Minimal serial, non-streaming Transformers HTTP baseline; loopback only.

Deliberately no SaddleLLM generation engine, batching, prefix cache or adapters.
Shares the official checkpoint/runtime with SaddleLLM. Not a production server.
"""

from __future__ import annotations

import argparse
import threading
import time
from importlib.metadata import version


def create_app(model, tokenizer, *, alias="qwen3-8b", context_limit=4096):
    import torch
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class Request(BaseModel):
        model: str = alias
        prompt: str = Field(min_length=1)
        max_tokens: int = Field(default=64, ge=1, le=512)
        temperature: float = 0.0
        top_p: float = 1.0
        stream: bool = False

    app = FastAPI(title="Raw Transformers experiment baseline")
    lock = threading.Lock()

    @app.get("/health")
    @app.get("/readyz")
    def health():
        return {
            "status": "ready",
            "model": alias,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "attention": model.config._attn_implementation,
            "torch": version("torch"),
            "transformers": version("transformers"),
            "streaming": False,
            "max_concurrent_generations": 1,
        }

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": alias, "object": "model"}]}

    def complete(request):
        if request.model != alias:
            raise HTTPException(404, "Unknown model alias")
        if request.stream or request.temperature != 0 or request.top_p != 1:
            raise HTTPException(
                400, "Baseline supports non-streaming greedy decoding only"
            )
        # Include preprocessing in the serial permit just like Saddle's HF path.
        with lock:
            encoded = tokenizer(request.prompt, return_tensors="pt")
            prompt_tokens = encoded["input_ids"].shape[-1]
            if prompt_tokens + request.max_tokens > context_limit:
                raise HTTPException(400, "Prompt and output exceed context budget")
            encoded = {name: value.to(model.device) for name, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=request.max_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            ids = output[0, prompt_tokens:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = (
            "stop"
            if len(ids) < request.max_tokens or ids[-1:] and ids[-1] in eos
            else "length"
        )
        return {
            "id": "raw-transformers",
            "object": "text_completion",
            "created": int(time.time()),
            "model": alias,
            "choices": [{"index": 0, "text": text, "finish_reason": reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(ids),
                "total_tokens": prompt_tokens + len(ids),
            },
        }

    # A local request type must be bound explicitly with postponed annotations.
    complete.__annotations__["request"] = Request
    app.post("/v1/completions")(complete)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8210)
    parser.add_argument("--alias", default="qwen3-8b")
    parser.add_argument("--context-limit", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.context_limit < 1:
        parser.error("invalid port or context limit")
    import torch
    import uvicorn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This strict BF16 GPU baseline requires CUDA; no CPU fallback"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    app = create_app(
        model, tokenizer, alias=args.alias, context_limit=args.context_limit
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
