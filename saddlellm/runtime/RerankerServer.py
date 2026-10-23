"""HTTP reranking service for local Qwen3 reranker artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any, Dict, List, Optional, Sequence


PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a query, retrieve the most relevant passage that answers it"


@dataclass
class RerankerServerSettings:
    model_path: str
    model_name: Optional[str] = None
    device: str = "auto"
    batch_size: int = 16
    max_concurrency: int = 1
    max_documents: int = 100
    max_document_chars: int = 32768
    max_total_chars: int = 524288
    max_length: int = 1024
    instruction: str = DEFAULT_INSTRUCTION
    local_files_only: bool = False
    trust_remote_code: bool = False
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        for name in (
            "batch_size",
            "max_concurrency",
            "max_documents",
            "max_document_chars",
            "max_total_chars",
            "max_length",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.instruction.strip():
            raise ValueError("instruction must not be blank")


class Qwen3RerankerEngine:
    """Thread-safe Qwen3 yes/no relevance scorer."""

    def __init__(self, settings: RerankerServerSettings) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Reranker serving requires torch and transformers; install the serve extra."
            ) from exc
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is unavailable")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            padding_side="left",
            local_files_only=settings.local_files_only,
            trust_remote_code=settings.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            torch_dtype=dtype,
            local_files_only=settings.local_files_only,
            trust_remote_code=settings.trust_remote_code,
        ).to(device).eval()
        self.torch = torch
        self.device = device
        self.settings = settings
        self.lock = RLock()
        self.false_token_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.true_token_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        if self.false_token_id < 0 or self.true_token_id < 0:
            raise RuntimeError("reranker tokenizer does not define yes/no decision tokens")
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.pair_budget = (
            settings.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        if self.pair_budget < 128:
            raise ValueError("max_length is too small for the reranker template")

    def score(
        self, query: str, documents: Sequence[str]
    ) -> tuple[List[float], Dict[str, Any]]:
        pairs = [
            f"<Instruct>: {self.settings.instruction}\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        started = time.perf_counter()
        maximum_tokens = 0
        scores: List[float] = []
        with self.lock:
            for start in range(0, len(pairs), self.settings.batch_size):
                tokenized = self.tokenizer(
                    pairs[start : start + self.settings.batch_size],
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=self.pair_budget,
                )
                input_ids = [
                    self.prefix_tokens + item + self.suffix_tokens
                    for item in tokenized["input_ids"]
                ]
                maximum_tokens = max(maximum_tokens, *(len(item) for item in input_ids))
                padded = self.tokenizer.pad(
                    {"input_ids": input_ids},
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                padded = {key: value.to(self.device) for key, value in padded.items()}
                with self.torch.inference_mode():
                    logits = self.model(**padded).logits[:, -1, :]
                    decision = self.torch.stack(
                        (
                            logits[:, self.false_token_id],
                            logits[:, self.true_token_id],
                        ),
                        dim=1,
                    )
                    values = self.torch.log_softmax(decision, dim=1)[:, 1].exp()
                scores.extend(float(value) for value in values.detach().cpu().tolist())
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        return scores, {
            "documents": len(documents),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "maximum_input_tokens": maximum_tokens,
        }


def _documents(payload: Dict[str, Any], settings: RerankerServerSettings) -> List[str]:
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("documents must be a non-empty array")
    if len(raw_documents) > settings.max_documents:
        raise ValueError(f"documents cannot exceed {settings.max_documents}")
    documents = [
        str(item.get("text") or "") if isinstance(item, dict) else str(item)
        for item in raw_documents
    ]
    if any(not item.strip() for item in documents):
        raise ValueError("document text must not be blank")
    if any(len(item) > settings.max_document_chars for item in documents):
        raise ValueError("a document exceeds the character limit")
    if sum(len(item) for item in documents) > settings.max_total_chars:
        raise ValueError("total document text exceeds the character limit")
    return documents


def create_reranker_app(
    settings: RerankerServerSettings,
    *,
    scorer: Any = None,
):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as exc:
        raise ImportError("FastAPI is required for reranker serving") from exc

    engine = scorer or Qwen3RerankerEngine(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    model_name = settings.model_name or Path(settings.model_path).name or settings.model_path
    app = FastAPI(title="SaddleLLM Reranker API", version="1.0.0")
    app.state.settings = settings
    app.state.engine = engine

    async def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not settings.api_key:
            return
        import secrets

        scheme, _, credential = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            credential, settings.api_key
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": model_name,
            "service": "reranker",
            "requires_auth": bool(settings.api_key),
            "instruction": settings.instruction,
            "max_length": settings.max_length,
        }

    async def rerank(payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        try:
            documents = _documents(payload, settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        top_n = int(payload.get("top_n") or len(documents))
        top_n = max(1, min(top_n, len(documents)))
        async with semaphore:
            scores, usage = await asyncio.to_thread(engine.score, query, documents)
        if len(scores) != len(documents):
            raise RuntimeError("reranker returned an unexpected score count")
        ranked = sorted(
            enumerate(scores), key=lambda item: (-float(item[1]), item[0])
        )[:top_n]
        return {
            "id": str(time.time_ns()),
            "model": model_name,
            "results": [
                {"index": index, "relevance_score": round(float(score), 8)}
                for index, score in ranked
            ],
            "usage": usage,
        }

    app.post("/rerank", dependencies=[Depends(authorize)])(rerank)
    app.post("/v1/rerank", dependencies=[Depends(authorize)])(rerank)
    return app


def api_key_from_env(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable {name!r} is empty or missing")
    return value
