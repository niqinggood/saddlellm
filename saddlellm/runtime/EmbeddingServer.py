"""OpenAI-compatible embedding service for local sentence-transformer models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class EmbeddingServerSettings:
    model_path: str
    model_name: Optional[str] = None
    device: str = "auto"
    batch_size: int = 32
    max_concurrency: int = 1
    max_inputs: int = 256
    max_input_chars: int = 32768
    max_total_chars: int = 262144
    max_input_tokens: int = 8192
    expected_dimensions: Optional[int] = None
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
            "max_inputs",
            "max_input_chars",
            "max_total_chars",
            "max_input_tokens",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.expected_dimensions is not None and self.expected_dimensions < 1:
            raise ValueError("expected_dimensions must be positive")


class SentenceEmbeddingEngine:
    """Thread-safe adapter with explicit input and output validation."""

    def __init__(self, settings: EmbeddingServerSettings) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Embedding serving requires sentence-transformers; install "
                "SaddleLLM with the embedding extra."
            ) from exc
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is unavailable")
        model_kwargs = {
            "local_files_only": settings.local_files_only,
            "trust_remote_code": settings.trust_remote_code,
        }
        self.model = SentenceTransformer(
            settings.model_path,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"local_files_only": settings.local_files_only},
        )
        self.settings = settings
        self.lock = RLock()
        configured_limit = int(getattr(self.model, "max_seq_length", 0) or 0)
        if configured_limit:
            self.model.max_seq_length = min(
                configured_limit, settings.max_input_tokens
            )

    def embed(self, texts: Sequence[str]) -> tuple[List[List[float]], int]:
        tokenizer = getattr(self.model, "tokenizer", None)
        prompt_tokens = 0
        if tokenizer is not None:
            tokenized = tokenizer(
                list(texts), add_special_tokens=True, truncation=False, padding=False
            )
            lengths = [len(row) for row in tokenized["input_ids"]]
            if max(lengths, default=0) > self.settings.max_input_tokens:
                raise ValueError(
                    "input token count exceeds the configured maximum of "
                    f"{self.settings.max_input_tokens}"
                )
            prompt_tokens = sum(lengths)
        with self.lock:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.settings.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return _normalized_rows(
            vectors,
            len(texts),
            self.settings.expected_dimensions,
        ), prompt_tokens


def _normalized_rows(
    vectors: Any,
    expected_rows: int,
    expected_dimensions: Optional[int] = None,
) -> List[List[float]]:
    rows = list(vectors)
    if len(rows) != expected_rows:
        raise RuntimeError("embedding model returned an unexpected vector count")
    dimensions = expected_dimensions
    output: List[List[float]] = []
    for raw in rows:
        values = [float(value) for value in list(raw)]
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError("embedding vectors must contain finite values")
        if dimensions is None:
            dimensions = len(values)
        if len(values) != dimensions:
            raise RuntimeError("embedding vectors have inconsistent dimensions")
        norm = math.sqrt(math.fsum(value * value for value in values))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("embedding vectors must have a positive finite norm")
        output.append([value / norm for value in values])
    return output


def create_embedding_app(
    settings: EmbeddingServerSettings,
    *,
    encoder: Any = None,
):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as exc:
        raise ImportError("FastAPI is required for embedding serving") from exc

    engine = encoder or SentenceEmbeddingEngine(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    model_name = settings.model_name or Path(settings.model_path).name or settings.model_path
    dimensions = settings.expected_dimensions

    app = FastAPI(title="SaddleLLM OpenAI-Compatible Embedding API", version="1.0.0")
    app.state.settings = settings
    app.state.engine = engine

    async def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not settings.api_key:
            return
        scheme, _, credential = (authorization or "").partition(" ")
        import secrets

        if scheme.lower() != "bearer" or not secrets.compare_digest(
            credential, settings.api_key
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": model_name,
            "dimensions": dimensions,
            "normalized": True,
            "requires_auth": bool(settings.api_key),
            "service": "embedding",
        }

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "saddlellm"}],
        }

    @app.post("/v1/embeddings", dependencies=[Depends(authorize)])
    async def embeddings(request: Dict[str, Any]) -> Dict[str, Any]:
        encoding_format = request.get("encoding_format", "float")
        requested_dimensions = request.get("dimensions")
        raw_input = request.get("input")
        if encoding_format != "float":
            raise HTTPException(status_code=400, detail="Only encoding_format='float' is supported")
        if (
            requested_dimensions is not None
            and dimensions is not None
            and requested_dimensions != dimensions
        ):
            raise HTTPException(status_code=400, detail=f"dimensions must equal {dimensions}")
        texts = [raw_input] if isinstance(raw_input, str) else raw_input
        if not isinstance(texts, list) or not texts or len(texts) > settings.max_inputs:
            raise HTTPException(status_code=400, detail=f"input must contain 1 to {settings.max_inputs} items")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise HTTPException(status_code=400, detail="input strings must not be blank")
        if any(len(text) > settings.max_input_chars for text in texts):
            raise HTTPException(status_code=400, detail="an input exceeds the character limit")
        if sum(len(text) for text in texts) > settings.max_total_chars:
            raise HTTPException(status_code=400, detail="total input exceeds the character limit")
        async with semaphore:
            try:
                vectors, prompt_tokens = await asyncio.to_thread(engine.embed, texts)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vector, "index": index}
                for index, vector in enumerate(vectors)
            ],
            "model": model_name,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

    return app


def api_key_from_env(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable {name!r} is empty or missing")
    return value
