"""Bounded multi-question typed API, using the existing server's limits/auth."""

# ruff: noqa: N999
from __future__ import annotations

import asyncio
from threading import RLock

from .TypedSchema import validate_question, validate_record


def create_typed_decision_app(settings, *, model=None):
    import secrets

    import torch
    from fastapi import Depends, FastAPI, Header, HTTPException

    from .Server import _BodyLimitMiddleware
    from .TypedModel import TypedDecisionModel

    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if settings.backbone_name_or_path is not None:
        raise ValueError(
            "typed artifacts bundle their backbone; --backbone is not applicable"
        )
    model = model or TypedDecisionModel.from_pretrained(
        settings.model_path, device=device
    )
    model.eval()
    lock = RLock()
    app = FastAPI(title="SaddleLLM Typed Decision API", version="2.0.0")
    app.add_middleware(_BodyLimitMiddleware, limit=settings.max_body_bytes)
    app.state.model, app.state.settings = model, settings

    async def authorize(authorization: str | None = Header(default=None)):
        if settings.api_key is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token, settings.api_key
        ):
            raise HTTPException(
                status_code=401, detail="Invalid or missing bearer token"
            )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "decision",
            "supported_types": ["Choice", "Score", "Noul"],
            "requires_auth": settings.api_key is not None,
            "backbone_bundled": True,
        }

    @app.get("/v1/decisions/schema", dependencies=[Depends(authorize)])
    async def schema():
        return {
            "supported_types": ["Choice", "Score", "Noul"],
            "input": {
                "state": "string",
                "questions": "array of independent typed questions",
                "instance_question": "optional nonblank string on each question",
            },
            "max_questions": settings.max_batch,
            "max_candidates": settings.max_candidates,
            "score": "levels require strictly increasing numeric value, id and text",
            "noul": "returns probability_true",
            "choice": "returns choice_id",
            "probability_summary": "max_probability is the maximum candidate probability",
        }

    def compute(rows):
        with lock:
            return {"data": model.predict(rows), "shared_state_encoding": True}

    @app.post("/v1/decisions", dependencies=[Depends(authorize)])
    async def decisions(request: dict):
        try:
            if set(request) != {"state", "questions"}:
                raise ValueError("provide exactly state and questions")
            state, questions = request["state"], request["questions"]
            if (
                not isinstance(state, str)
                or not state.strip()
                or len(state) > settings.max_state_chars
            ):
                raise ValueError("invalid state or state character limit exceeded")
            if (
                not isinstance(questions, list)
                or not 1 <= len(questions) <= settings.max_batch
            ):
                raise ValueError("question count exceeds max_batch or is empty")
            validated, ids, total = [], set(), len(state)
            for item in questions:
                question = validate_question(
                    item,
                    max_candidates=min(
                        settings.max_candidates, model.config.max_candidates
                    ),
                )
                if len(question["question"]) > settings.max_question_chars:
                    raise ValueError("question character limit exceeded")
                instance_question = question.get("instance_question")
                if (
                    instance_question is not None
                    and len(instance_question) > settings.max_question_chars
                ):
                    raise ValueError("instance question character limit exceeded")
                identifier = question.get("id")
                if identifier is not None:
                    if identifier in ids:
                        raise ValueError("question IDs must be unique")
                    ids.add(identifier)
                total += len(question["question"])
                if instance_question is not None:
                    total += len(instance_question)
                for c in question.get("choices", question.get("levels", [])):
                    if len(c["text"]) > settings.max_choice_chars:
                        raise ValueError("candidate character limit exceeded")
                    total += len(c["id"]) + len(c["text"])
                validated.append(
                    validate_record(
                        {**question, "state": state},
                        require_target=False,
                        max_candidates=model.config.max_candidates,
                    )
                )
            if total > settings.max_total_chars:
                raise ValueError("request exceeds total character limit")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return await asyncio.to_thread(compute, validated)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500, detail="Decision computation failed"
            ) from exc

    return app
