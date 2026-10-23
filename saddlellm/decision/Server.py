"""Bounded Choice decisions from one state encoding and a discriminative head."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import secrets
from threading import RLock
from typing import Any, Dict, Optional


@dataclass
class DecisionServerSettings:
    model_path: str
    device: str = "cpu"
    host: str = "127.0.0.1"
    api_key: Optional[str] = None
    max_batch: int = 16
    max_candidates: int = 100
    max_state_chars: int = 8192
    max_question_chars: int = 2048
    max_choice_chars: int = 2048
    max_total_chars: int = 262144
    max_body_bytes: int = 1048576
    candidate_cache_size: int = 32
    web_ui: bool = True
    encoder_dtype: str = "auto"
    backbone_name_or_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path is required")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda or auto")
        if self.encoder_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("invalid encoder_dtype")
        if self.backbone_name_or_path is not None and (not isinstance(self.backbone_name_or_path, str) or not self.backbone_name_or_path.strip()):
            raise ValueError("backbone_name_or_path must be a nonempty string")
        for name in ("max_batch", "max_candidates", "max_state_chars", "max_question_chars",
                     "max_choice_chars", "max_total_chars", "max_body_bytes", "candidate_cache_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_batch > 16 or not 2 <= self.max_candidates <= 100:
            raise ValueError("max_batch must be <=16 and max_candidates must be 2..100")
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key):
            raise ValueError("api_key must be a nonempty string when configured")


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds the character limit of {limit}")
    return value


def _choices(value: Any, settings: DecisionServerSettings) -> list[dict[str, str]]:
    if not isinstance(value, list) or not 2 <= len(value) <= settings.max_candidates:
        raise ValueError(f"choices must contain 2..{settings.max_candidates} candidates")
    result = []
    ids = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise ValueError("each choice must contain exactly id and text")
        identifier = _text(item["id"], "choice id", 128)
        description = _text(item["text"], "choice text", settings.max_choice_chars)
        if identifier in ids:
            raise ValueError("choice ids must be unique")
        ids.add(identifier)
        result.append({"id": identifier, "text": description})
    return result


def canonical_scope(question: str, choices: list[dict[str, str]]) -> tuple:
    """Order is irrelevant; changing any exact text, ID or question changes scope."""
    return question, tuple(sorted((item["id"], item["text"]) for item in choices))


class ChoiceDecisionEngine:
    def __init__(self, settings: DecisionServerSettings, model: Any = None) -> None:
        import torch

        self.settings = settings
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            from .Model import ChoiceDecisionModel

            dtype = settings.encoder_dtype
            if dtype == "auto":
                dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
            model = ChoiceDecisionModel.from_pretrained(
                settings.model_path, device=device, encoder_device=device,
                encoder_dtype=dtype, backbone_name_or_path=settings.backbone_name_or_path,
            )
            if settings.backbone_name_or_path is not None:
                from .dataset import sha256

                identity = model.config.backbone_identity
                if not identity:
                    raise ValueError("a backbone override requires a recorded backbone identity")
                root = Path(settings.backbone_name_or_path).expanduser().resolve()
                for name, expected in identity.items():
                    path = (root / name).resolve()
                    if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
                        raise ValueError("override backbone does not match the trained backbone identity")
        self.model = model
        self.device = device
        self.lock = RLock()
        self.candidate_cache: OrderedDict[tuple, Any] = OrderedDict()
        metadata = model.metadata
        self.default_question = _text(metadata.get("question"), "artifact question", settings.max_question_chars)
        self.default_choices = _choices(metadata.get("choices"), settings)
        calibration = model.calibration
        self.calibrated = calibration.get("calibrated") is True
        self.temperature = model.calibration_temperature
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (float, int)) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("artifact calibration temperature must be positive and finite")
        scope = calibration.get("metadata", {}).get("scope")
        self.calibration_scope = None
        if isinstance(scope, dict):
            question = _text(scope.get("question"), "calibration question", settings.max_question_chars)
            self.calibration_scope = canonical_scope(question, _choices(scope.get("choices"), settings))

    def validate(self, request: dict) -> tuple[list[str], str, list[dict[str, str]]]:
        unknown = set(request) - {"type", "state", "states", "question", "choices"}
        if unknown:
            raise ValueError("unsupported request fields: " + ", ".join(sorted(unknown)))
        if request.get("type", "Choice") not in {"Choice", "choice"}:
            raise ValueError("only Choice decisions are supported")
        if ("state" in request) == ("states" in request):
            raise ValueError("provide exactly one of state or states")
        states = [request["state"]] if "state" in request else request["states"]
        if not isinstance(states, list) or not 1 <= len(states) <= self.settings.max_batch:
            raise ValueError(f"states must contain 1..{self.settings.max_batch} items")
        states = [_text(state, "state", self.settings.max_state_chars) for state in states]
        question = _text(request.get("question", self.default_question), "question", self.settings.max_question_chars)
        choices = _choices(request.get("choices", self.default_choices), self.settings)
        total_chars = sum(map(len, states)) + len(question) + sum(len(item["id"]) + len(item["text"]) for item in choices)
        if total_chars > self.settings.max_total_chars:
            raise ValueError("request exceeds total character limit")
        return states, question, choices

    def decide(self, states: list[str], question: str, choices: list[dict[str, str]]) -> dict:
        import torch

        scope = canonical_scope(question, choices)
        if not self.calibrated:
            status, temperature = "uncalibrated", 1.0
        elif scope != self.calibration_scope:
            status, temperature = "out_of_scope", 1.0
        else:
            status, temperature = "calibrated", float(self.temperature)
        canonical_choices = scope[1]
        with self.lock, torch.inference_mode():
            candidates = self.candidate_cache.get(scope)
            if candidates is None:
                candidates = self.model.encode_choices([text for _, text in canonical_choices], question, device=self.device)
                self.candidate_cache[scope] = candidates
                while len(self.candidate_cache) > self.settings.candidate_cache_size:
                    self.candidate_cache.popitem(last=False)
            self.candidate_cache.move_to_end(scope)
            features = self.model.encode_states(states, question, device=self.device)
            logits = self.model.score(features, candidates)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (len(states), len(choices)) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("decision head returned invalid logits")
            logits = logits.detach().to(device="cpu", dtype=torch.float64)
            probabilities = torch.softmax((logits - logits.max(dim=-1, keepdim=True).values) / temperature, dim=-1)
            if not bool(torch.isfinite(probabilities).all()):
                raise RuntimeError("decision head returned invalid probabilities")
            rows = probabilities.tolist()
        indices = {identifier: index for index, (identifier, _) in enumerate(canonical_choices)}
        data = []
        for index, row in enumerate(rows):
            values = [{**item, "probability": row[indices[item["id"]]]} for item in choices]
            # Stable canonical tie breaking, independent of request presentation order.
            winner = canonical_choices[max(range(len(row)), key=row.__getitem__)][0]
            data.append({"index": index, "choice_id": winner, "probabilities": values})
        return {"type": "Choice", "data": data, "question": question,
                "calibration_status": status, "temperature_used": temperature}


class _BodyLimitMiddleware:
    def __init__(self, app, *, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        messages, count = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            count += len(message.get("body", b""))
            if count > self.limit:
                from starlette.responses import JSONResponse

                await JSONResponse({"detail": "request body exceeds byte limit"}, status_code=413)(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)


def create_decision_app(settings: DecisionServerSettings, *, model: Any = None):
    import json

    if model is not None:
        from .TypedModel import TypedDecisionModel
        typed = isinstance(model, TypedDecisionModel)
    else:
        payload = json.loads((Path(settings.model_path) / "decision_config.json").read_text(encoding="utf-8"))
        typed = payload.get("model_type") == "saddle_typed_decision"
    if typed:
        from .TypedServer import create_typed_decision_app
        return create_typed_decision_app(settings, model=model)
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse

    engine = ChoiceDecisionEngine(settings, model=model)
    app = FastAPI(title="SaddleLLM Choice Decision API", version="1.0.0")
    app.add_middleware(_BodyLimitMiddleware, limit=settings.max_body_bytes)
    app.state.settings, app.state.engine = settings, engine

    async def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        scheme, _, value = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "decision", "model": Path(settings.model_path).name,
                "supported_types": ["Choice"], "requires_auth": settings.api_key is not None}

    @app.get("/v1/decisions/schema", dependencies=[Depends(authorize)])
    async def schema() -> dict:
        return {"type": "Choice", "question": engine.default_question, "choices": engine.default_choices,
                "max_batch": settings.max_batch, "max_candidates": settings.max_candidates}

    @app.post("/v1/decisions", dependencies=[Depends(authorize)])
    async def decisions(request: Dict[str, Any]) -> dict:
        try:
            inputs = engine.validate(request)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return await asyncio.to_thread(engine.decide, *inputs)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail="Decision computation failed") from exc

    if settings.web_ui:
        @app.get("/", response_class=HTMLResponse)
        async def web() -> str:
            return _WEB_PAGE

    return app


_WEB_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Choice 决策</title>
<style>body{font:16px system-ui;max-width:780px;margin:40px auto;padding:0 18px;color:#18212f;background:#f7f8fa}textarea,input,button{font:inherit;box-sizing:border-box;padding:10px;width:100%;margin:8px 0}textarea{min-height:140px}button{background:#175cd3;color:white;border:0;border-radius:6px;cursor:pointer}pre{white-space:pre-wrap;background:white;padding:16px}label{display:block;margin-top:14px}</style>
<h1>Choice 决策</h1><p>输入当前状态，直接获得候选选项的概率。概率是否经过任务内校准会单独显示。</p>
<label>API Key（服务启用认证时填写）<input id="key" type="password" autocomplete="off"></label>
<label>当前状态<textarea id="state" placeholder="描述需要判断的状态"></textarea></label>
<button id="run">计算决策</button><p id="status" role="status"></p><pre id="result"></pre>
<script>const el=id=>document.getElementById(id);el('run').onclick=async()=>{el('run').disabled=true;el('status').textContent='正在计算…';try{const headers={'Content-Type':'application/json'};if(el('key').value)headers.Authorization='Bearer '+el('key').value;const r=await fetch('/v1/decisions',{method:'POST',headers,body:JSON.stringify({type:'Choice',state:el('state').value})});const d=await r.json();if(!r.ok)throw Error(d.detail||'请求失败');el('status').textContent='校准状态：'+d.calibration_status;el('result').textContent=d.question+'\\n\\n'+d.data[0].probabilities.map(c=>c.id+' · '+c.text+'：'+(100*c.probability).toFixed(2)+'%').join('\\n');}catch(e){el('status').textContent=String(e);el('result').textContent='';}finally{el('run').disabled=false;}};</script></html>"""


__all__ = ["DecisionServerSettings", "ChoiceDecisionEngine", "canonical_scope", "create_decision_app"]
