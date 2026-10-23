"""Dependency-light Web pages for SaddleLLM training runs and chat serving."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _latest(paths: Iterable[Path]) -> Optional[Path]:
    candidates = [path for path in paths if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_run_dir(target: str) -> Path:
    requested = Path(target).expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(f"Training output does not exist: {requested}")
    root = requested.parent if requested.is_file() else requested
    markers = ("pipeline_state.json", "training_summary.json", "summary.json")
    if any((root / marker).is_file() for marker in markers):
        return root
    marker = _latest(
        path
        for name in markers
        for path in root.rglob(name)
    )
    return marker.parent if marker is not None else root


def _read_jsonl(path: Optional[Path], limit: int = 2000) -> List[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    except (OSError, UnicodeDecodeError):
        return []
    return records


def _log_tail(path: Path, max_bytes: int = 65536, lines: int = 80) -> List[str]:
    if not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            content = handle.read().decode("utf-8", errors="replace")
        return content.splitlines()[-lines:]
    except OSError:
        return []


def _stage_rows(pipeline: Mapping[str, Any], summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    stages = pipeline.get("stages")
    if isinstance(stages, Mapping):
        return [
            {
                "name": str(name),
                "status": str(value.get("status", "unknown")) if isinstance(value, Mapping) else "unknown",
                "started_at": value.get("started_at") if isinstance(value, Mapping) else None,
                "finished_at": value.get("finished_at") if isinstance(value, Mapping) else None,
            }
            for name, value in stages.items()
        ]
    results = summary.get("results")
    if isinstance(results, Mapping):
        return [
            {
                "name": str(name),
                "status": str(value.get("status", "unknown")) if isinstance(value, Mapping) else "unknown",
                "started_at": None,
                "finished_at": None,
            }
            for name, value in results.items()
            if name != "pipeline"
        ]
    return []


def _model_path(summary: Mapping[str, Any]) -> Optional[str]:
    results = summary.get("results")
    if isinstance(results, Mapping):
        for stage in reversed(list(results.values())):
            if isinstance(stage, Mapping) and stage.get("model_path"):
                return str(stage["model_path"])
    return None


def collect_run_status(target: str) -> Dict[str, Any]:
    """Collect one stable dashboard payload from an output directory."""
    run_dir = _resolve_run_dir(target)
    summary_path = (
        (run_dir / "training_summary.json")
        if (run_dir / "training_summary.json").is_file()
        else (run_dir / "summary.json")
    )
    summary = _read_json(summary_path)
    pipeline = _read_json(run_dir / "pipeline_state.json")
    progress_path = _latest(run_dir.rglob("training_progress.json"))
    progress = _read_json(progress_path)
    metrics_path = _latest(run_dir.rglob("training_metrics.jsonl"))
    history = _read_jsonl(metrics_path)
    trainer_state_path = _latest(run_dir.rglob("trainer_state.json"))
    trainer_state = _read_json(trainer_state_path)
    if not history:
        raw_history = trainer_state.get("log_history", [])
        if isinstance(raw_history, list):
            history = [item for item in raw_history if isinstance(item, dict)][-2000:]

    train_history = [
        item for item in history if isinstance(item.get("loss"), (int, float))
    ]
    eval_history = [
        item for item in history if isinstance(item.get("eval_loss"), (int, float))
    ]
    latest_metrics: Dict[str, Any] = {}
    if train_history:
        latest_metrics.update(train_history[-1])
    if eval_history:
        latest_metrics.update(eval_history[-1])
    if isinstance(progress.get("latest"), Mapping):
        latest_metrics.update(progress["latest"])

    step = progress.get("global_step", trainer_state.get("global_step", 0))
    max_steps = progress.get("max_steps", trainer_state.get("max_steps", 0))
    try:
        step = int(step or 0)
        max_steps = int(max_steps or 0)
    except (TypeError, ValueError):
        step, max_steps = 0, 0

    status = str(pipeline.get("status") or progress.get("status") or "unknown")
    if status == "unknown" and summary:
        status = "completed"
    gate_path = _latest(run_dir.rglob("*gate*.json"))
    gate = _read_json(gate_path)
    if gate:
        from ..evaluation.ReleaseGate import release_gate_accepted

        gate_accepted = release_gate_accepted(gate)
    else:
        gate_accepted = None

    result_files = sorted(
        {
            str(path.relative_to(run_dir))
            for pattern in (
                "summary.json",
                "training_summary.json",
                "pipeline_state.json",
                "training_progress.json",
                "trainer_state.json",
                "release_manifest.json",
                "NOT_PROMOTED.md",
                "REJECTED.md",
                "*evaluation*.json",
                "*comparison*.json",
                "*gate*.json",
            )
            for path in run_dir.rglob(pattern)
            if path.is_file()
        }
    )[:100]
    return {
        "requested_path": str(Path(target).expanduser().resolve()),
        "run_dir": str(run_dir),
        "status": status,
        "active_stage": pipeline.get("active_stage"),
        "project": summary.get("project") or run_dir.name,
        "experiment": summary.get("experiment"),
        "model": summary.get("model"),
        "model_path": _model_path(summary),
        "progress": {
            "step": step,
            "max_steps": max_steps,
            "percent": round(step * 100 / max_steps, 2) if max_steps > 0 else None,
            "epoch": progress.get("epoch", trainer_state.get("epoch")),
        },
        "latest_metrics": latest_metrics,
        "train_history": train_history,
        "eval_history": eval_history,
        "stages": _stage_rows(pipeline, summary),
        "gate": {
            "accepted": gate_accepted,
            "path": str(gate_path) if gate_path else None,
        },
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "result_files": result_files,
        "log_tail": _log_tail(run_dir / "training.log"),
        "sources": {
            "summary": str(summary_path) if summary else None,
            "pipeline": str(run_dir / "pipeline_state.json") if pipeline else None,
            "progress": str(progress_path) if progress_path else None,
            "metrics": str(metrics_path) if metrics_path else None,
            "trainer_state": str(trainer_state_path) if trainer_state_path else None,
        },
    }


def create_dashboard_app(target: str):
    """Create the built-in, read-only training dashboard."""
    try:
        from fastapi import FastAPI, Response
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise ImportError("The dashboard requires the SaddleLLM serve extra.") from exc

    # Resolve once so a typo fails before the HTTP server starts.
    _resolve_run_dir(target)
    app = FastAPI(title="SaddleLLM Training Dashboard", version="1.0.0")
    app.state.output_target = target

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return dashboard_page()

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "saddlellm-dashboard"}

    @app.get("/api/status")
    async def status():
        return collect_run_status(target)

    return app


def _shell(title: str, body: str, script: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#f4f7fb;--panel:#fff;--ink:#172033;--muted:#68748a;--brand:#226d5a;--accent:#dff3ec;--line:#e2e8f0;--danger:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
.wrap{{max-width:1120px;margin:auto;padding:28px 20px 60px}}header{{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;margin-bottom:20px}}header>div{{min-width:0}}header .muted{{overflow-wrap:anywhere}}
h1{{font-size:25px;margin:0}}h2{{font-size:17px;margin:0 0 14px}}.muted{{color:var(--muted)}}.pill{{flex:none;white-space:nowrap;padding:5px 10px;border-radius:999px;background:var(--accent);color:var(--brand);font-weight:650}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:17px;box-shadow:0 5px 22px #23374d0b}}
.metric{{font-size:25px;font-weight:720;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.wide{{grid-column:1/-1}}.half{{grid-column:span 2}}
button,input{{font:inherit}}button{{border:0;border-radius:10px;background:var(--brand);color:white;padding:10px 16px;cursor:pointer}}button:disabled{{opacity:.55;cursor:not-allowed}}
input,select{{border:1px solid var(--line);border-radius:10px;padding:10px;background:white;color:var(--ink)}}pre{{margin:0;max-height:280px;overflow:auto;white-space:pre-wrap;font:12px/1.55 ui-monospace,Consolas,monospace}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line)}}canvas{{width:100%;height:260px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}.half{{grid-column:1/-1}}}}
</style></head><body><div class="wrap">{body}</div><script>{script}</script></body></html>"""


def dashboard_page() -> str:
    body = """
<header><div><h1>SaddleLLM 训练看板</h1><div class="muted" id="path">正在读取训练目录…</div><div class="muted" id="meta"></div></div><span class="pill" id="status">连接中</span></header>
<main class="grid">
  <section class="card"><div class="muted">训练进度</div><div class="metric" id="progress">—</div><div style="height:6px;background:#e5ebf2;border-radius:9px;margin-top:10px;overflow:hidden"><div id="progressBar" style="height:100%;width:0;background:#199473"></div></div><div class="muted" id="step" style="margin-top:6px;font-size:12px">—</div></section>
  <section class="card"><div class="muted">训练 Loss</div><div class="metric" id="loss">—</div></section>
  <section class="card"><div class="muted">验证 Loss</div><div class="metric" id="evalLoss">—</div></section>
  <section class="card"><div class="muted">发布门禁</div><div class="metric" id="gate">—</div></section>
  <section class="card wide"><h2>Loss 曲线</h2><canvas id="chart" width="1060" height="260"></canvas></section>
  <section class="card half"><h2>流水线阶段</h2><table><thead><tr><th>阶段</th><th>状态</th></tr></thead><tbody id="stages"></tbody></table></section>
  <section class="card half"><h2>产物</h2><pre id="files">—</pre></section>
  <section class="card wide"><h2>最近日志</h2><pre id="logs">—</pre></section>
</main>"""
    script = r"""
const $=id=>document.getElementById(id),labels={running:'训练中',completed:'已完成',failed:'失败',pending:'等待中',unknown:'未知'};const number=v=>Number.isFinite(Number(v))?Number(v).toFixed(4):'—';const duration=s=>Number.isFinite(Number(s))?(Number(s)/60).toFixed(1)+' 分钟':'—';
function draw(train,evaluation){const c=$('chart'),dpr=window.devicePixelRatio||1,w=Math.max(c.clientWidth,280),h=260;if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,w,h);x.font='11px system-ui';const items=[...train.map(v=>({step:Number(v.step)||0,value:Number(v.loss)})),...evaluation.map(v=>({step:Number(v.step)||0,value:Number(v.eval_loss)}))].filter(v=>v.value>0);if(!items.length){x.fillStyle='#68748a';x.fillText('等待训练指标…',55,55);return}const maxStep=Math.max(...items.map(v=>v.step),1),logs=items.map(v=>Math.log10(v.value)),max=Math.max(...logs),min=Math.min(...logs),range=Math.max(max-min,.1),left=w<500?38:55,right=w-18,top=28,bottom=225;x.strokeStyle='#e2e8f0';x.lineWidth=1;for(let i=0;i<5;i++){const py=top+(bottom-top)*i/4;x.beginPath();x.moveTo(left,py);x.lineTo(right,py);x.stroke();const value=Math.pow(10,max-range*i/4);x.fillStyle='#778398';x.fillText(value.toPrecision(2),2,py+4)}x.beginPath();x.moveTo(left,top);x.lineTo(left,bottom);x.lineTo(right,bottom);x.stroke();function line(list,key,color){const points=list.map(v=>({step:Number(v.step)||0,value:Number(v[key])})).filter(v=>v.value>0);if(!points.length)return;x.strokeStyle=color;x.lineWidth=2.5;x.beginPath();points.forEach((v,i)=>{const px=left+v.step/maxStep*(right-left),py=bottom-(Math.log10(v.value)-min)/range*(bottom-top);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}line(train,'loss','#147a64');line(evaluation,'eval_loss','#e07a00');x.fillStyle='#147a64';x.fillText('● 训练 Loss',left,17);x.fillStyle='#e07a00';x.fillText('● 验证 Loss',left+(w<500?78:92),17);x.fillStyle='#778398';x.fillText('step '+maxStep,Math.max(left,right-58),244)}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),d=await r.json();$('path').textContent=d.run_dir;$('meta').textContent=[d.project,d.model,d.elapsed_seconds==null?null:'运行 '+duration(d.elapsed_seconds)].filter(Boolean).join(' · ');const state=labels[d.status]||d.status;$('status').textContent=d.active_stage?state+' · '+d.active_stage:state;const p=d.progress||{},percent=p.percent==null?0:p.percent;$('progress').textContent=p.percent==null?(p.step||0)+' step':p.percent+'%';$('progressBar').style.width=Math.max(0,Math.min(100,percent))+'%';$('step').textContent=p.max_steps>0?'已完成 '+p.step+' / '+p.max_steps+' step':'当前 '+(p.step||0)+' step';$('loss').textContent=number(d.latest_metrics.loss??d.latest_metrics.train_loss);$('evalLoss').textContent=number(d.latest_metrics.eval_loss);$('gate').textContent=d.gate.accepted===true?'通过':d.gate.accepted===false?'未通过':'未设置';$('stages').replaceChildren(...d.stages.map(s=>{const tr=document.createElement('tr');const a=document.createElement('td'),b=document.createElement('td');a.textContent=s.name;b.textContent=labels[s.status]||s.status;tr.append(a,b);return tr}));$('files').textContent=d.result_files.join('\n')||'尚无评估或报告产物';$('logs').textContent=d.log_tail.join('\n')||'尚无日志';draw(d.train_history,d.eval_history)}catch(e){$('status').textContent='读取失败';$('logs').textContent=String(e)}}refresh();setInterval(refresh,2000);
"""
    return _shell("SaddleLLM 训练看板", body, script)


def chat_page(
    title: str = "SaddleLLM 对话",
    *,
    default_max_tokens: int = 256,
    max_tokens: int = 512,
    history_enabled: bool = True,
) -> str:
    """Render the self-contained entry page for the text-only chat interface."""
    template = (Path(__file__).parent / "assets" / "chat.html").read_text(encoding="utf-8")
    introduction = (Path(__file__).parent / "assets" / "saddle-intro.html").read_text(encoding="utf-8")
    limit = max(1, int(max_tokens))
    default = min(limit, max(1, int(default_max_tokens)))
    return (
        template.replace("{{PAGE_TITLE}}", html.escape(title, quote=True))
        .replace("<!-- SADDLE_INTRO -->", introduction)
        .replace("{{DEFAULT_MAX_TOKENS}}", str(default))
        .replace("{{MAX_TOKENS}}", str(limit))
        # Keep static defaults compatible with running older renderers reading
        # newly updated packaged assets; do not expose new unresolved tokens.
        .replace('data-history-enabled="true"', 'data-history-enabled="true"' if history_enabled else 'data-history-enabled="false"')
        .replace("历史保存在启动命令所在目录（共享工作区）", "历史保存在启动命令所在目录（共享工作区）" if history_enabled else "历史保存已关闭；刷新页面将清空当前对话")
    )


__all__ = ["chat_page", "collect_run_status", "create_dashboard_app", "dashboard_page"]
