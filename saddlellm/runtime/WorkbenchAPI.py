"""Read-only HTTP surface for evidence-backed experiment workspaces."""

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
from __future__ import annotations

from pathlib import Path


def create_workbench_app(project: str, study: str | None = None):
    """Serve local study artifacts without loading a model or running training."""
    from fastapi import FastAPI, HTTPException, Query, Response
    from fastapi.responses import FileResponse, HTMLResponse

    from .StudyWorkbench import StudyWorkbench

    reader = StudyWorkbench(project)
    if study is not None:
        reader.study_detail(study)
    assets = Path(__file__).parent / "assets"
    app = FastAPI(title="SaddleLLM 实验工作台", version="1.0.0")
    app.state.reader = reader
    app.state.initial_study = study

    def read(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=404, detail="Study or artifact not found."
            ) from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (assets / "workbench.html").read_text(encoding="utf-8")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "saddlellm-workbench", "read_only": True}

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str):
        if name not in {"workbench.css", "workbench.js"}:
            raise HTTPException(status_code=404, detail="Asset not found.")
        return FileResponse(assets / name)

    @app.get("/api/workbench/studies")
    def studies():
        payload = reader.list_studies()
        return {**payload, "initial_study": study, "read_only": True}

    @app.get("/api/workbench/studies/{study_id}")
    def detail(study_id: str):
        return read(reader.study_detail, study_id)

    @app.get("/api/workbench/studies/{study_id}/runs/{run_id}")
    def run(study_id: str, run_id: str):
        return read(reader.run_detail, study_id, run_id)

    @app.get("/api/workbench/studies/{study_id}/data")
    def data(
        study_id: str,
        source: str = Query("A", pattern="^[AB]$"),
        split: str = Query(
            "train", pattern="^(train|validation|test|original_train|original_test)$"
        ),
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
        query: str = Query("", max_length=500),
    ):
        return read(
            reader.data_samples,
            study_id,
            source=source,
            split=split,
            offset=offset,
            limit=limit,
            query=query,
        )

    @app.get("/api/workbench/studies/{study_id}/predictions")
    def predictions(
        study_id: str,
        run_id: str,
        task: str = Query("A", pattern="^[AB]$"),
        outcome: str = Query(
            "all", pattern="^(all|improved|regressed|both_correct|both_wrong|unpaired)$"
        ),
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
        query: str = Query("", max_length=500),
    ):
        return read(
            reader.predictions,
            study_id,
            run_id,
            task=task,
            outcome=outcome,
            offset=offset,
            limit=limit,
            query=query,
        )

    @app.get("/api/workbench/studies/{study_id}/artifacts/{artifact_id}")
    def artifact(study_id: str, artifact_id: str):
        path = read(reader.artifact_path, study_id, artifact_id)
        return FileResponse(path, filename=path.name)

    return app
