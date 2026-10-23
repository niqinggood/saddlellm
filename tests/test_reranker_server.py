from __future__ import annotations

import pytest

from saddlellm import cli
from saddlellm.runtime.RerankerServer import (
    RerankerServerSettings,
    create_reranker_app,
)


class FakeScorer:
    def score(self, query, documents):
        del query
        return [float(len(item)) / 10.0 for item in documents], {
            "documents": len(documents),
            "latency_ms": 1.0,
            "maximum_input_tokens": 12,
        }


def test_reranker_app_sorts_scores_and_checks_auth() -> None:
    from fastapi.testclient import TestClient

    app = create_reranker_app(
        RerankerServerSettings(
            model_path="fake",
            model_name="reranker-test",
            api_key="secret",
        ),
        scorer=FakeScorer(),
    )
    client = TestClient(app)
    assert client.get("/health").json()["service"] == "reranker"
    assert client.post("/rerank", json={"query": "q", "documents": ["a"]}).status_code == 401
    response = client.post(
        "/v1/rerank",
        headers={"Authorization": "Bearer secret"},
        json={"query": "q", "documents": ["a", {"text": "long"}], "top_n": 1},
    )
    assert response.status_code == 200
    assert response.json()["results"] == [{"index": 1, "relevance_score": 0.4}]


def test_reranker_settings_and_payload_limits() -> None:
    from fastapi.testclient import TestClient

    with pytest.raises(ValueError, match="instruction"):
        RerankerServerSettings(model_path="model", instruction=" ")
    app = create_reranker_app(
        RerankerServerSettings(model_path="fake", max_documents=1),
        scorer=FakeScorer(),
    )
    response = TestClient(app).post(
        "/rerank", json={"query": "q", "documents": ["a", "b"]}
    )
    assert response.status_code == 400


def test_cli_parses_reranker_service_options(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "_cmd_serve_reranker",
        lambda args: captured.update(vars(args)) or 0,
    )
    assert (
        cli.main(
            [
                "serve-reranker",
                "model",
                "--max-length",
                "2048",
                "--local-files-only",
            ]
        )
        == 0
    )
    assert captured["model_path"] == "model"
    assert captured["max_length"] == 2048
    assert captured["local_files_only"] is True
