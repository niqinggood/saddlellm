from __future__ import annotations

import math

import pytest

from saddlellm import cli
from saddlellm.runtime.EmbeddingServer import (
    EmbeddingServerSettings,
    _normalized_rows,
    create_embedding_app,
)


class FakeEncoder:
    def embed(self, texts):
        return [[0.6, 0.8] for _ in texts], sum(len(text) for text in texts)


def test_embedding_app_is_openai_compatible() -> None:
    from fastapi.testclient import TestClient

    app = create_embedding_app(
        EmbeddingServerSettings(
            model_path="fake",
            model_name="embedding-test",
            expected_dimensions=2,
            api_key="secret",
        ),
        encoder=FakeEncoder(),
    )
    client = TestClient(app)
    assert client.get("/health").json()["service"] == "embedding"
    assert client.post("/v1/embeddings", json={"input": "test"}).status_code == 401
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer secret"},
        json={"model": "embedding-test", "input": ["甲", "乙"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "embedding-test"
    assert [item["index"] for item in payload["data"]] == [0, 1]
    assert payload["data"][0]["embedding"] == [0.6, 0.8]


def test_normalized_rows_validates_dimensions_and_norm() -> None:
    rows = _normalized_rows([[3, 4]], expected_rows=1, expected_dimensions=2)
    assert math.isclose(sum(value * value for value in rows[0]), 1.0)
    with pytest.raises(RuntimeError, match="dimensions"):
        _normalized_rows([[1, 2, 3]], expected_rows=1, expected_dimensions=2)


def test_embedding_settings_reject_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_inputs"):
        EmbeddingServerSettings(model_path="model", max_inputs=0)


def test_cli_parses_embedding_service_options(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "_cmd_serve_embedding",
        lambda args: captured.update(vars(args)) or 0,
    )
    assert (
        cli.main(
            [
                "serve-embedding",
                "model",
                "--dimensions",
                "1024",
                "--local-files-only",
            ]
        )
        == 0
    )
    assert captured["model_path"] == "model"
    assert captured["dimensions"] == 1024
    assert captured["local_files_only"] is True
