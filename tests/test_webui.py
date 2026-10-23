import json
from types import SimpleNamespace


def test_local_progress_callback_writes_live_status_and_metrics(tmp_path):
    from saddlellm.training.TrainingProgress import LocalProgressCallback

    callback = LocalProgressCallback(str(tmp_path))
    state = SimpleNamespace(
        global_step=0,
        max_steps=10,
        epoch=0.0,
        is_world_process_zero=True,
    )
    control = object()
    callback.on_train_begin(None, state, control)
    state.global_step = 4
    state.epoch = 0.4
    callback.on_log(None, state, control, logs={"loss": 1.25, "learning_rate": 1e-5})

    progress = json.loads((tmp_path / "training_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "running"
    assert progress["global_step"] == 4
    assert progress["progress"] == 0.4
    assert progress["latest"]["loss"] == 1.25
    metrics = (tmp_path / "training_metrics.jsonl").read_text(encoding="utf-8")
    assert '"step": 4' in metrics

    callback.on_train_end(None, state, control)
    progress = json.loads((tmp_path / "training_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["finished_at"]


def test_progress_callback_resets_new_run_but_preserves_resumed_history(tmp_path):
    from saddlellm.training.TrainingProgress import LocalProgressCallback

    metrics = tmp_path / "training_metrics.jsonl"
    metrics.write_text('{"step": 9, "loss": 9}\n', encoding="utf-8")
    state = SimpleNamespace(
        global_step=0,
        max_steps=10,
        epoch=0.0,
        is_world_process_zero=True,
    )
    LocalProgressCallback(str(tmp_path)).on_train_begin(None, state, object())
    assert metrics.read_text(encoding="utf-8") == ""

    original_start = "2026-09-12T00:00:00Z"
    (tmp_path / "training_progress.json").write_text(
        json.dumps({"started_at": original_start}), encoding="utf-8"
    )
    metrics.write_text('{"step": 9, "loss": 0.5}\n', encoding="utf-8")
    state.global_step = 9
    callback = LocalProgressCallback(str(tmp_path))
    callback.on_train_begin(None, state, object())
    progress = json.loads(
        (tmp_path / "training_progress.json").read_text(encoding="utf-8")
    )
    assert metrics.read_text(encoding="utf-8").startswith('{"step": 9')
    assert progress["started_at"] == original_start


def test_collect_run_status_reads_pipeline_metrics_gate_and_log(tmp_path):
    from saddlellm.runtime.WebUI import collect_run_status

    run = tmp_path / "run"
    checkpoint = run / "sft_checkpoints"
    checkpoint.mkdir(parents=True)
    (run / "training_summary.json").write_text(
        json.dumps(
            {
                "project": "nursing-model",
                "model": "base-model",
                "elapsed_seconds": 12.5,
                "results": {"sft": {"status": "completed", "model_path": str(checkpoint)}},
            }
        ),
        encoding="utf-8",
    )
    (run / "pipeline_state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "active_stage": None,
                "stages": {"sft": {"status": "completed"}},
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "training_progress.json").write_text(
        json.dumps({"status": "completed", "global_step": 10, "max_steps": 10}),
        encoding="utf-8",
    )
    (checkpoint / "training_metrics.jsonl").write_text(
        '\n'.join(
            [
                json.dumps({"step": 5, "loss": 1.0}),
                json.dumps({"step": 10, "loss": 0.5}),
                json.dumps({"step": 10, "eval_loss": 0.4}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "release_gate.json").write_text(
        json.dumps({"decision": "promoted"}), encoding="utf-8"
    )
    (run / "training.log").write_text("start\ndone\n", encoding="utf-8")

    payload = collect_run_status(str(tmp_path))
    assert payload["run_dir"] == str(run.resolve())
    assert payload["status"] == "completed"
    assert payload["project"] == "nursing-model"
    assert payload["model_path"] == str(checkpoint)
    assert payload["progress"]["percent"] == 100.0
    assert payload["latest_metrics"]["loss"] == 0.5
    assert payload["latest_metrics"]["eval_loss"] == 0.4
    assert payload["gate"]["accepted"] is True
    assert payload["log_tail"][-1] == "done"


def test_dashboard_app_serves_page_and_status(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from saddlellm.runtime.WebUI import create_dashboard_app

    (tmp_path / "pipeline_state.json").write_text(
        json.dumps({"status": "running", "stages": {}}), encoding="utf-8"
    )
    client = TestClient(create_dashboard_app(str(tmp_path)))
    assert client.get("/health").json()["service"] == "saddlellm-dashboard"
    assert "SaddleLLM 训练看板" in client.get("/").text
    assert client.get("/api/status").json()["status"] == "running"


def test_chat_page_uses_packaged_ui_and_server_token_limits():
    from pathlib import Path

    from saddlellm.runtime.WebUI import chat_page

    page = chat_page("<SaddleLLM>", default_max_tokens=128, max_tokens=256)
    assert "&lt;SaddleLLM&gt;" in page
    assert "<SaddleLLM>" not in page
    assert 'id="maxTokens" type="number" min="1" max="256" value="128"' in page
    assert "新建对话" in page
    assert "开始对话" in page
    assert "/assets/chat.css" in page
    assert "/assets/chat.js" in page
    assert "/assets/katex/katex.min.css" in page
    assets = Path(__file__).resolve().parents[1] / "saddlellm" / "runtime" / "assets"
    assert (assets / "chat.css").is_file()
    assert (assets / "chat.js").is_file()
    assert (assets / "marked.min.js").is_file()
    assert (assets / "katex" / "katex.min.js").is_file()
    assert (assets / "katex" / "katex.min.css").is_file()
    assert (assets / "katex" / "fonts" / "KaTeX_Main-Regular.woff2").is_file()


def test_chat_assets_are_served_without_exposing_arbitrary_files():
    from fastapi.testclient import TestClient

    from saddlellm.runtime.InferenceServer import (
        InferenceServerSettings,
        create_inference_app,
    )

    settings = InferenceServerSettings(model_path="offline", web_ui=True)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128), device="cpu")
    with TestClient(create_inference_app(settings, model=model, tokenizer=object())) as client:
        for path in (
            "/assets/chat.css",
            "/assets/chat.js",
            "/assets/marked.min.js",
            "/assets/katex/katex.min.css",
            "/assets/katex/katex.min.js",
            "/assets/katex/fonts/KaTeX_Main-Regular.woff2",
        ):
            assert client.get(path).status_code == 200
        assert client.get("/assets/unknown.js").status_code == 404
        assert client.get("/assets/katex/unknown.js").status_code == 404
        assert client.get("/assets/katex/fonts/KaTeX_..%5Cprivate.woff2").status_code == 404


def test_chat_history_store_is_lazy_persistent_and_model_scoped(tmp_path):
    import uuid

    import pytest

    from saddlellm.runtime.ChatHistory import ChatHistoryStore

    store = ChatHistoryStore(tmp_path)
    session_id = str(uuid.uuid4())
    assert store.list_sessions("qwen") == []
    assert not store.path.exists()
    first = [{"role": "user", "content": "你好"},
             {"role": "assistant", "content": "你好！"}]
    saved = store.save_session(session_id, "qwen", first)
    assert store.path == tmp_path / ".saddlellm" / "chat_history.sqlite3"
    assert saved["title"] == "你好"
    assert ChatHistoryStore(tmp_path).get_session(session_id, "qwen")["messages"] == first
    assert store.list_sessions("deepseek") == []
    with pytest.raises(ValueError, match="another model"):
        store.save_session(session_id, "deepseek", first)
    second = first + [{"role": "user", "content": "再见"},
                      {"role": "assistant", "content": "再见！"}]
    store.save_session(session_id, "qwen", second)
    assert store.get_session(session_id, "qwen")["messages"] == second
    assert not store.delete_session(session_id, "deepseek")
    assert store.delete_session(session_id, "qwen")
    assert store.get_session(session_id, "qwen") is None


def test_chat_history_api_persists_and_restricts_remote_clients(tmp_path, monkeypatch):
    import asyncio
    import uuid

    import httpx

    from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app

    monkeypatch.chdir(tmp_path)
    settings = InferenceServerSettings(model_path="offline", model_name="ui-test", web_ui=True)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128), device="cpu")
    app = create_inference_app(settings, model=model, tokenizer=object())
    session_id = str(uuid.uuid4())
    path = f"/api/chat/sessions/{session_id}"
    messages = [{"role": "user", "content": "问候"},
                {"role": "assistant", "content": "你好"}]

    async def run():
        async with app.router.lifespan_context(app):
            local = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(transport=local, base_url="http://127.0.0.1") as client:
                assert (await client.get("/api/chat/sessions")).json()["sessions"] == []
                assert not (tmp_path / ".saddlellm" / "chat_history.sqlite3").exists()
                saved = await client.put(path, json={"messages": messages})
                assert saved.status_code == 200
                assert saved.json()["title"] == "问候"
                assert (await client.get("/api/chat/sessions")).json()["sessions"][0]["id"] == session_id
                assert (await client.get(path)).json()["messages"] == messages
                assert (await client.put(path, json={"messages": [
                    {"role": "system", "content": "invalid"}, messages[1],
                ]})).status_code == 422
                assert (await client.get("/api/chat/sessions/not-a-uuid")).status_code == 400
            remote = httpx.ASGITransport(app=app, client=("203.0.113.5", 12345))
            async with httpx.AsyncClient(transport=remote, base_url="http://example.test") as client:
                assert (await client.get("/api/chat/sessions")).status_code == 403
                assert (await client.get(path)).status_code == 403
                assert (await client.delete(path)).status_code == 403
            proxied = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(transport=proxied, base_url="http://public.example") as client:
                assert (await client.get("/api/chat/sessions")).status_code == 403
            local = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(transport=local, base_url="http://127.0.0.1") as client:
                assert (await client.delete(path)).status_code == 204
                assert (await client.get(path)).status_code == 404

    asyncio.run(run())
