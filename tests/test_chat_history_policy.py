from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app


@pytest.mark.parametrize("backend", ["local", "vllm"])
@pytest.mark.parametrize("existing", [False, True])
def test_history_disabled_never_reads_or_changes_store(tmp_path, monkeypatch, backend, existing):
    monkeypatch.chdir(tmp_path)
    history_path = tmp_path / ".saddlellm" / "chat_history.sqlite3"
    if existing:
        history_path.parent.mkdir()
        history_path.write_bytes(b"private existing history: leave untouched")
    monkeypatch.setattr("saddlellm.runtime.ChatWebRoutes.ChatHistoryStore",
                        lambda *_: pytest.fail("disabled history opened a store"))
    if backend == "vllm":
        from test_vllm_serving import setup_app

        app, _, _ = setup_app(web_ui=True, chat_history=False)
    else:
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128), device="cpu")
        app = create_inference_app(InferenceServerSettings(chat_history=False), model=model, tokenizer=object())
    with TestClient(app) as client:
        assert client.get("/health").json()["chat_history_enabled"] is False
        page = client.get("/").text
        assert 'data-history-enabled="false"' in page
        assert "历史保存已关闭" in page
        assert client.get("/assets/chat.js").status_code == 200
        for method, path in (("GET", "/api/chat/sessions"), ("GET", "/api/chat/sessions/anything"),
                             ("PUT", "/api/chat/sessions/anything"), ("DELETE", "/api/chat/sessions/anything")):
            assert client.request(method, path).status_code == 404
    assert history_path.exists() == existing
    if existing:
        assert history_path.read_bytes() == b"private existing history: leave untouched"
    else:
        assert not history_path.parent.exists()


def test_history_setting_requires_boolean():
    with pytest.raises(ValueError, match="chat_history"):
        InferenceServerSettings(chat_history="false")


def test_history_template_remains_usable_by_already_running_old_renderer():
    from pathlib import Path
    from saddlellm.runtime import WebUI

    template = (Path(WebUI.__file__).parent / "assets/chat.html").read_text(encoding="utf-8")
    old_renderer_output = template.replace("{{PAGE_TITLE}}", "SaddleLLM").replace("{{MAX_TOKENS}}", "8").replace("{{DEFAULT_MAX_TOKENS}}", "4")
    assert "{{" not in old_renderer_output
    assert 'data-history-enabled="true"' in old_renderer_output
    assert "共享工作区" in old_renderer_output


def test_cli_passes_history_setting_without_starting_server(monkeypatch):
    from saddlellm.cli import main
    from saddlellm.runtime import InferenceServer, ServingPlan

    captured = []
    monkeypatch.setattr(ServingPlan, "check_serving_address", lambda *args: None)
    monkeypatch.setattr(InferenceServer, "create_inference_app", lambda settings: captured.append(settings) or object())
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    assert main(["serve", "local-model", "--backend", "local", "--no-chat-history"]) == 0
    assert len(captured) == 1 and captured[0].chat_history is False
