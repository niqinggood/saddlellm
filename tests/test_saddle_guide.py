"""Read-only introduction contracts, with fake inference and no training jobs."""

import argparse
import builtins
import html
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


class PageElements(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids = Counter()
        self.links = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids[attrs["id"]] += 1
        if tag == "a":
            self.links.append(attrs.get("href"))


def fake_inference_app(backend, *, web_ui=True, chat_history=False):
    if backend == "vllm":
        from test_vllm_serving import setup_app

        app, engine, _ = setup_app(web_ui=web_ui, chat_history=chat_history)
        return app, engine

    from saddlellm.runtime.InferenceServer import (
        InferenceServerSettings,
        create_inference_app,
    )

    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=128), device="cpu"
    )
    settings = InferenceServerSettings(
        model_path="offline", web_ui=web_ui, chat_history=chat_history
    )
    return create_inference_app(settings, model=model, tokenizer=object()), None


@pytest.mark.parametrize("backend", ["local", "vllm"])
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_introduction_and_catalog_keep_chat_available_without_history(
    tmp_path, monkeypatch, backend, lang
):
    from saddlellm.runtime import InferenceServer, WebUI
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        InferenceServer,
        "load_inference_model",
        lambda *_: pytest.fail("introduction loaded another model"),
    )
    monkeypatch.setattr(
        "saddlellm.runtime.ChatWebRoutes.ChatHistoryStore",
        lambda *_: pytest.fail("introduction opened disabled chat history"),
    )
    app, engine = fake_inference_app(backend)
    assets = Path(WebUI.__file__).parent / "assets"
    # These are the elements the existing chat runtime consumes, rather than
    # an additional prescribed layout for the new introduction.
    chat_ids = set(re.findall(r"\$\('([^']+)'\)", (assets / "chat.js").read_text(encoding="utf-8")))
    assert {"form", "message", "messages", "historyList"} <= chat_ids
    with TestClient(app) as client:
        for route in ("/", "/saddle"):
            response = client.get(route)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            elements = PageElements(response.text)
            assert chat_ids <= elements.ids.keys()
            assert all(count == 1 for count in elements.ids.values())
            assert "/saddle" in elements.links
            assert 'data-history-enabled="false"' in response.text
            assert "/assets/chat.js" in response.text
            assert "/assets/saddle-intro.js" in response.text
            assert "/assets/saddle-intro.css" in response.text
            assert "{{SADDLE_INTRO}}" not in response.text
            assert "<!-- SADDLE_INTRO -->" not in response.text
        catalog = client.get("/api/saddle/catalog", params={"lang": lang})
        assert catalog.status_code == 200
        assert catalog.headers["content-type"].startswith("application/json")
        assert catalog.json()["lang"] == lang
        assert catalog.json() == saddle_catalog(lang=lang)
        assert client.get("/api/saddle/catalog").json() == saddle_catalog(lang="zh")
        assert client.get("/api/saddle/catalog", params={"lang": "invalid"}).status_code == 422
        assert client.get("/api/chat/sessions").status_code == 404
    assert not (tmp_path / ".saddlellm").exists()
    if engine is not None:
        assert engine.requests == []


@pytest.mark.parametrize("backend", ["local", "vllm"])
def test_api_only_service_does_not_install_introduction(backend):
    app, _ = fake_inference_app(backend, web_ui=False)
    with TestClient(app) as client:
        for path in (
            "/", "/saddle", "/api/saddle/catalog", "/assets/saddle-intro.js"
        ):
            assert client.get(path).status_code == 404


def test_introduction_assets_preserve_explicit_file_allowlist():
    app, _ = fake_inference_app("local")
    with TestClient(app) as client:
        for path, content_type in (
            ("/assets/saddle-intro.css", "text/css"),
            ("/assets/saddle-intro.js", "application/javascript"),
        ):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(content_type)
        for path in (
            "/assets/saddle-intro.html",
            "/assets/ChatWebRoutes.py",
            "/assets/%2e%2e%5cSaddleGuide.py",
            "/assets/%2e%2e%2fSaddleGuide.py",
            "/assets/unknown.js",
        ):
            assert client.get(path).status_code == 404


@pytest.mark.parametrize("history_enabled", [False, True])
def test_shared_chat_renderer_escapes_title_and_resolves_introduction(history_enabled):
    from saddlellm.runtime.WebUI import chat_page

    title = '<script>alert("title")</script><img src=x onerror=alert(1)>'
    page = chat_page(
        title, default_max_tokens=999, max_tokens=128,
        history_enabled=history_enabled,
    )
    assert f"<title>{html.escape(title, quote=True)}</title>" in page
    assert title not in page
    assert 'max="128" value="128"' in page
    assert "{{" not in page
    assert "<!-- SADDLE_INTRO -->" not in page
    expected_history = "true" if history_enabled else "false"
    assert f'data-history-enabled="{expected_history}"' in page


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_catalog_is_static_json_without_file_command_or_model_access(monkeypatch, lang):
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    original_import = builtins.__import__

    def deny_model_import(name, *args, **kwargs):
        if name.startswith(("torch", "transformers", "saddlellm.training", "saddlellm.tuning")):
            pytest.fail(f"static introduction imported model/training code: {name}")
        return original_import(name, *args, **kwargs)

    def deny_side_effect(*args, **kwargs):
        pytest.fail("static introduction accessed files or launched a command")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", deny_model_import)
        guard.setattr(builtins, "open", deny_side_effect)
        guard.setattr(Path, "read_text", deny_side_effect)
        guard.setattr(subprocess, "run", deny_side_effect)
        guard.setattr(subprocess, "Popen", deny_side_effect)
        guard.setattr(os, "system", deny_side_effect)
        first = saddle_catalog(lang=lang)
        serialized = json.dumps(first, ensure_ascii=False)
        second = saddle_catalog(lang=lang)
    assert json.loads(serialized) == second
    assert second["catalog_version"] == 1
    assert second["lang"] == lang
    assert second["workflow"] and second["features"]
    categories = {category["id"] for category in second["categories"]}
    assert {"prepare", "train", "evaluate", "deploy"} <= categories
    assert all(feature["category"] in categories for feature in second["features"])
    feature_ids = [feature["id"] for feature in second["features"]]
    assert len(feature_ids) == len(set(feature_ids))
    first["features"][0]["title"] = "changed only in one API response"
    assert saddle_catalog(lang=lang) == second


def test_bilingual_catalog_keeps_identical_features_commands_and_teaching_records():
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    zh, en = saddle_catalog(lang="zh"), saddle_catalog(lang="en")
    assert saddle_catalog() == zh
    assert zh.keys() == en.keys()
    for section in ("workflow", "categories", "features"):
        assert [item["id"] for item in zh[section]] == [item["id"] for item in en[section]]
    for chinese, english in zip(zh["features"], en["features"]):
        assert chinese.keys() == english.keys()
        assert chinese["category"] == english["category"]
        assert [item["command"] for item in chinese["commands"]] == [item["command"] for item in english["commands"]]
        # Switching interface language must not silently change the dataset or
        # the executable arguments that the reader is learning to reproduce.
        assert chinese["data_example"]["content"] == english["data_example"]["content"]
        assert len(chinese["troubleshooting"]) == len(english["troubleshooting"])

    def display_strings(value, path="catalog"):
        if isinstance(value, dict):
            for key, item in value.items():
                # Commands and teaching records are source/code, not translated
                # interface labels. Preserve those exact bytes across languages.
                if key not in {"command", "content"}:
                    yield from display_strings(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from display_strings(item, f"{path}[{index}]")
        elif isinstance(value, str):
            yield path, value

    for path, value in display_strings(en):
        assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value), path


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_feature_explains_checks_fixes_and_verification_for_common_problems(lang):
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    features = {feature["id"]: feature for feature in saddle_catalog(lang=lang)["features"]}
    assert {"pretrain", "sft", "preference", "grpo"} <= features.keys()
    minimums = {"sft": 4, "preference": 4, "grpo": 3}
    for feature_id, feature in features.items():
        problems = feature["troubleshooting"]
        assert len(problems) >= minimums.get(feature_id, 2), feature_id
        for problem in problems:
            assert set(problem) == {"symptom", "causes", "checks", "fixes", "verification"}
            assert isinstance(problem["symptom"], str) and problem["symptom"].strip()
            for field in ("causes", "checks", "fixes", "verification"):
                assert isinstance(problem[field], list) and problem[field], (feature_id, field)
                assert all(isinstance(item, str) and item.strip() for item in problem[field])


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_pretraining_offers_both_scratch_and_weights_only_continuation_examples(lang):
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    feature = next(item for item in saddle_catalog(lang=lang)["features"] if item["id"] == "pretrain")
    examples = [shlex.split(item["command"]) for item in feature["commands"]]
    repository = Path(__file__).resolve().parents[1]
    modes = set()
    for argv in examples:
        assert "--scratch" not in argv and "--warm-start" not in argv
        if argv[:2] != ["saddlellm", "train"]:
            continue
        if "--pretrain-mode" in argv:
            mode = argv[argv.index("--pretrain-mode") + 1]
            if mode == "scratch":
                assert "--model-config" in argv and "--tokenizer" in argv
            if mode == "continue":
                assert "--model" in argv
            modes.add(mode)
        elif argv[2] in {
            "configs/tinystories_tiny_scratch.yaml",
            "configs/tinystories_zh_16k_continue.yaml",
        }:
            config = repository / argv[2]
            assert config.is_file()
            match = re.search(r"^pretrain_mode:\s*(scratch|continue)\s*$", config.read_text(encoding="utf-8"), re.MULTILINE)
            assert match is not None
            modes.add(match.group(1))
    assert {"scratch", "continue"} <= modes


@pytest.mark.parametrize("lang", ["invalid", "zh-CN"])
def test_catalog_rejects_unsupported_languages(lang):
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    with pytest.raises(ValueError):
        saddle_catalog(lang=lang)


def test_catalog_cli_examples_parse_with_current_cli_without_execution(monkeypatch):
    from saddlellm import cli
    from saddlellm.experiments import ProjectRunner
    from saddlellm.runtime.SaddleGuide import saddle_catalog

    class ParsedOnly(Exception):
        pass

    original_parse = argparse.ArgumentParser.parse_args
    parsed_commands = []

    def parse_without_dispatch(parser, args=None, namespace=None):
        parsed = original_parse(parser, args, namespace)
        parsed_commands.append(parsed)
        # Stop before any CLI action can train, serve, read data, or write files.
        raise ParsedOnly

    def deny_execution(*args, **kwargs):
        pytest.fail("CLI example audit reached command execution")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_without_dispatch)
    monkeypatch.setattr(ProjectRunner, "main", deny_execution)
    monkeypatch.setattr(ProjectRunner, "project_command", deny_execution)
    for name, function in vars(cli).copy().items():
        if name.startswith("_cmd_") and callable(function):
            monkeypatch.setattr(cli, name, deny_execution)
    for name in ("run", "call", "Popen", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, deny_execution)
    examples = [
        item["command"]
        for feature in saddle_catalog()["features"]
        for item in feature["commands"]
    ]
    assert examples
    for command in examples:
        assert "\n" not in command
        argv = shlex.split(command)
        assert argv[0] == "saddlellm"
        if argv[1] == "experiment":
            # ProjectRunner forwards project-owned arguments before the general
            # CLI parser. Check only its outer --project grammar, with no call
            # to cli.main, ProjectRunner.main, or a project entry point.
            project_parser = argparse.ArgumentParser(prog="saddlellm experiment")
            project_parser.add_argument("--project", required=True)
            project_parser.add_argument("--show-command", action="store_true")
            project_args, project_arguments = project_parser.parse_known_args(argv[2:])
            assert project_args.project
            assert project_arguments
            continue
        previous_count = len(parsed_commands)
        with pytest.raises(ParsedOnly):
            cli.main(argv[1:])
        assert len(parsed_commands) == previous_count + 1
