from __future__ import annotations

import pytest

from saddlellm.models.ModelAdapter import (
    HuggingFaceModelAdapter,
    SaddleModelAdapter,
    get_model_adapter,
)
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.cli import main as cli_main
from saddlellm.framework import (
    BaseStage,
    DuplicatePluginError,
    PluginRegistry,
    StageCapabilities,
    StageRegistry,
    create_stage_registry,
)


def test_registry_discovers_entry_points_without_loading_until_resolve(monkeypatch):
    loaded = []

    class FakeEntryPoint:
        name = "external"
        value = "demo.plugin:External"

        def load(self):
            loaded.append(self.value)
            return object()

    class FakeEntryPoints(list):
        def select(self, **kwargs):
            assert kwargs == {"group": "saddlellm.tests"}
            return self

    monkeypatch.setattr(
        "saddlellm.framework.registry.metadata.entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint()]),
    )
    registry = PluginRegistry(kind="test plugin", entry_point_group="saddlellm.tests")

    assert registry.discover() == ("external",)
    assert registry.names() == ("external",)
    assert loaded == []
    assert registry.info()[0].loaded is False

    assert registry.resolve("external") is registry.resolve("external")
    assert loaded == ["demo.plugin:External"]
    assert registry.info()[0].loaded is True


def test_registry_preserves_builtin_on_entry_point_name_conflict(monkeypatch):
    class ConflictingEntryPoint:
        name = "builtin"
        value = "demo.plugin:Replacement"

        def load(self):  # pragma: no cover - conflict must stay lazy and ignored
            raise AssertionError("conflicting entry point must never load")

    class FakeEntryPoints(list):
        def select(self, **kwargs):
            return self

    monkeypatch.setattr(
        "saddlellm.framework.registry.metadata.entry_points",
        lambda: FakeEntryPoints([ConflictingEntryPoint()]),
    )
    registry = PluginRegistry(kind="test plugin", entry_point_group="saddlellm.tests")
    builtin = object()
    registry.register("builtin", builtin, source="builtin")

    assert registry.discover() == ()
    assert registry.resolve("builtin") is builtin
    assert "conflicts with builtin" in registry.discovery_errors[0]
    with pytest.raises(DuplicatePluginError):
        registry.register("builtin", object())


class EchoStage(BaseStage):
    name = "echo"
    capabilities = StageCapabilities(tags=frozenset({"test"}))

    def validate(self, context):
        if context.metadata.get("reject"):
            raise ValueError("metadata requested rejection")

    def run(self, context):
        return {
            "status": "completed",
            "message": context.metadata.get("message", "ok"),
            "previous": sorted(context.results),
        }


def _echo_registry() -> StageRegistry:
    registry = StageRegistry(entry_point_group=None)
    registry.register("echo", EchoStage, source="test")
    return registry


def test_orchestrator_executes_third_party_stage_without_core_changes(tmp_path):
    document = {
        "metadata": {"name": "echo-run", "message": "plugin-loaded"},
        "stages": ["echo"],
        "distributed": {"bf16": False, "fp16": False},
        "logging": {
            "backend": "local",
            "output_dir": str(tmp_path / "run"),
        },
    }

    orchestrator = TrainingOrchestrator.from_dict(
        document,
        stage_registry=_echo_registry(),
    )
    assert orchestrator.available_stages() == ("echo",)
    assert orchestrator.run()["echo"] == {
        "status": "completed",
        "message": "plugin-loaded",
        "previous": [],
    }


def test_stage_validation_and_result_contract_fail_closed(tmp_path):
    rejected = {
        "metadata": {"reject": True},
        "stages": ["echo"],
        "logging": {"output_dir": str(tmp_path / "rejected")},
    }
    with pytest.raises(ValueError, match="metadata requested rejection"):
        TrainingOrchestrator.from_dict(rejected, stage_registry=_echo_registry())

    class InvalidResultStage(EchoStage):
        name = "invalid_result"

        def run(self, context):
            return ["not", "a", "mapping"]

    registry = StageRegistry(entry_point_group=None)
    registry.register("invalid_result", InvalidResultStage)
    orchestrator = TrainingOrchestrator.from_dict(
        {
            "stages": ["invalid_result"],
            "logging": {"output_dir": str(tmp_path / "invalid")},
        },
        stage_registry=registry,
    )
    with pytest.raises(TypeError, match="expected a mapping or None"):
        orchestrator.run()


def test_builtin_stage_registry_declares_distributed_capabilities():
    registry = create_stage_registry(discover=False)
    assert "pretrain" in registry.names(discover=False)
    assert registry.create("pretrain").capabilities.supports_strategy("fsdp")
    assert not registry.create("sft").capabilities.supports_strategy("fsdp")
    assert registry.create("sft").capabilities.uses_model_adapter is True
    assert registry.create("operator").capabilities.uses_model_adapter is False


def test_model_adapters_remain_explicit_internal_choices():
    assert isinstance(get_model_adapter("hf"), HuggingFaceModelAdapter)
    assert isinstance(get_model_adapter("saddle"), SaddleModelAdapter)
    with pytest.raises(ValueError, match="expected one of: hf, saddle"):
        get_model_adapter("external")


def test_plugins_cli_lists_extension_points_without_heavy_loading(capsys):
    assert cli_main(["plugins"]) == 0
    output = capsys.readouterr().out
    assert '"entry_point_group": "saddlellm.stages"' in output
    assert '"name": "pretrain"' in output
    assert '"model_backends"' not in output


def test_static_config_validator_uses_stage_registry(tmp_path):
    result = TrainingConfigValidator.validate(
        {
            "stages": ["echo"],
            "logging": {"output_dir": str(tmp_path / "validated")},
        },
        inspect_data=False,
        stage_registry=_echo_registry(),
    )
    assert result.valid is True
    assert result.stages == ["echo"]


def test_train_cli_dry_run_reads_plain_config(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "metadata:",
                "  name: dry-run",
                "stages: [eval]",
                "eval:",
                "  enabled: false",
                "distributed:",
                "  bf16: false",
                "  fp16: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert cli_main(["train", str(config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"stages": [\n    "eval"\n  ]' in output
