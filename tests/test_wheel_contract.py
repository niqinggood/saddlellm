import re
import zipfile
from pathlib import Path

import pytest

from tools.check_wheel_contents import EXPECTED_EXTRAS, check_wheel


def _write_wheel(
    path,
    *,
    base_dependencies=("packaging>=23", "PyYAML>=6"),
    include_js=True,
    stale_flat_module=False,
    extras=EXPECTED_EXTRAS,
    missing_runtime_file=None,
):
    metadata = [
        "Metadata-Version: 2.4",
        "Name: saddlellm",
        "Version: 2.33",
    ]
    metadata.extend(f"Requires-Dist: {dependency}" for dependency in base_dependencies)
    metadata.extend(f"Provides-Extra: {extra}" for extra in sorted(extras))
    metadata.append("")
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("saddlellm/__init__.py", '__version__ = "2.33"\n')
        for package in (
            "agents",
            "alignment",
            "compression",
            "data",
            "distillation",
            "evaluation",
            "experiments",
            "factory",
            "framework",
            "models",
            "multimodal",
            "runtime",
            "spatial",
            "training",
            "tuning",
            "utils",
            "world_models",
        ):
            wheel.writestr(f"saddlellm/{package}/__init__.py", "")
        wheel.writestr("saddle_llm/__init__.py", "")
        for name in (
            "saddlellm/utils/EnvironmentDiagnostics.py",
            "saddlellm/runtime/ChatWebRoutes.py",
            "saddlellm/runtime/InferenceLifecycle.py",
            "saddlellm/models/NativeBatchDecode.py",
            "saddlellm/runtime/NativeStreamBatcher.py",
            "saddlellm/runtime/assets/chat.html",
            "saddlellm/runtime/assets/chat.js",
            "saddlellm/runtime/assets/chat.css",
            "saddlellm/runtime/StudyWorkbench.py",
            "saddlellm/runtime/WorkbenchAPI.py",
            "saddlellm/runtime/assets/workbench.html",
            "saddlellm/runtime/assets/workbench.js",
            "saddlellm/runtime/assets/workbench.css",
            "saddlellm/runtime/SaddleGuide.py",
            "saddlellm/runtime/assets/saddle-intro.html",
            "saddlellm/runtime/assets/saddle-intro.css",
            "saddlellm/runtime/assets/saddle-intro.js",
            "saddlellm/data/GroupSplit.py",
            "saddlellm/evaluation/EvaluationRunner.py",
            "saddlellm/evaluation/ModelIdentity.py",
            "saddlellm/compression/QuantizationPipeline.py",
            "saddlellm/compression/QuantizationStage.py",
            "saddlellm/compression/CalibratedPTQ.py",
            "saddlellm/compression/QuantizationAwareTraining.py",
        ):
            if name != missing_runtime_file:
                wheel.writestr(name, "")
        wheel.writestr("saddlellm/spatial_studio_web/index.html", "<main></main>")
        wheel.writestr("saddlellm/spatial_studio_web/assets/app.css", "")
        if include_js:
            wheel.writestr("saddlellm/spatial_studio_web/assets/app.js", "")
        if stale_flat_module:
            wheel.writestr("saddlellm/ModelLoader.py", "")
        wheel.writestr("saddlellm-2.33.dist-info/METADATA", "\n".join(metadata))
        wheel.writestr(
            "saddlellm-2.33.dist-info/entry_points.txt",
            "[console_scripts]\n"
            "saddle-llm = saddlellm.cli:main\n"
            "saddlellm = saddlellm.cli:main\n",
        )


def test_wheel_contract_accepts_lightweight_complete_wheel(tmp_path):
    wheel = tmp_path / "saddlellm-2.33-py3-none-any.whl"
    _write_wheel(wheel)

    assert check_wheel(wheel) == []


def test_wheel_contract_extras_match_declared_project_extras():
    source = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    section = re.search(
        r"^\[project\.optional-dependencies\]\s*\n(.*?)(?=^\[|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section is not None
    names = set(re.findall(r"^([a-z][a-z0-9-]*)\s*=", section[1], re.MULTILINE))
    assert names == EXPECTED_EXTRAS


@pytest.mark.parametrize("name", [
    "saddlellm/utils/EnvironmentDiagnostics.py", "saddlellm/runtime/ChatWebRoutes.py",
    "saddlellm/runtime/InferenceLifecycle.py",
    "saddlellm/models/NativeBatchDecode.py",
    "saddlellm/runtime/NativeStreamBatcher.py",
    "saddlellm/runtime/assets/chat.html", "saddlellm/runtime/assets/chat.js", "saddlellm/runtime/assets/chat.css",
    "saddlellm/runtime/StudyWorkbench.py", "saddlellm/runtime/WorkbenchAPI.py",
    "saddlellm/runtime/assets/workbench.html", "saddlellm/runtime/assets/workbench.js", "saddlellm/runtime/assets/workbench.css",
    "saddlellm/runtime/SaddleGuide.py", "saddlellm/runtime/assets/saddle-intro.html",
    "saddlellm/runtime/assets/saddle-intro.css", "saddlellm/runtime/assets/saddle-intro.js",
    "saddlellm/data/GroupSplit.py", "saddlellm/evaluation/EvaluationRunner.py",
    "saddlellm/evaluation/ModelIdentity.py",
])
def test_wheel_contract_rejects_missing_diagnostics_or_chat_files(tmp_path, name):
    wheel = tmp_path / "saddlellm-2.40-py3-none-any.whl"
    _write_wheel(wheel, missing_runtime_file=name)
    assert any("missing required files" in issue and name in issue for issue in check_wheel(wheel))


def test_wheel_contract_rejects_undeclared_extra(tmp_path):
    wheel = tmp_path / "saddlellm-2.33-py3-none-any.whl"
    _write_wheel(wheel, extras=EXPECTED_EXTRAS | {"unknown"})
    assert any("unexpected extras" in issue for issue in check_wheel(wheel))


def test_wheel_contract_rejects_heavy_base_dependency(tmp_path):
    wheel = tmp_path / "saddlellm-2.33-py3-none-any.whl"
    _write_wheel(wheel, base_dependencies=("packaging>=23", "PyYAML>=6", "torch>=2.6"))

    issues = check_wheel(wheel)

    assert any("base dependencies must stay lightweight" in issue for issue in issues)


def test_wheel_contract_requires_built_frontend_assets(tmp_path):
    wheel = tmp_path / "saddlellm-2.33-py3-none-any.whl"
    _write_wheel(wheel, include_js=False)

    assert "wheel must contain Spatial Studio CSS and JavaScript assets" in check_wheel(
        wheel
    )


def test_wheel_contract_rejects_stale_flat_implementation_modules(tmp_path):
    wheel = tmp_path / "saddlellm-2.33-py3-none-any.whl"
    _write_wheel(wheel, stale_flat_module=True)

    issues = check_wheel(wheel)

    assert any("stale flat implementation modules" in issue for issue in issues)
