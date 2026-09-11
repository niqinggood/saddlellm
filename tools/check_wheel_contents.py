"""Validate the lightweight SaddleLLM wheel contract without importing it."""

from __future__ import annotations

import argparse
import email
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


EXPECTED_BASE_DEPENDENCIES = {"packaging", "pyyaml"}
EXPECTED_EXTRAS = {
    "data",
    "dev",
    "monitor",
    "posttrain",
    "qlora",
    "serve",
    "spatial",
    "text",
    "train",
}
REQUIRED_FILES = {
    "saddlellm/__init__.py",
    "saddle_llm/__init__.py",
    "saddlellm/spatial_studio_web/index.html",
    *(
        f"saddlellm/{package}/__init__.py"
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
        )
    ),
}
ALLOWED_ROOT_MODULES = {
    "saddlellm/DataPipeline.py",
    "saddlellm/Lightweight.py",
    "saddlellm/ModelPruner.py",
    "saddlellm/ModelQuantizer.py",
    "saddlellm/TextProcess.py",
    "saddlellm/__init__.py",
    "saddlellm/_exports.py",
    "saddlellm/cli.py",
    "saddlellm/easy.py",
}


def _normalized_requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        raise ValueError(f"cannot parse requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def check_wheel(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.is_file() or path.suffix != ".whl":
        return [f"wheel does not exist: {path}"]
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            issues.append(f"wheel contains unsafe paths: {sorted(unsafe)}")
        missing = REQUIRED_FILES - names
        if missing:
            issues.append(f"wheel is missing required files: {sorted(missing)}")
        root_modules = {
            name
            for name in names
            if PurePosixPath(name).parent == PurePosixPath("saddlellm")
            and name.endswith(".py")
        }
        unexpected_root_modules = root_modules - ALLOWED_ROOT_MODULES
        if unexpected_root_modules:
            issues.append(
                "wheel contains stale flat implementation modules: "
                f"{sorted(unexpected_root_modules)}"
            )
        assets = {
            PurePosixPath(name).suffix
            for name in names
            if name.startswith("saddlellm/spatial_studio_web/assets/")
        }
        if not {".css", ".js"}.issubset(assets):
            issues.append("wheel must contain Spatial Studio CSS and JavaScript assets")

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        entry_point_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(metadata_names) != 1:
            issues.append(f"expected one METADATA file, found {len(metadata_names)}")
            return issues
        if len(entry_point_names) != 1:
            issues.append(
                f"expected one entry_points.txt, found {len(entry_point_names)}"
            )
        else:
            entry_points = wheel.read(entry_point_names[0]).decode("utf-8")
            for command in ("saddle-llm", "saddlellm"):
                if f"{command} = saddlellm.cli:main" not in entry_points:
                    issues.append(f"missing console script: {command}")

        metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
        if metadata.get("Name") != "saddlellm":
            issues.append(f"unexpected distribution name: {metadata.get('Name')!r}")
        base_dependencies = {
            _normalized_requirement_name(requirement)
            for requirement in metadata.get_all("Requires-Dist", [])
            if ";" not in requirement
        }
        if base_dependencies != EXPECTED_BASE_DEPENDENCIES:
            issues.append(
                "base dependencies must stay lightweight: "
                f"expected {sorted(EXPECTED_BASE_DEPENDENCIES)}, got {sorted(base_dependencies)}"
            )
        extras = set(metadata.get_all("Provides-Extra", []))
        if extras != EXPECTED_EXTRAS:
            issues.append(
                f"unexpected extras: expected {sorted(EXPECTED_EXTRAS)}, got {sorted(extras)}"
            )
    return issues


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    issues = check_wheel(args.wheel.resolve())
    if issues:
        for issue in issues:
            print(f"Wheel check failed: {issue}")
        return 1
    print(f"Wheel contract is valid: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
