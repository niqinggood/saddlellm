"""Freeze source/data for a SaddleLLM public-policy study without moving history."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--issue-log", help="Observed issues as a JSON array; defaults to an empty log"
    )
    args = parser.parse_args()
    framework = Path(__file__).resolve().parents[1]
    project = Path(args.project).resolve(strict=True)
    model = Path(args.model).resolve(strict=True)
    issues = (
        json.loads(Path(args.issue_log).read_text(encoding="utf-8"))
        if args.issue_log
        else []
    )
    if not isinstance(issues, list) or any(
        not isinstance(issue, dict) for issue in issues
    ):
        parser.error("--issue-log must contain a JSON array of observed issue objects")
    output = Path(args.output).resolve()
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "model": [],
    }
    framework_files = [
        *framework.glob("saddlellm/**/*.py"),
        framework / "pyproject.toml",
        framework / "README.md",
        framework / "docs/EXPERIMENT_PROJECTS.md",
        framework / "tools/prepare_public_policy_study.py",
        *framework.glob("tests/test*grpo*.py"),
        framework / "tests/test_experiment_project.py",
    ]
    project_files = [
        *project.glob("src/**/*.py"),
        *project.glob("configs/**/*.yaml"),
        *project.glob("tests/*.py"),
        project / "saddlellm.experiment.json",
        project / "pyproject.toml",
        project / "requirements.txt",
        project / "run.py",
        project / "README.md",
        *project.glob("docs/*.md"),
        *project.glob("docs/*.txt"),
        *project.glob("data/**/*.jsonl"),
        *project.glob("data/**/*.parquet"),
    ]
    for name, base, paths in (
        ("build-saddlellm", framework, framework_files),
        ("public_policy_rlvr", project, project_files),
    ):
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=base, text=True
        ).strip()
        manifest[name] = {
            "original_root": str(base),
            "base_commit": revision,
            "includes_uncommitted_workspace_changes": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=base, text=True
                ).strip()
            ),
        }
        for path in paths:
            if not path.is_file():
                continue
            relative = Path(name) / path.relative_to(base)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            manifest["files"].append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    for path in sorted(model.iterdir()):
        if path.is_file() and path.suffix in {
            ".json",
            ".safetensors",
            ".jinja",
            ".txt",
        }:
            manifest["model"].append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "issue_log.json").write_text(
        json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_snapshot": str(snapshot),
                "files": len(manifest["files"]),
                "model_files": len(manifest["model"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
