"""Run a task-owned research project through the SaddleLLM CLI.

The explicitly selected project's manifest supplies the CLI entry point. Task
data, prompts and rewards stay in that project, while algorithms stay here.
"""

# ruff: noqa: N999 -- Matches the repository's public module naming convention.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def project_command(
    project: str, arguments: list[str]
) -> tuple[list[str], dict[str, str]]:
    root = Path(project).expanduser().resolve(strict=True)
    manifest_path = root / "saddlellm.experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Project manifest requires schema_version=1")
    entry = manifest.get("entry_point", "")
    module, separator, function = entry.partition(":")
    if (
        not separator
        or function != "main"
        or not all(part.isidentifier() for part in module.split("."))
    ):
        raise ValueError(
            "Project entry_point must be a dotted Python module followed by :main"
        )
    paths = [str(Path(__file__).resolve().parents[2])]
    for relative in manifest.get("pythonpath", ["src"]):
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Project pythonpath entries must stay inside the project")
        paths.append(str(path))
    environment = os.environ.copy()
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONUTF8"] = "1"
    bootstrap = (
        "import importlib, sys; "
        "raise SystemExit(importlib.import_module(sys.argv[1]).main(sys.argv[2:]))"
    )
    return [sys.executable, "-c", bootstrap, module, *arguments], environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saddlellm experiment", description=__doc__)
    parser.add_argument(
        "--project", required=True, help="Project with saddlellm.experiment.json"
    )
    parser.add_argument(
        "--show-command",
        action="store_true",
        help="Print the command without executing it",
    )
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    try:
        command, environment = project_command(args.project, remaining)
        if args.show_command:
            print(
                json.dumps(
                    {
                        "command": command,
                        "project": str(Path(args.project).resolve()),
                        "pythonpath": environment["PYTHONPATH"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        return subprocess.call(command, env=environment)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
        return 2
