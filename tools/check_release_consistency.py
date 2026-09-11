"""Check that SaddleLLM release metadata uses one version everywhere.

This check intentionally uses only the Python standard library so it can run
before the heavyweight ML dependencies are installed.
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
README_VERSION_PATTERN = re.compile(r"^当前版本：`(?P<version>[^`]+)`\s*$", re.MULTILINE)


class ReleaseConsistencyError(ValueError):
    """Raised when release metadata cannot be read unambiguously."""


@dataclass(frozen=True)
class ReleaseVersions:
    package: str
    pyproject: str
    readme: str

    def as_dict(self) -> dict[str, str]:
        return {
            "saddlellm/__init__.py": self.package,
            "pyproject.toml": self.pyproject,
            "README.md": self.readme,
        }


def _constant_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            values.append(value.value)
    if len(values) != 1:
        raise ReleaseConsistencyError(
            f"expected one constant {name} assignment in {path}, found {len(values)}"
        )
    return values[0]


def _pyproject_version(path: Path) -> str:
    section = ""
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section != "project" or not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != "version":
            continue
        try:
            value = ast.literal_eval(raw_value.strip())
        except (SyntaxError, ValueError):
            value = None
        if isinstance(value, str):
            values.append(value)
    if len(values) != 1:
        raise ReleaseConsistencyError(
            f"expected one static project.version in {path}, found {len(values)}"
        )
    return values[0]


def _readme_version(path: Path) -> str:
    matches = README_VERSION_PATTERN.findall(path.read_text(encoding="utf-8-sig"))
    if len(matches) != 1:
        raise ReleaseConsistencyError(
            f"expected one '当前版本：`...`' marker in {path}, found {len(matches)}"
        )
    return matches[0]


def collect_release_versions(root: Path = ROOT) -> ReleaseVersions:
    root = root.resolve()
    return ReleaseVersions(
        package=_constant_assignment(root / "saddlellm" / "__init__.py", "__version__"),
        pyproject=_pyproject_version(root / "pyproject.toml"),
        readme=_readme_version(root / "README.md"),
    )


def consistency_issues(versions: ReleaseVersions) -> list[str]:
    by_version: dict[str, list[str]] = {}
    for source, version in versions.as_dict().items():
        by_version.setdefault(version, []).append(source)
    if len(by_version) == 1:
        return []
    rendered = "; ".join(
        f"{version}: {', '.join(sources)}" for version, sources in sorted(by_version.items())
    )
    return [f"release versions do not match ({rendered})"]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        versions = collect_release_versions(args.root)
    except (OSError, SyntaxError, ReleaseConsistencyError) as exc:
        print(f"Release consistency check failed: {exc}")
        return 1
    issues = consistency_issues(versions)
    if issues:
        for issue in issues:
            print(f"Release consistency check failed: {issue}")
        return 1
    print(f"Release metadata is consistent: {versions.package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
