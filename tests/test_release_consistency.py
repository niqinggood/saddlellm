from pathlib import Path

from tools.check_release_consistency import (
    ReleaseConsistencyError,
    ReleaseVersions,
    collect_release_versions,
    consistency_issues,
    main,
)


def _write_project(root: Path, *, package: str, pyproject: str, readme: str) -> None:
    package_dir = root / "saddlellm"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        f'__version__ = "{package}"\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "saddlellm"\nversion = "{pyproject}"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"# SaddleLLM\n\n当前版本：`{readme}`\n", encoding="utf-8"
    )


def test_repository_release_metadata_is_consistent():
    versions = collect_release_versions()

    assert versions.package
    assert consistency_issues(versions) == []


def test_release_consistency_reports_each_mismatched_source(tmp_path, capsys):
    _write_project(tmp_path, package="2.33", pyproject="2.34", readme="2.32")

    assert main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "release versions do not match" in output
    assert "saddlellm/__init__.py" in output
    assert "pyproject.toml" in output
    assert "README.md" in output


def test_release_consistency_rejects_dynamic_pyproject_version(tmp_path):
    _write_project(tmp_path, package="2.33", pyproject="2.33", readme="2.33")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "saddlellm"\ndynamic = ["version"]\n',
        encoding="utf-8",
    )

    try:
        collect_release_versions(tmp_path)
    except ReleaseConsistencyError as exc:
        assert "static project.version" in str(exc)
    else:
        raise AssertionError("dynamic pyproject version should fail closed")
