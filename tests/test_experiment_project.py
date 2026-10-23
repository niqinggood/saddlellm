import json
import subprocess
import sys

import pytest

from saddlellm.experiments.ProjectRunner import project_command


def test_project_command_uses_current_python_and_preserves_arguments(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "saddlellm.experiment.json").write_text(
        json.dumps(
            {"schema_version": 1, "entry_point": "task.cli:main", "pythonpath": ["src"]}
        )
    )
    command, environment = project_command(
        str(tmp_path), ["all", "--config", "path with spaces.yaml"]
    )
    assert command[0] == sys.executable
    assert command[-4:] == ["task.cli", "all", "--config", "path with spaces.yaml"]
    assert str(tmp_path / "src") in environment["PYTHONPATH"]


def test_project_manifest_rejects_escaping_import_paths(tmp_path):
    (tmp_path / "saddlellm.experiment.json").write_text(
        json.dumps(
            {"schema_version": 1, "entry_point": "task.cli:main", "pythonpath": [".."]}
        )
    )
    with pytest.raises(ValueError, match="inside"):
        project_command(str(tmp_path), [])


def test_project_calls_declared_main_and_preserves_exit_code(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    output = tmp_path / "received.json"
    (source / "example_task.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def main(argv):\n"
        f"    Path({str(output)!r}).write_text(json.dumps(argv))\n"
        "    return 7\n",
        encoding="utf-8",
    )
    (tmp_path / "saddlellm.experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entry_point": "example_task:main",
                "pythonpath": ["src"],
            }
        ),
        encoding="utf-8",
    )
    arguments = ["run", "--config", "path with spaces.yaml", "中文参数"]
    command, environment = project_command(str(tmp_path), arguments)
    result = subprocess.run(command, env=environment, check=False)
    assert result.returncode == 7
    assert json.loads(output.read_text()) == arguments
