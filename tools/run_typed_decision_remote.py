"""Run typed-decision regression and training on an authorized SSH host.

Password authentication reads SADDLE_REMOTE_PASSWORD from the environment;
credentials are never saved or logged. Each run gets an isolated remote tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--python", default="/home/medmodel/.venv/bin/python")
    parser.add_argument("--local-output", required=True)
    parser.add_argument("--trust-new-host", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.local_output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if args.trust_new_host:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        username=args.user,
        password=os.environ.get("SADDLE_REMOTE_PASSWORD"),
        timeout=15,
        auth_timeout=15,
    )
    remote = "/root/saddle-typed-" + datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    sftp = client.open_sftp()
    sftp.mkdir(remote)
    files = [
        *root.glob("saddlellm/**/*.py"),
        *root.glob("tests/test*decision*.py"),
        root / "tools/prepare_decision_typed.py",
        root / "tools/prepare_decision_banking77.py",
        root / "tools/verify_typed_decision_remote.py",
        root / "configs/decision_typed.yaml",
        *root.glob("data/decision_typed_example/*"),
    ]
    with tempfile.TemporaryDirectory(prefix="saddle-typed-upload-") as temp:
        archive = Path(temp) / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in files:
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(root).as_posix())
        sftp.put(str(archive), remote + "/source.tar.gz")
    _stdin, stdout, stderr = client.exec_command(
        f"tar -xzf {shlex.quote(remote + '/source.tar.gz')} -C {shlex.quote(remote)}",
        timeout=30,
    )
    if stdout.channel.recv_exit_status():
        raise RuntimeError(stderr.read().decode())
    command = (
        f"cd {shlex.quote(remote)} && PYTHONPATH={shlex.quote(remote)} "
        f"{shlex.quote(args.python)} tools/verify_typed_decision_remote.py"
    )
    log = sftp.open(remote + "/launch.sh", "w")
    log.write("#!/bin/sh\n" + command + " > verification.log 2>&1\n")
    log.close()
    _stdin, stdout, stderr = client.exec_command(
        f"cd {shlex.quote(remote)} && nohup sh launch.sh </dev/null >/dev/null 2>&1 &"
    )
    stdout.channel.recv_exit_status()
    _metadata = {
        "host": args.host,
        "remote_directory": remote,
        "python": args.python,
        "source_files": len(files),
    }
    (output / "remote_run.json").write_text(
        json.dumps(_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_metadata), flush=True)
    offset = 0
    deadline = time.monotonic() + 1800
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"remote verification did not finish; inspect {remote}/verification.log"
            )
        try:
            with sftp.open(remote + "/verification.log") as stream:
                stream.seek(offset)
                chunk = stream.read().decode("utf-8", errors="replace")
                offset = stream.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            print(chunk, end="", flush=True)
        try:
            sftp.stat(remote + "/verification_result.json")
            break
        except FileNotFoundError:
            time.sleep(3)

    # Pull complete small native artifacts and logs; HF checkpoints remain remote.
    def pull(remote_path, local_path):
        local_path.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_path):
            import stat

            if entry.filename in {
                "testdeps",
                "saddlellm",
                "tests",
                "tools",
                "data",
                "__pycache__",
            }:
                continue
            path = remote_path + "/" + entry.filename
            if stat.S_ISDIR(entry.st_mode):
                pull(path, local_path / entry.filename)
            elif (
                not entry.filename.endswith(".safetensors") or entry.st_size < 5_000_000
            ):
                sftp.get(path, str(local_path / entry.filename))

    pull(remote, output / "remote_evidence")
    result = json.loads(
        (output / "remote_evidence/verification_result.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps({"status": result["status"], "local_output": str(output)}),
        flush=True,
    )
    sftp.close()
    client.close()
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
