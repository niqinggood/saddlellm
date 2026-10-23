"""Freeze current source and upload a fresh L20 training workspace."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('Set SADDLE_REMOTE_PASSWORD for this process')
    if (out / 'remote_run.json').exists():
        raise RuntimeError('This local run has already been prepared; use a new directory')
    files = sorted(p for p in (repo / 'saddlellm').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts
                   and p.suffix in {'.py', '.json', '.yaml', '.html', '.css', '.js', '.txt'})
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'files': [{'path': p.relative_to(repo).as_posix(),
                           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in files]}
    source_hash = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    remote = f'/root/saddle-jev-smoke-{stamp}-{source_hash[:8]}'
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    package = out / 'source_snapshot.tar.gz'
    with tarfile.open(package, 'w:gz') as archive:
        for p in files:
            archive.add(p, arcname=p.relative_to(repo).as_posix())
        archive.add(out / 'source_manifest.json', arcname='source_manifest.json')
        archive.add(out / 'remote_workflow.py', arcname='remote_workflow.py')
    quoted = shlex.quote(remote)
    command = (f'set -e\ncd {quoted}\ntar xzf source_snapshot.tar.gz\nmkdir data\n'
               'cp -a /root/saddle-typed-20261003-020439/data/hf_joint_fixture data/hf_joint_fixture\n')
    launch = (f'#!/bin/sh\nset -e\ncd {quoted}\n'
              f'export PYTHONPATH={quoted}\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\n'
              'nohup /home/medmodel/.venv/bin/python remote_workflow.py >workflow.stdout.txt 2>workflow.stderr.txt </dev/null &\n'
              'echo $! > workflow.pid\ncat workflow.pid\n')
    (out / 'launch.sh').write_text(launch, encoding='utf-8', newline='\n')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(args.host, username='root', password=password, timeout=15)
        with client.open_sftp() as sftp:
            sftp.mkdir(remote)
            sftp.put(str(package), remote + '/source_snapshot.tar.gz')
            sftp.put(str(out / 'launch.sh'), remote + '/launch.sh')
        _, stdout, stderr = client.exec_command(command, timeout=60)
        stdout_text = stdout.read().decode()
        stderr_text = stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        (out / 'prepare.stdout.txt').write_text(stdout_text, encoding='utf-8')
        (out / 'prepare.stderr.txt').write_text(stderr_text, encoding='utf-8')
        if code:
            raise RuntimeError('Remote workspace preparation failed; see prepare.stderr.txt')
        metadata = {'host': args.host, 'remote_directory': remote,
                    'python': '/home/medmodel/.venv/bin/python',
                    'source_files': len(files), 'source_hash': source_hash,
                    'snapshot_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
                    'training_started': False}
        (out / 'remote_run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata))
    finally:
        client.close()


if __name__ == '__main__':
    main()
