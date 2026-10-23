"""Upload a frozen, isolated real-data experiment; credentials stay in memory."""
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
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out, data = args.output_dir.resolve(), args.data.resolve()
    if (out / 'remote_run.json').exists():
        raise FileExistsError('Use a fresh run directory')
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('Set SADDLE_REMOTE_PASSWORD for this process')
    source = sorted(p for p in (repo / 'saddlellm').rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix in {'.py', '.json', '.yaml', '.html', '.css', '.js', '.txt'})
    entries = [(p, p.relative_to(repo).as_posix()) for p in source]
    entries.append((repo / 'tools/evaluate_jev_real.py', 'tools/evaluate_jev_real.py'))
    entries.extend((out / name, name) for name in ('remote_workflow.py', 'remote_prepare.py', 'remote_probe.py'))
    entries.extend((p, 'data_candidate/' + p.relative_to(data).as_posix())
                   for p in sorted(data.rglob('*')) if p.is_file())
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'files': [{'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path, name in entries]}
    fingerprint = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    remote = f'/root/saddle-jev-real-{stamp}-{fingerprint[:8]}'
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    package = out / 'source_snapshot.tar.gz'
    with tarfile.open(package, 'w:gz') as archive:
        for path, name in entries:
            archive.add(path, arcname=name)
        archive.add(out / 'source_manifest.json', arcname='source_manifest.json')
    quoted = shlex.quote(remote)
    prepare = f'set -e\ncd {quoted}\ntar xzf source_snapshot.tar.gz\n'
    launch = (f'#!/bin/sh\nset -e\ncd {quoted}\nexport PYTHONPATH={quoted}\n'
              'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false '
              'PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n'
              'nohup /home/medmodel/.venv/bin/python remote_workflow.py '
              '>workflow.stdout.txt 2>workflow.stderr.txt </dev/null &\n'
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
        _, stdout, stderr = client.exec_command(prepare, timeout=60)
        stdout_text, stderr_text = stdout.read().decode(), stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        (out / 'prepare.stdout.txt').write_text(stdout_text, encoding='utf-8')
        (out / 'prepare.stderr.txt').write_text(stderr_text, encoding='utf-8')
        if code:
            raise RuntimeError('Remote preparation failed; inspect preserved logs')
        metadata = {'host': args.host, 'remote_directory': remote,
                    'python': '/home/medmodel/.venv/bin/python', 'snapshot_hash': fingerprint,
                    'snapshot_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
                    'source_files': len(entries), 'training_started': False}
        (out / 'remote_run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata))
    finally:
        client.close()


if __name__ == '__main__':
    main()
