"""Upload, start or collect the isolated optimization study; no stored credentials."""
import argparse
import datetime
import hashlib
import json
import os
import shlex
import tarfile
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['upload', 'start', 'collect'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--plan', default='development_plan.json')
    parser.add_argument('--runner', default='remote_study.py', help='Recorded remote runner path relative to study root')
    parser.add_argument('--summary-only', action='store_true', help='Collect status, current progress and completed reports only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('Supply SADDLE_REMOTE_PASSWORD for this process only')
    metadata = None
    if args.action != 'upload':
        metadata = json.loads((out / 'remote_run.json').read_text())
        if args.host != metadata['host']:
            raise ValueError('Host must match the original uploaded study')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(args.host, username='root', password=password, timeout=15)
        if metadata is not None:
            host_key = hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
            if host_key != metadata['host_key_sha256']:
                raise RuntimeError('Remote host key differs from the original study')
        if args.action == 'upload':
            if (out / 'remote_run.json').exists():
                raise FileExistsError('Already uploaded; use a fresh study directory')
            entries = [(p, p.relative_to(root).as_posix()) for p in sorted((root / 'saddlellm').rglob('*.py')) if '__pycache__' not in p.parts]
            entries.append((root / 'tools/evaluate_jev_real.py', 'tools/evaluate_jev_real.py'))
            entries.extend((out / name, name) for name in ('remote_setup.py', 'remote_study.py', 'development_plan.json'))
            entries.extend((p, 'data/' + p.relative_to(out / 'data').as_posix()) for p in sorted((out / 'data').rglob('*'))
                           if p.is_file() and not p.relative_to(out / 'data').parts[0].startswith('expanded'))
            manifest = {'files': [{'path': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p, name in entries]}
            fingerprint = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
            remote = f'/root/saddle-jev-opt-{stamp}-{fingerprint[:8]}'
            (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
            package = out / 'source_snapshot.tar.gz'
            with tarfile.open(package, 'w:gz') as archive:
                for path, name in entries:
                    archive.add(path, arcname=name)
                archive.add(out / 'source_manifest.json', arcname='source_manifest.json')
            with client.open_sftp() as sftp:
                sftp.mkdir(remote)
                sftp.put(str(package), remote + '/source_snapshot.tar.gz')
            _, stdout, stderr = client.exec_command(f'cd {shlex.quote(remote)} && tar xzf source_snapshot.tar.gz', timeout=60)
            stdout.read()
            error = stderr.read().decode()
            code = stdout.channel.recv_exit_status()
            if code:
                raise RuntimeError(error)
            metadata = {'host': args.host, 'remote_directory': remote, 'snapshot_hash': fingerprint,
                        'source_files': len(entries), 'training_started': False,
                        'host_key_sha256': hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()}
            (out / 'remote_run.json').write_text(json.dumps(metadata, indent=2))
            print(json.dumps(metadata))
            return
        remote = metadata['remote_directory']
        if not remote.startswith('/root/saddle-jev-opt-'):
            raise ValueError('Unexpected experiment root')
        if args.action == 'start':
            runner_path = Path(args.runner)
            if runner_path.is_absolute() or '..' in runner_path.parts or runner_path.suffix != '.py':
                raise ValueError('Runner must be a relative Python file within the study root')
            plan_path = out / args.plan
            with client.open_sftp() as sftp:
                sftp.put(str(plan_path), remote + '/' + args.plan)
            stem = Path(args.plan).stem
            command = (f'cd {shlex.quote(remote)}\nexport PYTHONPATH={shlex.quote(remote)}\n'
                'export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false\n'
                f'nohup /home/medmodel/.venv/bin/python {shlex.quote(args.runner)} --plan {shlex.quote(args.plan)} '
                f'>{shlex.quote(stem)}.stdout.txt 2>{shlex.quote(stem)}.stderr.txt </dev/null &\n'
                f'echo $! > {shlex.quote(stem)}.pid\ncat {shlex.quote(stem)}.pid\n')
            _, stdout, stderr = client.exec_command(command, timeout=30)
            pid, error = stdout.read().decode().strip(), stderr.read().decode()
            code = stdout.channel.recv_exit_status()
            record = {'command': command, 'exit_code': code, 'pid': pid,
                      'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()}
            (out / (stem + '_launch.json')).write_text(json.dumps(record, indent=2))
            if code:
                raise RuntimeError(error)
            print(json.dumps(record))
            return
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(30)
            def download(name):
                try:
                    entry = sftp.stat(remote + '/' + name)
                except OSError:
                    return
                target = out / 'result' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.download')
                sftp.get(remote + '/' + name, str(temporary), prefetch=False)
                os.utime(temporary, (entry.st_mtime, entry.st_mtime))
                temporary.replace(target)

            if args.summary_only:
                stem = Path(args.plan).stem
                for name in (stem + '_status.json', stem + '_failure.txt', 'doctor.json', 'setup.json', 'setup_v2.json'):
                    download(name)
                local_status = out / 'result' / (stem + '_status.json')
                current = json.loads(local_status.read_text()) if local_status.exists() else {}
                for experiment in current.get('experiments', []):
                    base = 'runs/' + experiment['name']
                    download(base + '/experiment.json')
                    download(base + '/config.yaml')
                    download(base + '/output/decision/attempt-001/run_config.json')
                    download(base + '/output/decision/attempt-001/report.json')
                    if experiment['status'] == 'running':
                        download(base + '/output/decision/attempt-001/training_history.jsonl')
                        download(base + '/' + experiment['name'] + '_train.stdout.txt')
                        download(base + '/' + experiment['name'] + '_train.stderr.txt')

            def collect(directory, relative=''):
                for entry in sftp.listdir_attr(directory):
                    name = (relative + '/' + entry.filename).lstrip('/')
                    if entry.st_mode & 0o170000 == 0o040000:
                        if name.split('/')[0] in {'runs', 'evaluations'} and entry.filename != 'backbone':
                            collect(directory + '/' + entry.filename, name)
                    elif Path(name).suffix in {'.json', '.yaml', '.txt', '.jsonl'} and entry.st_size < 8 * 1024**2:
                        target = out / 'result' / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if (not target.exists() or target.stat().st_size != entry.st_size
                                or int(target.stat().st_mtime) != entry.st_mtime):
                            temporary = target.with_name(target.name + '.download')
                            sftp.get(directory + '/' + entry.filename, str(temporary), prefetch=False)
                            os.utime(temporary, (entry.st_mtime, entry.st_mtime))
                            temporary.replace(target)
            if not args.summary_only:
                collect(remote)
        status = out / 'result' / (Path(args.plan).stem + '_status.json')
        value = json.loads(status.read_text()) if status.exists() else {'status': 'not_started'}
        print(json.dumps({'status': value.get('status'), 'stage': value.get('stage'), 'error': value.get('error'),
                          'experiments': [{k: row.get(k) for k in ('name', 'status', 'best_epoch', 'best_validation_nll')} for row in value.get('experiments', [])]}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
