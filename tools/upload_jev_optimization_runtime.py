"""Upload a separate immutable runtime and new data without modifying the running study."""
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
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    metadata = json.loads((out / 'remote_run.json').read_text())
    remote = metadata['remote_directory']
    if not remote.startswith('/root/saddle-jev-opt-'):
        raise ValueError('Unexpected experiment root')
    entries = [(p, 'runtime_v2/' + p.relative_to(repository).as_posix())
               for p in sorted((repository / 'saddlellm').rglob('*.py'))
               if '__pycache__' not in p.parts]
    entries.extend((out / name, 'runtime_v2/' + name)
                   for name in ('remote_followup.py', 'remote_runtime_setup.py', 'remote_finalize.py'))
    entries.extend((repository / 'tools' / name, 'runtime_v2/tools/' + name)
                   for name in ('prepare_jev_real_data.py', 'prepare_jev_optimization_data.py',
                                'enrich_jev_instance_questions.py', 'evaluate_jev_real.py',
                                'freeze_jev_optimization_selection.py'))
    for folder in ('expanded', 'instance_query'):
        data_root = out / 'data' / folder
        if not data_root.is_dir():
            raise FileNotFoundError(data_root)
        entries.extend((p, p.relative_to(out).as_posix())
                       for p in sorted(data_root.rglob('*')) if p.is_file())
    if len({name for _, name in entries}) != len(entries):
        raise ValueError('Duplicate archive paths')
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'purpose': 'Separate runtime; original frozen runtime remains unchanged',
                'files': [{'path': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                           'bytes': p.stat().st_size} for p, name in entries]}
    fingerprint = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    manifest['snapshot_hash'] = fingerprint
    manifest_path = out / 'runtime_v2_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    package = out / 'runtime_v2_snapshot.tar.gz'
    with tarfile.open(package, 'w:gz') as archive:
        for source, name in entries:
            archive.add(source, arcname=name)
        archive.add(manifest_path, arcname='runtime_v2_manifest.json')
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('Supply SADDLE_REMOTE_PASSWORD for this process only')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(metadata['host'], username='root', password=password, timeout=15)
        host_key = hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
        if host_key != metadata['host_key_sha256']:
            raise RuntimeError('Remote host key differs from the original study')
        with client.open_sftp() as sftp:
            # Every uploaded target must be new, including the frozen runtime directory.
            targets = {'runtime_v2', 'runtime_v2_manifest.json', 'runtime_v2_snapshot.tar.gz',
                       'data/expanded', 'data/instance_query'}
            for name in sorted(targets):
                try:
                    sftp.stat(remote + '/' + name)
                except FileNotFoundError:
                    continue
                raise FileExistsError('Refusing to overwrite ' + remote + '/' + name)
            sftp.put(str(package), remote + '/runtime_v2_snapshot.tar.gz')
        command = f'cd {shlex.quote(remote)} && tar xzf runtime_v2_snapshot.tar.gz'
        _, stdout, stderr = client.exec_command(command, timeout=60)
        stdout.read()
        error = stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        if code:
            raise RuntimeError(error)
        record = {'remote_directory': remote, 'runtime_directory': remote + '/runtime_v2',
                  'snapshot_hash': fingerprint, 'files': len(entries), 'archive_bytes': package.stat().st_size,
                  'command': command, 'exit_code': code, 'host_key_sha256': host_key,
                  'uploaded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (out / 'runtime_v2_upload.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))
    finally:
        client.close()


if __name__ == '__main__':
    main()
