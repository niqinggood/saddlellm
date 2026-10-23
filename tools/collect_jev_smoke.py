"""Fetch small L20 training evidence; leave full trained weights on the server."""
import argparse
import json
import os
from pathlib import Path
import stat

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads((args.output_dir / 'remote_run.json').read_text())
    remote = meta['remote_directory']
    local = args.output_dir / 'result'
    local.mkdir(exist_ok=True)
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('Set SADDLE_REMOTE_PASSWORD for this process')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(meta['host'], username='root', password=password, timeout=15)
        with client.open_sftp() as sftp:
            for item in sftp.listdir_attr(remote):
                if stat.S_ISREG(item.st_mode) and item.st_size < 8_000_000 and item.filename.endswith(('.json', '.txt', '.pid')):
                    sftp.get(remote + '/' + item.filename, str(local / item.filename))
            def collect(folder, target):
                target.mkdir(parents=True, exist_ok=True)
                for item in sftp.listdir_attr(folder):
                    if stat.S_ISDIR(item.st_mode):
                        collect(folder + '/' + item.filename, target / item.filename)
                    elif stat.S_ISREG(item.st_mode) and item.st_size < 8_000_000 and item.filename.endswith(('.json', '.jsonl', '.txt')):
                        sftp.get(folder + '/' + item.filename, str(target / item.filename))
            for dirname in ('outputs', 'data'):
                try:
                    sftp.stat(remote + '/' + dirname)
                except FileNotFoundError:
                    continue
                collect(remote + '/' + dirname, local / dirname)
        status_path = local / 'smoke_status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            print(json.dumps({key: status.get(key) for key in (
                'status', 'training_started', 'remote_directory', 'saved_model',
                'artifact_reload_verified', 'optimizer_delta_l2', 'counts', 'error',
                'finished_utc')}, indent=2))
            if status['status'] != 'passed':
                for name in ('train.stdout.txt', 'train.stderr.txt', 'workflow.stderr.txt'):
                    p = local / name
                    if p.exists():
                        print(name + '\n' + p.read_text()[-3500:])
        else:
            print('Workflow has not written a status file yet.')
    finally:
        client.close()


if __name__ == '__main__':
    main()
