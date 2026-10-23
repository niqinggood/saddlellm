"""Execute a recorded SSH experiment command without persisting credentials."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', default='root')
    parser.add_argument('--command-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('SADDLE_REMOTE_PASSWORD must be supplied for this process')
    command = args.command_file.read_text(encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    record = {'host': args.host, 'user': args.user,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command': command, 'command_sha256': hashlib.sha256(command.encode()).hexdigest()}
    try:
        client.connect(args.host, username=args.user, password=password,
                       timeout=15, auth_timeout=15, banner_timeout=15)
        record['host_key_sha256'] = hashlib.sha256(
            client.get_transport().get_remote_server_key().asbytes()).hexdigest()
        _, stdout, stderr = client.exec_command(command, timeout=args.timeout)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        code = stdout.channel.recv_exit_status()
        (args.output_dir / 'stdout.txt').write_text(out, encoding='utf-8')
        (args.output_dir / 'stderr.txt').write_text(err, encoding='utf-8')
        record['exit_code'] = code
        print(out)
        if err:
            print(err)
    finally:
        client.close()
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.output_dir / 'command.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
