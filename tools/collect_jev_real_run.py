"""Poll and collect owned experiment evidence without downloading model weights."""
import argparse
import json
import os
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    metadata = json.loads((out / 'remote_run.json').read_text())
    remote = metadata['remote_directory']
    if not remote.startswith('/root/saddle-jev-real-'):
        raise ValueError('Unexpected experiment root')
    password = os.environ.get('SADDLE_REMOTE_PASSWORD')
    if not password:
        raise RuntimeError('Set SADDLE_REMOTE_PASSWORD for this process')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(metadata['host'], username='root', password=password, timeout=15)
        with client.open_sftp() as sftp:
            paths = ['real_status.json', 'source_verification.json', 'doctor.json', 'tokenizer_audit.json',
                     'initialization.json', 'model_config.json', 'typed_real.yaml', 'workflow_failure.txt',
                     'workflow.stdout.txt', 'workflow.stderr.txt', 'comparison.json',
                     'evaluation_baseline/report.json', 'evaluation_trained/report.json', 'evaluation_reload/report.json',
                     'outputs/typed_real/decision/attempt-001/training_history.jsonl',
                     'outputs/typed_real/decision/attempt-001/run_config.json',
                     'outputs/typed_real/decision/attempt-001/report.json']
            paths.extend(f'probe_batch_{batch}.{suffix}' for batch in (8,4,2,1) for suffix in ('json', 'stdout.txt', 'stderr.txt'))
            paths.extend(f'{stage}.{suffix}.txt' for stage in ('doctor', 'tokenizer_data', 'initialize', 'baseline_evaluation',
                        'validate_config', 'dry_run', 'train', 'trained_evaluation', 'reload_evaluation') for suffix in ('stdout', 'stderr'))
            if args.final:
                paths.extend(f'data/{name}' for name in ('dataset.json', 'provenance.jsonl', 'train.jsonl',
                             'validation.jsonl', 'calibration.jsonl', 'test.jsonl', 'source_audit.json'))
                paths.extend(f'{directory}/{name}' for directory in ('evaluation_baseline','evaluation_trained','evaluation_reload')
                             for name in ('token_preflight.json','predictions.jsonl','logits.jsonl'))
                paths.extend(f'outputs/typed_real/decision/attempt-001/model/{name}' for name in
                             ('decision_config.json','calibration.json','manifest.json','metadata.json'))
            for name in paths:
                try:
                    size = sftp.stat(remote + '/' + name).st_size
                except OSError:
                    continue
                if size > 8 * 1024**2:
                    continue
                target = out / 'result' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote + '/' + name, str(target))
        path = out / 'result/real_status.json'
        if not path.exists():
            print(json.dumps({'stage': 'not_started'}))
            return
        status = json.loads(path.read_text())
        summary = {key: status.get(key) for key in ('status', 'stage', 'training_started', 'batch_size', 'best_epoch', 'error')}
        progress = []
        for log in (out / 'result/train.stdout.txt', out / 'result/train.stderr.txt'):
            if not log.exists():
                continue
            for line in log.read_text(errors='replace').splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get('phase') == 'training':
                    progress.append(value)
        if progress:
            summary['training_progress'] = max(progress, key=lambda value: (value['epoch'], value['step']))
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        client.close()


if __name__ == '__main__':
    main()
