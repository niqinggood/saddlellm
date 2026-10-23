"""Freeze a joint model by validation NLL before any fresh-test prediction."""
import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import paramiko

BASELINE_RUN = '/root/saddle-jev-real-20261004-011038-62eeb7b4/outputs/typed_real/decision/attempt-001'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidates', required=True, help='Comma-separated completed joint run names')
    parser.add_argument('--completed-plans', required=True, help='Comma-separated plan filenames')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if (out / 'selection_frozen.json').exists():
        raise FileExistsError('Selection already frozen; do not revise it using fresh-test results')
    names = args.candidates.split(',')
    if not names or any(not name.replace('_', '').isalnum() for name in names):
        raise ValueError('Invalid candidate run name')
    metadata = json.loads((out / 'remote_run.json').read_text())
    remote = metadata['remote_directory']
    if not remote.startswith('/root/saddle-jev-opt-'):
        raise ValueError('Unexpected remote study root')
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
            raise RuntimeError('Remote host key differs from original study')
        with client.open_sftp() as sftp:
            def read(path):
                with sftp.open(path, 'rb') as handle:
                    return handle.read()

            def load(path):
                return json.loads(read(path))

            try:
                sftp.stat(remote + '/selection_frozen.json')
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError('Remote selection already frozen')
            completed = {}
            for plan_name in args.completed_plans.split(','):
                plan_file = Path(plan_name)
                if plan_file.name != plan_name or plan_file.suffix != '.json':
                    raise ValueError('Plan must be a study-root JSON filename')
                plan = load(remote + '/' + plan_name)
                status = load(remote + '/' + plan_file.stem + '_status.json')
                if status['status'] != 'passed' or status['new_final_test_used_for_selection'] is not False:
                    raise RuntimeError('Development plan incomplete or test isolation absent: ' + plan_name)
                expected = {row['name'] for row in plan['experiments']}
                actual = {row['name'] for row in status['experiments'] if row['status'] == 'passed'}
                if expected != actual or hashlib.sha256(read(remote + '/' + plan_name)).hexdigest() != status['plan_sha256']:
                    raise RuntimeError('Completed plan differs from its recorded execution: ' + plan_name)
                completed.update({row['name']: row for row in status['experiments']})
            if any(name not in completed for name in names):
                raise ValueError('Candidate is absent from completed plans')
            from saddlellm.decision.TypedSchema import read_typed_dataset

            candidates = []
            for name, base in [('previous_trained_baseline', BASELINE_RUN)] + [
                    (name, remote + '/runs/' + name + '/output/decision/attempt-001') for name in names]:
                raw = read(base + '/report.json')
                report = json.loads(raw)
                if (report['counts']['validation'] != 384 or report['counts']['calibration'] != 384
                        or report['counts']['test'] != 768 or report['counts']['train'] < 1536):
                    raise ValueError('Candidate must cover the full joint validation/calibration set')
                if name != 'previous_trained_baseline' and (report['test_evaluated'] is not False or report['test'] != {}):
                    raise RuntimeError('Development candidate evaluated its test')
                data_name = 'development' if name == 'previous_trained_baseline' else completed[name]['data']
                if data_name not in {'development', 'expanded', 'instance_query/development', 'instance_query/expanded'}:
                    raise ValueError('Selection requires audited joint development data')
                _, candidate_splits, candidate_hashes = read_typed_dataset(out / 'data' / data_name)
                for split in ('validation', 'calibration'):
                    if Counter(row['type'] for row in candidate_splits[split]) != {'choice': 128, 'score': 128, 'noul': 128}:
                        raise ValueError('Joint validation/calibration requires 128 records per task')
                for split in candidate_splits:
                    digest = candidate_hashes[split]
                    if report['dataset_hashes'][split] != digest:
                        raise RuntimeError('Training report data differs from audited joint input: ' + split)
                    if hashlib.sha256(read(remote + '/data/' + data_name + '/' + split + '.jsonl')).hexdigest() != digest:
                        raise RuntimeError('Remote joint input differs from local input: ' + split)
                model = base + '/model'
                config = load(model + '/decision_config.json')['config']
                candidates.append({'name': name, 'model': model,
                                   'best_epoch': report['selection']['best_epoch'],
                                   'validation_nll': report['selection']['best_validation_nll'],
                                   'validation_samples': report['counts']['validation'],
                                   'report_sha256': hashlib.sha256(raw).hexdigest(),
                                   'model_manifest_sha256': hashlib.sha256(read(model + '/manifest.json')).hexdigest(),
                                   'query_pooling': config.get('query_pooling', 'mean'),
                                   'instance_question_in_query': config.get('instance_question_in_query', False)})
            chosen = min(candidates, key=lambda row: row['validation_nll'])
            dataset = remote + '/data/' + ('instance_query/final_holdout' if chosen['instance_question_in_query'] else 'final_holdout')
            local_dataset = out / 'data' / ('instance_query/final_holdout' if chosen['instance_question_in_query'] else 'final_holdout')
            _, splits, dataset_hashes = read_typed_dataset(local_dataset)
            for name, digest in dataset_hashes.items():
                file_name = name + '.jsonl' if name in splits else name
                if hashlib.sha256(read(dataset + '/' + file_name)).hexdigest() != digest:
                    raise RuntimeError('Fresh data differs between local and remote: ' + file_name)
            record = {'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'selection_metric': 'minimum uncalibrated overall validation NLL',
                      'selection_candidates': candidates, 'selected_name': chosen['name'],
                      'selected_validation_nll': chosen['validation_nll'],
                      'completed_plans': args.completed_plans.split(','),
                      'optimized_model': chosen['model'], 'baseline_model': BASELINE_RUN + '/model',
                      'optimized_manifest_sha256': chosen['model_manifest_sha256'],
                      'baseline_manifest_sha256': candidates[0]['model_manifest_sha256'],
                      'dataset_dir': dataset, 'dataset_hashes': dataset_hashes,
                      'dataset_counts': {k: len(v) for k, v in splits.items()},
                      'fresh_test_samples': len(splits['test']),
                      'runtime_manifest_sha256': hashlib.sha256(read(remote + '/runtime_v2_manifest.json')).hexdigest(),
                      'new_final_test_used_for_selection': False,
                      'baseline_is_previous_trained_model': True,
                      'host_key_sha256': host_key}
            payload = json.dumps(record, indent=2, allow_nan=False) + '\n'
            with sftp.open(remote + '/selection_frozen.json', 'wx') as handle:
                handle.write(payload)
            (out / 'selection_frozen.json').write_text(payload)
            print(json.dumps({'selected_name': chosen['name'], 'validation_nll': chosen['validation_nll'],
                              'new_final_test_used_for_selection': False}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
