"""Train SpeechCommands once with the validation-selected RAVDESS hyperparameters."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from variants.dinov3.common import atomic_json, validate_experiment, sha256
from scripts.report_dinov3_fixed import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='dinov3_speechcommands_ravdess_best_10p_20261018')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, default=4, help='Per-stage budget; default reduced for the submission deadline.')
    args = parser.parse_args()
    validate_experiment(args.experiment)
    if args.epochs < 1:
        parser.error('epochs must be positive')
    source_file = ROOT / 'artifacts/dinov3_search_10p_20261017/best_parameters_RAVDESS.json'
    source = json.loads(source_file.read_text())
    settings = dict(dataset='SpeechCommands', learning_rate=source['learning_rate'], weight_decay=source['weight_decay'],
                    batch_size=source['batch_size'], seed=source['seed'], epochs=args.epochs, label_percent=10,
                    initialization='pretrained', backbone_policy='trainable')
    output = ROOT / 'artifacts' / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'fixed_run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_file = output / 'fixed_run_status.json'
        if status_file.exists():
            status = json.loads(status_file.read_text())
            if not args.resume or status['settings'] != settings or status['selection_source'] != source:
                parser.error('Use matching --resume or a new experiment.')
            if status['state'] == 'completed':
                generate(ROOT, args.experiment)
                print('Already completed.', flush=True)
                return
            previous = {k: v for k, v in status.items() if k != 'previous_attempts'}
            status.setdefault('previous_attempts', []).append(previous)
            status.pop('error', None)
        else:
            status = dict(settings=settings, selection_source=source, source_selection_sha256=sha256(source_file),
                          started=time.time(), state='preparing')
            for filename in ('run_dinov3_fixed.py', 'report_dinov3_fixed.py'):
                shutil.copy2(ROOT / 'scripts' / filename, output / filename)
        def save():
            status.update(updated=time.time(), pid=os.getpid())
            atomic_json(status_file, status)
            try:
                generate(ROOT, args.experiment)
            except json.JSONDecodeError:
                pass  # A trainer is writing history; preserve last README and retry.
        command = [sys.executable, str(ROOT / 'variants/dinov3/pipeline.py'), '--experiment', args.experiment,
                   '--datasets', 'SpeechCommands', '--source-data-root', 'datasets/tensors/full',
                   '--label-percent', '10', '--epochs', str(settings['epochs']), '--batch-size', str(settings['batch_size']),
                   '--seed', str(settings['seed']), '--device', 'cuda:0', '--dinov3-repo', str(ROOT/'external/dinov3'),
                   '--dinov3-weights', str(ROOT/'weights/dinov3_vits16.pth'), '--initialization', 'pretrained',
                   '--backbone-policy', 'trainable', '--learning-rate', str(settings['learning_rate']),
                   '--weight-decay', str(settings['weight_decay']), '--validation-only']
        pipeline_file = ROOT / 'experiments_logs' / args.experiment / 'pipeline_status.json'
        if pipeline_file.exists():
            command.append('--resume')
        status.update(state='running', command=command)
        save()
        child = None
        def stop(_sig, _frame):
            raise KeyboardInterrupt('Supervisor stopped by signal')
        signal.signal(signal.SIGTERM, stop)
        try:
            pipe = json.loads(pipeline_file.read_text()) if pipeline_file.exists() else {}
            if pipe.get('state') != 'completed':
                with (output / 'supervisor.log').open('a') as stream:
                    stream.write('\nCOMMAND: ' + json.dumps(command) + '\n')
                    stream.flush()
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    status['pipeline_pid'] = child.pid
                    save()
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            save()
                    status['pipeline_exit_code'] = code
                    if code:
                        raise RuntimeError(f'Pipeline failed with exit code {code}; inspect supervisor.log/pipeline.log')
            status['state'] = 'evaluating_test'
            save()
            metrics_file = ROOT / 'experiments_logs' / args.experiment / 'SpeechCommands/ft_SupCon_10p_seed_0/winner_test_metrics.json'
            if metrics_file.exists():
                status['test'] = json.loads(metrics_file.read_text())
            else:
                from variants.dinov3.search import evaluate_winner
                status['test'] = evaluate_winner(args.experiment, 'SpeechCommands', settings['seed'])
            status.update(state='completed', finished=time.time())
            save()
            print(f'Fixed training completed: {output}', flush=True)
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            save()
            raise


if __name__ == '__main__':
    main()
