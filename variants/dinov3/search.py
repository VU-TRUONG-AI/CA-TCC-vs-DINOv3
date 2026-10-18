"""Sequential validation-only search, followed by full training and test of each winner."""
import argparse
import csv
import fcntl
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from variants.dinov3.common import ROOT, DATASETS, atomic_json, require_baseline_complete, sha256, validate_experiment


def validation_result(experiment, dataset, seed):
    folder = ROOT / 'experiments_logs' / experiment / dataset / f'ft_SupCon_10p_seed_{seed}'
    selection = json.loads((folder / 'selection.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    row = next(r for r in history if r['epoch'] == selection['epoch'])
    loss = float(selection['validation_loss'])
    if not math.isfinite(loss):
        raise ValueError('Nonfinite validation score')
    return dict(validation_loss=loss, validation_accuracy=row['validation_accuracy'],
                selected_epoch=selection['epoch'], stage_directory=str(folder))


def evaluate_winner(experiment, dataset, seed):
    # Import training dependencies only after the pretrained preflight has passed.
    import importlib
    import torch
    from torch.utils.data import DataLoader
    from dataloader.dataloader import Load_Dataset
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    from variants.dinov3.model import DinoAudioModel
    folder = ROOT / 'experiments_logs' / experiment / dataset / f'ft_SupCon_10p_seed_{seed}'
    config = json.loads((folder / 'run_config.json').read_text())
    configs = importlib.import_module(f'config_files.{dataset}_Configs').Config()
    torch.set_num_threads(config['threads'])
    device = torch.device(config['device'])
    model = DinoAudioModel(configs, repo=config['dinov3_repo'], initialization='random',
                           image_height=config['image_height'], clip_z=config['clip_z'],
                           backbone_policy=config['backbone_policy']).to(device)
    checkpoint = folder / 'saved_models/ckp_best.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()
    tensor = torch.load(Path(config['data_path']) / dataset / 'test.pt', map_location='cpu', weights_only=True)
    loader = DataLoader(Load_Dataset(tensor, configs, 'supervised'), batch_size=config['batch_size'], shuffle=False)
    actual, predicted = [], []
    with torch.inference_mode():
        for samples, labels, _, _ in loader:
            logits, _ = model(samples.to(device))
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    result = dict(accuracy=accuracy_score(actual, predicted),
                  macro_f1=f1_score(actual, predicted, labels=list(range(configs.num_classes)), average='macro', zero_division=0),
                  balanced_accuracy=balanced_accuracy_score(actual, predicted),
                  confusion_matrix=confusion_matrix(actual, predicted, labels=list(range(configs.num_classes))).tolist(),
                  samples=len(actual), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    with (folder / 'winner_test_predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['target', 'prediction'])
        writer.writerows(zip(actual, predicted))
    atomic_json(folder / 'winner_test_metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='dinov3_search_10p_20261017')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--dinov3-repo', default='external/dinov3')
    parser.add_argument('--dinov3-weights', default='weights/dinov3_vits16.pth')
    parser.add_argument('--source-data-root', default='datasets/tensors/full')
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[1e-5, 3e-5, 1e-4])
    parser.add_argument('--weight-decays', nargs='+', type=float, default=[1e-4, 3e-4])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--trial-epochs', type=int, default=10)
    parser.add_argument('--final-epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backbone-policy', choices=['trainable', 'frozen'], default='trainable')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    validate_experiment(args.experiment)
    if len(set(args.datasets)) != len(args.datasets) or args.batch_size < 2 or min(args.trial_epochs, args.final_epochs) < 1:
        parser.error('Datasets must be unique; batch-size >=2; epochs >=1.')
    if any(not math.isfinite(v) or v <= 0 for v in args.learning_rates) or any(not math.isfinite(v) or v < 0 for v in args.weight_decays):
        parser.error('Invalid learning rate or weight decay.')
    args.dinov3_repo = str(Path(args.dinov3_repo).resolve())
    args.dinov3_weights = str(Path(args.dinov3_weights).resolve())
    args.source_data_root = str(Path(args.source_data_root).resolve())
    output = ROOT / 'artifacts' / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'search.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This search is already running.')
    settings = {k: v for k, v in vars(args).items() if k != 'resume'}
    settings.update(label_percent=10, initialization='pretrained', selection='minimum final-stage validation cross entropy')
    path = output / 'search_status.json'
    if path.exists():
        status = json.loads(path.read_text())
        if not args.resume or status['settings'] != settings:
            parser.error('Use matching --resume or a new experiment name.')
    else:
        status = dict(settings=settings, state='preparing', trials=[], finalists={}, started=time.time())
    def save():
        status['updated'] = time.time()
        atomic_json(path, status)
        rows = status['trials']
        with (output / 'trials.csv').open('w') as stream:
            fields = ['experiment', 'dataset', 'learning_rate', 'weight_decay', 'batch_size', 'epochs', 'seed',
                      'state', 'exit_code', 'validation_loss', 'validation_accuracy', 'selected_epoch', 'error']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    save()
    try:
        require_baseline_complete()
        if not (Path(args.dinov3_repo) / 'hubconf.py').is_file():
            raise FileNotFoundError('Missing official local DINOv3 repo')
        if not Path(args.dinov3_weights).is_file():
            status.update(state='awaiting_weights', error='Official ViT-S/16 checkpoint required; no training was started.')
            save()
            print(status['error'], flush=True)
            return 2
        weights_hash = sha256(args.dinov3_weights)
        source_hash = {str(p.relative_to(args.dinov3_repo)): sha256(p)
                       for p in sorted(Path(args.dinov3_repo).rglob('*.py'))}
        provenance = dict(weights_sha256=weights_hash, source_sha256=source_hash)
        if status.get('provenance', provenance) != provenance:
            raise ValueError('Source or pretrained weights changed; use a new search.')
        status['provenance'] = provenance
        status.pop('error', None)
        status['state'] = 'running'
        save()
        def run(experiment, dataset, lr, wd, epochs):
            command = [sys.executable, str(ROOT / 'variants/dinov3/pipeline.py'), '--experiment', experiment,
                       '--datasets', dataset, '--source-data-root', args.source_data_root, '--label-percent', '10',
                       '--epochs', str(epochs), '--batch-size', str(args.batch_size), '--seed', str(args.seed),
                       '--device', args.device, '--dinov3-repo', args.dinov3_repo, '--dinov3-weights', args.dinov3_weights,
                       '--initialization', 'pretrained', '--backbone-policy', args.backbone_policy,
                       '--learning-rate', str(lr), '--weight-decay', str(wd), '--validation-only']
            if (ROOT / 'experiments_logs' / experiment / 'pipeline_status.json').exists():
                command.append('--resume')
            record = dict(experiment=experiment, dataset=dataset, learning_rate=lr, weight_decay=wd,
                          batch_size=args.batch_size, epochs=epochs, seed=args.seed, state='running',
                          started=time.time(), command=command)
            return record, command
        for dataset in args.datasets:
            finished = status['finalists'].get(dataset)
            if finished and finished['state'] == 'completed' and 'test' in finished:
                continue
            for index, (lr, wd) in enumerate(itertools.product(args.learning_rates, args.weight_decays)):
                experiment = f'{args.experiment}_{dataset.lower()}_t{index:02d}'
                old = next((r for r in status['trials'] if r['experiment'] == experiment), None)
                if old and old['state'] == 'completed':
                    continue
                record, command = run(experiment, dataset, lr, wd, args.trial_epochs)
                if old:
                    record['previous_attempts'] = old.get('previous_attempts', []) + [{k: v for k, v in old.items() if k != 'previous_attempts'}]
                    status['trials'].remove(old)
                status['trials'].append(record)
                save()
                print(f'Running {experiment}: lr={lr}, weight_decay={wd}', flush=True)
                with (output / f'{experiment}.log').open('a') as stream:
                    code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
                record.update(exit_code=code, finished=time.time())
                try:
                    if code:
                        raise RuntimeError(f'Pipeline exited {code}; see trial log')
                    record.update(validation_result(experiment, dataset, args.seed), state='completed')
                except Exception as error:
                    record.update(state='failed', error=str(error))
                save()
            candidates = [r for r in status['trials'] if r['dataset'] == dataset and r['state'] == 'completed']
            if not candidates:
                raise RuntimeError(f'No successful trials for {dataset}; inspect logs before retrying.')
            best = min(candidates, key=lambda r: r['validation_loss'])
            existing_final = status['finalists'].get(dataset)
            if existing_final and existing_final['selected_trial'] != best['experiment']:
                raise RuntimeError('Winner changed after final training started; use a new search experiment.')
            atomic_json(output / f'best_parameters_{dataset}.json', best)
            experiment = f'{args.experiment}_{dataset.lower()}_final'
            finalist = status['finalists'].get(dataset)
            if not finalist or finalist['state'] != 'completed':
                previous_finalist = finalist
                finalist, command = run(experiment, dataset, best['learning_rate'], best['weight_decay'], args.final_epochs)
                if previous_finalist:
                    finalist['previous_attempts'] = previous_finalist.get('previous_attempts', []) + [{k: v for k, v in previous_finalist.items() if k != 'previous_attempts'}]
                finalist['selected_trial'] = best['experiment']
                status['finalists'][dataset] = finalist
                save()
                with (output / f'{experiment}.log').open('a') as stream:
                    code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
                finalist['exit_code'] = code
                if code:
                    finalist['state'] = 'failed'
                    raise RuntimeError(f'Final training failed for {dataset}')
                finalist.update(validation_result(experiment, dataset, args.seed), state='completed', finished=time.time())
                save()
            if 'test' not in finalist:
                finalist['test'] = evaluate_winner(experiment, dataset, args.seed)
                save()
        status.update(state='completed', finished=time.time())
        save()
        return 0
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    sys.exit(main())
