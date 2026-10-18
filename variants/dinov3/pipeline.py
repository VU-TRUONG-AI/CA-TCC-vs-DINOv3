"""Explicit, isolated DINOv3 pipeline; never auto-started by the baseline monitor."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from variants.dinov3.common import (ROOT, DATASETS, atomic_json, prepare_data,
                                    require_baseline_complete, snapshot_sources, validate_experiment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--source-data-root', default='datasets/tensors/full')
    parser.add_argument('--label-percent', type=int, choices=[1, 5, 10, 50, 75], default=1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dinov3-repo', required=True)
    parser.add_argument('--dinov3-weights', default=None)
    parser.add_argument('--initialization', choices=['pretrained', 'random'], default='pretrained')
    parser.add_argument('--backbone-policy', choices=['trainable', 'frozen'], default='trainable')
    parser.add_argument('--image-height', type=int, default=64)
    parser.add_argument('--clip-z', type=float, default=4.0)
    parser.add_argument('--include-supervised', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    require_baseline_complete()
    validate_experiment(args.experiment)
    if (args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0)) or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Invalid learning rate or weight decay.')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Each dataset may appear only once.')
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size at least 2.')
    if args.image_height < 16 or args.image_height % 16 or args.clip_z <= 0:
        parser.error('image-height must be a positive multiple of 16; clip-z must be positive.')
    repo = Path(args.dinov3_repo).expanduser().resolve()
    if not (repo / 'hubconf.py').is_file():
        parser.error('Provide a local official DINOv3 checkout containing hubconf.py.')
    if args.initialization == 'pretrained':
        if not args.dinov3_weights or not Path(args.dinov3_weights).expanduser().is_file():
            parser.error('A local official ViT-S/16 checkpoint is required for pretrained initialization.')
    elif args.dinov3_weights:
        parser.error('Random initialization must not receive pretrained weights.')
    args.dinov3_repo = str(repo)
    args.source_data_root = str(Path(args.source_data_root).resolve())
    if args.dinov3_weights:
        args.dinov3_weights = str(Path(args.dinov3_weights).expanduser().resolve())
    output = ROOT / 'artifacts' / args.experiment
    status_path = ROOT / 'experiments_logs' / args.experiment / 'pipeline_status.json'
    settings = {k: v for k, v in vars(args).items() if k != 'resume'}
    settings['data_root'] = str(ROOT / 'datasets/tensors/experiments' / args.experiment)
    settings['backbone'] = 'dinov3_vits16'
    status = {'settings': settings, 'state': 'preparing', 'stages': [], 'started': time.time()}
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if not args.resume or old['settings'] != settings:
            parser.error('Run exists or settings changed. Use a new experiment or matching --resume.')
        status = old
        for stage in status['stages']:
            if stage['state'] == 'running':
                stage['state'] = 'interrupted'
    elif args.resume:
        parser.error('Cannot resume an experiment without a pipeline status file.')
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        snapshot_sources(output)
    status['state'] = 'preparing'
    atomic_json(status_path, status)
    private_data = prepare_data(args.source_data_root, args.experiment, args.datasets, args.label_percent)
    # Capture this interpreter's environment, without installation or downloading.
    with (output / 'packages.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    status['state'] = 'running'
    atomic_json(status_path, status)
    try:
        for dataset in args.datasets:
            percent = args.label_percent
            stages = ['self_supervised', f'train_linear_{percent}p', f'ft_{percent}p',
                      'gen_pseudo_labels', 'SupCon', f'train_linear_SupCon_{percent}p', f'ft_SupCon_{percent}p']
            if args.include_supervised:
                stages.append('supervised')
            for stage in stages:
                if any(s['dataset'] == dataset and s['stage'] == stage and s['state'] == 'completed'
                       for s in status['stages']):
                    continue
                command = [sys.executable, str(ROOT / 'variants/dinov3/main.py'),
                           '--selected_dataset', dataset, '--run_description', dataset,
                           '--experiment_description', args.experiment, '--data_path', str(private_data),
                           '--training_mode', stage, '--epochs', str(args.epochs),
                           '--batch_size', str(args.batch_size), '--seed', str(args.seed),
                           '--label_percent', str(percent), '--device', args.device,
                           '--dinov3-repo', args.dinov3_repo, '--initialization', args.initialization,
                           '--backbone-policy', args.backbone_policy, '--image-height', str(args.image_height),
                           '--clip-z', str(args.clip_z)]
                command += ['--weight-decay', str(args.weight_decay)]
                if args.learning_rate is not None:
                    command += ['--learning-rate', str(args.learning_rate)]
                if args.validation_only:
                    command.append('--validation-only')
                if args.dinov3_weights:
                    command += ['--dinov3-weights', args.dinov3_weights]
                if args.resume:
                    command.append('--resume')
                record = {'dataset': dataset, 'stage': stage, 'state': 'running',
                          'started': time.time(), 'command': command}
                status['stages'].append(record)
                status['updated_local'] = datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()
                atomic_json(status_path, status)
                with (output / 'pipeline.log').open('a') as stream:
                    stream.write('\nCOMMAND: ' + json.dumps(command) + '\n')
                    stream.flush()
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                record.update(state='completed' if result.returncode == 0 else 'failed',
                              exit_code=result.returncode, finished=time.time())
                atomic_json(status_path, status)
                if result.returncode:
                    raise RuntimeError(f'{dataset}/{stage} failed; see {output / "pipeline.log"}')
            # Existing reporting code consumes stage metrics independently of backbone type.
            if not args.validation_only:
                subprocess.run([sys.executable, str(ROOT / 'scripts/report_full_audio.py'),
                                '--experiment', args.experiment, '--datasets', dataset], check=True, cwd=ROOT)
        status.update(state='completed', finished=time.time())
        atomic_json(status_path, status)
        print(f'DINOv3 training and reports completed: {output}')
    except BaseException as error:
        status.update(state='failed', error=str(error), finished=time.time())
        atomic_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
