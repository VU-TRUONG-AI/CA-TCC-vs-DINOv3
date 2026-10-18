"""Run CA-TCC stages with durable stage status and epoch-level resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--datasets', nargs='+', default=['SpeechCommands', 'RAVDESS'], choices=['SpeechCommands', 'RAVDESS'])
p.add_argument('--data-root', default='datasets/tensors/full')
p.add_argument('--experiment', default='audio_baseline')
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--label-percent', type=int, default=1, choices=[1, 5, 10, 50, 75])
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='auto')
p.add_argument('--include-supervised', action='store_true')
p.add_argument('--resume', action='store_true')
args = p.parse_args()
status_path = ROOT / 'experiments_logs' / args.experiment / 'pipeline_status.json'
status_path.parent.mkdir(parents=True, exist_ok=True)
settings = {k:v for k,v in vars(args).items() if k != 'resume'}
status = {'settings':settings, 'stages':[], 'state':'running', 'started':time.time()}
if status_path.exists():
    old = json.loads(status_path.read_text())
    if not args.resume or old['settings'] != settings:
        p.error('Experiment already exists or settings changed. Use a new experiment or matching --resume.')
    status = old
    status['state'] = 'running'


def save():
    temp = status_path.with_suffix('.tmp')
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(status_path)


save()
for dataset in args.datasets:
    stages = ['self_supervised', f'train_linear_{args.label_percent}p', f'ft_{args.label_percent}p',
              'gen_pseudo_labels', 'SupCon', f'train_linear_SupCon_{args.label_percent}p',
              f'ft_SupCon_{args.label_percent}p']
    if args.include_supervised:
        stages += ['supervised']
    for stage in stages:
        if any(s['dataset']==dataset and s['stage']==stage and s['state']=='completed' for s in status['stages']):
            print(f'SKIP completed: {dataset}/{stage}', flush=True)
            continue
        command = [sys.executable, str(ROOT / 'main.py'), '--selected_dataset', dataset,
                   '--data_path', str(Path(args.data_root).resolve()), '--training_mode', stage,
                   '--experiment_description', args.experiment, '--run_description', dataset,
                   '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
                   '--seed', str(args.seed), '--device', args.device, '--label_percent', str(args.label_percent)]
        if args.resume:
            command.append('--resume')
        record = dict(dataset=dataset, stage=stage, state='running', started=time.time(), command=command)
        status['stages'].append(record)
        save()
        print('RUN:', ' '.join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        record.update(finished=time.time(), exit_code=result.returncode,
                      state='completed' if result.returncode==0 else 'failed')
        if result.returncode:
            status['state']='failed'
            save()
            raise SystemExit(result.returncode)
        save()
status.update(state='completed', finished=time.time())
save()
print(f'Pipeline completed: {status_path}', flush=True)
