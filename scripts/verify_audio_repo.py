"""Reproducible end-to-end smoke verification on both new audio datasets."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment', default='smoke_audio_20261017')
p.add_argument('--device', default='auto')
p.add_argument('--resume', action='store_true')
args = p.parse_args()
output = ROOT / 'artifacts' / args.experiment
output.mkdir(parents=True, exist_ok=True)
status = dict(experiment=args.experiment, purpose='smoke_test_only', started=time.time(), state='running', commands=[])
status_path = output / 'verification_status.json'


def save():
    status_path.write_text(json.dumps(status, indent=2))


def run(name, command):
    record = dict(name=name, command=command, started=time.time(), log=str(output / f'{name}.log'))
    status['commands'].append(record)
    save()
    print(f'[{name}] Running; log={record["log"]}', flush=True)
    with open(record['log'], 'w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, seconds=time.time()-record['started'])
    save()
    if result.returncode:
        status.update(state='failed', failed_step=name, finished=time.time())
        save()
        print(Path(record['log']).read_text()[-8000:], flush=True)
        raise SystemExit(result.returncode)
    print(f'[{name}] PASS ({record["seconds"]:.1f}s)', flush=True)


python = sys.executable
run('environment', [python, '-c',
    "import torch,numpy,scipy,sklearn,einops; print('torch',torch.__version__);print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('sklearn',sklearn.__version__);print('einops',einops.__version__); print('CUDA',torch.cuda.is_available());print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); x=torch.ones(2,device='cuda' if torch.cuda.is_available() else 'cpu'); print('tensor', (x+x).cpu().tolist())"])
run('package', [python, 'scripts/package_audio_data.py', '--source', 'datasets/processed/smoke', '--output', 'datasets/tensors/smoke'])
command = [python, 'scripts/run_audio_pipeline.py', '--data-root', 'datasets/tensors/smoke', '--experiment', args.experiment,
           '--epochs', '1', '--label-percent', '10', '--device', args.device, '--include-supervised']
if args.resume:
    command.append('--resume')
run('pipeline', command)
for dataset in ('SpeechCommands', 'RAVDESS'):
    checkpoint = f'experiments_logs/{args.experiment}/{dataset}/ft_SupCon_10p_seed_0/saved_models/ckp_best.pt'
    run('export_'+dataset, [python, 'scripts/export_audio_model.py', '--dataset', dataset, '--data-root', 'datasets/tensors/smoke',
                           '--checkpoint', checkpoint, '--output', str(output/'bundles'/dataset)])
    # Verify epoch resume state actually loads, without training extra epochs.
    run('resume_'+dataset, [python, 'main.py', '--data_path', str(ROOT/'datasets/tensors/smoke'), '--selected_dataset', dataset,
                            '--experiment_description', args.experiment, '--run_description', dataset,
                            '--training_mode', 'ft_SupCon_10p', '--epochs', '1', '--batch_size', '32',
                            '--label_percent', '10', '--device', args.device, '--resume'])
run('wav_parity', [python, 'scripts/verify_audio_bundles.py', '--data-root', 'datasets/tensors/smoke',
                   '--bundles', str(output/'bundles'), '--output', str(output/'wav_parity.json')])
rows=[]
for dataset in ('SpeechCommands', 'RAVDESS'):
    source=ROOT/'experiments_logs'/args.experiment/dataset
    for path in sorted(source.glob('*/metrics.json')):
        metrics=json.loads(path.read_text())
        rows.append(dict(dataset=dataset, stage=path.parent.name, accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                         balanced_accuracy=metrics['balanced_accuracy'], test_count=metrics['test_count']))
with (output/'metrics_summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
(output/'metrics_summary.json').write_text(json.dumps(rows,indent=2))
manifest={}
for folder in ('models','trainer','dataloader','config_files','scripts'):
    for path in (ROOT/folder).glob('*.py'):
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest['main.py']=hashlib.sha256((ROOT/'main.py').read_bytes()).hexdigest()
(output/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
run('packages',[python,'-m','pip','freeze'])
status.update(state='completed', finished=time.time(), metrics_file=str(output/'metrics_summary.csv'))
save()
print(f'VERIFIED: {output}',flush=True)
