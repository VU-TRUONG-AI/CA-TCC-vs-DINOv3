"""Supervise a full-data run and save reports automatically as each dataset finishes."""
import argparse
from datetime import datetime
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment',required=True)
parser.add_argument('--epochs',type=int,default=40)
parser.add_argument('--label-percent',type=int,default=10,choices=[1,5,10,50,75])
parser.add_argument('--batch-size',type=int,default=32)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--data-root',default='datasets/tensors/full')
parser.add_argument('--skip-supervised',action='store_true',help='Reuse an existing 100%% supervised reference instead of retraining it.')
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
os.chdir(ROOT)
output=ROOT/'artifacts'/args.experiment
output.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(output/'mpl-cache'))
status_path=output/'full_run_status.json'
settings={k:v for k,v in vars(args).items() if k!='resume'}
status=dict(state='preparing',settings=settings,pid=os.getpid(),started=time.time(),reported_datasets=[],checks=[])
if status_path.exists():
    previous=json.loads(status_path.read_text())
    previous['settings'].setdefault('data_root', 'datasets/tensors/full')
    previous['settings'].setdefault('skip_supervised', False)
    if not args.resume or previous['settings']!=settings:
        parser.error('Run already exists or settings changed. Use a new experiment or matching --resume.')
    status=previous
    status.update(state='resuming',pid=os.getpid())


def save():
    status['updated_local']=datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()
    temporary=status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status,ensure_ascii=False,indent=2))
    temporary.replace(status_path)


def run(name,command):
    print(f'[{name}] {command}',flush=True)
    with (output/f'{name}.log').open('a') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    status['checks'].append(dict(name=name,command=command,exit_code=result.returncode))
    save()
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {output/name}.log')


child=None
try:
    save()
    # Preserve the actual source and environment before training starts.
    if not args.resume:
        files=[ROOT/'main.py',ROOT/'utils.py',ROOT/'requirements-audio.txt',ROOT/'requirements-report.txt',ROOT/'datasets/pipeline/audio_features.py']
        for name in ('models','dataloader','trainer','config_files','scripts'):
            files+=list((ROOT/name).glob('*.py'))
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (output/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
        with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in files:archive.write(path,str(path.relative_to(ROOT)))
    run('environment',[sys.executable,'-c',
        "import torch,numpy,scipy,sklearn;print('torch',torch.__version__);print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('sklearn',sklearn.__version__);assert torch.cuda.is_available();print('GPU',torch.cuda.get_device_name(0));print('CUDA tensor',torch.ones(2,device='cuda').cpu().tolist())"])
    run('packages',[sys.executable,'-m','pip','freeze'])
    run('package',[sys.executable,'scripts/package_audio_data.py','--source','datasets/processed/full','--output',args.data_root])
    for dataset in ('RAVDESS','SpeechCommands'):
        manifest=json.loads((ROOT/args.data_root/dataset/'manifest.json').read_text())
        if manifest['smoke_subset']:
            raise RuntimeError('Full-data run cannot use smoke artifacts.')
    command=[sys.executable,'scripts/run_audio_pipeline.py','--data-root',args.data_root,'--datasets','RAVDESS','SpeechCommands',
             '--experiment',args.experiment,'--epochs',str(args.epochs),'--label-percent',str(args.label_percent),
             '--batch-size',str(args.batch_size),'--seed',str(args.seed),'--device','cuda:0']
    if not args.skip_supervised:command.append('--include-supervised')
    if args.resume:command.append('--resume')
    status.update(state='training',pipeline_command=command)
    save()
    with (output/'pipeline.log').open('a') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status['pipeline_pid']=child.pid
        save()
        while True:
            pipeline_path=ROOT/'experiments_logs'/args.experiment/'pipeline_status.json'
            if pipeline_path.exists():
                try:
                    pipeline=json.loads(pipeline_path.read_text())
                except json.JSONDecodeError:
                    pipeline=None
                if pipeline:
                    active=[s for s in pipeline['stages'] if s['state']=='running']
                    status['active_stage']={k:active[-1][k] for k in ('dataset','stage','started')} if active else None
                    for dataset in ('RAVDESS','SpeechCommands'):
                        done=[s for s in pipeline['stages'] if s['dataset']==dataset and s['state']=='completed']
                        if len(done)==(7 if args.skip_supervised else 8) and dataset not in status['reported_datasets']:
                            run('report_'+dataset,[sys.executable,'scripts/report_full_audio.py','--experiment',args.experiment,'--datasets',dataset])
                            checkpoint=f'experiments_logs/{args.experiment}/{dataset}/ft_SupCon_{args.label_percent}p_seed_{args.seed}/saved_models/ckp_best.pt'
                            run('export_'+dataset,[sys.executable,'scripts/export_audio_model.py','--dataset',dataset,'--data-root',args.data_root,
                                                   '--checkpoint',checkpoint,'--output',str(output/'bundles'/dataset)])
                            status['reported_datasets'].append(dataset)
                    save()
            code=child.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f'Training pipeline exited {code}; see pipeline.log.')
                if len(status['reported_datasets'])==2:
                    break
            time.sleep(5)
    run('wav_parity',[sys.executable,'scripts/verify_audio_bundles.py','--data-root',args.data_root,
                       '--bundles',str(output/'bundles'),'--output',str(output/'wav_parity.json')])
    status.update(state='completed',finished=time.time(),active_stage=None)
    save()
    print(f'FULL TRAINING COMPLETED: {output}',flush=True)
except BaseException as error:
    # Avoid a silent orphan if report/export fails while the pipeline is running.
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=30)
    status.update(state='failed',error=str(error),finished=time.time())
    save()
    traceback.print_exc()
    raise
