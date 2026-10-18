"""One entry point for data work; training lives in the same project."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
STEPS={
    'audit':('prepare_audio.py',['--audit-only']),
    'prepare':('prepare_audio.py',[]),
    'validate':('validate_preprocessed.py',[]),
    'check-stages':('report_preprocessing_stages.py',[]),
    'figures':('render_preprocessing_report.py',[]),
    'smoke':('make_smoke_data.py',[]),
    'package':('package_tensors.py',[]),
}
p=argparse.ArgumentParser(description='Run one clear data step; --step all runs audit through smoke, without requiring PyTorch.')
p.add_argument('--step',choices=[*STEPS,'all'],default='validate')
args=p.parse_args()
steps=['audit','prepare','validate','check-stages','figures','smoke'] if args.step=='all' else [args.step]
(ROOT/'logs').mkdir(exist_ok=True)
for step in steps:
    script,flags=STEPS[step]
    timestamp=datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y%m%d_%H%M%S')
    log=ROOT/'logs'/f'{timestamp}_{step}.log'
    command=[sys.executable,str(ROOT/'pipeline'/script),*flags]
    print(f'[{step}] Running {script}\nLog: {log}',flush=True)
    with log.open('w') as output:
        output.write('Command: '+' '.join(command)+'\n');output.flush()
        result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-5000:],file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f'[{step}] PASS',flush=True)
    if step=='package':
        subprocess.run([sys.executable,str(ROOT/'pipeline'/script),'--data-root',str(ROOT/'processed/smoke')],cwd=ROOT,check=True)
