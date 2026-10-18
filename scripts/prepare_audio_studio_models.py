"""Prepare hosted browser inference assets after the fixed SpeechCommands run completes."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from variants.dinov3.common import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    experiment='dinov3_speechcommands_ravdess_best_10p_20261018'
    status_file=ROOT/'artifacts'/experiment/'fixed_run_status.json'
    target=ROOT/'artifacts/audio_studio_web_20261018/browser_assets_status.json'
    atomic_json(target,dict(state='waiting_for_training',dataset='SpeechCommands'))
    while True:
        state=json.loads(status_file.read_text())['state']
        if state=='completed':break
        if state in ('failed','interrupted'):
            atomic_json(target,dict(state='training_requires_attention',training_state=state))
            return
        if not args.watch:return
        time.sleep(30)
    try:
        from scripts.export_dinov3_browser import export
        output=ROOT/'artifacts/dinov3_browser_models/SpeechCommands'
        atomic_json(target,dict(state='exporting',dataset='SpeechCommands'))
        checkpoint=ROOT/'experiments_logs'/experiment/'SpeechCommands/ft_SupCon_10p_seed_0/saved_models/ckp_best.pt'
        export('SpeechCommands',checkpoint,output)
        destination=ROOT/'sites/ca-tcc-audio-studio/dist/models/SpeechCommands'
        destination.mkdir(parents=True,exist_ok=True)
        for file in [*output.glob('weights-*.bin'),output/'manifest.json']:
            shutil.copy2(file,destination/file.name)
        atomic_json(target,dict(state='ready_to_publish',project_id='appgprj_6ad441e65b748191a1d715dcdf0ff360',
                               dataset='SpeechCommands',source=str(destination),finished=time.time(),
                               note='Source assets ready. Obtain a fresh Site source credential, package exact source, save and deploy; do not claim updated live until native deployment succeeds.'))
        print('SpeechCommands browser assets ready; Site publication required.',flush=True)
    except BaseException as error:
        atomic_json(target,dict(state='failed',error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':main()
