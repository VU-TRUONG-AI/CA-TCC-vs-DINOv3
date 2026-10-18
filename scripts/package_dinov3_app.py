"""After fixed training completes, export both DINOv3 models and build the app ZIP."""
import argparse
import atexit
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from variants.dinov3.common import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    output = ROOT/'artifacts/dinov3_app_bundles'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'package.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A packaging worker is already active.', flush=True)
        lock.close()
        return
    atexit.register(lock.close)
    state_path = output/'package_status.json'
    state = dict(state='waiting_for_training', speechcommands_experiment='dinov3_speechcommands_ravdess_best_10p_20261018')
    atomic_json(state_path, state)
    fixed_status = ROOT/'artifacts'/state['speechcommands_experiment']/'fixed_run_status.json'
    while True:
        status = json.loads(fixed_status.read_text()) if fixed_status.exists() else {}
        if status.get('state') == 'completed':
            break
        if status.get('state') in ('failed', 'interrupted'):
            state.update(state='training_requires_attention', error=status.get('error'))
            atomic_json(state_path, state)
            return
        if not args.watch:
            print('Training is not finished; use --watch or run again after completion.', flush=True)
            return
        time.sleep(30)
    try:
        state['state'] = 'exporting'
        atomic_json(state_path, state)
        from scripts.export_dinov3_audio import export
        checkpoints = {
            'RAVDESS': 'dinov3_search_10p_20261017_ravdess_final',
            'SpeechCommands': state['speechcommands_experiment'],
        }
        for dataset, experiment in checkpoints.items():
            checkpoint = ROOT/'experiments_logs'/experiment/dataset/'ft_SupCon_10p_seed_0/saved_models/ckp_best.pt'
            export(dataset, checkpoint, output/dataset)
        from scripts.verify_dinov3_app_api import verify
        verify(output, list(checkpoints))
        (output/'README.md').write_text(
            '# DINOv3 app models\n\nTwo verified CPU TorchScript bundles: RAVDESS and SpeechCommands. '
            'Each folder contains the model, labels/metadata, training normalization, WAV frontend, '
            'standalone infer.py and verification.json.\n\n'
            'For each model install requirements.txt, then run python infer.py --wav /absolute/path.wav. '
            'The repo web/API integration is documented in web/README.md. '
            'HTTP and model parity checks are in api_verification.json.\n',encoding='utf-8')
        state['state'] = 'completed'
        atomic_json(state_path, state)
        archive = shutil.make_archive(str(ROOT/'artifacts/dinov3_app_models'), 'zip', output)
        app_archive = ROOT/'artifacts/dinov3_demo_app.zip'
        with zipfile.ZipFile(app_archive, 'w', zipfile.ZIP_DEFLATED) as app:
            for file in (ROOT/'web').rglob('*'):
                if file.is_file() and '__pycache__' not in file.parts:
                    app.write(file, str(file.relative_to(ROOT)))
            for name in ('scripts/audio_features.py', 'datasets/pipeline/audio_features.py'):
                app.write(ROOT/name, name)
            app.writestr('scripts/__init__.py', '')
            app.writestr('datasets/pipeline/__init__.py', '')
            for dataset in checkpoints:
                for file in (output/dataset).rglob('*'):
                    if file.is_file():
                        app.write(file, 'models/'+str(file.relative_to(output)))
            app.writestr('requirements.txt', 'torch==2.5.1\nnumpy>=1.26,<3\nscipy>=1.11,<2\nscikit-learn>=1.4,<2\n')
            app.writestr('start_app.py', 'from pathlib import Path\nimport os,runpy,sys\nos.chdir(Path(__file__).resolve().parent)\nsys.argv=["web.server","--bundles","models","--jobs","runtime_jobs","--port","8000"]\nrunpy.run_module("web.server",run_name="__main__")\n')
            app.writestr('README.md', '# DINOv3 Audio App\n\nInstall requirements.txt, then run python start_app.py. Open http://127.0.0.1:8000. Models: RAVDESS and SpeechCommands. CPU inference; no training data or DINOv3 checkout needed. See web/README.md for API endpoints. Do not run the training commands from that development README in this inference-only archive.\n')
        with zipfile.ZipFile(app_archive) as app:
            assert app.testzip() is None
        readme = ROOT/'artifacts'/state['speechcommands_experiment']/'README.md'
        with readme.open('a') as stream:
            stream.write('\n## 8. Gói xuất và nhúng app\n\nCả hai mô hình DINOv3 đã xuất TorchScript, '
                         'kiểm tra logits/WAV parity và API localhost bằng WAV thật. '
                         '[Gói model](../dinov3_app_models.zip), [app demo kèm model](../dinov3_demo_app.zip), '
                         '[hướng dẫn API](../../web/README.md). Suy luận CPU ở backend Python.\n')
        state.update(state='completed', archive=archive, app_archive=str(app_archive), finished=time.time())
        atomic_json(state_path, state)
        print(f'Both DINOv3 models exported and packaged: {archive}', flush=True)
    except BaseException as error:
        state.update(state='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
