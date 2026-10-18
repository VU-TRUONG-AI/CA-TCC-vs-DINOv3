"""Run isolation, source provenance and prerequisites for the DINOv3 variant."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE_EXPERIMENT = 'full_audio_1p_seed0_20261017'
DATASETS = ('RAVDESS', 'SpeechCommands')


def require_baseline_complete():
    """Do not use resources for the new experiment until the current run finishes."""
    folder = ROOT / 'artifacts' / BASELINE_EXPERIMENT
    status_path = folder / 'full_run_status.json'
    if not status_path.is_file():
        raise RuntimeError(f'Missing baseline status: {status_path}')
    status = json.loads(status_path.read_text())
    if status.get('state') != 'completed' or set(status.get('reported_datasets', [])) != set(DATASETS):
        raise RuntimeError('The current 1% baseline has not finished both datasets. DINOv3 is deferred.')
    checks = {c['name']: c['exit_code'] for c in status.get('checks', [])}
    for name in ('report_RAVDESS', 'report_SpeechCommands', 'export_RAVDESS', 'export_SpeechCommands', 'wav_parity'):
        if checks.get(name) != 0:
            raise RuntimeError(f'Baseline post-processing is incomplete: {name}')
    parity = json.loads((folder / 'wav_parity.json').read_text())
    if not all(parity.get(d, {}).get('passed') is True for d in DATASETS):
        raise RuntimeError('Baseline WAV parity must pass for both datasets first.')


def validate_experiment(name):
    if not name.startswith('dinov3_') or Path(name).name != name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Use a single experiment name beginning with dinov3_, containing letters, digits, _ or -.')


def experiment_data_root(experiment):
    validate_experiment(experiment)
    return ROOT / 'datasets/tensors/experiments' / experiment


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def prepare_data(source_root, experiment, datasets, label_percent):
    """Share immutable tensors, but keep generated pseudo labels private to this run."""
    source_root = Path(source_root).resolve()
    destination = experiment_data_root(experiment)
    for dataset in datasets:
        src, dst = source_root / dataset, destination / dataset
        if src.resolve() == dst.resolve():
            raise ValueError('Source and experiment data must be different folders.')
        dst.mkdir(parents=True, exist_ok=True)
        names = ('train.pt', 'val.pt', 'test.pt', f'train_{label_percent}perc.pt')
        for name in names:
            source, target = src / name, dst / name
            if not source.is_file():
                raise FileNotFoundError(source)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeError(f'Existing link points at a different source: {target}')
            elif target.exists():
                raise RuntimeError(f'Refusing to overwrite a non-link tensor: {target}')
            else:
                # Relative links allow the complete project to move together.
                import os
                target.symlink_to(os.path.relpath(source, dst))
        for name in ('manifest.json', 'normalization.npz'):
            source, target = src / name, dst / name
            if target.is_symlink():
                raise RuntimeError(f'Experiment metadata must be private: {target}')
            if target.exists():
                if sha256(target) != sha256(source):
                    raise RuntimeError(f'Source data changed since experiment preparation: {target}')
            else:
                shutil.copy2(source, target)
        manifest = json.loads((dst / 'manifest.json').read_text())
        if manifest['smoke_subset']:
            raise ValueError('This experiment pipeline expects full data, not smoke data.')
        for name in ('pseudo_train_data.pt', 'pseudo_label_diagnostics.json'):
            if (dst / name).is_symlink():
                raise RuntimeError(f'Pseudo labels must not share a baseline target: {dst / name}')
    return destination


def snapshot_sources(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = [ROOT / 'utils.py', ROOT / 'datasets/pipeline/audio_features.py']
    for folder in ('variants', 'models', 'config_files', 'dataloader', 'trainer'):
        files.extend((ROOT / folder).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}
    atomic_json(destination / 'source_sha256.json', hashes)
    with zipfile.ZipFile(destination / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
