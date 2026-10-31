"""Prepare, hash, download and verify versioned deployment assets; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('RAVDESS', 'SpeechCommands')

def safe_path(directory, name):
    if not isinstance(name, str) or '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe asset path')
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Unsafe asset path')
    return path

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'deployment-manifest.json').read_text())
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported deployment manifest')
    for name, expected in manifest['files'].items():
        path = safe_path(directory, name)
        if sha(path) != expected:
            raise ValueError(f'Asset hash mismatch: {name}')
    for dataset, classes in [('RAVDESS', 8), ('SpeechCommands', 35)]:
        meta = json.loads((directory / 'models' / dataset / 'metadata.json').read_text())
        if meta['dataset'] != dataset or len(meta['classes']) != classes:
            raise ValueError(f'Invalid classes: {dataset}')
        if meta['backbone'] != 'dinov3_vits16':
            raise ValueError(f'Unexpected backbone: {dataset}')
        model = directory / 'models' / dataset / 'classifier.ts'
        if sha(model) != meta['classifier_sha256']:
            raise ValueError(f'Model identity mismatch: {dataset}')
        browser = json.loads((directory / 'browser' / dataset / 'manifest.json').read_text())
        for chunk in browser['chunks']:
            file = safe_path(directory / 'browser' / dataset, chunk['file'])
            if file.stat().st_size != chunk['bytes'] or sha(file) != chunk['sha256']:
                raise ValueError(f'Browser model mismatch: {file.name}')
        for name in ('classifier.ts', 'normalization.npz', 'metadata.json'):
            if f'models/{dataset}/{name}' not in manifest['files']:
                raise ValueError(f'Untracked required model asset: {dataset}/{name}')
        for name in ['manifest.json', *[c['file'] for c in browser['chunks']]]:
            if f'browser/{dataset}/{name}' not in manifest['files']:
                raise ValueError(f'Untracked browser asset: {dataset}/{name}')
    for name in ('fixtures/reference.json', 'fixtures/RAVDESS.wav', 'fixtures/SpeechCommands.wav', 'fixtures/NOTICE.md', 'DINOv3_LICENSE.md'):
        if name not in manifest['files']:
            raise ValueError(f'Missing required release asset: {name}')
    return manifest

def prepare_fixtures(target):
    # Two unmodified validation WAVs, used only for inference parity, never accuracy.
    import sys
    sys.path.insert(0, str(ROOT))
    from web.batch import BundlePredictor
    folder = target / 'fixtures'
    folder.mkdir(exist_ok=True)
    references = {}
    for dataset in DATASETS:
        source = json.loads((ROOT / 'datasets/tensors/full' / dataset / 'manifest.json').read_text())
        wav = Path(source['splits']['val']['paths'][0])
        destination = folder / f'{dataset}.wav'
        shutil.copy2(wav, destination)
        predictor = BundlePredictor(target / 'models' / dataset, dataset)
        scores = predictor.predict([destination])[0]
        references[dataset] = {'file': destination.name, 'source_filename': wav.name,
                               'partition': 'validation', 'sha256': sha(destination),
                               'model_version': predictor.version, 'classes': predictor.classes,
                               'probabilities': [float(x) for x in scores]}
    (folder / 'reference.json').write_text(json.dumps(references, indent=2))
    shutil.copy2(ROOT / 'deployment/FIXTURE_NOTICE.md', folder / 'NOTICE.md')

def prepare():
    target = ROOT / 'build'
    target.mkdir(exist_ok=True)
    for dataset in DATASETS:
        source = ROOT / 'artifacts/dinov3_app_bundles' / dataset
        destination = target / 'models' / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('classifier.ts', 'metadata.json', 'normalization.npz'):
            shutil.copy2(source / name, destination / name)
        metadata_path = destination / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('checkpoint'):
            metadata['checkpoint'] = Path(metadata['checkpoint']).name
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        browser = ROOT / 'artifacts/dinov3_browser_models' / dataset
        destination = target / 'browser' / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for file in [browser / 'manifest.json', *browser.glob('weights-*.bin')]:
            shutil.copy2(file, destination / file.name)
    shutil.copy2(ROOT / 'deployment/DINOv3_LICENSE.md', target / 'DINOv3_LICENSE.md')
    prepare_fixtures(target)
    paths = []
    for dataset in DATASETS:
        paths.extend(target / 'models' / dataset / name for name in ('classifier.ts', 'metadata.json', 'normalization.npz'))
        browser = target / 'browser' / dataset
        browser_meta = json.loads((browser / 'manifest.json').read_text())
        paths.extend(safe_path(browser, name) for name in ['manifest.json', *[c['file'] for c in browser_meta['chunks']]])
    paths.extend(target / 'fixtures' / name for name in ('RAVDESS.wav', 'SpeechCommands.wav', 'reference.json', 'NOTICE.md'))
    files = {str(p.relative_to(target)): sha(p) for p in sorted(paths)}
    files['DINOv3_LICENSE.md'] = sha(target / 'DINOv3_LICENSE.md')
    (target / 'deployment-manifest.json').write_text(json.dumps({'schema': 1, 'files': files}, indent=2))
    verify(target)
    output = ROOT / 'output/deployment'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'deployment-assets.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'deployment-manifest.json']:
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, (target / name).read_bytes())
    checksum = sha(archive)
    (output / 'deployment-assets.zip.sha256').write_text(checksum + '  deployment-assets.zip\n')
    print(json.dumps({'archive': str(archive), 'sha256': checksum, 'files': len(files)}, indent=2))

def fetch(url, checksum):
    if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Set MODEL_ASSETS_SHA256 to the trusted 64-character SHA256')
    if not url.startswith('https://'):
        raise ValueError('MODEL_ASSETS_URL must be an HTTPS release-asset URL')
    target = ROOT / 'build'
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'assets.zip'
        urllib.request.urlretrieve(url, archive)
        if sha(archive) != checksum:
            raise ValueError('Downloaded archive checksum mismatch')
        stage = Path(tmp) / 'verified'
        stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            seen, total = set(), 0
            for item in z.infolist():
                safe_path(stage, item.filename)
                total += item.file_size
                if item.filename in seen or stat.S_ISLNK(item.external_attr >> 16) or total > 1024**3:
                    raise ValueError('Unsafe asset archive')
                seen.add(item.filename)
            z.extractall(stage)
        manifest = verify(stage)
        if seen != {*manifest['files'], 'deployment-manifest.json'}:
            raise ValueError('Archive contains undeclared files')
        target.mkdir(exist_ok=True)
        for name in ('models', 'browser', 'fixtures'):
            destination = target / name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(stage / name, destination)
        for name in ('deployment-manifest.json', 'DINOv3_LICENSE.md'):
            shutil.copy2(stage / name, target / name)
    print('Verified release assets')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fetch', 'verify'])
    parser.add_argument('--url')
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'fetch':
        fetch(args.url or '', args.sha256 or '')
    else:
        verify(ROOT / 'build')
        print('Verified local release assets')
