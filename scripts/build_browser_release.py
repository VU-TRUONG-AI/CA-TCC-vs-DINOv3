"""Assemble a complete static web release from verified model assets."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deployment_assets import ROOT, DATASETS, verify, sha


def build(source, output, revision):
    assets = ROOT / 'build'
    manifest = verify(assets)
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Use separate source and output directories')
    if output.exists():
        raise ValueError('Output already exists; choose a new empty release directory')
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('weights-*.bin', '*.onnx'))
    for dataset in DATASETS:
        destination = output / 'models' / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for item in (assets / 'browser' / dataset).iterdir():
            shutil.copy2(item, destination / item.name)
    shutil.copy2(assets / 'DINOv3_LICENSE.md', output / 'models/DINOv3_LICENSE.md')
    (output / '.nojekyll').touch()
    report = {'code_revision': revision, 'model_manifest_sha256': sha(assets / 'deployment-manifest.json'),
              'engine': 'onnxruntime-web-wasm', 'requires_local_backend': False, 'models': {}}
    for dataset in DATASETS:
        meta = json.loads((output / 'models' / dataset / 'manifest.json').read_text())
        for chunk in meta['chunks']:
            file = output / 'models' / dataset / chunk['file']
            if sha(file) != chunk['sha256']:
                raise ValueError(f'Corrupt staged model: {dataset}')
        report['models'][dataset] = meta['model_sha256']
    for name in ('index.html', 'app.js', 'style.css', 'runtime/ort.min.js'):
        if not (output / name).is_file():
            # The repository uses ort.wasm.min.js instead in some versions.
            if name == 'runtime/ort.min.js' and any((output / 'runtime').glob('ort*.js')):
                continue
            raise ValueError(f'Missing static asset: {name}')
    (output / 'deployment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'directory': str(output), **report}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT / 'sites/ca-tcc-audio-studio/dist'))
    parser.add_argument('--output', default=str(ROOT / 'build/web-dist'))
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    build(args.source, args.output, args.revision)
