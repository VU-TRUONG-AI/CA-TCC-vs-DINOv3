"""Build a labeled WAV ZIP strictly from the saved test split for web verification."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def build(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    with np.load(source / 'test.npz', allow_pickle=False) as data:
        labels = data['labels']
    paths = manifest['splits']['test']['paths']
    if len(labels) != len(paths):
        raise ValueError('Test paths and labels do not match.')
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['file', 'label'])
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for i, (path, label) in enumerate(zip(paths, labels)):
            name = f'test/{i:05d}_{Path(path).name}'
            archive.write(path, name)
            writer.writerow([name, manifest['classes'][int(label)]])
        archive.writestr('labels.csv', buffer.getvalue())
    print(f'{output}: {len(paths)} WAV from test split; smoke={manifest["smoke_subset"]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    build(args.source, args.output)
