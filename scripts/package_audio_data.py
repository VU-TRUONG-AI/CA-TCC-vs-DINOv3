"""Package existing features into a writable training directory; never reprocess WAV."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch


def package(source, output):
    for dataset in ('SpeechCommands', 'RAVDESS'):
        src, dst = Path(source) / dataset, Path(output) / dataset
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('manifest.json', 'normalization.npz'):
            shutil.copy2(src / name, dst / name)
        for path in sorted(src.glob('*.npz')):
            if path.stem == 'normalization':
                continue
            target = dst / path.with_suffix('.pt').name
            if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
                continue
            with np.load(path, allow_pickle=False) as values:
                samples, labels = values['samples'], values['labels']
                if not np.isfinite(samples).all() or len(samples) != len(labels):
                    raise ValueError(f'Invalid features: {path}')
                torch.save({'samples': torch.from_numpy(samples), 'labels': torch.from_numpy(labels)}, target)
            print(f'Packaged {target}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    package(args.source, args.output)
