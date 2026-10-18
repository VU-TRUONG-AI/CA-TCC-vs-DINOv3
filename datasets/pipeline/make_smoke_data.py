"""Select small verification subsets from already normalized full datasets."""
import argparse
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import shutil
from collections import Counter
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--data-root', default=str(ROOT/'processed/full'))
p.add_argument('--output-root', default=str(ROOT/'processed/smoke'))
p.add_argument('--per-class', type=int, default=20)
args = p.parse_args()
for name in ['SpeechCommands', 'RAVDESS']:
    source, target = Path(args.data_root)/name, Path(args.output_root)/name
    target.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source/'manifest.json').read_text())
    rng = np.random.default_rng(meta['seed'])
    train = None
    for split in ['train', 'val', 'test']:
        with np.load(source/f'{split}.npz') as data:
            labels = data['labels']
            indices = np.concatenate([rng.permutation(np.flatnonzero(labels == i))[:args.per_class]
                                      for i in range(len(meta['classes']))])
            subset = {k: data[k][indices] for k in ['samples', 'labels']}
        np.savez(target/f'{split}.npz', **subset)
        if split == 'train':
            train = subset
        meta['splits'][split]['count'] = len(indices)
        meta['splits'][split]['shape'] = list(subset['samples'].shape)
        meta['splits'][split]['paths'] = [meta['splits'][split]['paths'][i] for i in indices]
        meta['splits'][split]['class_counts'] = dict(Counter(meta['classes'][i] for i in subset['labels']))
    meta['label_budgets'] = {}
    for pct in [1,5,10,50,75]:
        labels = train['labels']
        n = max(1, round(len(labels) * pct / 100))
        if n >= len(meta['classes']):
            chosen = [int(rng.choice(np.flatnonzero(labels == c))) for c in range(len(meta['classes']))]
            others = np.setdiff1d(np.arange(len(labels)), chosen)
            chosen += rng.choice(others,n-len(chosen),replace=False).tolist()
        else:
            chosen = rng.choice(len(labels),n,replace=False).tolist()
        np.savez(target/f'train_{pct}perc.npz', **{k:v[chosen] for k,v in train.items()})
        meta['label_budgets'][str(pct)] = {'count':n, 'classes_present':sorted(set(labels[chosen].tolist()))}
    shutil.copyfile(source/'normalization.npz',target/'normalization.npz')
    meta['smoke_subset'] = True
    meta['normalization'] = 'full training partition statistics, reused for smoke verification'
    (target/'manifest.json').write_text(json.dumps(meta,indent=2))
    if importlib.util.find_spec('torch'):
        from package_tensors import package
        package(target)
    print(name, {s: v['count'] for s,v in meta['splits'].items()})
