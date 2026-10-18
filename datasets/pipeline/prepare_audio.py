"""Audit full corpora and build reproducible, speaker-disjoint CA-TCC tensors."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.audio_features import read_audio, extract_features

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def records_for(name, root, seed):
    if name == 'SpeechCommands':
        source = root / 'speech_commands_v0.02'
        classes = sorted(p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith('_'))
        validation = set((source / 'validation_list.txt').read_text().splitlines())
        testing = set((source / 'testing_list.txt').read_text().splitlines())
        assert not validation & testing
        records = []
        for label, cls in enumerate(classes):
            for path in sorted((source / cls).glob('*.wav')):
                rel = path.relative_to(source).as_posix()
                split = 'val' if rel in validation else 'test' if rel in testing else 'train'
                records.append({'path': path, 'label': label, 'class': cls,
                                'speaker': path.stem.split('_nohash_')[0], 'split': split})
        observed = {r['path'].relative_to(source).as_posix() for r in records}
        if (validation | testing) - observed:
            raise ValueError('Official validation/test files missing from local corpus')
        return records, classes, 1.0
    source = root / 'Audio_Speech_Actors_01-24'
    # Group assignment stratified by sex (odd=male, even=female), deterministic.
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for actors in (np.arange(1, 25, 2), np.arange(2, 25, 2)):
        actors = rng.permutation(actors).tolist()
        train += actors[:8]
        val += actors[8:10]
        test += actors[10:]
    records = []
    for path in sorted(source.glob('Actor_*/*.wav')):
        fields = [int(v) for v in path.stem.split('-')]
        if len(fields) != 7 or fields[:2] != [3, 1]:
            raise ValueError(f'Unexpected RAVDESS audio speech filename: {path}')
        actor = fields[-1]
        if path.parent.name != f'Actor_{actor:02d}' or not 1 <= fields[2] <= 8:
            raise ValueError(f'Invalid actor/emotion: {path}')
        split = 'train' if actor in train else 'val' if actor in val else 'test'
        records.append({'path': path, 'label': fields[2]-1, 'class': EMOTIONS[fields[2]-1],
                        'speaker': str(actor), 'split': split})
    if len(records) != 1440 or len({r['speaker'] for r in records}) != 24:
        raise ValueError('Expected 1440 speech files from 24 RAVDESS actors')
    return records, EMOTIONS, 6.0


def prepare(name, args):
    records, classes, seconds = records_for(name, Path(args.raw_root), args.seed)
    out = Path(args.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_root) / name
    audit_dir.mkdir(parents=True, exist_ok=True)
    groups = {s: {r['speaker'] for r in records if r['split'] == s} for s in ('train', 'val', 'test')}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if groups[a] & groups[b]:
            raise ValueError(f'Speaker leakage in {name}: {a}/{b}')
    # A smoke subset is selected AFTER splitting, and explicitly marked as such.
    selected = records
    if args.limit_per_class:
        selected = []
        rng = np.random.default_rng(args.seed)
        for split in ('train', 'val', 'test'):
            for label in range(len(classes)):
                pool = [r for r in records if r['split'] == split and r['label'] == label]
                indices = rng.permutation(len(pool))[:args.limit_per_class]
                selected.extend(pool[i] for i in indices)
    selected_paths = set() if args.audit_only else {r['path'] for r in selected}
    features = defaultdict(list)
    labels = defaultdict(list)
    paths = defaultdict(list)
    seen = {}
    duplicate_pairs = []
    audit = []
    for i, record in enumerate(records):
        path = record['path']
        audio, stats = read_audio(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            other = seen[digest]
            duplicate_pairs.append([str(other['path']), str(path)])
            if other['split'] != record['split']:
                raise ValueError(f'Exact duplicate across splits: {path}')
        else:
            seen[digest] = record
        audit.append({**record, 'path': str(path), **stats, 'sha256': digest})
        if path in selected_paths:
            split = record['split']
            features[split].append(extract_features(audio, seconds))
            labels[split].append(record['label'])
            paths[split].append(str(path))
        if (i+1) % 5000 == 0:
            print(f'{name}: audited {i+1}/{len(records)}', flush=True)
    if args.audit_only:
        with (audit_dir / 'audit.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        summary = {'dataset': name, 'raw_count': len(records),
                   'class_counts': dict(Counter(r['class'] for r in records)),
                   'sample_rates': dict(Counter(r['sample_rate'] for r in audit)),
                   'channels': dict(Counter(r['channels'] for r in audit)),
                   'duration_range': [min(r['duration'] for r in audit), max(r['duration'] for r in audit)],
                   'silent_files': sum(r['rms'] < 1e-6 for r in audit),
                   'files_with_clipping': sum(r['clipped_fraction'] > 0 for r in audit),
                   'over_window_count': sum(r['duration'] > seconds for r in audit),
                   'duplicate_pairs': duplicate_pairs,
                   'splits': {s: {'count': sum(r['split'] == s for r in records), 'speakers': sorted(groups[s])} for s in groups}}
        (audit_dir / 'audit_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        return
    import importlib.util
    torch_available = importlib.util.find_spec("torch") is not None
    train = np.stack(features['train'])
    # Training partition only, global statistics for each mel channel.
    mean = train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[None, :, None]
    std = np.maximum(train.std(axis=(0, 2), dtype=np.float64), 1e-5).astype(np.float32)[None, :, None]
    np.savez(out / 'normalization.npz', mean=mean, std=std)
    datasets = {}
    for split in ('train', 'val', 'test'):
        x = (train if split == 'train' else np.stack(features[split]))
        x = ((x - mean) / std).astype(np.float32)
        assert np.isfinite(x).all()
        data = {'samples': x, 'labels': np.asarray(labels[split], dtype=np.int64)}
        np.savez(out / f'{split}.npz', **data)
        datasets[split] = data
    rng = np.random.default_rng(args.seed)
    subset_counts = {}
    for percent in (1, 5, 10, 50, 75):
        # Exact rounded global budget; guarantee one/class if budget permits.
        y = datasets['train']['labels']
        budget = max(1, int(round(len(y) * percent / 100)))
        if budget >= len(classes):
            chosen = [int(rng.choice(np.flatnonzero(y == c))) for c in range(len(classes))]
            remaining = np.setdiff1d(np.arange(len(y)), chosen)
            chosen += rng.choice(remaining, budget-len(chosen), replace=False).tolist()
        else:
            chosen = rng.choice(len(y), budget, replace=False).tolist()
        indices = np.asarray(sorted(chosen))
        np.savez(out / f'train_{percent}perc.npz', **{k: v[indices] for k, v in datasets['train'].items()})
        subset_counts[str(percent)] = {'count': budget, 'classes_present': sorted(set(y[chosen].tolist()))}
    with (audit_dir / 'audit.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    manifest = {'dataset': name, 'seed': args.seed, 'classes': classes, 'sample_rate': 16000,
                'seconds': seconds, 'n_mels': 40, 'n_fft': 512, 'win_length': 400, 'hop_length': 160,
                'feature': 'natural log mel power, HTK mel, scipy STFT',
                'normalization': 'per mel channel, training partition only',
                'smoke_subset': bool(args.limit_per_class), 'raw_count': len(records),
                'raw_class_counts': dict(Counter(r['class'] for r in records)),
                'sample_rates': dict(Counter(r['sample_rate'] for r in audit)),
                'channels': dict(Counter(r['channels'] for r in audit)),
                'duration_range': [min(r['duration'] for r in audit), max(r['duration'] for r in audit)],
                'over_window_count': sum(r['duration'] > seconds for r in audit),
                'silent_files': sum(r['rms'] < 1e-6 for r in audit),
                'files_with_clipping': sum(r['clipped_fraction'] > 0 for r in audit),
                'exact_duplicate_pairs': duplicate_pairs,
                'label_budgets': subset_counts,
                'splits': {s: {'count': len(labels[s]), 'shape': list(datasets[s]['samples'].shape),
                               'class_counts': dict(Counter(classes[y] for y in labels[s])),
                               'speakers': sorted(groups[s]), 'paths': paths[s]} for s in groups}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if torch_available:
        from pipeline.package_tensors import package
        package(out)
    else:
        print('NumPy data ready. Run scripts/package_tensors.py after installing PyTorch.', flush=True)
    print(json.dumps({k: manifest[k] for k in ('dataset', 'raw_count', 'duration_range', 'label_budgets')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['SpeechCommands', 'RAVDESS'], choices=['SpeechCommands', 'RAVDESS'])
    parser.add_argument('--raw-root', default=str(ROOT/'raw'))
    parser.add_argument('--output-root', default=str(ROOT/'processed/full'))
    parser.add_argument('--audit-root', default=str(ROOT/'results/01_raw_audit'))
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit-per-class', type=int, default=0, help='Only for smoke tests; audit still covers full corpus')
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name, args)
