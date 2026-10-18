"""Check data invariants and train/inference frontend parity without PyTorch."""
import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.audio_features import read_audio, extract_features

p=argparse.ArgumentParser()
p.add_argument('--data-root', default=str(ROOT/'processed/full'))
p.add_argument('--report', default=str(ROOT/'results/03_validation/full_data_validation.json'))
args=p.parse_args()
report={}
for name in ['SpeechCommands','RAVDESS']:
    folder=Path(args.data_root)/name
    manifest=json.loads((folder/'manifest.json').read_text())
    stats=np.load(folder/'normalization.npz')
    detail={}
    for split in ['train','val','test']:
        with np.load(folder/f'{split}.npz') as d:
            x,y=d['samples'],d['labels']
            assert x.dtype==np.float32 and y.dtype==np.int64
            assert np.isfinite(x).all() and len(x)==len(y)==manifest['splits'][split]['count']
            assert x.shape[1]==40 and y.min()>=0 and y.max()<len(manifest['classes'])
            if split=='train' and not manifest['smoke_subset']:
                mean=x.mean(axis=(0,2),dtype=np.float64)
                std=x.std(axis=(0,2),dtype=np.float64)
                assert np.max(abs(mean))<1e-4 and np.max(abs(std-1))<1e-4
                detail['normalized_train_max_mean_error']=float(np.max(abs(mean)))
                detail['normalized_train_max_std_error']=float(np.max(abs(std-1)))
            source=manifest['splits'][split]['paths'][0]
            audio,_=read_audio(source)
            feats=extract_features(audio,manifest['seconds'])
            np.testing.assert_allclose(x[0],((feats[None]-stats['mean'])/stats['std'])[0],rtol=1e-5,atol=1e-5)
    for pct in [1,5,10,50,75]:
        with np.load(folder/f'train_{pct}perc.npz') as d:
            assert len(d['labels'])==manifest['label_budgets'][str(pct)]['count']
            assert set(d['labels'].tolist())==set(manifest['label_budgets'][str(pct)]['classes_present'])
    groups=[set(manifest['splits'][s]['speakers']) for s in ['train','val','test']]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    detail['checks']='finite features, dtypes, shapes, label budgets, train normalization, shared frontend, speaker separation passed'
    report[name]=detail
Path(args.report).parent.mkdir(parents=True,exist_ok=True)
Path(args.report).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
