"""Re-run and measure each frontend stage against the saved full dataset.

Does not alter raw WAVs, splits, normalization, or saved training arrays.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.audio_features import SAMPLE_RATE, N_MELS, N_FFT, WIN_LENGTH, HOP_LENGTH, MEL_BANK

STAGES = ['raw_pcm', 'pcm_to_float', 'mono', 'resample_16k', 'pad_window', 'stft_power', 'mel_power', 'log_mel', 'normalized']


class Aggregate:
    def __init__(self):
        self.files = self.count = self.nonfinite = 0
        self.total = self.squared = 0.
        self.minimum, self.maximum = math.inf, -math.inf
        self.shapes, self.dtypes = Counter(), Counter()
    def add(self, array):
        a = np.asarray(array)
        self.files += 1
        self.shapes[str(list(a.shape))] += 1
        self.dtypes[str(a.dtype)] += 1
        self.count += a.size
        nonfinite = int(a.size - np.isfinite(a).sum())
        self.nonfinite += nonfinite
        if nonfinite:
            raise ValueError('Nonfinite values at frontend stage')
        lo, hi = float(a.min()), float(a.max())
        self.minimum = min(self.minimum,lo)
        self.maximum = max(self.maximum,hi)
        self.total += float(a.sum(dtype=np.float64))
        self.squared += float(np.square(a,dtype=np.float64).sum())
        return {'shape': list(a.shape), 'dtype':str(a.dtype), 'min':lo, 'max':hi,
                'rms':float(np.sqrt(np.square(a,dtype=np.float64).mean()))}
    def result(self):
        mean = self.total / self.count
        return {'files':self.files,'scalar_values':self.count,'nonfinite_values':self.nonfinite,
                'min':self.minimum,'max':self.maximum,'global_mean':mean,
                'global_std':math.sqrt(max(0,self.squared/self.count - mean**2)),
                'shape_counts':dict(self.shapes),'dtype_counts':dict(self.dtypes)}


def quantiles(values):
    v=np.asarray(values,dtype=float)
    return {'min':float(v.min()),'p25':float(np.quantile(v,.25)), 'median':float(np.median(v)),
            'p75':float(np.quantile(v,.75)),'max':float(v.max()),'mean':float(v.mean())}


def decode(raw):
    if np.issubdtype(raw.dtype,np.integer):
        info=np.iinfo(raw.dtype)
        if info.min == 0:
            return (raw.astype(np.float32)-(info.max+1)/2)/((info.max+1)/2)
        return raw.astype(np.float32)/max(abs(info.min),info.max)
    return raw.astype(np.float32)


def run(name, args):
    source=Path(args.data_root)/name
    out=Path(args.output)/name
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['smoke_subset']:
        raise ValueError('This report requires the full preprocessed dataset')
    norm=np.load(source/'normalization.npz')
    audit={r['path']:r for r in csv.DictReader((Path(args.audit_root)/name/'audit.csv').open())}
    # Illustration is the median-duration nonclipped training recording.
    eligible=sorted((p for p in manifest['splits']['train']['paths'] if float(audit[p]['clipped_fraction'])==0),
                    key=lambda p:(float(audit[p]['duration']),p))
    example_path=eligible[len(eligible)//2]
    aggregates=defaultdict(Aggregate)
    rates=defaultdict(Counter)
    lengths=defaultdict(list)
    padding=[]
    duration_errors=[]
    pcm_clipped=mono_clipped=post_resample_overrange=0
    stereo_count=changed_rate=hash_mismatch=0
    max_parity_error=0.
    class_counts=defaultdict(Counter)
    stages_all=Counter()
    t0=time.time()
    with (out/'per_file_stages.jsonl').open('w') as output:
        for split in ['train','val','test']:
            with np.load(source/f'{split}.npz') as saved:
                x,y=saved['samples'],saved['labels']
                assert len(x)==len(manifest['splits'][split]['paths'])
                for i,path in enumerate(manifest['splits'][split]['paths']):
                    sr,raw=wavfile.read(path)
                    if raw.size==0:
                        raise ValueError(f'Empty WAV: {path}')
                    hash_ok=hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit[path]['sha256']
                    hash_mismatch+=not hash_ok
                    fp32=decode(raw)
                    channels=1 if fp32.ndim==1 else fp32.shape[1]
                    stereo_count+=channels>1
                    mono=fp32 if channels==1 else fp32.mean(axis=1)
                    if sr!=SAMPLE_RATE:
                        d=math.gcd(int(sr),SAMPLE_RATE)
                        resampled=resample_poly(mono,SAMPLE_RATE//d,int(sr)//d).astype(np.float32)
                        changed_rate+=1
                    else:
                        resampled=mono
                    duration_errors.append(abs(len(resampled)/SAMPLE_RATE-len(mono)/sr))
                    target=round(manifest['seconds']*SAMPLE_RATE)
                    pad=max(0,target-len(resampled));crop=max(0,len(resampled)-target)
                    if crop:
                        start=crop//2;window=resampled[start:start+target]
                    else:
                        window=np.pad(resampled,(0,pad))
                    _,_,z=stft(window,fs=SAMPLE_RATE,window='hann',nperseg=WIN_LENGTH,
                               noverlap=WIN_LENGTH-HOP_LENGTH,nfft=N_FFT,boundary='zeros',padded=True)
                    power=(np.abs(z)**2).astype(np.float32)
                    mel=MEL_BANK@power
                    logmel=np.log(np.maximum(mel,1e-10)).astype(np.float32)
                    normalized=((logmel[None]-norm['mean'])/norm['std'])[0].astype(np.float32)
                    error=float(np.max(np.abs(normalized-x[i])))
                    max_parity_error=max(max_parity_error,error)
                    np.testing.assert_allclose(normalized,x[i],rtol=1e-5,atol=1e-5)
                    assert int(y[i])==int(audit[path]['label'])
                    arrays=dict(zip(STAGES,[raw,fp32,mono,resampled,window,power,mel,logmel,normalized]))
                    row={'path':path,'split':split,'label':int(y[i]),'source_sha256_matches':hash_ok,
                         'source_sample_rate':int(sr),'source_channels':channels,
                         'padding_samples':pad,'cropped_samples':crop,'parity_max_abs_error':error,'stages':{}}
                    for stage,a in arrays.items():
                        row['stages'][stage]=aggregates[(split,stage)].add(a)
                        stages_all[stage]+=1
                    padding.append(pad/target)
                    pcm_clipped+=bool(np.any(abs(fp32)>=.999))
                    mono_clipped+=bool(np.any(abs(mono)>=.999))
                    post_resample_overrange+=bool(np.any(abs(resampled)>1.0))
                    rates['source'][str(sr)]+=1
                    rates['resampled']['16000']+=1
                    lengths['raw_duration'].append(len(mono)/sr)
                    lengths['resampled_samples'].append(len(resampled))
                    lengths['padding_samples'].append(pad)
                    lengths['cropped_samples'].append(crop)
                    class_counts[split][manifest['classes'][int(y[i])]]+=1
                    output.write(json.dumps(row,separators=(',',':'))+'\n')
                    if path==example_path:
                        np.savez(out/'illustration_arrays.npz',raw_pcm=raw,mono=mono,resampled=resampled,
                                 padded=window,power=power,mel=mel,logmel=logmel,normalized=normalized,
                                 sample_rate=np.asarray(sr))
                    if (i+1)%5000==0:
                        print(f'{name}/{split}: checked {i+1}/{len(x)}',flush=True)
                # Check each mel channel on train, and separately on held-out splits.
                means=x.mean(axis=(0,2),dtype=np.float64)
                stds=x.std(axis=(0,2),dtype=np.float64)
                np.savez(out/f'{split}_normalized_channel_stats.npz',mean=means,std=stds)
                if split=='train':
                    assert max(abs(means))<1e-4 and max(abs(stds-1))<1e-4
    assert hash_mismatch==0
    assert len(set(stages_all.values()))==1
    groups={s:set(manifest['splits'][s]['speakers']) for s in ['train','val','test']}
    overlaps={f'{a}_{b}':sorted(groups[a]&groups[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlaps.values())
    budget_report={}
    train_labels=np.load(source/'train.npz')['labels']
    for pct in [1,5,10,50,75]:
        with np.load(source/f'train_{pct}perc.npz') as saved:
            counts=dict(Counter(manifest['classes'][int(v)] for v in saved['labels']))
            assert len(saved['labels'])==manifest['label_budgets'][str(pct)]['count']
            budget_report[str(pct)]={'count':len(saved['labels']),'actual_percent':100*len(saved['labels'])/len(train_labels),
                                    'classes_present':len(counts),'class_counts':counts}
    result={'dataset':name,'report_method':'Full re-run from original WAV; each normalized output compared to saved NPZ. No training performed.',
            'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
            'elapsed_seconds':time.time()-t0,'raw_files':len(audit),'stages_file_counts':dict(stages_all),
            'source_hash_mismatches':hash_mismatch,'speaker_overlaps':overlaps,
            'stereo_to_mono_files':stereo_count,'resampled_files':changed_rate,
            'pcm_float_files_with_abs_ge_0_999':pcm_clipped,'mono_files_with_abs_ge_0_999':mono_clipped,
            'resampled_files_with_abs_gt_1':post_resample_overrange,
            'sample_rate_counts':{s:dict(v) for s,v in rates.items()},
            'raw_duration_seconds':quantiles(lengths['raw_duration']),
            'resampled_sample_counts':quantiles(lengths['resampled_samples']),
            'max_duration_error_seconds':max(duration_errors),'padding_sample_counts':quantiles(lengths['padding_samples']),
            'padding_fraction':quantiles(padding),'cropped_files':sum(v>0 for v in lengths['cropped_samples']),
            'max_normalized_parity_abs_error':max_parity_error,
            'split_class_counts':{s:dict(v) for s,v in class_counts.items()},'label_budgets':budget_report,
            'illustration':{'path':example_path,'selection':'median duration among nonclipped train WAVs',
                            'class':audit[example_path]['class'],'split':'train'},
            'stage_statistics':{s:{stage:aggregates[(s,stage)].result() for stage in STAGES} for s in groups}}
    (out/'stage_summary.json').write_text(json.dumps(result,indent=2))
    print(f'{name}: PASS; {len(audit)} WAV, parity max abs error={max_parity_error}',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default=str(ROOT/'processed/full'))
    p.add_argument('--audit-root', default=str(ROOT/'results/01_raw_audit'))
    p.add_argument('--output',default=str(ROOT/'results/02_stage_checks'))
    p.add_argument('--datasets',nargs='+',default=['SpeechCommands','RAVDESS'])
    args=p.parse_args()
    # RAVDESS has ancillary WAV chunks ignored by SciPy; do not hide other warnings.
    warnings.filterwarnings('ignore',message='Chunk \\(non-data\\) not understood, skipping it.',category=wavfile.WavFileWarning)
    results={name:run(name,args) for name in args.datasets}
    (Path(args.output)/'summary.json').write_text(json.dumps(results,indent=2))
