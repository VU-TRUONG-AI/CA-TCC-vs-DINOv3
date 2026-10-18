"""Prepare raw-only Kaggle upload packages, relative-path catalogs, and checksums."""
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import shutil
import wave
import zipfile
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/kaggle_raw_upload'
RAW = ROOT / 'datasets/raw'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def make(dataset, dirname, slug, archive_name, expected_words):
    src, out = RAW / dirname, OUTPUT / slug
    out.mkdir(parents=True, exist_ok=True)
    archive = out / archive_name
    if archive.exists():
        raise RuntimeError(f'Package exists; inspect before rebuilding: {archive}')
    # Audit hashes provide an independent check against the previously verified raw files.
    audit = {}
    with (ROOT / 'datasets/results/01_raw_audit' / dataset / 'audit.csv').open() as f:
        for row in csv.DictReader(f):
            audit[str(Path(row['path']).relative_to(src))] = row['sha256']
    manifest = json.loads((ROOT / 'datasets/processed/full' / dataset / 'manifest.json').read_text())
    classes = manifest['classes']
    official_val = set((src / 'validation_list.txt').read_text().splitlines()) if dataset == 'SpeechCommands' else set()
    official_test = set((src / 'testing_list.txt').read_text().splitlines()) if dataset == 'SpeechCommands' else set()
    rows, hashes = [], {}
    files = sorted(p for p in src.rglob('*') if p.is_file() and not p.name.startswith('.') and '__MACOSX' not in p.parts)
    temp = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for index, p in enumerate(files):
            relative = str(p.relative_to(src))
            data = p.read_bytes()
            sha = digest(data)
            hashes[relative] = sha
            if relative in audit and audit[relative] != sha:
                raise ValueError(f'Raw file differs from audited version: {relative}')
            z.writestr(dirname + '/' + relative, data)
            if p.suffix.lower() == '.wav':
                with wave.open(io.BytesIO(data)) as wav:
                    rate, channels, frames, width = wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth()
                if dataset == 'SpeechCommands':
                    label = p.parent.name
                    is_background = label == '_background_noise_'
                    label_id = '' if is_background else classes.index(label)
                    speaker = '' if is_background else p.stem.split('_nohash_')[0]
                    split = 'not_applicable' if is_background else 'validation' if relative in official_val else 'test' if relative in official_test else 'train'
                    extra = {'emotion_code':'','intensity_code':'','statement_code':'','repetition_code':''}
                else:
                    codes = p.stem.split('-')
                    label_id = int(codes[2]) - 1
                    label = classes[label_id]
                    speaker = codes[6]
                    split = 'not_provided'
                    is_background = False
                    extra = dict(zip(['emotion_code','intensity_code','statement_code','repetition_code'],codes[2:6]))
                rows.append({'relative_path':dirname + '/' + relative, 'label':label, 'label_id':label_id,
                             'speaker_id':speaker, 'sample_rate_hz':rate,'channels':channels,'sample_width_bytes':width,
                             'frame_count':frames, 'duration_seconds':frames/rate,'bytes':len(data),'sha256':sha,
                             'official_split':split,'is_background_noise':is_background, **extra})
            if index % 10000 == 0:
                print(f'{dataset}: {index}/{len(files)} files packed', flush=True)
    temp.replace(archive)
    assert len([r for r in rows if not r['is_background_noise']]) == expected_words
    assert set(audit) <= set(hashes)
    # Read every archive member again to verify round-trip bytes, not only filenames.
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            relative = entry.filename[len(dirname)+1:]
            if digest(z.read(entry)) != hashes[relative]:
                raise ValueError(f'Archive round-trip mismatch: {entry.filename}')
    with (out / 'files.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out / 'class_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['label','count']);w.writerows(sorted(Counter(r['label'] for r in rows).items()))
    with (out / 'FILE_SHA256SUMS.txt').open('w') as f:
        for name, sha in hashes.items():
            f.write(sha+'  '+dirname+'/'+name+'\n')
    archive_hash = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):archive_hash.update(block)
    summary={'dataset':dataset,'raw_audio_preserved':True,'wav_count':len(rows),'labeled_wav_count':expected_words,
             'background_wav_count':sum(r['is_background_noise'] for r in rows),'classes':classes,
             'channels':dict(Counter(r['channels'] for r in rows)),
             'sample_rates':dict(Counter(r['sample_rate_hz'] for r in rows)),
             'duration_min_seconds':min(r['duration_seconds'] for r in rows),'duration_max_seconds':max(r['duration_seconds'] for r in rows),
             'official_splits':dict(Counter(r['official_split'] for r in rows)),
             'archive_file_count':len(hashes),'archive_bytes':archive.stat().st_size,'archive_sha256':archive_hash.hexdigest(),
             'source_audit_hashes_matched':len(audit),'archive_round_trip_all_files_verified':True,
             'excluded':['.DS_Store and hidden operating-system files'],
             'no_resampling_no_mono_conversion_no_padding_no_normalization':True,
             'prepared_local':datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()}
    dump(out/'inventory_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for args in [('RAVDESS','Audio_Speech_Actors_01-24','ravdess-speech-raw','ravdess_speech_raw.zip',1440),
                 ('SpeechCommands','speech_commands_v0.02','speech-commands-v002-raw','speech_commands_v002_raw.zip',105829)]:
        summaries.append(make(*args))
    dump(OUTPUT/'preparation_status.json',{'state':'prepared_not_uploaded','datasets':summaries})
