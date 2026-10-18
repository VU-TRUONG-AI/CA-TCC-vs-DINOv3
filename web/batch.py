"""Bounded ZIP ingestion, cached inference and reproducible batch reports."""
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import threading
import time
import uuid
import zipfile

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'SpeechCommands': 1.0, 'RAVDESS': 6.0}
MAX_UPLOAD = 512 * 1024**2
MAX_EXPANDED = 2 * 1024**3
MAX_FILE = 32 * 1024**2
MAX_FILES = 50000


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_labels(content, classes):
    rows = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    if not rows.fieldnames or not {'file', 'label'} <= set(rows.fieldnames):
        raise ValueError('CSV cần hai cột file,label.')
    labels = {}
    for row in rows:
        name, label = row['file'], row['label']
        if not name or not label or label not in classes:
            raise ValueError(f'Nhãn không hợp lệ tại file {name!r}: {label!r}.')
        if name in labels:
            raise ValueError(f'CSV có file trùng: {name}.')
        labels[name] = label
    return labels


def inspect_archive(archive, folder, seconds):
    """Reject unsafe archives before extraction; record bad audio individually."""
    records, seen, total = [], set(), 0
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > MAX_FILES:
            raise ValueError(f'ZIP vượt quá {MAX_FILES} mục.')
        for entry in entries:
            name = entry.filename
            parts = PurePosixPath(name)
            mode = entry.external_attr >> 16
            if (not name or '\\' in name or parts.is_absolute() or '..' in parts.parts
                    or ':' in name or stat.S_ISLNK(mode) or name in seen or entry.flag_bits & 1):
                raise ValueError(f'Đường dẫn ZIP không an toàn hoặc bị trùng: {name!r}.')
            seen.add(name)
            total += entry.file_size
            if entry.file_size > MAX_FILE or total > MAX_EXPANDED:
                raise ValueError('ZIP vượt giới hạn: 32 MiB/file hoặc 2 GiB sau giải nén.')
        for entry in entries:
            if entry.is_dir():
                continue
            name = entry.filename
            if not name.lower().endswith('.wav') and name != 'labels.csv':
                continue
            destination = folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as source, destination.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            if name == 'labels.csv':
                continue
            record = {'file': name, 'error': None}
            try:
                sr, audio = wavfile.read(destination)
                if sr <= 0 or audio.ndim not in (1, 2) or not len(audio) or not np.isfinite(audio).all():
                    raise ValueError('WAV rỗng hoặc chứa dữ liệu không hợp lệ.')
                duration = len(audio) / sr
                record.update(duration=round(duration, 4), sample_rate=int(sr),
                              channels=1 if audio.ndim == 1 else audio.shape[1])
                if duration > seconds + 1 / sr:
                    raise ValueError(f'Âm thanh {duration:.3f}s vượt cửa sổ {seconds:g}s; hãy chia đoạn trước khi tải.')
            except Exception as error:
                record['error'] = str(error)
            records.append(record)
    if not records:
        raise ValueError('ZIP không có file WAV.')
    return records


class BundlePredictor:
    def __init__(self, folder, dataset):
        import torch
        from scripts.audio_features import SAMPLE_RATE, N_MELS, N_FFT, WIN_LENGTH, HOP_LENGTH
        self.torch = torch
        torch.set_num_threads(4)
        self.metadata = json.loads((folder / 'metadata.json').read_text())
        expected = dict(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
                        win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
        if self.metadata['dataset'] != dataset or self.metadata['seconds'] != DATASETS[dataset]:
            raise ValueError('Bundle không khớp tác vụ/thời lượng.')
        if any(self.metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Frontend trong bundle không khớp phiên bản hiện tại.')
        self.classes = self.metadata['classes']
        self.model = torch.jit.load(str(folder / 'classifier.ts'), map_location='cpu').eval()
        with np.load(folder / 'normalization.npz', allow_pickle=False) as stats:
            self.mean, self.std = stats['mean'], stats['std']
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Thống kê chuẩn hóa không hợp lệ.')
        self.version = hashlib.sha256((folder / 'classifier.ts').read_bytes()).hexdigest()[:16]

    def predict(self, paths):
        from scripts.audio_features import read_audio, extract_features
        arrays = []
        for path in paths:
            audio, _ = read_audio(path)
            features = extract_features(audio, self.metadata['seconds'])
            arrays.append(features)
        with self.torch.inference_mode():
            normalized = ((np.stack(arrays) - self.mean) / self.std).astype(np.float32)
            scores = self.model(self.torch.from_numpy(normalized))[0].softmax(dim=1).numpy()
        if scores.shape != (len(paths), len(self.classes)) or not np.isfinite(scores).all():
            raise ValueError('Đầu ra mô hình không hợp lệ.')
        return scores


class JobStore:
    def __init__(self, root, bundles, predictor_factory=BundlePredictor):
        self.root, self.bundles = Path(root), Path(bundles)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.worker_lock = threading.Lock()
        self.predictors = {}
        self.factory = predictor_factory
        # A server restart cannot silently leave a job appearing active forever.
        for path in self.root.glob('*/job.json'):
            value = json.loads(path.read_text())
            if value['status'] in ('queued', 'running'):
                value.update(status='failed', error='Server đã khởi động lại; hãy tạo lượt chạy mới.')
                write_json(path, value)

    def classes(self, dataset):
        metadata = self.bundles / dataset / 'metadata.json'
        if metadata.is_file():
            classes = json.loads(metadata.read_text()).get('classes')
            if isinstance(classes, list) and classes:
                return classes
        return json.loads((ROOT / 'datasets/processed/full' / dataset / 'manifest.json').read_text())['classes']

    def models(self):
        output = []
        for dataset, seconds in DATASETS.items():
            folder = self.bundles / dataset
            item = dict(dataset=dataset, seconds=seconds, classes=self.classes(dataset), ready=False)
            try:
                meta = json.loads((folder / 'metadata.json').read_text())
                if (folder / 'classifier.ts').is_file() and (folder / 'normalization.npz').is_file():
                    item.update(ready=True, purpose=meta['purpose'], backbone=meta.get('backbone', 'cnn'))
            except (OSError, ValueError, KeyError):
                pass
            output.append(item)
        return output

    def directory(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('Mã tác vụ không hợp lệ.')
        return self.root / job_id

    def predict_one(self, path, dataset):
        if dataset not in DATASETS:
            raise ValueError('Tác vụ không hợp lệ.')
        from scripts.audio_features import read_audio
        _, audio_info = read_audio(path)
        if audio_info['duration'] > DATASETS[dataset] + 1/audio_info['sample_rate']:
            raise ValueError(f'Âm thanh vượt cửa sổ {DATASETS[dataset]:g}s; hãy chia đoạn trước khi tải.')
        with self.worker_lock:
            if dataset not in self.predictors:
                self.predictors[dataset] = self.factory(self.bundles/dataset, dataset)
            predictor = self.predictors[dataset]
            scores = predictor.predict([path])[0]
            indices = np.argsort(scores)[-min(5, len(scores)):][::-1]
            return dict(dataset=dataset, backbone=predictor.metadata.get('backbone', 'cnn'),
                        version=predictor.version, duration=audio_info['duration'],
                        top_k=[dict(label=predictor.classes[i], probability=float(scores[i])) for i in indices])

    def get(self, job_id):
        with self.lock:
            return json.loads((self.directory(job_id) / 'job.json').read_text())

    def save(self, job):
        with self.lock:
            write_json(self.directory(job['id']) / 'job.json', job)

    def create(self, archive, dataset):
        if dataset not in DATASETS:
            raise ValueError('Tác vụ không hợp lệ.')
        job_id = uuid.uuid4().hex
        folder = self.directory(job_id)
        folder.mkdir()
        try:
            records = inspect_archive(archive, folder / 'audio', DATASETS[dataset])
            labels_path = folder / 'audio' / 'labels.csv'
            labels = read_labels(labels_path.read_bytes(), self.classes(dataset)) if labels_path.exists() else {}
            job = dict(id=job_id, dataset=dataset, status='inspected', created=time.time(),
                       total=len(records), processed=0, succeeded=0, records=records, labels=labels,
                       error=None, metrics=None)
            self.apply_labels(job, labels)
            self.save(job)
            return job
        except Exception:
            shutil.rmtree(folder)
            raise

    def apply_labels(self, job, labels):
        files = {r['file'] for r in job['records']}
        extra = set(labels) - files
        if extra:
            raise ValueError(f'CSV có {len(extra)} đường dẫn không có trong ZIP: {sorted(extra)[:3]}')
        job['labels'] = labels
        job['valid'] = sum(not r['error'] for r in job['records'])
        job['invalid'] = job['total'] - job['valid']
        job['labeled'] = len(labels)
        for row in job['records']:
            row['true_label'] = labels.get(row['file'])

    def set_labels(self, job_id, content):
        with self.lock:
            job = self.get(job_id)
            if job['status'] != 'inspected':
                raise ValueError('Chỉ cập nhật nhãn trước khi bắt đầu chạy.')
            self.apply_labels(job, read_labels(content, self.classes(job['dataset'])))
            self.save(job)
            return job

    def start(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] != 'inspected' or not job['valid']:
                raise ValueError('Tác vụ không có file hợp lệ hoặc đã được chạy.')
            model = next(m for m in self.models() if m['dataset'] == job['dataset'])
            if not model['ready']:
                raise ValueError('Chưa có bundle mô hình cho tác vụ này. Có thể kiểm tra dữ liệu trước.')
            job.update(status='queued')
            self.save(job)
            threading.Thread(target=self.run, args=(job_id,), daemon=True).start()
            return job

    def run(self, job_id):
        with self.worker_lock:
            job = self.get(job_id)
            try:
                job.update(status='running', started=time.time())
                self.save(job)
                dataset = job['dataset']
                if dataset not in self.predictors:
                    self.predictors[dataset] = self.factory(self.bundles / dataset, dataset)
                predictor = self.predictors[dataset]
                job['model'] = dict(version=predictor.version, purpose=predictor.metadata['purpose'])
                valid = [r for r in job['records'] if not r['error']]
                job['processed'] = job['invalid']
                for offset in range(0, len(valid), 32):
                    batch = valid[offset:offset + 32]
                    paths = [self.directory(job_id) / 'audio' / r['file'] for r in batch]
                    try:
                        scores = predictor.predict(paths)
                        for row, probabilities in zip(batch, scores):
                            indices = np.argsort(probabilities)[::-1][:5]
                            row.update(predicted_label=predictor.classes[indices[0]], score=float(probabilities[indices[0]]),
                                       top_k=[dict(label=predictor.classes[i], score=float(probabilities[i])) for i in indices])
                            row['correct'] = (row['predicted_label'] == row['true_label']) if row['true_label'] else None
                            job['succeeded'] += 1
                    except Exception as error:
                        for row in batch:
                            row['error'] = f'Lỗi dự đoán batch: {error}'
                    job['processed'] += len(batch)
                    self.save(job)
                if not job['succeeded']:
                    raise ValueError('Không có file nào được dự đoán thành công; xem danh sách lỗi.')
                job['metrics'] = calculate_metrics(job['records'], predictor.classes)
                job.update(status='completed', finished=time.time())
                self.reports(job)
                self.save(job)
            except Exception as error:
                job.update(status='failed', error=str(error), finished=time.time())
                self.save(job)

    def reports(self, job):
        folder = self.directory(job['id'])
        for filename, fields, rows in (
            ('predictions.csv', ['file', 'true_label', 'predicted_label', 'score', 'correct'],
             [r for r in job['records'] if 'predicted_label' in r]),
            ('errors.csv', ['file', 'error'], [r for r in job['records'] if r['error']])):
            with (folder / filename).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for row in rows:
                    # Prevent spreadsheet formula execution from uploaded filenames.
                    writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
                                     for k, v in row.items()})
        if job['metrics']:
            write_json(folder / 'metrics.json', job['metrics'])
        rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r.get(k, "")))}</td>' for k in
                       ('file', 'true_label', 'predicted_label', 'score', 'error')) + '</tr>' for r in job['records'])
        body = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Báo cáo CA-TCC</title>
        <style>body{{font:15px system-ui;margin:40px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:8px}}pre{{white-space:pre-wrap}}</style>
        <h1>Báo cáo phân tích bộ âm thanh</h1><p>{html.escape(job['dataset'])} · {job['succeeded']}/{job['total']} file thành công</p>
        <p>Phiên bản mô hình: {html.escape(str(job['model']))}</p><p>Điểm softmax chưa được hiệu chỉnh xác suất. Mô hình smoke chỉ để kiểm tra chương trình.</p>
        <h2>Đánh giá</h2><pre>{html.escape(json.dumps(job['metrics'], ensure_ascii=False, indent=2))}</pre>
        <table><tr><th>File</th><th>Nhãn thật</th><th>Dự đoán</th><th>Điểm</th><th>Lỗi</th></tr>{rows}</table></html>'''
        (folder / 'report.html').write_text(body, encoding='utf-8')


def calculate_metrics(records, classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    labeled = [r for r in records if not r['error'] and r.get('predicted_label') and r.get('true_label')]
    if not labeled:
        return None
    true, pred = [r['true_label'] for r in labeled], [r['predicted_label'] for r in labeled]
    return dict(evaluated=len(labeled), excluded=len(records) - len(labeled), classes=classes,
                accuracy=float(accuracy_score(true, pred)),
                macro_f1=float(f1_score(true, pred, labels=classes, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(true, pred)),
                macro_f1_definition='Mean over all model classes; missing classes contribute zero.',
                confusion_matrix=confusion_matrix(true, pred, labels=classes).tolist())
