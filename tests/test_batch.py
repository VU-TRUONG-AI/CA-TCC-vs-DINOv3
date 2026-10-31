"""Integration checks use an explicit test double, never fake research metrics."""
import io
import json
from pathlib import Path
from unittest.mock import patch
import tempfile
import time
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile
from web.batch import JobStore, inspect_archive, read_labels, calculate_metrics


def wav(seconds=.2):
    out = io.BytesIO()
    wavfile.write(out, 16000, np.zeros(round(16000 * seconds), dtype=np.int16))
    return out.getvalue()


def archive(path, entries):
    with zipfile.ZipFile(path, 'w') as z:
        for name, content in entries.items():
            z.writestr(name, content)


class FakePredictor:
    def __init__(self, folder, dataset):
        self.classes = ['yes', 'no']
        self.version = 'test-double'
        self.metadata = {'purpose': 'test_double_only'}
    def predict(self, paths):
        return np.tile([.8, .2], (len(paths), 1))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Tests must not read training data from the developer's checkout.
        self.source_patch = patch('web.batch.ROOT', self.root / 'empty-source')
        self.source_patch.start()
    def tearDown(self):
        self.source_patch.stop()
        self.tmp.cleanup()
    def test_single_wav_prediction_and_duration_guard(self):
        store = JobStore(self.root/'jobs', self.root/'bundles', predictor_factory=FakePredictor)
        sample = self.root/'sample.wav'
        sample.write_bytes(wav())
        result = store.predict_one(sample, 'SpeechCommands')
        self.assertEqual(result['dataset'], 'SpeechCommands')
        self.assertEqual(result['top_k'][0]['label'], 'yes')
        self.assertAlmostEqual(result['top_k'][0]['probability'], .8)
        with self.assertRaises(ValueError):
            store.predict_one(sample, 'unknown')
        sample.write_bytes(wav(1.2))
        with self.assertRaises(ValueError):
            store.predict_one(sample, 'SpeechCommands')
    def test_classes_can_be_loaded_from_portable_bundle(self):
        bundles = self.root/'bundles'
        (bundles/'SpeechCommands').mkdir(parents=True)
        (bundles/'SpeechCommands/metadata.json').write_text(json.dumps({'classes':['yes','no']}))
        store = JobStore(self.root/'jobs', bundles, predictor_factory=FakePredictor)
        self.assertEqual(store.classes('SpeechCommands'), ['yes','no'])
    def test_zip_traversal_rejected(self):
        for name in ('../escape.wav', '/escape.wav', 'a\\escape.wav'):
            archive(self.root/'input.zip', {name:wav()})
            with self.assertRaises(ValueError):
                inspect_archive(self.root/'input.zip', self.root/'audio', 1)
    def test_duration_and_corrupt_audio_are_individual_errors(self):
        archive(self.root/'input.zip', {'good.wav':wav(), 'long.wav':wav(1.2), 'broken.wav':b'bad'})
        rows=inspect_archive(self.root/'input.zip', self.root/'audio', 1)
        self.assertEqual(sum(not r['error'] for r in rows),1)
        self.assertEqual(sum(bool(r['error']) for r in rows),2)
    def test_csv_invalid_and_duplicate_labels(self):
        for data in (b'file,label\na.wav,unknown',b'file,label\na.wav,yes\na.wav,no'):
            with self.assertRaises(ValueError): read_labels(data,['yes','no'])
    def test_unlabeled_has_no_metrics(self):
        self.assertIsNone(calculate_metrics([dict(error=None,predicted_label='yes',true_label=None)],['yes']))
    def test_batch_job_reports_and_partial_labels(self):
        store=JobStore(self.root/'jobs', self.root/'models', FakePredictor)
        bundle=self.root/'models'/'SpeechCommands';bundle.mkdir(parents=True)
        (bundle/'metadata.json').write_text(json.dumps({'purpose':'test_double_only', 'classes':['yes','no']}))
        for name in ('classifier.ts','normalization.npz'): (bundle/name).touch()
        archive(self.root/'input.zip', {'a.wav':wav(),'b.wav':wav(),'bad.wav':b'bad',
                                      'labels.csv':b'file,label\na.wav,yes\nbad.wav,no'})
        job=store.create(self.root/'input.zip','SpeechCommands')
        store.start(job['id'])
        deadline=time.time()+15
        while time.time()<deadline:
            result=store.get(job['id'])
            if result['status'] not in ('queued','running'):break
            time.sleep(.02)
        self.assertEqual(result['status'],'completed',result.get('error'))
        self.assertEqual(result['processed'],3)
        self.assertEqual(result['succeeded'],2)
        self.assertEqual(result['metrics']['evaluated'],1)
        self.assertEqual(result['metrics']['excluded'],2)
        self.assertEqual(result['metrics']['accuracy'],1)
        self.assertEqual(result['metrics']['confusion_matrix'],[[1,0],[0,0]])
        self.assertTrue((store.directory(job['id'])/'report.html').is_file())
        with self.assertRaises(ValueError):store.start(job['id'])
    def test_extra_csv_path_rejected(self):
        store=JobStore(self.root/'jobs',self.root/'models')
        bundle=self.root/'models/SpeechCommands';bundle.mkdir(parents=True)
        (bundle/'metadata.json').write_text(json.dumps({'classes':['yes','no']}))
        archive(self.root/'input.zip',{'a.wav':wav(),'labels.csv':b'file,label\nmissing.wav,yes'})
        with self.assertRaises(ValueError):store.create(self.root/'input.zip','SpeechCommands')
    def test_no_bundle_blocks_start_but_not_inspection(self):
        store=JobStore(self.root/'jobs',self.root/'models')
        archive(self.root/'input.zip',{'a.wav':wav()})
        job=store.create(self.root/'input.zip','SpeechCommands')
        with self.assertRaises(ValueError):store.start(job['id'])


if __name__=='__main__':unittest.main(verbosity=2)
