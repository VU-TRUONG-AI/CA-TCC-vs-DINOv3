"""Check exported ONNX models numerically; this is not a browser UI test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audio_features import read_audio


def verify(directory, fixtures):
    references = json.loads((fixtures / 'reference.json').read_text())
    report = {'passed': False, 'runtime': ort.__version__, 'test_type': 'ONNX CPU parity; browser UI not exercised', 'datasets': {}}
    for dataset, ref in references.items():
        folder = directory / 'models' / dataset
        meta = json.loads((folder / 'manifest.json').read_text())
        parts = []
        for chunk in meta['chunks']:
            data = (folder / chunk['file']).read_bytes()
            assert len(data) == chunk['bytes'] and hashlib.sha256(data).hexdigest() == chunk['sha256']
            parts.append(data)
        model = b''.join(parts)
        assert hashlib.sha256(model).hexdigest() == meta['model_sha256']
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
        audio, _ = read_audio(fixtures / ref['file'])
        waveform = np.zeros((1, meta['input_samples']), dtype=np.float32)
        waveform[0, :min(len(audio), waveform.shape[1])] = audio[:waveform.shape[1]]
        logits = session.run([meta['output_name']], {meta['input_name']: waveform})[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        expected = np.array(ref['probabilities'])
        error = float(np.max(np.abs(probabilities - expected)))
        assert np.isfinite(probabilities).all() and len(probabilities) == len(ref['classes'])
        assert error < 0.002, (dataset, error)
        assert probabilities.argmax() == expected.argmax()
        report['datasets'][dataset] = {'passed': True, 'maximum_probability_error': error,
            'top_label': ref['classes'][int(probabilities.argmax())], 'model_sha256': meta['model_sha256']}
        del session, model, parts
    report['passed'] = True
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'build/web-dist')
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'build/fixtures')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/deployment/onnx-verification.json')
    args = parser.parse_args()
    result = verify(args.directory, args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
