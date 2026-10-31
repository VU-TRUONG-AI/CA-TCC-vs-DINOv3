"""Exercise a running API with deterministic synthetic WAVs (availability, not accuracy)."""
import argparse
import io
import json
import math
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
import hashlib


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def run(base, fixtures=None):
    deadline = time.monotonic() + 90
    while True:
        try:
            health = get(base + '/healthz')
            assert health['ready']
            break
        except (OSError, AssertionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b''.join(struct.pack('<h', round(1000 * math.sin(i * 2 * math.pi * 440 / 16000))) for i in range(3200)))
    report = {'passed': False, 'fixture': 'synthetic 440 Hz tone; no accuracy claim', 'datasets': {}}
    for dataset in ('RAVDESS', 'SpeechCommands'):
        request = urllib.request.Request(base + '/api/predict?dataset=' + dataset,
                                         data=audio.getvalue(), headers={'Content-Type': 'audio/wav'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        assert result['dataset'] == dataset and result['backbone'] == 'dinov3_vits16'
        assert len(result['top_k']) == 5 and len(result['version']) == 16
        assert all(math.isfinite(x['probability']) and 0 <= x['probability'] <= 1 for x in result['top_k'])
        report['datasets'][dataset] = result
    for body, dataset in [(audio.getvalue(), 'unknown'), (b'invalid wav', 'RAVDESS')]:
        try:
            urllib.request.urlopen(urllib.request.Request(base + '/api/predict?dataset=' + dataset, data=body))
            raise AssertionError('Invalid input accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 400, error.code
    try:
        urllib.request.urlopen(urllib.request.Request(base + '/api/predict?dataset=RAVDESS',
            data=audio.getvalue(), headers={'Origin': 'https://untrusted.example'}))
        raise AssertionError('Unexpected Origin accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 403
    report.update(passed=True, invalid_input_rejected=True, unexpected_origin_rejected=True)
    # An oversized waveform must be rejected before model inference.
    oversized = io.BytesIO()
    with wave.open(oversized, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x00\x00' * 19200)
    try:
        urllib.request.urlopen(urllib.request.Request(base + '/api/predict?dataset=SpeechCommands', data=oversized.getvalue()))
        raise AssertionError('Oversized WAV accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
    report['duration_guard_passed'] = True
    if fixtures:
        folder = Path(fixtures)
        references = json.loads((folder / 'reference.json').read_text())
        report['real_wav_parity'] = {}
        for dataset, reference in references.items():
            data = (folder / reference['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == reference['sha256']
            request = urllib.request.Request(base + '/api/predict?dataset=' + dataset,
                data=data, headers={'Content-Type': 'audio/wav'})
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            assert result['version'] == reference['model_version']
            expected = dict(zip(reference['classes'], reference['probabilities']))
            best = max(expected, key=expected.get)
            assert result['top_k'][0]['label'] == best
            maximum = max(abs(item['probability'] - expected[item['label']]) for item in result['top_k'])
            assert maximum < 1e-4, maximum
            report['real_wav_parity'][dataset] = {'passed': True, 'max_probability_error': maximum,
                'response': result, 'fixture_sha256': reference['sha256']}
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--output')
    parser.add_argument('--fixtures', help='Release fixtures directory for real WAV numerical parity')
    args = parser.parse_args()
    report = run(args.base.rstrip('/'), args.fixtures)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text)
    print(text)
