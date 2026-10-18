"""Check the actual HTTP API with exported DINOv3 models and real validation WAVs."""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.batch import JobStore, BundlePredictor
from web.server import handler_for
from variants.dinov3.common import atomic_json


def verify(bundles, datasets):
    bundles = Path(bundles)
    with tempfile.TemporaryDirectory() as folder:
        store = JobStore(Path(folder)/'jobs', bundles)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        report = dict(passed=False, datasets={}, models=[])
        try:
            with urllib.request.urlopen(base+'/api/models') as response:
                report['models'] = json.load(response)
            with urllib.request.urlopen(base+'/') as response:
                assert response.status == 200 and b'<html' in response.read()
            for dataset in datasets:
                manifest = json.loads((ROOT/'datasets/tensors/full'/dataset/'manifest.json').read_text())
                wav = Path(manifest['splits']['val']['paths'][0])
                request = urllib.request.Request(base+'/api/predict?dataset='+dataset,
                                                data=wav.read_bytes(), headers={'Content-Type':'audio/wav'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                assert result['dataset'] == dataset and result['backbone'] == 'dinov3_vits16'
                reference = BundlePredictor(bundles/dataset, dataset)
                scores = reference.predict([wav])[0]
                best = int(scores.argmax())
                assert result['top_k'][0]['label'] == reference.classes[best]
                assert abs(result['top_k'][0]['probability']-float(scores[best])) < 1e-5
                report['datasets'][dataset] = dict(passed=True, http_status=200, reference_parity=True, response=result)
            request = urllib.request.Request(base+'/api/predict?dataset=unknown', data=wav.read_bytes())
            try:
                urllib.request.urlopen(request)
                raise AssertionError('Invalid dataset unexpectedly accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 400
            report.update(passed=True, invalid_dataset_rejected=True, frontend_http_status=200)
            atomic_json(bundles/'api_verification.json', report)
            return report
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles', default='artifacts/dinov3_app_bundles')
    parser.add_argument('--datasets', nargs='+', choices=['RAVDESS','SpeechCommands'], default=['RAVDESS','SpeechCommands'])
    args = parser.parse_args()
    result = verify(ROOT/args.bundles, args.datasets)
    print(json.dumps(result,ensure_ascii=False,indent=2))
