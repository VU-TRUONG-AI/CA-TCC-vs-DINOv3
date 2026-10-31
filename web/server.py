"""HTTP server for the CA-TCC demo; loopback by default, configurable for Docker."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, unquote, urlparse

from web.batch import JobStore, BundlePredictor, DATASETS, MAX_UPLOAD, MAX_FILE, ROOT

STATIC = Path(__file__).parent / 'static'
DOWNLOADS = {'predictions.csv': 'text/csv; charset=utf-8', 'errors.csv': 'text/csv; charset=utf-8',
             'metrics.json': 'application/json', 'report.html': 'text/html; charset=utf-8'}


def handler_for(store, allowed_hosts=None, allowed_origins=None):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type='application/json; charset=utf-8', attachment=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path == '/healthz':
                    models = store.models()
                    ready = len(models) == 2 and all(item['ready'] for item in models)
                    return self.respond(200 if ready else 503, {'ready': ready, 'models': models})
                if path == '/api/models':
                    return self.respond(200, store.models())
                if path == '/api/jobs':
                    jobs = [json.loads(p.read_text()) for p in store.root.glob('*/job.json')]
                    return self.respond(200, [{k: v for k, v in job.items() if k not in ('records', 'labels')}
                                             for job in sorted(jobs, key=lambda j: j['created'], reverse=True)])
                parts = path.strip('/').split('/')
                if len(parts) >= 3 and parts[:2] == ['api', 'jobs']:
                    job = store.get(parts[2])
                    if len(parts) == 3:
                        return self.respond(200, job)
                    if len(parts) == 5 and parts[3] == 'download' and parts[4] in DOWNLOADS:
                        name = parts[4]
                        if job['status'] != 'completed':
                            raise ValueError('Báo cáo chưa sẵn sàng.')
                        return self.respond(200, (store.directory(job['id']) / name).read_bytes(), DOWNLOADS[name], name)
                    if len(parts) >= 5 and parts[3] == 'audio':
                        name = unquote('/'.join(parts[4:]))
                        if name not in {r['file'] for r in job['records']}:
                            raise FileNotFoundError()
                        return self.respond(200, (store.directory(job['id']) / 'audio' / name).read_bytes(), 'audio/wav')
                files = {'/': ('index.html', 'text/html; charset=utf-8'),
                         '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                         '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path in files:
                    name, content_type = files[path]
                    return self.respond(200, (STATIC / name).read_bytes(), content_type)
                if path.startswith(('/runtime/', '/models/')):
                    asset = (STATIC / unquote(path.lstrip('/'))).resolve()
                    if not asset.is_relative_to(STATIC.resolve()) or not asset.is_file():
                        raise FileNotFoundError()
                    types = {'.js':'text/javascript; charset=utf-8', '.mjs':'text/javascript; charset=utf-8',
                             '.json':'application/json; charset=utf-8', '.wasm':'application/wasm'}
                    return self.respond(200, asset.read_bytes(), types.get(asset.suffix, 'application/octet-stream'))
                self.respond(404, {'error': 'Không tìm thấy.'})
            except FileNotFoundError:
                self.respond(404, {'error': 'Không tìm thấy tác vụ hoặc tệp.'})
            except ValueError as error:
                self.respond(400, {'error': str(error)})

        def do_POST(self):
            try:
                # Requests from other websites must not submit local jobs.
                host = self.headers.get('Host', '')
                hosts = allowed_hosts or {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
                origins = allowed_origins or {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
                if host not in hosts:
                    return self.respond(403, {'error': 'Host không hợp lệ.'})
                origin = self.headers.get('Origin')
                if origin and origin not in origins:
                    return self.respond(403, {'error': 'Origin không hợp lệ.'})
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or length > MAX_UPLOAD:
                    return self.respond(413, {'error': 'Tệp tải lên vượt giới hạn 512 MiB.'})
                parsed = urlparse(self.path)
                if parsed.path == '/api/predict':
                    if length < 1 or length > MAX_FILE:
                        return self.respond(413, {'error': 'Gửi một file WAV không quá 32 MiB.'})
                    dataset = parse_qs(parsed.query).get('dataset', [''])[0]
                    with tempfile.NamedTemporaryFile(suffix='.wav', dir=store.root) as stream:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024*1024, remaining))
                            if not chunk:
                                raise ValueError('Tải lên bị ngắt.')
                            stream.write(chunk)
                            remaining -= len(chunk)
                        stream.flush()
                        return self.respond(200, store.predict_one(stream.name, dataset))
                if parsed.path == '/api/jobs':
                    dataset = parse_qs(parsed.query).get('dataset', [''])[0]
                    with tempfile.NamedTemporaryFile(suffix='.zip', dir=store.root) as stream:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError('Tải lên bị ngắt.')
                            stream.write(chunk)
                            remaining -= len(chunk)
                        stream.flush()
                        return self.respond(201, store.create(stream.name, dataset))
                parts = parsed.path.strip('/').split('/')
                if len(parts) == 4 and parts[:2] == ['api', 'jobs']:
                    if parts[3] == 'labels':
                        if length > 8 * 1024**2:
                            raise ValueError('CSV vượt giới hạn 8 MiB.')
                        return self.respond(200, store.set_labels(parts[2], self.rfile.read(length)))
                    if parts[3] == 'start':
                        return self.respond(202, store.start(parts[2]))
                self.respond(404, {'error': 'Không tìm thấy.'})
            except FileNotFoundError:
                self.respond(404, {'error': 'Không tìm thấy tác vụ.'})
            except Exception as error:
                self.respond(400, {'error': str(error)})
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--allowed-host', action='append', help='Exact HTTP Host, including port; repeat as needed.')
    parser.add_argument('--allowed-origin', action='append', help='Exact browser Origin; repeat as needed.')
    parser.add_argument('--preload', action='store_true', help='Load both models before accepting requests.')
    parser.add_argument('--jobs', type=Path, default=ROOT / 'runtime_jobs')
    parser.add_argument('--bundles', type=Path, default=ROOT / 'artifacts' / 'models')
    args = parser.parse_args()
    store = JobStore(args.jobs, args.bundles)
    if args.preload:
        for dataset in DATASETS:
            store.predictors[dataset] = BundlePredictor(args.bundles / dataset, dataset)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(
        store, args.allowed_host, args.allowed_origin))
    print(f'CA-TCC Audio Lab: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
