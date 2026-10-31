"""HTTP readiness and explicit Host/Origin policy for container deployment."""
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
import urllib.error
import urllib.request
from web.server import handler_for

class Store:
    ready = True
    def models(self):
        return [{'dataset': name, 'ready': self.ready} for name in ('RAVDESS', 'SpeechCommands')]
    def predict_one(self, path, dataset):
        return {'dataset': dataset}
    @property
    def root(self):
        import tempfile
        return tempfile.gettempdir()

class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(
            self.store, ['audio.example'], ['https://audio.example']))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
    def test_readiness_changes_status(self):
        with urllib.request.urlopen(self.base + '/healthz') as response:
            self.assertTrue(json.load(response)['ready'])
        self.store.ready = False
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base + '/healthz')
        self.assertEqual(error.exception.code, 503)
    def test_explicit_host_origin_and_rejection(self):
        for host, origin, expected in [('audio.example','https://audio.example',200),
                                        ('bad.example','https://audio.example',403),
                                        ('audio.example','https://bad.example',403)]:
            request = urllib.request.Request(self.base + '/api/predict?dataset=RAVDESS', data=b'wav',
                headers={'Host': host, 'Origin': origin})
            if expected == 200:
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, expected)
            else:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, expected)
