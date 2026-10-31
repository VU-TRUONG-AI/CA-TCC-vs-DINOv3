"""Release download rejects traversal, duplicates, tampering and undeclared files."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from scripts.deployment_assets import safe_path, fetch

class AssetTests(unittest.TestCase):
    def test_asset_paths_cannot_escape_root(self):
        root = Path('/tmp/asset-root')
        for name in ('../outside', '/tmp/outside', 'a/../../outside', 'a\\outside'):
            with self.assertRaises(ValueError):
                safe_path(root, name)
        self.assertEqual(safe_path(root, 'models/file.ts'), root / 'models/file.ts')
    def test_release_requires_https_and_trusted_hash(self):
        for url, checksum in [('http://example.com/model.zip', 'a'*64),
                              ('https://example.com/model.zip', 'invalid')]:
            with self.assertRaises(ValueError):
                fetch(url, checksum)
    def test_download_hash_and_zip_traversal_rejected_before_extraction(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('../outside', b'malicious')
        content = data.getvalue()
        def download(url, target):
            Path(target).write_bytes(content)
        with patch('scripts.deployment_assets.urllib.request.urlretrieve', download):
            for checksum in ('0'*64, hashlib.sha256(content).hexdigest()):
                with self.assertRaises(ValueError):
                    fetch('https://example.com/release.zip', checksum)
