"""Export classifier + preprocessing to a portable inference bundle."""
import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import base_Model

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=['SpeechCommands', 'RAVDESS'])
p.add_argument('--data-root', default='datasets/tensors/full')
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
config = importlib.import_module(f'config_files.{args.dataset}_Configs').Config()
model = base_Model(config).eval()
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
source = Path(args.data_root) / args.dataset
manifest = json.loads((source / 'manifest.json').read_text())
metadata = {k: manifest[k] for k in ['dataset', 'classes', 'sample_rate', 'seconds', 'n_mels',
                                    'n_fft', 'win_length', 'hop_length', 'feature', 'normalization', 'smoke_subset']}
metadata['manifest_sha256'] = hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest()
metadata['checkpoint'] = str(Path(args.checkpoint).resolve())
metadata['purpose'] = 'smoke_test_only' if metadata['smoke_subset'] else 'research_model'
normalization = np.load(source / 'normalization.npz')
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
np.savez(output / 'normalization.npz', mean=normalization['mean'], std=normalization['std'])
(output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
scripted = torch.jit.script(model)
scripted.save(str(output / 'classifier.ts'))
example = torch.load(source / 'test.pt', map_location='cpu', weights_only=True)['samples'][:2]
with torch.no_grad():
    expected = model(example)[0]
    actual = torch.jit.load(str(output / 'classifier.ts'))(example)[0]
torch.testing.assert_close(expected, actual)
print(f'Export validated: {output}; purpose={metadata["purpose"]}')
