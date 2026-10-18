"""Predict a WAV using precisely the training preprocessing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audio_features import read_audio, extract_features


def predict(bundle, wav):
    bundle = Path(bundle)
    metadata = json.loads((bundle / 'metadata.json').read_text())
    stats = np.load(bundle / 'normalization.npz')
    audio, _ = read_audio(wav)
    x = extract_features(audio, metadata['seconds'])[None]
    x = torch.from_numpy(((x - stats['mean']) / stats['std']).astype(np.float32))
    model = torch.jit.load(str(bundle / 'classifier.ts'), map_location='cpu').eval()
    with torch.no_grad():
        probabilities = model(x)[0].softmax(dim=1)[0]
    values, indices = probabilities.topk(min(5, len(probabilities)))
    return {'dataset': metadata['dataset'], 'purpose': metadata['purpose'],
            'top_k': [{'label': metadata['classes'][i], 'probability': float(v)} for v, i in zip(values, indices)]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--wav', required=True)
    args = p.parse_args()
    print(json.dumps(predict(args.bundle, args.wav), indent=2))
