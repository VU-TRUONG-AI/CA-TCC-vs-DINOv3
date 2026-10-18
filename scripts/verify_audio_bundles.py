"""Verify real WAV inference matches cached test tensors after export."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audio_features import read_audio, extract_features

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', required=True)
p.add_argument('--bundles', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
torch.set_num_threads(4)
report = {}
for dataset in ('SpeechCommands', 'RAVDESS'):
    source = Path(args.data_root) / dataset
    bundle = Path(args.bundles) / dataset
    manifest = json.loads((source / 'manifest.json').read_text())
    meta = json.loads((bundle / 'metadata.json').read_text())
    model = torch.jit.load(str(bundle / 'classifier.ts')).eval()
    test = torch.load(source / 'test.pt', map_location='cpu', weights_only=True)
    with np.load(bundle / 'normalization.npz') as stats:
        mean, std = stats['mean'], stats['std']
    # Verify the first test example of every class, including all RAVDESS emotions.
    indices = [int(torch.where(test['labels'] == i)[0][0]) for i in range(len(meta['classes']))]
    largest_feature_error, largest_logit_error, predictions = 0.0, 0.0, []
    for index in indices:
        path = manifest['splits']['test']['paths'][index]
        audio, _ = read_audio(path)
        features = ((extract_features(audio, meta['seconds'])[None] - mean) / std).astype(np.float32)
        stored = test['samples'][index:index+1]
        np.testing.assert_allclose(features, stored.numpy(), rtol=1e-5, atol=1e-5)
        with torch.inference_mode():
            logits = model(torch.from_numpy(features))[0]
            reference = model(stored)[0]
        torch.testing.assert_close(logits, reference)
        largest_feature_error = max(largest_feature_error, float(np.max(np.abs(features-stored.numpy()))))
        largest_logit_error = max(largest_logit_error, float((logits-reference).abs().max()))
        predictions.append(dict(file=str(path), true_label=meta['classes'][int(test['labels'][index])],
                                predicted_label=meta['classes'][int(logits.argmax())]))
    report[dataset] = dict(passed=True, checked_files=len(indices), purpose=meta['purpose'],
                           max_feature_error=largest_feature_error, max_logit_error=largest_logit_error,
                           predictions=predictions)
output=Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='predictions'} for k,v in report.items()},indent=2))
