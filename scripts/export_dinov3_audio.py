"""Export an evaluated DINOv3 audio checkpoint to a verified CPU TorchScript bundle."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.audio_features import read_audio, extract_features
from variants.dinov3.common import sha256, atomic_json
from variants.dinov3.model import DinoAudioModel


def export(dataset, checkpoint_path, output_path):
    torch.set_num_threads(2)
    checkpoint_path, output = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    config_path = checkpoint_path.parent.parent / 'run_config.json'
    run = json.loads(config_path.read_text())
    if run['selected_dataset'] != dataset or run['backbone'] != 'dinov3_vits16':
        raise ValueError('Checkpoint/config does not match the requested DINOv3 dataset.')
    checkpoint_hash = sha256(checkpoint_path)
    if output.exists():
        metadata = json.loads((output / 'metadata.json').read_text())
        verification = json.loads((output / 'verification.json').read_text())
        if metadata['checkpoint_sha256'] != checkpoint_hash or not verification.get('passed') or sha256(output/'classifier.ts') != metadata['classifier_sha256']:
            raise ValueError('Existing bundle differs or is invalid; select a new output directory.')
        print(f'Already exported and verified: {output}', flush=True)
        return output
    configs = importlib.import_module(f'config_files.{dataset}_Configs').Config()
    model = DinoAudioModel(configs, repo=run['dinov3_repo'], initialization='random',
                           image_height=run['image_height'], clip_z=run['clip_z'],
                           backbone_policy=run['backbone_policy']).eval()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    source = Path(run['data_path']) / dataset
    manifest = json.loads((source / 'manifest.json').read_text())
    validation = torch.load(source / 'val.pt', map_location='cpu', weights_only=True)
    examples = validation['samples'][:3].float()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + dataset + '-export-', dir=output.parent))
    records = []
    with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # Image/time dimensions are fixed by frontend. Check dynamic batch explicitly below.
        traced = torch.jit.trace(model, examples[:2], check_inputs=[(examples[:1],), (examples[:3],)], strict=True)
        traced = torch.jit.freeze(traced.eval())
        traced.save(str(staging / 'classifier.ts'))
        portable = torch.jit.load(str(staging / 'classifier.ts'), map_location='cpu').eval()
        for batch in (1, 2, 3):
            expected = model(examples[:batch])[0]
            actual = portable(examples[:batch])[0]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            records.append(dict(kind='normalized_tensor', batch=batch, max_logit_error=float((actual-expected).abs().max())))
        with np.load(source / 'normalization.npz', allow_pickle=False) as stats:
            mean, std = stats['mean'], stats['std']
            np.savez(staging / 'normalization.npz', mean=mean, std=std)
        for class_id in range(len(manifest['classes'])):
            candidates = torch.where(validation['labels'] == class_id)[0]
            if not len(candidates):
                continue
            index = int(candidates[0])
            wav = manifest['splits']['val']['paths'][index]
            audio, _ = read_audio(wav)
            x = torch.from_numpy(((extract_features(audio, manifest['seconds'])[None] - mean) / std).astype(np.float32))
            torch.testing.assert_close(x, validation['samples'][index:index+1], rtol=1e-5, atol=1e-5)
            expected, actual = model(x)[0], portable(x)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            records.append(dict(kind='real_validation_wav', class_id=class_id, file=str(wav),
                                max_logit_error=float((actual-expected).abs().max())))
    metadata = {k: manifest[k] for k in ('dataset', 'classes', 'sample_rate', 'seconds', 'n_mels',
                                        'n_fft', 'win_length', 'hop_length', 'feature', 'normalization', 'smoke_subset')}
    metadata.update(backbone='dinov3_vits16', format='torchscript_trace', purpose='research_model',
                    checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                    classifier_sha256=sha256(staging/'classifier.ts'), manifest_sha256=sha256(source/'manifest.json'),
                    input_contract=dict(dtype='float32', shape=['batch', 40, examples.shape[2]],
                                        normalization='Apply normalization.npz to log-mel before model; adapter is inside model'),
                    output_contract='Tuple(logits [B,num_classes], temporal_features [B,128,L]); use softmax(logits)',
                    torch_version=str(torch.__version__), label_percent=run['label_percent'],
                    training_epochs=run['epochs'], learning_rate=run['learning_rate'], weight_decay=run['weight_decay'],
                    initialization=run['initialization'], pretrained_weights_sha256=run['dinov3_weights_sha256'],
                    image_height=run['image_height'], clip_z=run['clip_z'],
                    runtime_note='CPU deployment verified; upstream DINOv3 source and pretrained checkpoint are not needed for TorchScript inference.')
    atomic_json(staging/'metadata.json', metadata)
    atomic_json(staging/'verification.json', dict(passed=True, checks=records,
                trace_warnings=sorted(set(str(w.message) for w in caught)),
                limitations='Fixed channel/time dimensions; dynamic batches 1,2,3 verified on CPU. Export parity does not measure prediction quality.'))
    shutil.copy2(ROOT/'datasets/pipeline/audio_features.py', staging/'audio_features.py')
    shutil.copy2(config_path, staging/'training_config.json')
    license_file = ROOT/'external/dinov3/LICENSE.md'
    if license_file.exists():
        shutil.copy2(license_file, staging/'DINOv3_LICENSE.txt')
    (staging/'infer.py').write_text('''import argparse, json
from pathlib import Path
import numpy as np
import torch
from audio_features import read_audio, extract_features

def predict(wav, bundle=None):
    bundle = Path(bundle) if bundle else Path(__file__).resolve().parent
    meta = json.loads((bundle/'metadata.json').read_text())
    audio, _ = read_audio(wav)
    with np.load(bundle/'normalization.npz', allow_pickle=False) as stats:
        features = ((extract_features(audio, meta['seconds'])[None]-stats['mean'])/stats['std']).astype(np.float32)
    torch.set_num_threads(2)
    model = torch.jit.load(str(bundle/'classifier.ts'), map_location='cpu').eval()
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(features))[0].softmax(1)[0]
    values, indices = probabilities.topk(min(5, len(meta['classes'])))
    return {'dataset':meta['dataset'], 'top_k':[{'label':meta['classes'][int(i)], 'probability':float(v)} for v,i in zip(values,indices)]}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--wav',required=True);a=p.parse_args()
    print(json.dumps(predict(a.wav),ensure_ascii=False,indent=2))
''', encoding='utf-8')
    (staging/'requirements.txt').write_text('torch==2.5.1\nnumpy>=1.26,<3\nscipy>=1.11,<2\n')
    (staging/'README.md').write_text(f'''# DINOv3 audio bundle — {dataset}

Verified CPU TorchScript export of the validation-selected checkpoint. Accuracy is documented in the training report; export verification only checks numerical equivalence.

Files: classifier.ts, normalization.npz, metadata.json, verification.json, audio_features.py, infer.py, training_config.json, requirements.txt and upstream license when available.

Run `python infer.py --wav /absolute/path/to/audio.wav`. Input WAV is converted to mono 16 kHz, cropped/padded to {manifest['seconds']} seconds, converted to 40-channel log-mel, then normalized with training statistics. The image adapter is embedded in classifier.ts. Do not normalize RGB externally.

In an API, load and cache classifier.ts once with torch.jit.load(..., map_location="cpu").eval(). Forward returns (logits, features); apply softmax to logits. Fixed input shape [B,40,{examples.shape[2]}]; batches 1,2,3 verified on CPU. No upstream DINOv3 checkout or pretrained checkpoint required at runtime. This Python/PyTorch bundle is suitable for server-side app inference; native mobile conversion is a separate target.
''', encoding='utf-8')
    staging.replace(output)
    print(f'Export verified on CPU: {output}; checks={len(records)}', flush=True)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['RAVDESS', 'SpeechCommands'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    export(a.dataset, a.checkpoint, a.output)
