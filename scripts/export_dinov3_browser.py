"""Export fixed-waveform DINOv3 ONNX models, validate parity and split static assets."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from datasets.pipeline.audio_features import read_audio, extract_features, MEL_BANK
from variants.dinov3.model import DinoAudioModel
from variants.dinov3.common import atomic_json, sha256


class WaveformClassifier(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        n = np.arange(400, dtype=np.float64)
        k = np.arange(257, dtype=np.float64)[:, None]
        window = .5-.5*np.cos(2*np.pi*n/400)
        angle = 2*np.pi*k*n[None]/512
        kernels = np.concatenate([np.cos(angle), -np.sin(angle)])*window[None]/window.sum()
        self.register_buffer('fft_kernel', torch.from_numpy(kernels.astype(np.float32))[:, None])
        self.register_buffer('mel', torch.from_numpy(MEL_BANK))
        self.register_buffer('mean', torch.from_numpy(mean).float())
        self.register_buffer('std', torch.from_numpy(std).float())

    def features(self, waveform):
        spectrum = F.conv1d(waveform[:, None], self.fft_kernel, stride=160, padding=200)
        power = spectrum[:, :257].square()+spectrum[:, 257:].square()
        logmel = torch.matmul(self.mel, power).clamp_min(1e-10).log()
        return (logmel-self.mean)/self.std

    def forward(self, waveform):
        return self.model(self.features(waveform))[0]


def export(dataset, checkpoint, directory):
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(2)
    checkpoint = Path(checkpoint).resolve()
    run = json.loads((checkpoint.parent.parent/'run_config.json').read_text())
    config = importlib.import_module(f'config_files.{dataset}_Configs').Config()
    model = DinoAudioModel(config, repo=run['dinov3_repo'], initialization='random',
                           image_height=run['image_height'], clip_z=run['clip_z'], backbone_policy=run['backbone_policy']).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict'])
    source = Path(run['data_path'])/dataset
    manifest = json.loads((source/'manifest.json').read_text())
    with np.load(source/'normalization.npz') as stats:
        mean, std = stats['mean'], stats['std']
    wrapped = WaveformClassifier(model, mean, std).eval()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_file = directory/'waveform.onnx'
    size = int(manifest['seconds']*16000)
    paths = manifest['splits']['val']['paths'][:3]
    def fixed(path):
        audio, _ = read_audio(path)
        if len(audio)>size:
            offset=(len(audio)-size)//2;audio=audio[offset:offset+size]
        return np.pad(audio,(0,size-len(audio))).astype(np.float32)[None]
    example = torch.from_numpy(fixed(paths[0]))
    with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
        torch.onnx.export(wrapped, example, str(model_file), input_names=['waveform'], output_names=['logits'],
                          opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(model_file))
    options = ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session = ort.InferenceSession(str(model_file), options, providers=['CPUExecutionProvider'])
    checks=[]
    with torch.inference_mode():
        for path in paths:
            waveform=fixed(path)
            audio,_=read_audio(path)
            normalized=((extract_features(audio,manifest['seconds'])[None]-mean)/std).astype(np.float32)
            actual_features=wrapped.features(torch.from_numpy(waveform)).numpy()
            np.testing.assert_allclose(actual_features,normalized,rtol=2e-3,atol=2e-3)
            reference=model(torch.from_numpy(normalized))[0].numpy()
            actual=session.run(None,{'waveform':waveform})[0]
            np.testing.assert_allclose(actual,reference,rtol=2e-3,atol=2e-3)
            assert int(actual.argmax())==int(reference.argmax())
            checks.append(dict(file=str(path),max_feature_error=float(np.max(abs(actual_features-normalized))),
                               max_logit_error=float(np.max(abs(actual-reference)))))
    chunks=[]
    with model_file.open('rb') as stream:
        index=0
        while chunk:=stream.read(16*1024**2):
            filename=f'weights-{index:02d}.bin';part=directory/filename;part.write_bytes(chunk)
            chunks.append(dict(file=filename,bytes=len(chunk),sha256=sha256(part)));index+=1
    meta=dict(dataset=dataset,backbone='dinov3_vits16',classes=manifest['classes'],seconds=manifest['seconds'],
              sample_rate=16000,input_name='waveform',output_name='logits',input_samples=size,
              model_sha256=sha256(model_file),checkpoint_sha256=sha256(checkpoint),chunks=chunks,ready=True,
              embedded_frontend='mono float32 16kHz waveform; padding/cropping is performed by caller; log-mel/normalization/adapter inside ONNX',
              onnx_opset=17,training_epochs=run['epochs'],learning_rate=run['learning_rate'],weight_decay=run['weight_decay'])
    atomic_json(directory/'verification.json',dict(passed=True,checks=checks,trace_warnings=sorted(set(str(w.message) for w in caught))))
    atomic_json(directory/'manifest.json',meta)
    print(f'Browser model ready: {dataset}; chunks={len(chunks)}; checks={checks}',flush=True)
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['RAVDESS','SpeechCommands'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();export(a.dataset,a.checkpoint,a.output)
