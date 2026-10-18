from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
"""Shared, deterministic WAV -> log-mel frontend for training and inference."""
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160


def read_audio(path):
    sr, raw = wavfile.read(path)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if info.min == 0:
            audio = (raw.astype(np.float32) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            audio = raw.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = raw.astype(np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    if channels > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f'Empty/nonfinite audio: {path}')
    stats = {'sample_rate': int(sr), 'channels': channels, 'duration': len(audio) / sr,
             'rms': float(np.sqrt(np.mean(audio ** 2))),
             'peak': float(np.max(np.abs(audio))),
             'clipped_fraction': float(np.mean(np.abs(audio) >= 0.999)),
             'near_silent_fraction': float(np.mean(np.abs(audio) < 1e-4))}
    if sr != SAMPLE_RATE:
        divisor = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    return audio, stats


def mel_bank():
    # HTK mel conversion. Constants are stored with the exported model.
    mel = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), N_MELS + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    bins = np.linspace(0, 8000, N_FFT // 2 + 1)
    bank = np.zeros((N_MELS, len(bins)), dtype=np.float32)
    for i in range(N_MELS):
        bank[i] = np.maximum(0, np.minimum((bins - hz[i]) / (hz[i+1] - hz[i]),
                                          (hz[i+2] - bins) / (hz[i+2] - hz[i+1])))
    return bank


MEL_BANK = mel_bank()


def extract_features(audio, seconds):
    target = round(SAMPLE_RATE * seconds)
    # Center crop only if necessary; pad waveform before STFT otherwise.
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    else:
        audio = np.pad(audio, (0, target - len(audio)))
    _, _, spectrum = stft(audio, fs=SAMPLE_RATE, window='hann', nperseg=WIN_LENGTH,
                          noverlap=WIN_LENGTH-HOP_LENGTH, nfft=N_FFT,
                          boundary='zeros', padded=True)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    return np.log(np.maximum(MEL_BANK @ power, 1e-10)).astype(np.float32)
