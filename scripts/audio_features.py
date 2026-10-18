"""Compatibility import: canonical frontend lives in the consolidated datasets folder."""
from pathlib import Path
import sys
DATA_WORKSPACE = Path(__file__).resolve().parents[1] / 'datasets'
sys.path.insert(0, str(DATA_WORKSPACE))
from pipeline.audio_features import read_audio, extract_features, SAMPLE_RATE, N_MELS, N_FFT, WIN_LENGTH, HOP_LENGTH, MEL_BANK
