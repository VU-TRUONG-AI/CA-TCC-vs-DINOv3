"""Package NPZ into the shared tensors folder using the canonical packager."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from scripts.package_audio_data import package

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(ROOT / 'processed/full'))
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()
    destination = Path(args.output_root) if args.output_root else ROOT / 'tensors' / Path(args.data_root).name
    package(args.data_root, destination)
