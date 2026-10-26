# CA-TCC with DINOv3 for Audio Classification

This repository implements a CA-TCC-style audio classification pipeline with a DINOv3 ViT-S/16 encoder adapted for speech/audio tasks. The project is designed for two benchmark datasets:

- Speech Commands v0.02 (35 classes)
- RAVDESS (8 emotion classes)

The workflow combines:

- log-mel audio preprocessing
- DINOv3 backbone integration for audio-to-visual token adaptation
- temporal/contextual CA-TCC training stages
- pseudo-label and supervised contrastive fine-tuning
- validation/test evaluation and experiment tracking

## Why this project

The baseline CA-TCC pipeline uses a CNN encoder. This project replaces the encoder with a local pretrained DINOv3 backbone while keeping the original CA-TCC temporal/contextual learning structure. The integration points are isolated in the DINOv3 variant under the `variants/dinov3/` folder, while the baseline training and data flow remain available in the main project code.

## Project structure

```text
CA-TCC-main/
├── main.py                          # Baseline CA-TCC entry point
├── variants/
│   └── dinov3/
│       ├── model.py                # DINOv3 audio encoder + classifier
│       ├── main.py                 # DINOv3-specific training entry point
│       ├── trainer.py              # training loop and validation logic
│       ├── pipeline.py             # managed experiment stages
│       └── common.py               # provenance and validation helpers
├── models/
│   ├── model.py                    # original CNN baseline
│   ├── TC.py                      # temporal/contextual module
│   └── loss.py                    # NT-Xent / SupCon loss code
├── dataloader/
├── config_files/
├── datasets/
├── experiments_logs/
├── artifacts/
├── external/
├── weights/
├── requirements-audio.txt
├── README_DINOV3_ORIGINAL.md
├── docs/
└── web/
```

## Supported tasks

- speech keyword classification
- speech emotion recognition
- self-supervised representation learning
- linear probing and fine-tuning with limited labels
- pseudo-label generation and SupCon-based refinement

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-audio.txt
```

### 2. Prepare data

The project expects prepared tensorized dataset folders under a structure similar to:

```text
datasets/tensors/full/
├── SpeechCommands/
└── RAVDESS/
```

Check the dataset guide in `datasets/README.md` and the reproduction notes in `docs/REPRODUCE_AUDIO_VI.md` before training.

### 3. Run a standard training job

Baseline entry point:

```bash
python main.py \
  --experiment_description demo_run \
  --run_description SpeechCommands \
  --selected_dataset SpeechCommands \
  --training_mode self_supervised \
  --data_path datasets/tensors/full \
  --device cuda:0 \
  --seed 0
```

DINOv3-specific entry point:

```bash
python variants/dinov3/main.py \
  --experiment_description dinov3_demo \
  --run_description SpeechCommands \
  --selected_dataset SpeechCommands \
  --training_mode self_supervised \
  --data_path datasets/tensors/full \
  --device cuda:0 \
  --seed 0 \
  --dinov3-repo /path/to/local/dinov3 \
  --dinov3-weights /path/to/dinov3_vits16.pth \
  --initialization pretrained \
  --backbone-policy trainable
```

## Training modes

The DINOv3 variant supports stage-based training patterns such as:

- `self_supervised`
- `SupCon`
- `gen_pseudo_labels`
- `train_linear_10p`
- `ft_10p`
- `train_linear_SupCon_10p`
- `ft_SupCon_10p`

These modes are designed to mimic the CA-TCC learning pipeline: pretrain representation, generate pseudo-labels, then fine-tune the classifier with selective labeled data.

## Important notes

- The official DINOv3 repository and pretrained weights are external local assets; they are not bundled with Git source.
- Trained runs and experiment metadata are stored under `experiments_logs/` and `artifacts/`.
- Detailed methodology, implementation notes, and experiment interpretation are documented in `README_DINOV3_ORIGINAL.md`.
- This project is research-oriented and intended for reproducible experiments, not as a production-ready package by default.

## Reproducibility and artifacts

The repo includes saved experiment outputs, validation logs, and run metadata under:

```text
artifacts/
experiments_logs/
```

These artifacts capture configuration, source snapshots, checkpoint history, and evaluation summaries for each run.

## License and dependencies

This project relies on the upstream DINOv3 implementation and dataset-specific licenses. Please respect the terms for:

- DINOv3
- Speech Commands
- RAVDESS
- any local model checkpoints used for training

## Related documentation

- [README_DINOV3_ORIGINAL.md](README_DINOV3_ORIGINAL.md)
- [datasets/README.md](datasets/README.md)
- [docs/REPRODUCE_AUDIO_VI.md](docs/REPRODUCE_AUDIO_VI.md)
- [web/README.md](web/README.md)

## Summary

This repository is a research training codebase for CA-TCC audio classification with DINOv3 backbone integration. It is best used for reproducing and extending experiments on speech keyword and emotion tasks, with a focus on staged self-supervised and contrastive learning.

