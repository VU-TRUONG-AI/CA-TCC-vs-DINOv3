Versioned deployment artifacts for CA-TCC + DINOv3 Audio.

Includes verified CPU TorchScript bundles, ONNX chunks, normalization, metadata,
SHA256 manifest and two validation WAV fixtures for inference regression.

- RAVDESS: 8 emotion classes, maximum 6 seconds.
- SpeechCommands: 35 keyword classes, maximum 1 second.
- These are existing research models; this release does not retrain them or claim new accuracy results.
- DINOv3 license is included in the archive.
- RAVDESS fixture: Livingstone and Russo (2018), CC BY-NC-SA 4.0, https://zenodo.org/records/1188976 .
- Speech Commands fixture: Pete Warden / Google, CC BY 4.0, https://arxiv.org/abs/1804.03209 .
- Fixture attribution and license links are included under fixtures/NOTICE.md.

Download deployment-assets.zip and verify its SHA256 against the accompanying file.
Do not modify assets under an existing version; publish a new model-assets version instead.
