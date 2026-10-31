# Inference regression fixtures

Two unmodified WAVs are selected from the existing validation partition. They test
numerical agreement and HTTP behavior, not classification accuracy. Reference
probabilities are generated with the exact released TorchScript model and frontend.
See reference.json for original filenames, SHA256 and model identities.

## RAVDESS
Steven R. Livingstone and Frank A. Russo (2018), The Ryerson Audio-Visual Database
of Emotional Speech and Song (RAVDESS), version 1.0.0.
Source: https://zenodo.org/records/1188976
Paper: https://doi.org/10.1371/journal.pone.0196391
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
License terms: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en
The WAV is unchanged. Use this fixture for noncommercial educational/research tests.

## Speech Commands v0.02
Pete Warden / Google, Speech Commands: A Dataset for Limited-Vocabulary Speech
Recognition (2018). Source: https://arxiv.org/abs/1804.03209
Dataset: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
License: Creative Commons Attribution 4.0 International.
License terms: https://creativecommons.org/licenses/by/4.0/legalcode.en
The WAV is unchanged.

The fixture licenses apply to the WAVs, separately from the software and DINOv3
model license. Fixtures are distributed in the release asset, not inside the API image.
