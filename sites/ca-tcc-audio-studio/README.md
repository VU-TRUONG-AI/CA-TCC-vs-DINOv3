# CA-TCC Audio Studio — hosted browser inference

Buildless HTML/CSS/JS in dist/. Three-panel design follows the user's screenshot. Select RAVDESS, SpeechCommands, or a duration-based suggestion (≤1 second means SpeechCommands; longer means RAVDESS). This suggestion is not a learned task classifier.

Hosted inference uses ONNX Runtime Web WASM with the trained DINOv3 models and embedded log-mel/train normalization. Model chunks are 16 MiB each to fit static-asset limits. Never upload the unchunked waveform.onnx file as a static asset. PCM parsing and Kaiser polyphase resampling match the Python frontend. Real WAV browser output is compared against the verified Python model in artifacts/audio_studio_web_20261018/browser_verification.json.

Audio stays in the browser on the hosted Site. No local-machine API or tunnel is required for hosted inference. In the local Python app, the same UI can use its same-origin API instead. Large model files download on first use; model sessions are reused.

Deployment identity is in .openai/hosting.json. Reuse that project ID; never create another Site for this checkout. Use the installed Sites skill to obtain a temporary source write credential, pass it only through hidden stdin to site-workflow.mjs, push/package the exact source and deploy the saved version. Do not store credentials here. Current live/native deployment state is recorded in artifacts/audio_studio_web_20261018/deployment_state.json.

RAVDESS assets are ready. scripts/prepare_audio_studio_models.py --watch prepares SpeechCommands chunks after fixed training completes. Re-publish this same Site after browser_assets_status.json says ready_to_publish. A pending model is labelled honestly in the UI and cannot run.

Testing: local HTTP preview, scripts/qa_audio_studio.mjs (desktop/mobile, real WAV inference parity, model routing and CSV). CPU ONNX checks are in artifacts/dinov3_browser_models/<dataset>/verification.json. WASM runtime pinned to 1.22.0; exporter uses onnxruntime 1.22.1 and opset 17.

Audio conversion runs locally with Web Audio: compressed recordings are decoded and encoded as mono 16 kHz PCM16 WAV, with a free WAV download button. Supported codecs depend on the browser. Sequential upload/confirm/run/result flow and existing inference duration limits are preserved. Browser conversion QA is recorded in `artifacts/audio_studio_web_20261018/conversion_verification.json` in the parent workspace.
