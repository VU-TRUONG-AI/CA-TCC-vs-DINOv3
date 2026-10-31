FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60
ARG SOURCE_REVISION=working-tree
ARG MODEL_ASSETS_SHA256=local
LABEL org.opencontainers.image.title="CA-TCC DINOv3 Audio API" \
      org.opencontainers.image.source="https://github.com/VU-TRUONG-AI/CA-TCC-vs-DINOv3" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      ai.model.assets.sha256=$MODEL_ASSETS_SHA256
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
WORKDIR /app
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/deployment/requirements.txt \
    && useradd --uid 10001 --create-home appuser \
    && mkdir /jobs && chown appuser:appuser /jobs
COPY web /app/web
COPY scripts/audio_features.py /app/scripts/audio_features.py
COPY datasets/pipeline/__init__.py datasets/pipeline/audio_features.py /app/datasets/pipeline/
COPY deployment/DINOv3_LICENSE.md /app/licenses/DINOv3_LICENSE.md
COPY build/models/RAVDESS /app/artifacts/dinov3_app_bundles/RAVDESS
COPY build/models/SpeechCommands /app/artifacts/dinov3_app_bundles/SpeechCommands
USER 10001:10001
EXPOSE 8000
VOLUME ["/jobs"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=8)"]
CMD ["python", "-m", "web.server", "--host", "0.0.0.0", "--port", "8000", "--bundles", "/app/artifacts/dinov3_app_bundles", "--jobs", "/jobs", "--preload"]
