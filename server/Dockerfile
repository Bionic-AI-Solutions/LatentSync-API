# syntax=docker/dockerfile:1.7
#
# LatentSync inference API — GPU service.
#
# Base: Bionic AI Solutions' ai-template (Ubuntu 22.04 + CUDA 12.8.1 + cuDNN 9.8 +
# conda env "wan22" with torch 2.9.0+cu128). Same base as linly/musetalk/flashhead.
#
# Build from the LatentSync repo root:
#   docker build -t docker4zerocool/ai-latentsync:latest -f server/Dockerfile .
#
# Checkpoints and HF cache are VOLUME-MOUNTED at runtime (see docker-compose.yaml).
# Nothing from /mnt/ai-models is baked into the image.

FROM docker4zerocool/ai-template:runtime-v2

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH=/opt/conda/envs/wan22/bin:/opt/conda/bin:$PATH \
    APP_HOME=/app/latentsync \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR $APP_HOME

# libgl1 — InsightFace / OpenCV; libsndfile1 — soundfile/librosa
# build-essential — g++ required by insightface's Cython mesh extension
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libsndfile1 ca-certificates build-essential \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# LatentSync deps — strip torch/torchvision pins so we don't downgrade the
# base image's torch 2.9.0+cu128 to 2.5.1+cu121.
COPY requirements.txt /tmp/latentsync-reqs.txt
RUN sed '/^torch[^a-z]/d; /^torchvision/d; /^--extra-index-url/d' \
        /tmp/latentsync-reqs.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Server-only deps (fastapi, uvicorn, python-multipart)
COPY server/requirements.txt /tmp/server-reqs.txt
RUN pip install --no-cache-dir -r /tmp/server-reqs.txt

# LatentSync source — configs/, latentsync/, scripts/, server/ all land here.
# Checkpoint files are NOT copied; they are volume-mounted at /models.
COPY . $APP_HOME

EXPOSE 8014

COPY server/entrypoint.sh /usr/local/bin/latentsync-entrypoint.sh
RUN chmod +x /usr/local/bin/latentsync-entrypoint.sh
ENTRYPOINT ["latentsync-entrypoint.sh"]
