#!/usr/bin/env bash
# One-shot bootstrap for LatentSync-API on a fresh GPU instance (Brev / bare-metal).
#
# Prerequisites (Brev provides these automatically):
#   - Docker 29+
#   - NVIDIA Container Toolkit 1.18+
#   - NVIDIA driver (L40S / A6000 / A100 / H100 tested)
#
# Required env var:
#   DOCKER_PAT  — Docker Hub PAT for docker4zerocool (needed to pull ai-template base image)
#                 export DOCKER_PAT=<your-pat> before running, or pass on the command line:
#                 DOCKER_PAT=xxx bash setup.sh
#
# Optional env vars:
#   DOCKER_USER   — Docker Hub username (default: docker4zerocool)
#   MODEL_DIR     — model mount root (default: /mnt/ai-models)
#   BASE_URL      — poll_url base in API responses (default: http://<public-ip>:8014)
#   HF_TOKEN      — Hugging Face token for authenticated downloads (optional)

set -euo pipefail

DOCKER_USER="${DOCKER_USER:-docker4zerocool}"
MODEL_DIR="${MODEL_DIR:-/mnt/ai-models}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. Validate ────────────────────────────────────────────────────────────────
if [[ -z "${DOCKER_PAT:-}" ]]; then
    echo "ERROR: DOCKER_PAT is required to pull the base image."
    echo "  export DOCKER_PAT=<your-docker-hub-pat> && bash setup.sh"
    exit 1
fi

echo "=== LatentSync-API bootstrap ==="
echo "Repo:      $REPO_DIR"
echo "Model dir: $MODEL_DIR"

# ── 2. Docker login ────────────────────────────────────────────────────────────
echo "[1/5] Docker login..."
echo "$DOCKER_PAT" | docker login -u "$DOCKER_USER" --password-stdin

# ── 3. Model directories + download ───────────────────────────────────────────
echo "[2/5] Model directories..."
sudo mkdir -p "$MODEL_DIR/models/latentsync/whisper" \
              "$MODEL_DIR/models/latentsync/insightface" \
              "$MODEL_DIR/models/hf_cache"
sudo chown -R "$(whoami)" "$MODEL_DIR"

UNET="$MODEL_DIR/models/latentsync/latentsync_unet.pt"
WHISPER="$MODEL_DIR/models/latentsync/whisper/tiny.pt"

if [[ -f "$UNET" && -f "$WHISPER" ]]; then
    echo "[2/5] Models already present — skipping download."
else
    echo "[2/5] Downloading models (~4.9 GB)..."
    bash "$REPO_DIR/server/download_models.sh"
fi

# ── 4. Build Docker image ──────────────────────────────────────────────────────
echo "[3/5] Building Docker image..."
docker build -t "${DOCKER_USER}/ai-latentsync:latest" \
    -f "$REPO_DIR/server/Dockerfile" \
    "$REPO_DIR"

# ── 5. Derive BASE_URL if not set ─────────────────────────────────────────────
if [[ -z "${BASE_URL:-}" ]]; then
    PUBLIC_IP=$(curl -sf --max-time 5 ifconfig.me || hostname -I | awk '{print $1}')
    BASE_URL="http://${PUBLIC_IP}:8014"
fi
echo "[4/5] BASE_URL = $BASE_URL"

# ── 6. Expose port through UFW ─────────────────────────────────────────────────
echo "[4/5] Opening port 8014..."
sudo ufw allow 8014/tcp 2>/dev/null || true

# ── 7. Start container ─────────────────────────────────────────────────────────
echo "[5/5] Starting container..."
BASE_URL="$BASE_URL" \
MODEL_DIR="$MODEL_DIR" \
docker compose -f "$REPO_DIR/server/docker-compose.yaml" up -d

# ── 8. Health check ────────────────────────────────────────────────────────────
echo "Waiting for pipeline to load (this takes ~30s)..."
for i in $(seq 1 24); do
    status=$(curl -sf http://localhost:8014/healthz 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    if [[ "$status" == "ok" ]]; then
        echo ""
        echo "✓ LatentSync-API is live!"
        echo "  Health:  http://localhost:8014/healthz"
        echo "  Lipsync: POST ${BASE_URL}/v1/video/lipsync"
        exit 0
    fi
    printf "."
    sleep 5
done

echo ""
echo "Timed out waiting — check logs with: docker logs ai-latentsync"
exit 1
