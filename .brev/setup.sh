#!/usr/bin/env bash
# Brev startup hook — runs automatically when the instance is created or reset.
# Idempotent: skips steps that are already complete.
#
# Brev provides Docker + NVIDIA Container Toolkit out of the box.
# This script handles everything else: models, image build, and container start.
#
# Set DOCKER_PAT as a Brev secret:
#   brev secret set DOCKER_PAT <your-pat>
# Or pass it via environment before the instance is created.

set -euo pipefail
LOG=/tmp/brev-setup.log
exec > >(tee -a "$LOG") 2>&1

echo "=== LatentSync-API Brev setup $(date) ==="

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_USER="${DOCKER_USER:-docker4zerocool}"
MODEL_DIR="${MODEL_DIR:-/mnt/ai-models}"

# ── Docker login ───────────────────────────────────────────────────────────────
if [[ -n "${DOCKER_PAT:-}" ]]; then
    echo "[1/5] Docker login..."
    echo "$DOCKER_PAT" | docker login -u "$DOCKER_USER" --password-stdin
else
    echo "[1/5] DOCKER_PAT not set — skipping login (will use cached layers if image exists)"
fi

# ── Model directories ──────────────────────────────────────────────────────────
echo "[2/5] Model directories..."
sudo mkdir -p "$MODEL_DIR/models/latentsync/whisper" \
              "$MODEL_DIR/models/latentsync/insightface" \
              "$MODEL_DIR/models/hf_cache"
sudo chown -R "$(whoami)" "$MODEL_DIR"

# ── Download models if missing ─────────────────────────────────────────────────
UNET="$MODEL_DIR/models/latentsync/latentsync_unet.pt"
WHISPER="$MODEL_DIR/models/latentsync/whisper/tiny.pt"

if [[ -f "$UNET" && -f "$WHISPER" ]]; then
    echo "[3/5] Models already present — skipping download."
else
    echo "[3/5] Downloading models (~4.9 GB)..."
    bash "$REPO_DIR/server/download_models.sh"
fi

# ── Build image if not present ─────────────────────────────────────────────────
if docker image inspect "${DOCKER_USER}/ai-latentsync:latest" &>/dev/null; then
    echo "[4/5] Docker image already built — skipping."
else
    echo "[4/5] Building Docker image..."
    docker build -t "${DOCKER_USER}/ai-latentsync:latest" \
        -f "$REPO_DIR/server/Dockerfile" \
        "$REPO_DIR"
fi

# ── Start container ────────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q '^ai-latentsync$'; then
    echo "[5/5] Container already running — skipping."
else
    echo "[5/5] Starting container..."
    PUBLIC_IP=$(curl -sf --max-time 5 ifconfig.me || hostname -I | awk '{print $1}')
    BASE_URL="http://${PUBLIC_IP}:8014"
    sudo ufw allow 8014/tcp 2>/dev/null || true
    BASE_URL="$BASE_URL" MODEL_DIR="$MODEL_DIR" \
        docker compose -f "$REPO_DIR/server/docker-compose.yaml" up -d
fi

echo "=== Setup complete. Logs: docker logs ai-latentsync ==="
