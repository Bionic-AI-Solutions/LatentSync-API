#!/usr/bin/env bash
# Download LatentSync checkpoints to /mnt/ai-models/models/latentsync/
# Run this ONCE on the GPU host before starting the container.
#
# Requires: pip install huggingface_hub  (installs the `hf` CLI)
# Optional: HF_TOKEN env var for authenticated downloads (higher rate limits)
#
# After this script, the layout will be:
#   /mnt/ai-models/models/latentsync/
#     latentsync_unet.pt        (~4.8 GB  — LatentSync-1.6)
#     whisper/
#       tiny.pt                 (~73 MB)
#     insightface/              (empty dir; face-detection weights auto-download on first run)

set -euo pipefail

DEST=/mnt/ai-models/models/latentsync

# Ensure hf CLI is available
if ! command -v hf &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q --break-system-packages huggingface_hub
fi

echo "Creating directories under $DEST ..."
mkdir -p "$DEST/whisper"
mkdir -p "$DEST/insightface"
mkdir -p /mnt/ai-models/models/hf_cache

echo "Downloading ByteDance/LatentSync-1.6 checkpoints ..."
hf download ByteDance/LatentSync-1.6 \
    latentsync_unet.pt \
    whisper/tiny.pt \
    --local-dir "$DEST"

echo ""
echo "Done. Checkpoint sizes:"
ls -lh "$DEST/latentsync_unet.pt"
ls -lh "$DEST/whisper/tiny.pt"
echo ""
echo "InsightFace face-detection weights will auto-download to $DEST/insightface/ on first container run."
