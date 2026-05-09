#!/usr/bin/env bash
# Download LatentSync checkpoints to /mnt/ai-models/models/latentsync/
# Run this ONCE on the GPU host before starting the container.
#
# Requires: huggingface-cli (pip install huggingface_hub)
# Optional: HF_TOKEN env var for private repos (ByteDance/LatentSync-1.6 is public)
#
# After this script, the layout will be:
#   /mnt/ai-models/models/latentsync/
#     latentsync_unet.pt        (~1.8 GB)
#     whisper/
#       tiny.pt                 (~39 MB)
#     insightface/              (empty dir; face-detection weights auto-download on first run)

set -euo pipefail

DEST=/mnt/ai-models/models/latentsync

echo "Creating directories under $DEST ..."
mkdir -p "$DEST/whisper"
mkdir -p "$DEST/insightface"

echo "Downloading ByteDance/LatentSync-1.6 checkpoints ..."
huggingface-cli download ByteDance/LatentSync-1.6 \
    latentsync_unet.pt \
    whisper/tiny.pt \
    --local-dir "$DEST"

echo ""
echo "Done. Checkpoint sizes:"
ls -lh "$DEST/latentsync_unet.pt"
ls -lh "$DEST/whisper/tiny.pt"
echo ""
echo "sd-vae-ft-mse will be reused from /mnt/ai-models/models/hf_cache (already present)."
echo "InsightFace face-detection weights will auto-download to $DEST/insightface/ on first container run."
