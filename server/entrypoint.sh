#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so "configs/" relative path resolves for DDIMScheduler.
cd /app/latentsync

exec uvicorn server.server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8014}" \
    --log-level "${LOG_LEVEL:-info}"
