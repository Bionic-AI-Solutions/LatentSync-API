# LatentSync API

GPU-accelerated lip-sync inference service wrapping [ByteDance/LatentSync](https://github.com/bytedance/LatentSync).

Exposes an async REST API with job queuing so multiple callers can submit lipsync jobs without blocking each other.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/video/lipsync` | Submit a lipsync job (multipart: `video` + `audio` files) |
| `GET`  | `/v1/video/lipsync/jobs/{job_id}` | Poll job status |
| `GET`  | `/v1/video/lipsync/jobs/{job_id}/result` | Download result MP4 |
| `GET`  | `/healthz` | 503 while loading, 200 once pipeline ready |
| `GET`  | `/readyz` | 200 idle, 202 busy |

### Submit a job

```bash
curl -X POST https://mcp.baisoln.com/gpu-ai/v1/video/lipsync \
  -H "X-API-Key: $KEY" \
  -F "video=@input.mp4" \
  -F "audio=@speech.wav" \
  -G -d "inference_steps=20" -d "guidance_scale=1.5"
```

Response `202`:
```json
{
  "job_id": "abc123...",
  "status": "queued",
  "poll_url": "https://mcp.baisoln.com/gpu-ai/v1/video/lipsync/jobs/abc123...",
  "queue_depth": 0
}
```

### Poll and download

```bash
# Poll until status == "completed"
curl https://mcp.baisoln.com/gpu-ai/v1/video/lipsync/jobs/$JOB_ID

# Download result
curl -o result.mp4 https://mcp.baisoln.com/gpu-ai/v1/video/lipsync/jobs/$JOB_ID/result
```

## Deploying on Brev (one command)

The repo includes `.brev/setup.sh` which Brev runs automatically on instance creation.

```bash
# Create instance (L40S recommended — 48 GB VRAM, matches tested config)
brev create latentsync --gpu-name L40S

# Set your Docker Hub PAT so the hook can pull the private base image
brev secret set DOCKER_PAT <your-docker4zerocool-pat>
```

Brev will:
1. Provision the GPU instance
2. Run `.brev/setup.sh` automatically
3. Download models (~4.9 GB from HuggingFace), build the image, start the container

Once healthy (`/healthz → {"status":"ok"}`), port-forward to test locally:

```bash
brev port-forward latentsync -p 8014:8014
curl http://localhost:8014/healthz
```

To expose it publicly, open the port:
```bash
brev exec latentsync "sudo ufw allow 8014/tcp"
# Then hit http://<brev-public-ip>:8014/v1/video/lipsync
```

## Running on any GPU host

### Prerequisites

- NVIDIA GPU with CUDA 12.x
- Docker 29+ with NVIDIA Container Toolkit
- Docker Hub credentials for `docker4zerocool` (private base image)

### One-shot setup

```bash
git clone https://github.com/Bionic-AI-Solutions/LatentSync-API.git
cd LatentSync-API
DOCKER_PAT=<your-pat> bash setup.sh
```

`setup.sh` handles everything: Docker login, model download, image build, container start, UFW rule, and health check. It is idempotent — safe to re-run.

### Manual steps

```bash
# 1. Download models
bash server/download_models.sh

# 2. Build image (from repo root)
docker build -t docker4zerocool/ai-latentsync:latest -f server/Dockerfile .

# 3. Start
BASE_URL=http://<your-ip>:8014 docker compose -f server/docker-compose.yaml up -d
```

Service listens on **port 8014**. `MODEL_DIR` and `BASE_URL` are overridable env vars.

## Architecture

- FastAPI + uvicorn on port 8014
- Single GPU worker: jobs queue in-process (`asyncio.Queue`), processed one at a time
- [DeepCache](https://github.com/horseee/DeepCache) enabled (cache_interval=3) — ~3× speedup with negligible quality loss
- Job TTL: 1 hour (temp files auto-cleaned)
- Base image: `docker4zerocool/ai-template:runtime-v2` (Ubuntu 22.04 + CUDA 12.8.1 + cuDNN 9.8 + torch 2.9.0+cu128)

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LATENTSYNC_CKPT_PATH` | `/models/latentsync_unet.pt` | UNet checkpoint |
| `LATENTSYNC_WHISPER_PATH` | `/models/whisper/tiny.pt` | Whisper tiny encoder |
| `LATENTSYNC_CONFIG` | `/app/latentsync/configs/unet/stage2_512.yaml` | UNet config YAML |
| `LATENTSYNC_STEPS` | `20` | Default inference steps |
| `LATENTSYNC_GUIDANCE` | `1.5` | Default guidance scale |
| `JOB_TTL_SECONDS` | `3600` | Result cleanup delay |
| `BASE_URL` | `https://mcp.baisoln.com/gpu-ai` | Used to build `poll_url` in responses |
| `LOG_LEVEL` | `info` | uvicorn log level |

## Upstream

Based on [ByteDance/LatentSync](https://github.com/bytedance/LatentSync) (Apache-2.0).
