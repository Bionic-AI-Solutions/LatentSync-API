"""LatentSync inference API — GPU lipsync service.

Endpoints
---------
POST /v1/video/lipsync                   submit job → 202 {job_id, poll_url, status}
GET  /v1/video/lipsync/jobs/{job_id}     poll       → {status, video_url?, duration_s?, error?}
GET  /v1/video/lipsync/jobs/{job_id}/result  download binary MP4
GET  /healthz                            503 while loading, 200 once ready
GET  /readyz                             200 idle, 202 busy

n8n vd.lipsync workflow submits a manually-built multipart/form-data body
(fields: `video`, `audio`) with inference_steps + model as query params.
The poll + download URLs are returned so n8n can poll at its own cadence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf

# LatentSync repo root is one level above this file (server/server.py → ../)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment)
# ---------------------------------------------------------------------------

CKPT_PATH     = Path(os.environ.get("LATENTSYNC_CKPT_PATH",    "/models/latentsync_unet.pt"))
WHISPER_PATH  = Path(os.environ.get("LATENTSYNC_WHISPER_PATH", "/models/whisper/tiny.pt"))
CONFIG_PATH   = Path(os.environ.get("LATENTSYNC_CONFIG",       "/app/latentsync/configs/unet/stage2_512.yaml"))
DEFAULT_STEPS = int(os.environ.get("LATENTSYNC_STEPS",         "20"))
DEFAULT_GUID  = float(os.environ.get("LATENTSYNC_GUIDANCE",    "1.5"))
JOB_TTL       = int(os.environ.get("JOB_TTL_SECONDS",          "3600"))
BASE_URL      = os.environ.get("BASE_URL",                      "https://mcp.baisoln.com/gpu-ai")

JOBS_DIR = Path("/tmp/latentsync_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("latentsync.server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------------

_pipeline: LipsyncPipeline | None = None
_config = None
_ready = False
_job_queue: asyncio.Queue = asyncio.Queue()
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pipeline loader (runs in thread pool at startup)
# ---------------------------------------------------------------------------

def _load_pipeline() -> tuple[LipsyncPipeline, object]:
    logger.info("Loading LatentSync UNet from %s", CKPT_PATH)
    config = OmegaConf.load(str(CONFIG_PATH))

    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16 else torch.float32
    logger.info("dtype=%s  fp16_supported=%s", dtype, is_fp16)

    # DDIMScheduler lives in configs/ inside the repo
    scheduler = DDIMScheduler.from_pretrained(str(_REPO_ROOT / "configs"))

    audio_encoder = Audio2Feature(
        model_path=str(WHISPER_PATH),
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    # sd-vae-ft-mse: downloaded to HF_HOME on first run (or reused from cache)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        str(CKPT_PATH),
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # DeepCache: skip every 3rd denoising step → ~3x speedup at negligible quality cost
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    logger.info("Pipeline ready")
    return pipeline, config


# ---------------------------------------------------------------------------
# Inference runner (called from executor so it doesn't block the event loop)
# ---------------------------------------------------------------------------

def _run_inference(job_id: str, video_path: str, audio_path: str,
                   steps: int, guidance: float) -> str:
    global _pipeline, _config
    assert _pipeline is not None, "pipeline not loaded"
    config = _config

    out_path = str(JOBS_DIR / job_id / "result.mp4")
    set_seed(42)

    dtype = torch.float16 if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    ) else torch.float32

    _pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )
    return out_path


def _probe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip()) if r.returncode == 0 else 0.0


# ---------------------------------------------------------------------------
# Background job worker — processes one job at a time (GPU is single-tenant)
# ---------------------------------------------------------------------------

async def _job_worker():
    loop = asyncio.get_event_loop()
    while True:
        job_id, payload = await _job_queue.get()
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["started_at"] = time.time()
        logger.info("Processing job %s (queue remaining: %d)", job_id, _job_queue.qsize())
        try:
            out_path = await loop.run_in_executor(
                None,
                _run_inference,
                job_id,
                payload["video_path"],
                payload["audio_path"],
                payload["steps"],
                payload["guidance"],
            )
            duration_s = _probe_duration(out_path)
            _jobs[job_id].update({
                "status": "completed",
                "video_url": f"{BASE_URL}/v1/video/lipsync/jobs/{job_id}/result",
                "result_url": f"{BASE_URL}/v1/video/lipsync/jobs/{job_id}/result",
                "result_path": out_path,
                "duration_s": duration_s,
                "finished_at": time.time(),
            })
            logger.info("Job %s done — %.1fs output video", job_id, duration_s)
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            _jobs[job_id].update({
                "status": "failed",
                "error": str(exc),
                "finished_at": time.time(),
            })
        finally:
            _job_queue.task_done()
            asyncio.create_task(_cleanup_after(job_id, JOB_TTL))


async def _cleanup_after(job_id: str, delay: int):
    await asyncio.sleep(delay)
    import shutil
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    _jobs.pop(job_id, None)
    logger.info("Cleaned up job %s", job_id)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _config, _ready
    loop = asyncio.get_event_loop()
    logger.info("Starting pipeline load…")
    _pipeline, _config = await loop.run_in_executor(None, _load_pipeline)
    _ready = True
    worker = asyncio.create_task(_job_worker())
    yield
    worker.cancel()


app = FastAPI(title="LatentSync Lipsync API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    if not _ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if not _ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    busy = any(j["status"] == "processing" for j in _jobs.values())
    if busy:
        return JSONResponse(
            {"status": "busy", "queue_depth": _job_queue.qsize()},
            status_code=202,
        )
    return {"status": "idle", "queue_depth": _job_queue.qsize()}


# ---------------------------------------------------------------------------
# Lipsync endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/video/lipsync", status_code=202)
async def submit_lipsync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_steps: int = Query(DEFAULT_STEPS, ge=10, le=50),
    guidance_scale: float = Query(DEFAULT_GUID, ge=1.0, le=3.0),
    model: str = Query("latentsync-1.5"),  # accepted for API compat, ignored
):
    if not _ready:
        raise HTTPException(503, "Pipeline not ready")

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(job_dir / "source.mp4")
    audio_path = str(job_dir / "audio.wav")

    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "queued_at": time.time(),
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
    }
    await _job_queue.put((job_id, {
        "video_path": video_path,
        "audio_path": audio_path,
        "steps": inference_steps,
        "guidance": guidance_scale,
    }))

    poll_url = f"{BASE_URL}/v1/video/lipsync/jobs/{job_id}"
    logger.info("Queued job %s (queue depth: %d)", job_id, _job_queue.qsize())
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "poll_url": poll_url,
        "queue_depth": _job_queue.qsize(),
    }, status_code=202)


@app.get("/v1/video/lipsync/jobs/{job_id}")
def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return job


@app.get("/v1/video/lipsync/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id!r} not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']!r}, not completed")
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(500, "Result file missing from disk")
    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=f"lipsync_{job_id[:8]}.mp4",
    )
