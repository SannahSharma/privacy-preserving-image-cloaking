"""
FastAPI backend — Person B's track.

Endpoints:
    POST /cloak    - main endpoint, image + strength + model -> cloaked result
    GET  /health   - liveness check, also reports whether engine is mock/live
    GET  /models   - lists supported target models (for the frontend dropdown)

CORS is wide open (allow_origins=["*"]) since the frontend isn't deployed
yet and its origin isn't known. Tighten this to the real Vercel URL once
Person C deploys (Day 4).
"""

import base64
import time

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import CloakResponse, HealthResponse, ModelsResponse, ModelInfo
from app.engine_interface import apply_cloak, ENGINE_MODE, SUPPORTED_MODELS
from app import storage

app = FastAPI(title="Image Cloaking API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to the Vercel frontend origin once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_VALID_MODEL_NAMES = {m["name"] for m in SUPPORTED_MODELS}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB, generous for free-tier CPU limits


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "engine_mode": ENGINE_MODE}


@app.get("/models", response_model=ModelsResponse)
def models():
    return {"models": SUPPORTED_MODELS}


_VALID_METHODS = {"fgsm", "pgd"}


@app.post("/cloak", response_model=CloakResponse)
async def cloak(
    image: UploadFile = File(...),
    strength: float = Form(...),
    model_name: str = Form("resnet50"),
    method: str = Form("fgsm"),
):
    # --- validation ---
    if model_name not in _VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"model_name must be one of {sorted(_VALID_MODEL_NAMES)}",
        )
    if method not in _VALID_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"method must be one of {sorted(_VALID_METHODS)}",
        )
    if not (0.0 <= strength <= 1.0):
        raise HTTPException(status_code=422, detail="strength must be between 0.0 and 1.0")
    if image.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="image must be png, jpg, or webp")

    image_bytes = await image.read()
    if len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 10MB)")

    # --- call the engine ---
    start = time.perf_counter()
    try:
        result = apply_cloak(image_bytes=image_bytes, strength=strength, model_name=model_name, method=method)
    except NotImplementedError as e:
        # model_name is valid but the engine doesn't support it yet (e.g. mobilenetv3/vit)
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        # Never leak internal stack traces to the frontend; log server-side instead.
        raise HTTPException(status_code=500, detail="cloaking engine failed to process image") from e
    elapsed_ms = (time.perf_counter() - start) * 1000

    # --- optional: persist to Supabase temp storage (safe no-op if disabled) ---
    storage.save_temp_image(result["cloaked_image_bytes"])

    return {
        "cloaked_image_base64": base64.b64encode(result["cloaked_image_bytes"]).decode("ascii"),
        "model_used": model_name,
        "method_used": method,
        "strength": strength,
        "original_predictions": result["original_predictions"],
        "cloaked_predictions": result["cloaked_predictions"],
        "activation_map": result["activation_map"],
        "ssim_score": result["ssim_score"],
        "processing_time_ms": round(elapsed_ms, 2),
    }