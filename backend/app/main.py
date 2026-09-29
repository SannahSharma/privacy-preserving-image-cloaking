"""
FastAPI backend — Person B's track.

Endpoints:
    POST /cloak
        Main endpoint: image + strength + model -> cloaked result

    POST /evaluate/transferability
        Generate an attack using a source model and evaluate the
        same cloaked image on a different target model.

    GET /health
        Liveness check, also reports whether engine is mock/live

    GET /models
        Lists supported target models for the frontend dropdown

CORS is wide open (allow_origins=["*"]) since the frontend isn't deployed
yet and its origin isn't known. Tighten this to the real Vercel URL once
Person C deploys.
"""

import base64
import io
import time
import traceback

from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    CloakResponse,
    HealthResponse,
    ModelsResponse,
)

from app.engine_interface import (
    apply_cloak,
    ENGINE_MODE,
    SUPPORTED_MODELS,
)

from app.evaluation import evaluate_transferability

from app import storage

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Image Cloaking API",
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONSTANTS
# ============================================================

_VALID_MODEL_NAMES = {m["name"] for m in SUPPORTED_MODELS}

_VALID_METHODS = {
    "fgsm",
    "pgd",
}

_VALID_PROTECTION_MODES = {
    "standard",
    "strong",
}

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ============================================================
# HEALTH
# ============================================================


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "ok",
        "engine_mode": ENGINE_MODE,
    }


# ============================================================
# MODELS
# ============================================================


@app.get(
    "/models",
    response_model=ModelsResponse,
)
def models():
    return {
        "models": SUPPORTED_MODELS,
    }


# ============================================================
# CLOAK IMAGE
# ============================================================


@app.post(
    "/cloak",
    response_model=CloakResponse,
)
async def cloak(
    image: UploadFile = File(...),
    strength: float = Form(...),
    model_name: str = Form("resnet50"),
    method: str = Form("fgsm"),
    protection_mode: str = Form("standard"),
):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if model_name not in _VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(f"model_name must be one of " f"{sorted(_VALID_MODEL_NAMES)}"),
        )

    if method not in _VALID_METHODS:
        raise HTTPException(
            status_code=422,
            detail=(f"method must be one of " f"{sorted(_VALID_METHODS)}"),
        )

    if not (0.0 <= strength <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="strength must be between 0.0 and 1.0",
        )

    if protection_mode not in _VALID_PROTECTION_MODES:
        raise HTTPException(
            status_code=422,
            detail=(
                "protection_mode must be one of " f"{sorted(_VALID_PROTECTION_MODES)}"
            ),
        )

    if image.content_type not in (
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    ):
        raise HTTPException(
            status_code=415,
            detail="image must be png, jpg, or webp",
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bytes = await image.read()

    if len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="image too large (max 10MB)",
        )

    # --------------------------------------------------------
    # Call cloaking engine
    # --------------------------------------------------------

    start = time.perf_counter()

    try:

        result = apply_cloak(
            image_bytes=image_bytes,
            strength=strength,
            model_name=model_name,
            method=method,
            protection_mode=protection_mode,
        )

    except NotImplementedError as e:

        raise HTTPException(
            status_code=501,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail="cloaking engine failed to process image",
        ) from e

    elapsed_ms = (time.perf_counter() - start) * 1000

    # --------------------------------------------------------
    # Optional Supabase storage
    # --------------------------------------------------------

    storage.save_temp_image(result["cloaked_image_bytes"])

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "cloaked_image_base64": base64.b64encode(result["cloaked_image_bytes"]).decode(
            "ascii"
        ),
        "model_used": model_name,
        "method_used": method,
        "protection_mode": protection_mode,
        "strength": strength,
        "epsilon": result["epsilon"],
        "misclassified": result["misclassified"],
        "confidence_drop": result["confidence_drop"],
        "original_class_confidence_after": (result["original_class_confidence_after"]),
        "original_predictions": (result["original_predictions"]),
        "cloaked_predictions": (result["cloaked_predictions"]),
        "activation_map": (result["activation_map"]),
        "ssim_score": (result["ssim_score"]),
        "processing_time_ms": round(
            elapsed_ms,
            2,
        ),
    }


# ============================================================
# TRANSFERABILITY EVALUATION
# ============================================================


@app.post("/evaluate/transferability")
async def transferability(
    image: UploadFile = File(...),
    source_model: str = Form(...),
    target_model: str = Form(...),
    strength: float = Form(1.0),
    method: str = Form("fgsm"),
    protection_mode: str = Form("standard"),
):

    # --------------------------------------------------------
    # Validate source model
    # --------------------------------------------------------

    if source_model not in _VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(f"source_model must be one of " f"{sorted(_VALID_MODEL_NAMES)}"),
        )

    # --------------------------------------------------------
    # Validate target model
    # --------------------------------------------------------

    if target_model not in _VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(f"target_model must be one of " f"{sorted(_VALID_MODEL_NAMES)}"),
        )

    # --------------------------------------------------------
    # Source and target must be different
    # --------------------------------------------------------

    if source_model == target_model:
        raise HTTPException(
            status_code=422,
            detail=("source_model and target_model " "must be different"),
        )

    # --------------------------------------------------------
    # Validate strength
    # --------------------------------------------------------

    if not (0.0 <= strength <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="strength must be between 0.0 and 1.0",
        )

    # --------------------------------------------------------
    # Validate attack method
    # --------------------------------------------------------

    if method not in _VALID_METHODS:
        raise HTTPException(
            status_code=422,
            detail=(f"method must be one of " f"{sorted(_VALID_METHODS)}"),
        )

    # --------------------------------------------------------
    # Validate protection mode
    # --------------------------------------------------------

    if protection_mode not in _VALID_PROTECTION_MODES:
        raise HTTPException(
            status_code=422,
            detail=(
                "protection_mode must be one of " f"{sorted(_VALID_PROTECTION_MODES)}"
            ),
        )

    # --------------------------------------------------------
    # Validate image type
    # --------------------------------------------------------

    if image.content_type not in (
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    ):
        raise HTTPException(
            status_code=415,
            detail="image must be png, jpg, or webp",
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bytes = await image.read()

    if len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="image too large (max 10MB)",
        )

    # --------------------------------------------------------
    # Convert bytes -> PIL image
    # --------------------------------------------------------

    try:

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    except Exception as e:

        raise HTTPException(
            status_code=422,
            detail="invalid image file",
        ) from e

    # --------------------------------------------------------
    # Run transferability evaluation
    # --------------------------------------------------------

    start = time.perf_counter()

    try:

        result = evaluate_transferability(
            image=pil_image,
            source_model=source_model,
            target_model=target_model,
            strength=strength,
            method=method,
            protection_mode=protection_mode,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    except Exception as e:

        # Print the real error in the Uvicorn terminal
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(f"transferability evaluation failed: {str(e)}"),
        ) from e

    elapsed_ms = (time.perf_counter() - start) * 1000

    # --------------------------------------------------------
    # Add processing time
    # --------------------------------------------------------

    result["processing_time_ms"] = round(
        elapsed_ms,
        2,
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return result
