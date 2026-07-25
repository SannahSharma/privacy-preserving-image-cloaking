"""
ENGINE CONTRACT
===============
This is the ONLY file that changes when the engine's internals change.
Nothing in main.py or schemas.py needs to know what cloak_image() actually
looks like — they only ever see apply_cloak()'s output below.

apply_cloak(image_bytes: bytes, strength: float, model_name: str, method: str) -> dict
returns a dict with keys:
    cloaked_image_bytes, original_predictions, cloaked_predictions,
    activation_map, ssim_score
(exact shapes are documented inline below, next to where each is built.)

STATUS: LIVE — wired to Person A's real app/cloak_engine.py (2026-07-22).
"""

import io
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

try:
    from torchvision.models import ResNet50_Weights
    _IMAGENET_CATEGORIES = ResNet50_Weights.DEFAULT.meta["categories"]
except ImportError:
    # Fallback labels when torchvision is unavailable.
    _IMAGENET_CATEGORIES = [f"class_{i}" for i in range(1000)]

from app.cloak_engine import cloak_image as _cloak_image

ENGINE_MODE = "live"

# Person A's engine currently only implements ResNet-50. MobileNetV3/ViT are
# listed for the frontend to show as "coming soon" but aren't callable yet —
# requesting them raises NotImplementedError, which main.py turns into a 501.
SUPPORTED_MODELS = [
    {"name": "resnet50", "display_name": "ResNet-50", "supports_face_mode": True, "implemented": True},
    {"name": "mobilenetv3", "display_name": "MobileNetV3", "supports_face_mode": True, "implemented": False},
    {"name": "vit", "display_name": "Vision Transformer", "supports_face_mode": False, "implemented": False},
]
_IMPLEMENTED_MODELS = {m["name"] for m in SUPPORTED_MODELS if m["implemented"]}


def _label_for(class_idx: int) -> str:
    try:
        return _IMAGENET_CATEGORIES[class_idx]
    except (IndexError, TypeError):
        return f"class_{class_idx}"


def _grid_to_points(grid: list[list[float]]) -> list[dict]:
    """
    The real engine returns a 7x7 grid of floats (ResNet-50's layer4
    activation, downsampled). The frontend/Canvas contract expects a flat
    list of {x, y, intensity} points in 0-1 space, so we flatten the grid
    here — one point per cell, centered in its cell.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    points = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            points.append({
                "x": round((c + 0.5) / cols, 3),
                "y": round((r + 0.5) / rows, 3),
                "intensity": round(float(val), 3),
            })
    return points


def _compute_ssim(original: Image.Image, cloaked: Image.Image) -> float | None:
    """
    The engine doesn't return an SSIM score itself, but the project's core
    quality metric is SSIM > 0.95, so the backend computes it here. Both
    images are the same resolution (the engine preserves original size),
    so this is a direct grayscale comparison.
    """
    try:
        original_arr = np.array(original.convert("L"))
        cloaked_arr = np.array(cloaked.convert("L"))
        return round(float(ssim_metric(original_arr, cloaked_arr, data_range=255)), 4)
    except Exception:
        return None


def apply_cloak(image_bytes: bytes, strength: float, model_name: str, method: str = "fgsm") -> dict:
    if model_name not in _IMPLEMENTED_MODELS:
        raise NotImplementedError(
            f"'{model_name}' is not implemented yet by the cloaking engine — only "
            f"{sorted(_IMPLEMENTED_MODELS)} are currently supported."
        )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = _cloak_image(image, strength=strength, method=method)
    cloaked_image = result["cloaked_image"]

    out_buf = io.BytesIO()
    cloaked_image.save(out_buf, format="PNG")
    cloaked_bytes = out_buf.getvalue()

    return {
        "cloaked_image_bytes": cloaked_bytes,
        "original_predictions": [{
            "label": _label_for(result["original_class"]),
            "confidence": result["original_confidence"],
        }],
        "cloaked_predictions": [{
            "label": _label_for(result["cloaked_class"]),
            "confidence": result["cloaked_confidence"],
        }],
        "activation_map": _grid_to_points(result["activation_map"]),
        "ssim_score": _compute_ssim(image, cloaked_image),
    }