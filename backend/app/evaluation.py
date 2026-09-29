"""
Transferability evaluation.

Generates an adversarial image using a source model and evaluates
that same cloaked image on a different target model.
"""

import io

import torch
from PIL import Image

from app.cloak_engine import (
    cloak_image,
    _get_model,
    _PREPROCESS,
)

# ============================================================
# TARGET MODEL PREDICTION
# ============================================================


def _predict(image: Image.Image, model_name: str):
    """
    Run inference using the target model.

    Returns:
        class_index:
            Top predicted class.

        confidence:
            Confidence of the top prediction.

        probabilities:
            Full probability vector.
    """

    model = _get_model(model_name)

    x = _PREPROCESS(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    confidence, class_index = torch.max(
        probabilities,
        dim=1,
    )

    return {
        "class_index": int(class_index.item()),
        "confidence": float(confidence.item()),
        "probabilities": probabilities,
    }


# ============================================================
# TRANSFERABILITY
# ============================================================


def evaluate_transferability(
    image: Image.Image,
    source_model: str,
    target_model: str,
    strength: float = 1.0,
    method: str = "fgsm",
    protection_mode: str = "standard",
):

    # --------------------------------------------------------
    # Validate models
    # --------------------------------------------------------

    if source_model == target_model:
        raise ValueError("Source model and target model must be different.")

    # --------------------------------------------------------
    # Generate adversarial image using SOURCE model
    # --------------------------------------------------------

    source_result = cloak_image(
        image,
        strength=strength,
        method=method,
        model_name=source_model,
        protection_mode=protection_mode,
    )

    # --------------------------------------------------------
    # Extract cloaked image
    # --------------------------------------------------------

    if "cloaked_image" in source_result:

        cloaked_image = source_result["cloaked_image"]

    elif "cloaked_image_bytes" in source_result:

        cloaked_image = Image.open(
            io.BytesIO(source_result["cloaked_image_bytes"])
        ).convert("RGB")

    else:

        raise RuntimeError("Cloaking engine did not return " "a cloaked image.")

    # --------------------------------------------------------
    # TARGET MODEL - ORIGINAL IMAGE
    # --------------------------------------------------------

    target_original = _predict(
        image,
        target_model,
    )

    # --------------------------------------------------------
    # TARGET MODEL - CLOAKED IMAGE
    # --------------------------------------------------------

    target_cloaked = _predict(
        cloaked_image,
        target_model,
    )

    # --------------------------------------------------------
    # Original target class
    # --------------------------------------------------------

    original_target_class = target_original["class_index"]

    # --------------------------------------------------------
    # Confidence of ORIGINAL target class
    # after the attack
    # --------------------------------------------------------

    target_cloaked_original_class_confidence = float(
        target_cloaked["probabilities"][
            0,
            original_target_class,
        ].item()
    )

    # --------------------------------------------------------
    # Confidence drop
    # --------------------------------------------------------

    confidence_drop = (
        target_original["confidence"] - target_cloaked_original_class_confidence
    )

    # --------------------------------------------------------
    # Prediction changed?
    # --------------------------------------------------------

    prediction_changed = target_original["class_index"] != target_cloaked["class_index"]

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        # Attack configuration
        "source_model": source_model,
        "target_model": target_model,
        "method": method,
        "protection_mode": protection_mode,
        "strength": strength,
        "epsilon": source_result["epsilon"],
        # Target model results
        "target_original_class": (target_original["class_index"]),
        "target_cloaked_class": (target_cloaked["class_index"]),
        "target_original_confidence": (target_original["confidence"]),
        "target_cloaked_confidence": (target_cloaked["confidence"]),
        # Important:
        # confidence of the ORIGINAL target
        # class after the attack
        "target_cloaked_original_class_confidence": (
            target_cloaked_original_class_confidence
        ),
        "target_confidence_drop": (confidence_drop),
        "prediction_changed": (prediction_changed),
        # Source model results
        "source_misclassified": (source_result["misclassified"]),
        "source_confidence_drop": (source_result["confidence_drop"]),
    }
