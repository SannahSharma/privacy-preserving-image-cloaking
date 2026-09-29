"""
Batch Transferability Evaluation

Purpose:
    Generate a cloaked/adversarial image using one source model,
    then test that SAME cloaked image on a different target model.

Models:
    ResNet-50
    MobileNetV3
    Vision Transformer (ViT)

First test:
    10 images × 6 model pairs = 60 experiments

Output:
    results/transferability_results.csv
"""

import csv
import io
import sys
from pathlib import Path

import torch
from PIL import Image

# ============================================================
# PATH SETUP
# ============================================================

# Project root:
# C:\MajorProject\privacy-preserving-image-cloaking
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Backend:
# C:\MajorProject\privacy-preserving-image-cloaking\backend
BACKEND_DIR = PROJECT_ROOT / "backend"

# Allow imports such as:
# from app.cloak_engine import ...
sys.path.insert(0, str(BACKEND_DIR))


from app.cloak_engine import (
    cloak_image,
    _get_model,
    _PREPROCESS,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODELS = [
    "resnet50",
    "mobilenetv3",
    "vit",
]

METHOD = "fgsm"

PROTECTION_MODE = "standard"

STRENGTH = 1.0

# Your actual image folder
IMAGE_DIR = PROJECT_ROOT / "datasets" / "test_images"

# First test only 10 images.
# Later change this to 120 for the full dataset.
MAX_IMAGES = 120

# CSV output
OUTPUT_FILE = PROJECT_ROOT / "results" / "transferability_results.csv"


# ============================================================
# PREDICTION
# ============================================================


def predict(image: Image.Image, model_name: str):
    """
    Run prediction using a cached model.

    Returns:
        class_index:
            Top predicted ImageNet class.

        confidence:
            Confidence of the top prediction.

        probabilities:
            Full probability vector.
    """

    model = _get_model(model_name)

    # Use the same preprocessing as the cloaking engine
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
# SINGLE MODEL-PAIR EVALUATION
# ============================================================


def evaluate_pair(
    image: Image.Image,
    source_model: str,
    target_model: str,
):
    """
    1. Create the attack using source_model.
    2. Get the cloaked image.
    3. Test original image on target_model.
    4. Test cloaked image on target_model.
    5. Measure transferability.
    """

    # --------------------------------------------------------
    # STEP 1:
    # Generate cloaked image using SOURCE model
    # --------------------------------------------------------

    source_result = cloak_image(
        image,
        strength=STRENGTH,
        method=METHOD,
        model_name=source_model,
        protection_mode=PROTECTION_MODE,
    )

    # --------------------------------------------------------
    # STEP 2:
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
    # STEP 3:
    # TARGET MODEL prediction on ORIGINAL image
    # --------------------------------------------------------

    target_original = predict(
        image,
        target_model,
    )

    # --------------------------------------------------------
    # STEP 4:
    # TARGET MODEL prediction on CLOAKED image
    # --------------------------------------------------------

    target_cloaked = predict(
        cloaked_image,
        target_model,
    )

    # --------------------------------------------------------
    # STEP 5:
    # Get original target class
    # --------------------------------------------------------

    original_target_class = target_original["class_index"]

    # --------------------------------------------------------
    # STEP 6:
    # Get confidence of ORIGINAL class
    # after cloaking
    # --------------------------------------------------------

    target_cloaked_original_class_confidence = float(
        target_cloaked["probabilities"][
            0,
            original_target_class,
        ].item()
    )

    # --------------------------------------------------------
    # STEP 7:
    # Calculate confidence drop
    # --------------------------------------------------------

    target_confidence_drop = (
        target_original["confidence"] - target_cloaked_original_class_confidence
    )

    # --------------------------------------------------------
    # STEP 8:
    # Did target prediction change?
    # --------------------------------------------------------

    prediction_changed = target_original["class_index"] != target_cloaked["class_index"]

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        # Attack information
        "epsilon": source_result["epsilon"],
        "source_misclassified": (source_result["misclassified"]),
        "source_confidence_drop": (source_result["confidence_drop"]),
        # Target model
        "target_original_class": (target_original["class_index"]),
        "target_cloaked_class": (target_cloaked["class_index"]),
        "target_original_confidence": (target_original["confidence"]),
        "target_cloaked_original_class_confidence": (
            target_cloaked_original_class_confidence
        ),
        "target_confidence_drop": (target_confidence_drop),
        "prediction_changed": (prediction_changed),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)
    print("BATCH TRANSFERABILITY EVALUATION")
    print("=" * 70)

    print(f"Project root     : {PROJECT_ROOT}")

    print(f"Image directory  : {IMAGE_DIR}")

    print(f"Method           : {METHOD}")

    print(f"Protection mode  : {PROTECTION_MODE}")

    print(f"Strength         : {STRENGTH}")

    print(f"Maximum images   : {MAX_IMAGES}")

    print()

    # ========================================================
    # CHECK IMAGE DIRECTORY
    # ========================================================

    if not IMAGE_DIR.exists():

        print("ERROR: Image directory does not exist:")

        print(IMAGE_DIR)

        return

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    image_paths = sorted(
        [
            path
            for path in IMAGE_DIR.rglob("*")
            if path.suffix.lower() in image_extensions
        ]
    )

    if not image_paths:

        print("ERROR: No images found.")

        return

    # Take only the first MAX_IMAGES
    image_paths = image_paths[:MAX_IMAGES]

    print(f"Images selected: {len(image_paths)}")

    print()

    # ========================================================
    # MODEL PAIRS
    # ========================================================

    model_pairs = [
        (source, target) for source in MODELS for target in MODELS if source != target
    ]

    print("Model pairs:")

    for source, target in model_pairs:

        print(f"  {source} -> {target}")

    total_experiments = len(image_paths) * len(model_pairs)

    print()

    print(f"Total experiments: " f"{total_experiments}")

    print()

    # ========================================================
    # CREATE RESULTS DIRECTORY
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CSV COLUMNS
    # ========================================================

    fieldnames = [
        "image",
        "source_model",
        "target_model",
        "method",
        "strength",
        "epsilon",
        "source_misclassified",
        "source_confidence_drop",
        "target_original_class",
        "target_cloaked_class",
        "target_original_confidence",
        "target_cloaked_original_class_confidence",
        "target_confidence_drop",
        "prediction_changed",
    ]

    rows = []

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    current_experiment = 0

    for image_path in image_paths:

        print(f"\nImage: {image_path.name}")

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as e:

            print(f"  ERROR opening image: {e}")

            continue

        # ----------------------------------------------------
        # Test all six model pairs
        # ----------------------------------------------------

        for source_model, target_model in model_pairs:

            current_experiment += 1

            print(
                f"  [{current_experiment}/"
                f"{total_experiments}] "
                f"{source_model} -> "
                f"{target_model}",
                end=" ... ",
                flush=True,
            )

            try:

                result = evaluate_pair(
                    image=image,
                    source_model=source_model,
                    target_model=target_model,
                )

                rows.append(
                    {
                        "image": image_path.name,
                        "source_model": source_model,
                        "target_model": target_model,
                        "method": METHOD,
                        "strength": STRENGTH,
                        "epsilon": result["epsilon"],
                        "source_misclassified": result["source_misclassified"],
                        "source_confidence_drop": result["source_confidence_drop"],
                        "target_original_class": result["target_original_class"],
                        "target_cloaked_class": result["target_cloaked_class"],
                        "target_original_confidence": result[
                            "target_original_confidence"
                        ],
                        "target_cloaked_original_class_confidence": result[
                            "target_cloaked_original_class_confidence"
                        ],
                        "target_confidence_drop": result["target_confidence_drop"],
                        "prediction_changed": result["prediction_changed"],
                    }
                )

                # ------------------------------------------------
                # Print result
                # ------------------------------------------------

                if result["prediction_changed"]:

                    print("PREDICTION CHANGED")

                else:

                    print(
                        "same prediction | "
                        f"confidence drop = "
                        f"{result['target_confidence_drop']:.4f}"
                    )

            except Exception as e:

                print(f"ERROR: {e}")

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TRANSFERABILITY SUMMARY")
    print("=" * 70)

    print(f"Successful evaluations: " f"{len(rows)}/{total_experiments}")

    print()

    if not rows:

        print("No successful evaluations.")

        return

    # --------------------------------------------------------
    # Summary for every model pair
    # --------------------------------------------------------

    for source_model, target_model in model_pairs:

        pair_rows = [
            row
            for row in rows
            if row["source_model"] == source_model
            and row["target_model"] == target_model
        ]

        if not pair_rows:
            continue

        # Number of changed predictions
        changed_count = sum(1 for row in pair_rows if row["prediction_changed"] is True)

        # Transfer rate
        transfer_rate = (changed_count / len(pair_rows)) * 100

        # Average target confidence drop
        average_confidence_drop = sum(
            float(row["target_confidence_drop"]) for row in pair_rows
        ) / len(pair_rows)

        print(
            f"{source_model:15} -> "
            f"{target_model:15} | "
            f"changed: "
            f"{changed_count}/"
            f"{len(pair_rows)} | "
            f"transfer rate: "
            f"{transfer_rate:.2f}% | "
            f"avg confidence drop: "
            f"{average_confidence_drop:.4f}"
        )

    print()
    print(f"CSV saved to:")

    print(OUTPUT_FILE)

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
