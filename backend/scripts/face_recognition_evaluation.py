"""
Face-recognition privacy evaluation using LFW + FaceNet.

Pipeline:

    LFW face image
          |
          v
    FaceNet embedding
          |
          v
    Gallery matching
          |
          v
    Original vs Cloaked comparison

Requirements:
    pip install facenet-pytorch scikit-learn pandas

Run from project root:
    python backend\\scripts\\face_recognition_evaluation.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from facenet_pytorch import InceptionResnetV1

# ================================================================
# PROJECT PATH
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "backend"))

from app.cloak_engine import cloak_image

# ================================================================
# CONFIGURATION
# ================================================================

MAX_PROBES = 120

STRENGTH = 1.0
METHOD = "fgsm"
PROTECTION_MODE = "standard"

OUT = ROOT / "results" / "face_recognition_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# LFW -> PIL IMAGE
# ================================================================


def lfw_to_pil(image_array):
    """
    Convert sklearn LFW image data into a PIL RGB image.

    fetch_lfw_people() returns normalized floating-point
    image values approximately in the range [0, 1].
    """

    image_array = np.asarray(image_array, dtype=np.float32)

    # Convert [0,1] -> [0,255]
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(image_array).convert("RGB")


# ================================================================
# FACE EMBEDDING
# ================================================================


def embed(model, image):
    """
    Generate a normalized FaceNet embedding.

    LFW images are already face-cropped/funneled,
    so MTCNN is not required here.
    """

    image = image.convert("RGB")

    # FaceNet expects 160x160
    image = image.resize((160, 160))

    image_array = np.asarray(image, dtype=np.float32)

    # [0,255] -> [0,1]
    image_array = image_array / 255.0

    x = torch.from_numpy(image_array)

    # HWC -> CHW
    x = x.permute(2, 0, 1)

    # FaceNet normalization
    x = (x - 0.5) / 0.5

    # Add batch dimension
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        embedding = model(x)

    embedding = embedding.cpu().numpy()[0]

    # L2 normalization
    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

    return embedding


# ================================================================
# COSINE SIMILARITY
# ================================================================


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between
    two face embeddings.
    """

    denominator = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12

    return float(np.dot(a, b) / denominator)


# ================================================================
# MAIN
# ================================================================


def main():

    # ============================================================
    # LOAD LFW
    # ============================================================

    print("Downloading/loading LFW dataset if necessary...")

    data = fetch_lfw_people(
        min_faces_per_person=2, resize=0.5, color=True, funneled=True
    )

    names = data.target_names
    images = data.images
    targets = data.target

    print(f"LFW images loaded: {len(images)}")

    print(f"Identities: {len(names)}")

    # ============================================================
    # GROUP IMAGES BY PERSON
    # ============================================================

    grouped = defaultdict(list)

    for index, target in enumerate(targets):

        grouped[int(target)].append(index)

    usable = [indices for indices in grouped.values() if len(indices) >= 2]

    if not usable:

        raise RuntimeError("No identities with at least two images found.")

    print(f"Identities with >=2 images: " f"{len(usable)}")

    # ============================================================
    # LOAD FACENET
    # ============================================================

    print("Loading FaceNet recognition model...")

    recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    # ============================================================
    # BUILD GALLERY
    #
    # First image of each person = gallery image.
    #
    # Remaining images = probe images.
    # ============================================================

    print("Building face-recognition gallery...")

    gallery_embeddings = {}

    gallery_count = 0

    for indices in usable:

        gallery_index = indices[0]

        identity = int(targets[gallery_index])

        gallery_image = lfw_to_pil(images[gallery_index])

        gallery_embedding = embed(recognizer, gallery_image)

        gallery_embeddings[identity] = gallery_embedding

        gallery_count += 1

    print(f"Gallery identities: " f"{gallery_count}")

    if gallery_count == 0:

        raise RuntimeError("Gallery is empty.")

    # ============================================================
    # PROCESS PROBES
    # ============================================================

    rows = []

    probes = 0

    skipped = 0

    for indices in usable:

        identity = int(targets[indices[0]])

        # --------------------------------------------------------
        # Remaining images are probes
        # --------------------------------------------------------

        for probe_index in indices[1:]:

            if probes >= MAX_PROBES:

                break

            # ----------------------------------------------------
            # Original LFW image
            # ----------------------------------------------------

            original = lfw_to_pil(images[probe_index])

            # ----------------------------------------------------
            # FaceNet embedding - original
            # ----------------------------------------------------

            original_embedding = embed(recognizer, original)

            # ----------------------------------------------------
            # Cloak image
            # ----------------------------------------------------

            cloaked_result = cloak_image(
                original,
                strength=STRENGTH,
                method=METHOD,
                model_name="resnet50",
                protection_mode=PROTECTION_MODE,
            )

            # ----------------------------------------------------
            # Get cloaked image
            # ----------------------------------------------------

            if "cloaked_image" in cloaked_result:

                cloaked = cloaked_result["cloaked_image"]

            elif "cloaked_image_bytes" in cloaked_result:

                cloaked = Image.open(
                    BytesIO(cloaked_result["cloaked_image_bytes"])
                ).convert("RGB")

            else:

                raise RuntimeError("cloak_image() did not return " "a cloaked image.")

            # ----------------------------------------------------
            # FaceNet embedding - cloaked
            # ----------------------------------------------------

            cloaked_embedding = embed(recognizer, cloaked)

            # ----------------------------------------------------
            # Compare original image against gallery
            # ----------------------------------------------------

            original_scores = {}

            for gallery_identity, gallery_embedding in gallery_embeddings.items():

                original_scores[gallery_identity] = cosine_similarity(
                    original_embedding, gallery_embedding
                )

            # ----------------------------------------------------
            # Compare cloaked image against gallery
            # ----------------------------------------------------

            cloaked_scores = {}

            for gallery_identity, gallery_embedding in gallery_embeddings.items():

                cloaked_scores[gallery_identity] = cosine_similarity(
                    cloaked_embedding, gallery_embedding
                )

            # ----------------------------------------------------
            # Determine predictions
            # ----------------------------------------------------

            original_prediction = max(original_scores, key=original_scores.get)

            cloaked_prediction = max(cloaked_scores, key=cloaked_scores.get)

            # ----------------------------------------------------
            # Similarity to TRUE identity
            # ----------------------------------------------------

            original_true_similarity = original_scores[identity]

            cloaked_true_similarity = cloaked_scores[identity]

            similarity_drop = original_true_similarity - cloaked_true_similarity

            # ----------------------------------------------------
            # Save row
            # ----------------------------------------------------

            rows.append(
                {
                    "identity": names[identity],
                    "original_recognized_correctly": original_prediction == identity,
                    "original_top_identity": names[original_prediction],
                    "cloaked_top_identity": names[cloaked_prediction],
                    "identity_changed": cloaked_prediction != original_prediction,
                    "original_similarity_to_true": original_true_similarity,
                    "cloaked_similarity_to_true": cloaked_true_similarity,
                    "similarity_drop": similarity_drop,
                }
            )

            probes += 1

        print(f"Processed " f"{probes}/{MAX_PROBES}", flush=True)

        if probes >= MAX_PROBES:

            break

    # ============================================================
    # CHECK RESULTS
    # ============================================================

    if not rows:

        raise RuntimeError("No face-recognition results generated.")

    # ============================================================
    # SAVE CSV
    # ============================================================

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))

        writer.writeheader()

        writer.writerows(rows)

    # ============================================================
    # CALCULATE SUMMARY
    # ============================================================

    valid = [row for row in rows if row["original_recognized_correctly"]]

    changed = sum(bool(row["identity_changed"]) for row in valid)

    if valid:

        average_similarity_drop = sum(
            float(row["similarity_drop"]) for row in valid
        ) / len(valid)

    else:

        average_similarity_drop = 0.0

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print()

    print("FACE-RECOGNITION SUMMARY")

    print("=" * 70)

    print(f"Total probes: " f"{len(rows)}")

    print(f"Baseline correctly recognized: " f"{len(valid)}/{len(rows)}")

    if valid:

        print(
            f"Identity changed after cloaking: "
            f"{changed}/{len(valid)} "
            f"({100 * changed / len(valid):.2f}%)"
        )

    else:

        print("Identity changed after cloaking: " "No valid baseline probes")

    print(
        f"Average similarity drop to "
        f"true identity: "
        f"{average_similarity_drop:.6f}"
    )

    print(f"Skipped probes: " f"{skipped}")

    print(f"CSV saved to: " f"{OUT}")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()
