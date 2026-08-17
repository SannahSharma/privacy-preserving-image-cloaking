"""
batch_eval_epsilon8.py

Bypasses the strength slider entirely and tests a literature-standard
epsilon (~8/255 in real pixel terms, which is roughly 0.14 in the
model's normalized space) directly, to see whether misclassification
rate improves substantially beyond what the UI's strength=1.0 max
(epsilon=0.06) can produce. Reuses the existing small test_images set.

Does NOT modify cloak_engine.py -- only imports additional pieces from
it (the same internal attack functions cloak_image() already uses).
"""
import csv
import statistics as stats
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from cloak_engine import (
    _model,
    _preprocess,
    _STD,
    _get_prediction,
    _fgsm_attack,
    _pgd_attack,
    _full_tensor_to_image,
)

INPUT_DIR = Path("../datasets/test_images")
OUTPUT_CSV = Path("../datasets/results_epsilon8.csv")
EPSILON = 0.14  # ~8/255 in real pixel terms, well beyond the UI's max of 0.06
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


def cloak_image_custom_epsilon(image, epsilon, method="fgsm"):
    original_width, original_height = image.size
    input_tensor = _preprocess(image).unsqueeze(0)
    original_class, original_confidence = _get_prediction(input_tensor)
    true_label = torch.tensor([original_class])

    if method == "pgd":
        alpha = epsilon / 4
        cloaked_tensor_small = _pgd_attack(input_tensor, true_label, epsilon, alpha, steps=10)
    else:
        cloaked_tensor_small = _fgsm_attack(input_tensor, true_label, epsilon)

    delta_small = (cloaked_tensor_small - input_tensor) * _STD
    delta_full = F.interpolate(
        delta_small, size=(original_height, original_width),
        mode="bilinear", align_corners=False,
    )

    original_full_tensor = transforms.ToTensor()(image).unsqueeze(0)
    cloaked_full_tensor = torch.clamp(original_full_tensor + delta_full, 0, 1)
    cloaked_image = _full_tensor_to_image(cloaked_full_tensor)

    recheck_tensor = _preprocess(cloaked_image).unsqueeze(0)
    cloaked_class, cloaked_confidence = _get_prediction(recheck_tensor)

    with torch.no_grad():
        probs = torch.softmax(_model(recheck_tensor), dim=1)
    conf_in_original_after = probs[0, original_class].item()

    return {
        "cloaked_image": cloaked_image,
        "original_class": original_class,
        "original_confidence": original_confidence,
        "cloaked_class": cloaked_class,
        "cloaked_confidence": cloaked_confidence,
        "confidence_in_original_class_after": conf_in_original_after,
    }


def evaluate_image(path):
    rows = []
    image = Image.open(path).convert("RGB")
    width, height = image.size
    original_array = np.array(image)

    for method in ["fgsm", "pgd"]:
        start = time.time()
        result = cloak_image_custom_epsilon(image, EPSILON, method=method)
        elapsed = time.time() - start

        cloaked_array = np.array(result["cloaked_image"])
        score = ssim(original_array, cloaked_array, channel_axis=2)

        rows.append({
            "filename": path.name,
            "width": width,
            "height": height,
            "method": method,
            "original_class": result["original_class"],
            "original_confidence": result["original_confidence"],
            "cloaked_class": result["cloaked_class"],
            "cloaked_confidence": result["cloaked_confidence"],
            "confidence_in_original_class_after": result["confidence_in_original_class_after"],
            "ssim": score,
            "seconds": round(elapsed, 2),
        })
    return rows


def main():
    image_paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    print(f"Found {len(image_paths)} images in {INPUT_DIR}")
    print(f"Using epsilon={EPSILON} (~8/255 in real pixel terms)\n")

    all_rows = []
    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {path.name} ...", end=" ", flush=True)
        try:
            rows = evaluate_image(path)
            all_rows.extend(rows)
            print("done")
        except Exception as e:
            print(f"SKIPPED ({e})")

    if not all_rows:
        print("No results produced.")
        return

    fieldnames = list(all_rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_CSV}")

    print("\n===== SUMMARY =====")
    for method in ["fgsm", "pgd"]:
        method_rows = [r for r in all_rows if r["method"] == method]
        n = len(method_rows)
        flipped = sum(1 for r in method_rows if r["cloaked_class"] != r["original_class"])
        success_rate = flipped / n

        ssim_values = [r["ssim"] for r in method_rows]
        drops = [
            r["original_confidence"] - r["confidence_in_original_class_after"]
            for r in method_rows
        ]

        print(f"\n--- {method.upper()} (n={n}) ---")
        print(f"Misclassification rate: {success_rate:.1%}")
        print(f"Mean confidence drop:   {stats.mean(drops):.4f} (std {stats.pstdev(drops):.4f})")
        print(f"Mean SSIM:              {stats.mean(ssim_values):.4f} (std {stats.pstdev(ssim_values):.4f})")


if __name__ == "__main__":
    main()