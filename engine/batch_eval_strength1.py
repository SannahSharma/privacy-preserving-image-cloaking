"""
batch_eval_strength1.py

Same as batch_eval.py, but at STRENGTH = 1.0 (epsilon = 0.06, double the
previous 0.03) instead of 0.5, to test whether a larger perturbation
budget produces meaningfully higher misclassification rates. Reuses the
existing small test_images set.
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from cloak_engine import cloak_image, _model, _preprocess

INPUT_DIR = Path("../datasets/test_images")
OUTPUT_CSV = Path("../datasets/results_strength1.csv")
STRENGTH = 1.0
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


def confidence_in_class(pil_image, class_index):
    tensor = _preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
    return probs[0, class_index].item()


def evaluate_image(path):
    rows = []
    image = Image.open(path).convert("RGB")
    width, height = image.size
    original_array = np.array(image)

    for method in ["fgsm", "pgd"]:
        start = time.time()
        result = cloak_image(image, strength=STRENGTH, method=method)
        elapsed = time.time() - start

        cloaked_array = np.array(result["cloaked_image"])
        score = ssim(original_array, cloaked_array, channel_axis=2)

        conf_in_original_after = confidence_in_class(
            result["cloaked_image"], result["original_class"]
        )

        rows.append({
            "filename": path.name,
            "width": width,
            "height": height,
            "method": method,
            "original_class": result["original_class"],
            "original_confidence": result["original_confidence"],
            "cloaked_class": result["cloaked_class"],
            "cloaked_confidence": result["cloaked_confidence"],
            "confidence_in_original_class_after": conf_in_original_after,
            "ssim": score,
            "seconds": round(elapsed, 2),
        })
    return rows


def print_summary(rows):
    import statistics as stats
    print("\n===== SUMMARY (mean / std / min / max) =====")
    for method in ["fgsm", "pgd"]:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        print(f"\n--- {method.upper()} (n={len(method_rows)}) ---")
        for field in ["ssim", "confidence_in_original_class_after", "cloaked_confidence"]:
            values = [r[field] for r in method_rows]
            mean = stats.mean(values)
            std = stats.pstdev(values) if len(values) > 1 else 0.0
            print(f"{field:38s} mean={mean:.4f}  std={std:.4f}  min={min(values):.4f}  max={max(values):.4f}")


def main():
    image_paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    print(f"Found {len(image_paths)} images in {INPUT_DIR}")

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

    print_summary(all_rows)


if __name__ == "__main__":
    main()