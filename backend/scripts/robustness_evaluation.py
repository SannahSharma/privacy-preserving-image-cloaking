"""Evaluate cloaking robustness after common upload-pipeline transformations.

Run from project root:
    python backend\\scripts\\robustness_evaluation.py

Default: 120 Imagenette test images x 3 source models x 4 transforms.
"""
import csv, io, sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
from app.cloak_engine import cloak_image, _get_model, _PREPROCESS

MODELS = ["resnet50", "mobilenetv3", "vit"]
METHOD = "fgsm"
PROTECTION_MODE = "standard"
STRENGTH = 1.0
MAX_IMAGES = 120
IMAGE_DIR = PROJECT_ROOT / "datasets" / "test_images"
OUTPUT_FILE = PROJECT_ROOT / "results" / "robustness_results.csv"


def predict(image, model_name):
    model = _get_model(model_name)
    x = _PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)
    conf, cls = torch.max(p, dim=1)
    return int(cls.item()), float(conf.item()), p


def jpeg_transform(image, quality):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_transforms(image):
    w, h = image.size
    resized = image.resize((max(32, int(w * 0.75)), max(32, int(h * 0.75))), Image.Resampling.LANCZOS)
    resized = resized.resize((w, h), Image.Resampling.LANCZOS)
    return {
        "none": image,
        "jpeg_q90": jpeg_transform(image, 90),
        "jpeg_q70": jpeg_transform(image, 70),
        "resize_75pct": resized,
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:MAX_IMAGES]
    pairs = []
    for source in MODELS:
        for target in MODELS:
            if source != target:
                pairs.append((source, target))

    fields = ["image", "source_model", "target_model", "transform", "prediction_changed", "target_original_confidence", "target_cloaked_original_class_confidence", "target_confidence_drop"]
    rows = []
    print(f"Images: {len(paths)} | experiments: {len(paths)*len(pairs)*4}")

    for idx, path in enumerate(paths, 1):
        print(f"[{idx}/{len(paths)}] {path.name}", flush=True)
        image = Image.open(path).convert("RGB")
        for source, target in pairs:
            source_result = cloak_image(image, strength=STRENGTH, method=METHOD, model_name=source, protection_mode=PROTECTION_MODE)
            if "cloaked_image" in source_result:
                cloaked = source_result["cloaked_image"]
            else:
                cloaked = Image.open(io.BytesIO(source_result["cloaked_image_bytes"])).convert("RGB")
            target_cls, target_conf, target_probs = predict(image, target)
            for name, transformed in make_transforms(cloaked).items():
                cls2, conf2, probs2 = predict(transformed, target)
                original_class_conf_after = float(probs2[0, target_cls].item())
                rows.append({
                    "image": path.name,
                    "source_model": source,
                    "target_model": target,
                    "transform": name,
                    "prediction_changed": cls2 != target_cls,
                    "target_original_confidence": target_conf,
                    "target_cloaked_original_class_confidence": original_class_conf_after,
                    "target_confidence_drop": target_conf - original_class_conf_after,
                })

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print("\\nROBUSTNESS SUMMARY")
    print("=" * 70)
    for source, target in pairs:
        print(f"\\n{source} -> {target}")
        subset = [r for r in rows if r["source_model"] == source and r["target_model"] == target]
        for t in ["none", "jpeg_q90", "jpeg_q70", "resize_75pct"]:
            x = [r for r in subset if r["transform"] == t]
            changed = sum(bool(r["prediction_changed"]) for r in x)
            avg = sum(float(r["target_confidence_drop"]) for r in x) / len(x)
            print(f"  {t:14} changed: {changed:3}/{len(x)} | rate: {100*changed/len(x):6.2f}% | avg drop: {avg:.4f}")
    print(f"\\nCSV saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
