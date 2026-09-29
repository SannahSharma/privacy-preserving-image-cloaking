"""Compare standard vs strong protection across the 120-image dataset."""
import csv, io, sys
from pathlib import Path
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.cloak_engine import cloak_image, _get_model, _PREPROCESS

MODELS = ["resnet50", "mobilenetv3", "vit"]
METHODS = ["fgsm", "pgd"]
MODES = ["standard", "strong"]
STRENGTH = 1.0
MAX_IMAGES = 120
IMAGE_DIR = PROJECT_ROOT / "datasets" / "test_images"
OUTPUT_FILE = PROJECT_ROOT / "results" / "protection_comparison.csv"


def predict(image, model_name):
    model = _get_model(model_name)
    x = _PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)
    c, k = torch.max(p, dim=1)
    return int(k.item()), float(c.item()), p


def main():
    paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:MAX_IMAGES]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "model", "method", "protection_mode", "epsilon", "prediction_changed", "confidence_drop", "original_confidence", "cloaked_original_class_confidence", "ssim"]
    rows = []
    total = len(paths) * len(MODELS) * len(METHODS) * len(MODES)
    print(f"Experiments: {total}")
    n = 0
    for path in paths:
        image = Image.open(path).convert("RGB")
        for model in MODELS:
            for method in METHODS:
                for mode in MODES:
                    n += 1; print(f"[{n}/{total}] {path.name} | {model} | {method} | {mode}", flush=True)
                    r = cloak_image(image, strength=STRENGTH, method=method, model_name=model, protection_mode=mode)
                    original = predict(image, model)
                    if "cloaked_image" in r:
                        cloaked = r["cloaked_image"]
                    else:
                        cloaked = Image.open(io.BytesIO(r["cloaked_image_bytes"])).convert("RGB")
                    _, _, probs = predict(cloaked, model)
                    original_class = original[0]
                    after_original = float(probs[0, original_class].item())
                    rows.append({
                        "image": path.name, "model": model, "method": method, "protection_mode": mode,
                        "epsilon": r["epsilon"], "prediction_changed": r["cloaked_class"] != original_class,
                        "confidence_drop": original[1] - after_original, "original_confidence": original[1],
                        "cloaked_original_class_confidence": after_original, "ssim": r.get("ssim_score", ""),
                    })
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("\\nPROTECTION COMPARISON")
    print("="*70)
    for model in MODELS:
        for method in METHODS:
            print(f"\\n{model} | {method}")
            for mode in MODES:
                x=[r for r in rows if r["model"]==model and r["method"]==method and r["protection_mode"]==mode]
                changed=sum(bool(r["prediction_changed"]) for r in x)
                drop=sum(float(r["confidence_drop"]) for r in x)/len(x)
                print(f"  {mode:8} changed: {changed}/{len(x)} ({100*changed/len(x):.2f}%) | avg drop: {drop:.4f}")
    print(f"\\nCSV saved to: {OUTPUT_FILE}")

if __name__ == "__main__": main()
