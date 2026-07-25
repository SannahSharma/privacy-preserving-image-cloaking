# Track A — Adversarial Cloaking Engine

## What this does
`cloak_image()` takes an uploaded image and a "cloaking strength" (0–1) and returns
a visually-unchanged image that fools a ResNet-50 image classifier, plus the
confidence scores and activation data needed for the frontend's live visualization.

## How to use it

```python
from PIL import Image
from cloak_engine import cloak_image

image = Image.open("some_upload.jpg").convert("RGB")

result = cloak_image(image, strength=0.5, method="fgsm")  # or method="pgd"
```

## Function signature

```python
cloak_image(image: PIL.Image.Image, strength: float, method: str = "fgsm") -> dict
```

**Inputs**
- `image`: a PIL Image, already opened and converted to RGB (`.convert("RGB")`).
  Any size or aspect ratio is supported — output keeps the exact same dimensions.
- `strength`: float between `0.0` (no change) and `1.0` (maximum protection).
  Maps internally to an epsilon value; higher = more disruption to the classifier.
- `method`: `"fgsm"` (default, fast, one-step) or `"pgd"` (slower, iterative,
  much stronger — see benchmark notes below).

**Returns** a dict with these keys:

| Key | Type | Meaning |
|---|---|---|
| `cloaked_image` | `PIL.Image.Image` | The protected image, same size as input. This is what gets sent back to the frontend/saved for download. |
| `original_class` | `int` | ImageNet class index the model predicted BEFORE cloaking. |
| `original_confidence` | `float` | Confidence (0–1) in that original prediction. |
| `cloaked_class` | `int` | ImageNet class index predicted AFTER cloaking. |
| `cloaked_confidence` | `float` | Confidence (0–1) in the new (usually wrong) prediction. |
| `activation_map` | `list[list[float]]` | 7x7 grid of floats (0–1), ResNet-50's `layer4` activation — feed this to the frontend Canvas overlay. |

## Example benchmark result (test image: 666x1000 portrait/object photo)

| Method | Original class/confidence | Cloaked class/confidence |
|---|---|---|
| FGSM | 281 / 0.139 | 287 / 0.128 (confidence dropped) |
| PGD  | 281 / 0.139 | 283 / 0.921 (confidently WRONG — stronger effect) |

## Known limitations / things to know before integrating

- **Performance**: FGSM is near-instant. PGD runs 10 gradient steps internally, so
  it is noticeably slower — good candidate to run in a background task or show a
  loading state on the frontend for PGD specifically.
- **Model loads once**: `_model` loads at import time, not per-call. Don't reload
  the module repeatedly per-request — import `cloak_image` once at app startup.
- **Only ResNet-50 is supported right now.** MobileNetV3/ViT were stretch goals
  not yet implemented — if needed, ping me before assuming they exist.
- **Resolution fix**: earlier versions of this function cropped output to 224x224.
  This is fixed as of the "Day 4: fix cloak_image to preserve original image
  resolution" commit — always pull latest before integrating.
- **Dependencies**: `torch`, `torchvision`, `pillow` (see `requirements.txt`).

## Files in this folder
- `cloak_engine.py` — the real deliverable, contains `cloak_image()`.
- `day1_load_model.py`, `day2_fgsm.py` — early development scripts, kept for
  history/reference, not used by the final function.
- `day4_multi_test.py` — test script proving the function works across multiple
  images, not just one.
- `test.jpg`, `test2.jpg`, `test3.jpg` — sample test images.