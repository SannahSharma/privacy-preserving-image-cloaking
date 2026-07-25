# Image Cloaking API (Backend — Person B's track)

**Status: LIVE and VERIFIED WORKING** (2026-07-22). Wired to Person A's real
`cloak_image()` in `app/cloak_engine.py`, confirmed end-to-end with a real
image (tiger photo): server correctly loaded ResNet-50, classified the
original image correctly (`"tiger"`, real confidence score), returned a
cloaked version, and computed a real SSIM score (0.996 — well above the
project's >0.95 target).

## For Person C (frontend) — what you need to know

### Base URL
Running locally at `http://127.0.0.1:8000` for now. Will share the Render
URL once deployed.

### The request you need to send — `POST /cloak`
`multipart/form-data` with these fields:

| Field | Type | Notes |
|---|---|---|
| `image` | file | png/jpg/webp, max 10MB. **Set the correct MIME type explicitly** when uploading (`image/png`, `image/jpeg`, or `image/webp`) — some HTTP clients guess wrong and get rejected with a 415. |
| `strength` | float, 0.0-1.0 | from your slider |
| `model_name` | string | send `"resnet50"` — it's the only one that works right now (see note below) |
| `method` | string | **new field, not in the original plan** — send `"fgsm"` (fast) or `"pgd"` (slower, stronger effect). See the FGSM vs PGD note below — you'll probably want this as a toggle, not just the strength slider. |

### The response you'll get back — `200 OK`
```json
{
  "cloaked_image_base64": "...",
  "model_used": "resnet50",
  "method_used": "fgsm",
  "strength": 0.5,
  "original_predictions": [{"label": "tiger", "confidence": 0.34}],
  "cloaked_predictions": [{"label": "tiger", "confidence": 0.33}],
  "activation_map": [{"x": 0.07, "y": 0.07, "intensity": 0.0}, ...],
  "ssim_score": 0.996,
  "processing_time_ms": 15.0
}
```
- `cloaked_image_base64`: decode this and display/download it — it's a real PNG.
- `activation_map`: a flat list of 49 points (`{x, y, intensity}`, all 0-1 range)
  for your Canvas overlay — this is a flattened 7x7 grid from the model's
  internal layer, so expect a fairly blocky/low-res look, not a smooth heatmap.
- `original_predictions` / `cloaked_predictions`: currently just ONE
  prediction each (not top-k), since that's what the engine returns today.

### Real behavior to design around — FGSM vs PGD
This came up in testing: at moderate strength, `fgsm` sometimes barely moves
the prediction (confidence drops slightly, label may not even change) —
this is expected, not a bug. `pgd` is much more likely to flip the label
and tank the confidence, but it's noticeably slower (multiple gradient
steps internally). Practical suggestion: give the user an explicit
FGSM/PGD choice (not just a strength slider), and show a loading state
for PGD specifically.

### Errors to handle
| Code | Meaning |
|---|---|
| `422` | bad `strength`/`method`/`model_name` value |
| `415` | wrong/missing image content-type |
| `413` | image over 10MB |
| `501` | valid `model_name` but not implemented yet (`mobilenetv3`/`vit`) |
| `500` | engine crashed unexpectedly |

### Other endpoints
- `GET /health` → `{"status": "ok", "engine_mode": "live"}`
- `GET /models` → list of models with an `"implemented"` flag — use this to
  gray out mobilenetv3/vit in your model picker until Person A builds them.

### CORS
Wide open (`allow_origins=["*"]`) for now since there's no deployed frontend
origin yet. Will tighten to your Vercel URL once you deploy — just flag me
when that URL exists.

---

## For Person A (engine) — status and open items

Your `cloak_image()` is fully wired in and confirmed working against a real
image end-to-end (tiger photo, correct classification, SSIM 0.996). Nothing
needed from you right now for what's already built. A few things flagged
for awareness, not action:

- **Only resnet50 is exposed in the API right now**, matching what's
  actually implemented. If/when you build mobilenetv3 or vit, ping me and
  I'll wire them into `engine_interface.py` — should be a small change as
  long as they return the same shape (`cloaked_image`, `*_class`,
  `*_confidence`, `activation_map`).
- **FGSM vs PGD confidence swing**: at strength 0.5, FGSM barely moved the
  tiger's confidence (0.339 → 0.334, same label) in my test. That matches
  your own README's example (FGSM caused only a partial drop; PGD flipped
  the label confidently). Not reporting this as a bug — just confirming
  I saw the same pattern you documented, so nothing looks broken.
- **Face-recognition benchmark (LFW/CelebA)** from the original project
  proposal isn't tested yet — that's still open if it's still in scope.

## Run it locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs

Note: on first run, torchvision downloads ResNet-50's pretrained weights
(~100MB) from `download.pytorch.org` — needs real internet access, won't
work in network-restricted environments, but works fine on Render, a
normal laptop, or any standard cloud host.

If you're on Python 3.13, use unpinned versions of torch/torchvision (see
`requirements.txt`) — the exact pinned versions originally used don't have
Python 3.13 wheels yet.

## What changed from the original mock contract

Person A's real engine differed from the initially assumed contract —
all absorbed inside `app/engine_interface.py` so nothing else had to change:

| Assumed (mock) | Actual (Person A's real engine) |
|---|---|
| `apply_cloak(bytes, strength, model_name)` | `cloak_image(PIL.Image, strength, method)` |
| model_name selects architecture (resnet50/mobilenetv3/vit) | only resnet50 is implemented; `method` (fgsm/pgd) is the real second axis |
| predictions returned as labeled lists directly | engine returns raw ImageNet class indices - mapped to labels here via `ResNet50_Weights.DEFAULT.meta["categories"]` |
| `ssim_score` returned by the engine | engine doesn't compute this - backend computes it with `scikit-image` |
| `activation_map` as flat `{x,y,intensity}` points | engine returns a 7x7 grid - flattened into 49 points here so the frontend contract didn't need to change |

## How this connects to Person A's engine

All engine calls go through **one function**: `apply_cloak()` in
`app/engine_interface.py`. It imports Person A's real `cloak_image()` from
`app/cloak_engine.py`, converts bytes<->PIL Image, maps class indices to
labels, flattens the activation grid, and computes SSIM. If Person A ships
updates to `cloak_engine.py` (new models, bug fixes), drop the new file in
- `engine_interface.py` only needs changes if the function's signature or
return keys change.

## Supabase (optional, cut-first per the build plan)

`app/storage.py` only activates if `SUPABASE_URL` and `SUPABASE_KEY` env
vars are set. If missing, storage silently no-ops - `/cloak` still works
fully without it.

## Deployment (Render)

The included `Dockerfile` reads `$PORT` (Render sets this automatically).
First boot will be slower than usual while ResNet-50's weights download
and cache - this is normal and only happens once per instance.
