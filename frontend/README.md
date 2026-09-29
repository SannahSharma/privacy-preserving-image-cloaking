# Privacy-Preserving Image Cloaking — React + Vite Frontend

This frontend replaces the original single-file HTML page with a React + Vite dashboard for the current FastAPI backend.

## Requirements
- Node.js 18+
- Backend running at `http://127.0.0.1:8001`

## Run

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Backend URL

By default the app uses `http://127.0.0.1:8001`. To change it, create `.env.local`:

```text
VITE_API_BASE=http://127.0.0.1:8001
```

## Live features

- Backend health indicator
- Dynamic model discovery from `GET /models`
- ResNet-50, MobileNetV3 and ViT model selection when implemented by backend
- FGSM / PGD selection
- Standard / Strong protection mode
- Strength slider
- Image upload and preview
- Live `/cloak` API call
- Original vs cloaked predictions and confidence
- Epsilon, confidence drop, SSIM and processing time
- 7×7 activation-map visualization from the backend's 49 points
- Download cloaked PNG
- Research-result dashboard for transferability, robustness, face recognition and statistical evaluation

## Design

This build carries a redesigned visual identity — a darkroom/optics theme (safelight red accent, viewfinder corner brackets on upload and image panes, film-sprocket divider between the original/cloaked pair) — built on the same FastAPI endpoints and state logic as the original implementation. Fonts (Fraunces, Space Grotesk, IBM Plex Mono) load from Google Fonts at runtime.
