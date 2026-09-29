# privacy-preserving-image-cloaking

# Privacy-Preserving Image Cloaking using Adversarial Machine Learning

## Data Source

The raw test images used for the batch evaluation (`datasets/test_images/` and `datasets/test_images_large/`) come from **Imagenette**, a small, free, permissively-licensed subset of ImageNet created by Jeremy Howard (fast.ai). No login or account is required to download it.

* **160px variant (`test_images/`):** [Imagenette 160 px on Kaggle](https://www.kaggle.com/datasets/jhoward/imagenette-160-px)
* **320px variant (`test_images_large/`):** [imagenette2-320 on Kaggle](https://www.kaggle.com/datasets/xbinchen/imagenette2-320)
* **Original/canonical source:** [fastai/imagenette on GitHub](https://github.com/fastai/imagenette)

The images are downloaded and sampled automatically by:

```text
engine/download_images.py
engine/download_large_images.py
```

---

## Overview

**Privacy-Preserving Image Cloaking** is a full-stack machine-learning system for generating adversarially cloaked images using established adversarial attack techniques such as **FGSM** and **PGD**.

The system allows users to upload an image, select an AI model and cloaking method, generate a cloaked image, and inspect the effect through prediction confidence, SSIM, perturbation strength, processing time, and model activation visualization.

The project also includes research evaluation modules for **transferability, post-processing robustness, face-recognition analysis, and statistical analysis**.

---

## Objectives

The main objectives of the project are:

1. Build an end-to-end image-cloaking system.
2. Implement adversarial image perturbation using FGSM and PGD.
3. Support multiple image-classification architectures.
4. Preserve the visual similarity of the original image.
5. Measure changes in model confidence and prediction.
6. Visualize internal model activations.
7. Evaluate adversarial transferability between models.
8. Evaluate the effect of image post-processing on cloaked images.
9. Perform face-recognition analysis on cloaked images.
10. Provide a web-based interface for demonstrating the complete system.

---

## Key Features

### Image Cloaking

Users can:

* Upload an image.
* Select an image-classification model.
* Select an adversarial attack.
* Adjust cloaking strength.
* Choose standard or strong protection.
* Generate a cloaked image.
* Compare the original and cloaked images.

### Supported Models

* ResNet-50
* MobileNetV3-Large
* Vision Transformer (ViT-B/16)

### Supported Attack Methods

* FGSM
* PGD

### Metrics

The system displays:

* Original prediction
* Cloaked prediction
* Original confidence
* Cloaked confidence
* Confidence drop
* Epsilon
* SSIM
* Processing time
* Activation map

### Research Evaluation

The project also includes:

* Transferability evaluation
* Robustness evaluation
* Face-recognition evaluation
* Statistical analysis
* Research result CSV files
* Consolidated Excel research workbook

---

## System Architecture

```text
┌──────────────────────────────────────┐
│          React + Vite Frontend       │
│                                      │
│  Upload • Controls • Results • UI    │
└──────────────────┬───────────────────┘
                   │ HTTP
                   ↓
┌──────────────────────────────────────┐
│            FastAPI Backend            │
│                                      │
│  /health                             │
│  /models                             │
│  /cloak                              │
│  /evaluate/transferability           │
└──────────────────┬───────────────────┘
                   │
                   ↓
┌──────────────────────────────────────┐
│          PyTorch ML Engine            │
│                                      │
│ ResNet-50 • MobileNetV3 • ViT        │
│ FGSM • PGD                            │
└──────────────────┬───────────────────┘
                   │
                   ↓
┌──────────────────────────────────────┐
│        Evaluation & Research         │
│                                      │
│ Transferability • Robustness         │
│ Face Recognition • Statistics       │
└──────────────────────────────────────┘
```

---

## Machine Learning Models

### ResNet-50

ResNet-50 is a deep **Convolutional Neural Network (CNN)** that uses residual/skip connections.

### MobileNetV3

MobileNetV3 is a lightweight CNN architecture designed for efficient image classification.

### Vision Transformer

The project uses **ViT-B/16**. Unlike CNN-based models, Vision Transformer processes an image using image patches and Transformer-based processing.

---

## Adversarial Attack Methods

### FGSM

**FGSM — Fast Gradient Sign Method**

FGSM generates an adversarial perturbation using the gradient of the model's loss with respect to the input image.

FGSM is primarily a **one-step attack**.

### PGD

**PGD — Projected Gradient Descent**

PGD applies multiple smaller gradient-based updates while keeping the perturbation within the permitted epsilon bound.

---

## How to Run

### Backend

```powershell
cd C:\MajorProject\privacy-preserving-image-cloaking\backend
..\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8001
```

### Frontend

Open another PowerShell terminal:

```powershell
cd C:\MajorProject\privacy-preserving-image-cloaking\frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

## Research Results

The completed research outputs are stored in:

```text
results/
├── face_recognition_results.csv
├── final_research_results.xlsx
├── robustness_results.csv
├── statistical_analysis.csv
└── transferability_results.csv
```

The research evaluation includes:

* Transferability between ResNet-50, MobileNetV3, and ViT
* JPEG compression and resizing robustness
* Face-recognition evaluation
* Statistical comparison of attack configurations

---

## Technology Stack

### Frontend

* React
* Vite
* JavaScript
* HTML
* CSS

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* PyTorch
* Torchvision
* ResNet-50
* MobileNetV3-Large
* Vision Transformer
* FGSM
* PGD

### Research & Evaluation

* Pandas
* OpenPyXL
* NumPy
* Pillow
* scikit-image

### Development

* Git
* GitHub
* VS Code

