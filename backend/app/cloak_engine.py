"""Adversarial cloaking engine used by the FastAPI backend.

Supports ResNet-50, MobileNetV3-Large and ViT-B/16 with FGSM/PGD.
The original image resolution is preserved by generating the perturbation at
224x224 and resizing only the perturbation back onto the original image.
"""
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_MODELS = {}
_ACTIVATIONS = {}


def _build_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        layer = model.layer4
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        layer = model.features[-1]
    elif model_name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        layer = model.encoder.ln
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")
    model.eval()

    def hook(_module, _inputs, output):
        _ACTIVATIONS[model_name] = output.detach()
    layer.register_forward_hook(hook)
    return model


def _get_model(model_name):
    if model_name not in _MODELS:
        _MODELS[model_name] = _build_model(model_name)
    return _MODELS[model_name]


_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN.flatten().tolist(), std=STD.flatten().tolist()),
])


def _predict(model, tensor):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, predicted = probs.max(1)
    return predicted.item(), confidence.item()


def _fgsm(model, x, label, epsilon):
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), label)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return (x + epsilon * x.grad.sign()).detach()


def _pgd(model, x, label, epsilon, alpha, steps=10):
    original = x.detach().clone()
    adv = original.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), label)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adv = adv.detach() + alpha * adv.grad.sign()
        delta = torch.clamp(adv - original, -epsilon, epsilon)
        adv = (original + delta).detach()
    return adv


def _activation_grid(model_name):
    act = _ACTIVATIONS.get(model_name)
    if act is None:
        return [[0.0] * 7 for _ in range(7)]

    if model_name == "vit":
        # ViT encoder output: [1, 197, 768]. Drop CLS token and reshape
        # 196 patch tokens to 14x14, then resize to the common 7x7 contract.
        grid = act[:, 1:, :].norm(dim=-1).reshape(1, 1, 14, 14)
        grid = F.interpolate(grid, size=(7, 7), mode="bilinear", align_corners=False)[0, 0]
    else:
        if act.ndim == 4:
            grid = act[0].mean(dim=0)
        else:
            grid = act.mean(dim=-1).reshape(1, 1, -1)
            grid = F.interpolate(grid, size=49, mode="linear", align_corners=False).reshape(7, 7)

    grid_min, grid_max = grid.min(), grid.max()
    return ((grid - grid_min) / (grid_max - grid_min + 1e-8)).tolist()


def cloak_image(image: Image.Image, strength: float, method: str = "fgsm",
                model_name: str = "resnet50", protection_mode: str = "standard"):
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")
    if method not in {"fgsm", "pgd"}:
        raise ValueError("method must be 'fgsm' or 'pgd'")
    if protection_mode not in {"standard", "strong"}:
        raise ValueError("protection_mode must be 'standard' or 'strong'")

    model = _get_model(model_name)
    # Standard matches the original project contract: max epsilon = 0.06.
    # Strong extends the bounded budget while keeping the user-facing slider 0-1.
    epsilon_max = 0.06 if protection_mode == "standard" else 0.125
    epsilon = strength * epsilon_max
    image = image.convert("RGB")
    original_width, original_height = image.size

    x = _PREPROCESS(image).unsqueeze(0)
    original_class, original_confidence = _predict(model, x)
    label = torch.tensor([original_class])

    if method == "pgd":
        adv_small = _pgd(model, x, label, epsilon, epsilon / 4, steps=10)
    else:
        adv_small = _fgsm(model, x, label, epsilon)

    delta_small = (adv_small - x) * STD
    delta_full = F.interpolate(delta_small, size=(original_height, original_width),
                               mode="bilinear", align_corners=False)
    original_full = transforms.ToTensor()(image).unsqueeze(0)
    cloaked = torch.clamp(original_full + delta_full, 0, 1)
    cloaked_image = Image.fromarray(
        (cloaked.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
    )

    recheck = _PREPROCESS(cloaked_image).unsqueeze(0)
    cloaked_class, cloaked_confidence = _predict(model, recheck)

    # Re-check probability of the original class; this is more informative than
    # only looking at the new top-1 class.
    with torch.no_grad():
        probs = torch.softmax(model(recheck), dim=1)
        original_class_conf_after = probs[0, original_class].item()

    return {
        "cloaked_image": cloaked_image,
        "original_class": original_class,
        "original_confidence": original_confidence,
        "cloaked_class": cloaked_class,
        "cloaked_confidence": cloaked_confidence,
        "original_class_confidence_after": original_class_conf_after,
        "confidence_drop": original_confidence - original_class_conf_after,
        "misclassified": original_class != cloaked_class,
        "epsilon": epsilon,
        "activation_map": _activation_grid(model_name),
    }
