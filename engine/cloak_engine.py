import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Load the pretrained ResNet-50 model once, when this module is first imported.
# Reused by every call to cloak_image() — we don't want to reload it every time.
_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
_model.eval()

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# The exact mean/std used above — needed again later to convert normalized
# differences back into real pixel-scale differences.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Activation hook setup ---
_activation = {}


def _save_activation(name):
    """Returns a hook function that stores a layer's output into _activation[name]."""
    def hook(module, layer_input, layer_output):
        _activation[name] = layer_output.detach()
    return hook


_model.layer4.register_forward_hook(_save_activation("layer4"))


def _get_prediction(tensor):
    """Return (predicted_class, confidence) for a preprocessed 224x224 tensor."""
    with torch.no_grad():
        output = _model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    return predicted_class.item(), confidence.item()


def _fgsm_attack(input_tensor, true_label, epsilon):
    """Nudge every pixel slightly to increase the model's loss, bounded by epsilon. One step."""
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad = True

    output = _model(input_tensor)
    loss = F.cross_entropy(output, true_label)
    _model.zero_grad()
    loss.backward()

    perturbation = epsilon * input_tensor.grad.sign()
    cloaked = input_tensor + perturbation
    cloaked = torch.clamp(cloaked, input_tensor.min(), input_tensor.max())
    return cloaked.detach()


def _pgd_attack(input_tensor, true_label, epsilon, alpha, steps):
    """
    Same idea as FGSM, but repeated in small steps ('alpha' sized), each time
    projecting the total change back into the epsilon boundary. Stronger, slower.
    """
    original = input_tensor.clone().detach()
    cloaked = input_tensor.clone().detach()

    for _ in range(steps):
        cloaked.requires_grad = True
        output = _model(cloaked)
        loss = F.cross_entropy(output, true_label)
        _model.zero_grad()
        loss.backward()

        cloaked = cloaked + alpha * cloaked.grad.sign()
        perturbation = torch.clamp(cloaked - original, -epsilon, epsilon)
        cloaked = torch.clamp(original + perturbation,
                               original.min(), original.max()).detach()

    return cloaked


def _full_tensor_to_image(tensor):
    """Convert a raw 0-1 pixel-scale tensor (no normalization applied) into a PIL image."""
    tensor = tensor.squeeze(0).clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")  # CHW -> HWC, scale to 0-255
    return Image.fromarray(array)


def _activation_to_grid(activation_tensor):
    """
    Convert layer4's raw activation tensor into a small 2D grid of numbers (0-1),
    simple enough for the frontend to draw as a heatmap/overlay.
    """
    grid = activation_tensor[0].mean(dim=0)  # [2048, 7, 7] -> [7, 7]
    grid_min, grid_max = grid.min(), grid.max()
    grid = (grid - grid_min) / (grid_max - grid_min + 1e-8)
    return grid.tolist()


def cloak_image(image: Image.Image, strength: float, method: str = "fgsm"):
    """
    Main engine function — the contract Person B (backend) will call.

    Args:
        image: a PIL Image (already opened, RGB), ANY size
        strength: float from 0.0 to 1.0, controls how aggressive the cloaking is
        method: "fgsm" (fast, default) or "pgd" (stronger, slower)

    Returns:
        dict with:
            cloaked_image: PIL Image, SAME size as the input image
            original_class: int, predicted class index before cloaking
            original_confidence: float
            cloaked_class: int, predicted class index after cloaking (re-checked on the final image)
            cloaked_confidence: float
            activation_map: 7x7 nested list of floats (0-1)
    """
    epsilon = strength * 0.06
    original_width, original_height = image.size  # PIL gives (width, height)

    # Step 1: build the small 224x224 view the model actually needs
    input_tensor = _preprocess(image).unsqueeze(0)

    original_class, original_confidence = _get_prediction(input_tensor)
    true_label = torch.tensor([original_class])

    # Step 2: run the attack on that small 224x224 view
    if method == "pgd":
        alpha = epsilon / 4
        cloaked_tensor_small = _pgd_attack(input_tensor, true_label, epsilon, alpha, steps=10)
    else:
        cloaked_tensor_small = _fgsm_attack(input_tensor, true_label, epsilon)

    # Step 3: extract just the CHANGE the attack made, converted from normalized
    # space back into real pixel-scale space (multiplying by std undoes Normalize's scaling).
    delta_small = (cloaked_tensor_small - input_tensor) * _STD  # shape [1, 3, 224, 224]

    # Step 4: stretch that small perturbation up to the original image's real size
    delta_full = F.interpolate(
        delta_small,
        size=(original_height, original_width),  # interpolate wants (H, W)
        mode="bilinear",
        align_corners=False,
    )

    # Step 5: add that perturbation onto the ORIGINAL, full-resolution image
    original_full_tensor = transforms.ToTensor()(image).unsqueeze(0)  # raw 0-1 pixels, full size
    cloaked_full_tensor = torch.clamp(original_full_tensor + delta_full, 0, 1)
    cloaked_image = _full_tensor_to_image(cloaked_full_tensor)

    # Step 6: re-check the prediction on the ACTUAL final image (same way a real
    # downstream classifier would: it resizes/crops whatever it receives first).
    recheck_tensor = _preprocess(cloaked_image).unsqueeze(0)
    cloaked_class, cloaked_confidence = _get_prediction(recheck_tensor)

    activation_map = _activation_to_grid(_activation["layer4"])

    return {
        "cloaked_image": cloaked_image,
        "original_class": original_class,
        "original_confidence": original_confidence,
        "cloaked_class": cloaked_class,
        "cloaked_confidence": cloaked_confidence,
        "activation_map": activation_map,
    }


# Quick manual test when running this file directly (not when imported by teammates)
if __name__ == "__main__":
    img = Image.open("test.jpg").convert("RGB")
    print("Original image size:", img.size)

    result = cloak_image(img, strength=0.5, method="fgsm")

    print("Cloaked image size:", result["cloaked_image"].size)
    print("Original class:", result["original_class"], "confidence:", result["original_confidence"])
    print("Cloaked class:", result["cloaked_class"], "confidence:", result["cloaked_confidence"])

    result["cloaked_image"].save("cloaked_output_fgsm.jpg")