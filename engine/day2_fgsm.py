import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Load the same pretrained ResNet-50 model as Day 1
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

img = Image.open('test.jpg').convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)


def get_prediction(model, tensor):
    """Run the model on a tensor and return (predicted_class, confidence)."""
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    return predicted_class.item(), confidence.item()


def get_confidence_for_class(model, tensor, class_index):
    """Run the model on a tensor and return its confidence for one specific class."""
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    return probs[0, class_index].item()


def fgsm_attack(model, input_tensor, true_label, epsilon):
    """Nudge every pixel slightly to increase the model's loss, bounded by epsilon."""
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad = True

    output = model(input_tensor)
    loss = F.cross_entropy(output, true_label)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * input_tensor.grad.sign()
    cloaked = input_tensor + perturbation
    cloaked = torch.clamp(cloaked, input_tensor.min(), input_tensor.max())
    return cloaked.detach()


# --- Step 1: get the original prediction (before attack) ---
original_class, original_confidence = get_prediction(model, input_tensor)
print("BEFORE ATTACK")
print("Predicted class index:", original_class)
print("Confidence:", original_confidence)

# --- Step 2: run FGSM using that predicted class as the "true_label" ---
true_label = torch.tensor([original_class])
epsilon = 0.03  # small bound, tune later
cloaked_tensor = fgsm_attack(model, input_tensor, true_label, epsilon)

# --- Step 3: get the new top prediction on the cloaked tensor ---
cloaked_class, cloaked_confidence = get_prediction(model, cloaked_tensor)
print("\nAFTER ATTACK")
print("Predicted class index:", cloaked_class)
print("Confidence (on new top class):", cloaked_confidence)

# --- Step 4: the real sanity check — confidence on the ORIGINAL class, after attack ---
confidence_on_original_class = get_confidence_for_class(model, cloaked_tensor, original_class)
print("\nSANITY CHECK")
print(f"Confidence on ORIGINAL class ({original_class}) after attack:", confidence_on_original_class)