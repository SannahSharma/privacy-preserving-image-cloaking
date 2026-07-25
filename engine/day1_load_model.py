import torch
from torchvision import models, transforms
from PIL import Image

# Load the pretrained ResNet-50 model with its default trained weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Put the model in evaluation mode (turns off training-only behavior)
model.eval()

# Define the exact preprocessing steps ResNet-50 expects
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load our test image and make sure it's in RGB color format
img = Image.open('test.jpg').convert('RGB')

# Apply preprocessing, then add a "batch" dimension (models expect batches, not single images)
input_tensor = preprocess(img).unsqueeze(0)

# Run the image through the model without tracking gradients (we don't need them yet)
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

print("Predicted class index:", predicted_class.item())
print("Confidence:", confidence.item())