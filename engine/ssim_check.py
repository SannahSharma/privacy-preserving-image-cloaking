from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from cloak_engine import cloak_image

test_images = ["test.jpg", "test2.jpg", "test3.jpg"]

for filename in test_images:
    original = Image.open(filename).convert("RGB")
    original_array = np.array(original)

    print(f"\n===== {filename} =====")

    for method in ["fgsm", "pgd"]:
        result = cloak_image(original, strength=0.5, method=method)
        cloaked_array = np.array(result["cloaked_image"])

        score = ssim(original_array, cloaked_array, channel_axis=2)

        status = "PASS (>0.95)" if score > 0.95 else "BELOW TARGET"
        print(f"{method.upper()} SSIM: {score:.4f}  [{status}]")