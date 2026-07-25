from PIL import Image
from cloak_engine import cloak_image

test_images = ["test.jpg", "test2.jpg", "test3.jpg"]

for filename in test_images:
    img = Image.open(filename).convert("RGB")

    print(f"\n===== {filename} =====")

    for method in ["fgsm", "pgd"]:
        result = cloak_image(img, strength=0.5, method=method)

        print(f"--- {method.upper()} ---")
        print("Original class:", result["original_class"],
              "| confidence:", round(result["original_confidence"], 4))
        print("Cloaked class: ", result["cloaked_class"],
              "| confidence:", round(result["cloaked_confidence"], 4))

        # Save each result so we can visually check all 6 outputs afterward
        output_name = f"{filename.split('.')[0]}_{method}_cloaked.jpg"
        result["cloaked_image"].save(output_name)
        print("Saved:", output_name)