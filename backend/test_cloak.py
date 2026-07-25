import requests

with open("amur-tiger-01-01.webp", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/cloak",
        files={"image": ("amur-tiger-01-01.webp", f, "image/webp")},
        data={"strength": "0.5", "model_name": "resnet50", "method": "fgsm"}
    )

data = r.json()
print("STATUS:", r.status_code)
print("original:", data.get("original_predictions"))
print("cloaked:", data.get("cloaked_predictions"))
print("ssim:", data.get("ssim_score"))
print("detail:", data.get("detail"))