"""
download_images.py

Downloads Imagenette (a small, free, permissively-licensed subset of
ImageNet, no login required) and copies 120 random real photographs into
datasets/test_images/ for large-scale batch testing.
"""
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
N_IMAGES = 120

work_dir = Path("../datasets/imagenette_download")
work_dir.mkdir(parents=True, exist_ok=True)
archive_path = work_dir / "imagenette2-160.tgz"
extract_dir = work_dir / "extracted"

if not archive_path.exists():
    print("Downloading dataset (about 94 MB, may take a minute)...")
    urllib.request.urlretrieve(IMAGENETTE_URL, archive_path)
    print("Download complete.")
else:
    print("Archive already downloaded, skipping.")

if not extract_dir.exists():
    print("Extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(extract_dir)
    print("Extraction complete.")
else:
    print("Already extracted, skipping.")

all_images = list(extract_dir.rglob("*.JPEG")) + list(extract_dir.rglob("*.jpg"))
print(f"Found {len(all_images)} candidate images.")

random.seed(42)
sample = random.sample(all_images, min(N_IMAGES, len(all_images)))

out_dir = Path("../datasets/test_images")
out_dir.mkdir(parents=True, exist_ok=True)
for i, img_path in enumerate(sample):
    dest = out_dir / f"img_{i:04d}{img_path.suffix.lower()}"
    shutil.copy(img_path, dest)

print(f"\nCopied {len(sample)} images into {out_dir.resolve()}")