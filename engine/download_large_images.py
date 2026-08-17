"""
download_large_images.py

Same as download_images.py, but pulls the 320px variant of Imagenette
(shortest side 320px, above the model's 224x224 working resolution)
into a separate folder, so we can compare attack strength on
"small" vs "large" images.

Includes automatic retry + size verification, since large downloads
can get cut off partway on an unstable connection.
"""
import random
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
N_IMAGES = 120
MAX_RETRIES = 5

work_dir = Path("../datasets/imagenette_download_large")
work_dir.mkdir(parents=True, exist_ok=True)
archive_path = work_dir / "imagenette2-320.tgz"
extract_dir = work_dir / "extracted"


def download_with_retry(url, dest_path, max_retries=MAX_RETRIES):
    with urllib.request.urlopen(url) as response:
        expected_size = int(response.getheader("Content-Length", 0))
    print(f"Expected file size: {expected_size / 1_000_000:.1f} MB")

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        print(f"Download attempt {attempt}/{max_retries}...")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            actual_size = tmp_path.stat().st_size
            if expected_size and actual_size != expected_size:
                raise IOError(f"Incomplete: got {actual_size} of {expected_size} bytes")
            tmp_path.rename(dest_path)
            print("Download verified complete.")
            return
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(3)

    raise RuntimeError(f"Failed to download after {max_retries} attempts")


if not archive_path.exists():
    print("Downloading dataset (larger, may take a few minutes)...")
    download_with_retry(IMAGENETTE_URL, archive_path)
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

out_dir = Path("../datasets/test_images_large")
out_dir.mkdir(parents=True, exist_ok=True)
for i, img_path in enumerate(sample):
    dest = out_dir / f"img_{i:04d}{img_path.suffix.lower()}"
    shutil.copy(img_path, dest)

print(f"\nCopied {len(sample)} images into {out_dir.resolve()}")