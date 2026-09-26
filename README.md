# privacy-preserving-image-cloaking
Privacy-Preserving Image Cloaking using Adversarial Machine Learning

## Data Source

The raw test images used for the batch evaluation (`datasets/test_images/` and `datasets/test_images_large/`) come from **Imagenette**, a small, free, permissively-licensed subset of ImageNet created by Jeremy Howard (fast.ai). No login or account is required to download it.

- 160px variant (`test_images/`): [Imagenette 160 px on Kaggle](https://www.kaggle.com/datasets/jhoward/imagenette-160-px)
- 320px variant (`test_images_large/`): [imagenette2-320 on Kaggle](https://www.kaggle.com/datasets/xbinchen/imagenette2-320)
- Original/canonical source: [fastai/imagenette on GitHub](https://github.com/fastai/imagenette)

The images are downloaded and sampled automatically by `engine/download_images.py` and `engine/download_large_images.py`.
