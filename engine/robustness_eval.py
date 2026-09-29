"""Quick robustness evaluation: save/reload JPEG and resize transformations."""
import csv, io, time
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from cloak_engine import cloak_image, _preprocess, _model
import torch

INPUT_DIR=Path('../datasets/test_images')
OUT=Path('../datasets/robustness_results.csv')

def predict(im):
    x=_preprocess(im).unsqueeze(0)
    with torch.no_grad():
        p=torch.softmax(_model(x),dim=1); c=p.argmax(1).item()
    return c

rows=[]
for path in sorted(INPUT_DIR.glob('*.jpeg')):
    image=Image.open(path).convert('RGB')
    for method in ('fgsm','pgd'):
        r=cloak_image(image,0.5,method)
        protected=r['cloaked_image']
        transforms={'original':protected}
        b=io.BytesIO(); protected.save(b,format='JPEG',quality=75); b.seek(0)
        transforms['jpeg75']=Image.open(b).convert('RGB')
        small=protected.resize((160,160)); transforms['resize160']=small.resize(protected.size)
        for name,im in transforms.items():
            rows.append({'filename':path.name,'method':method,'transformation':name,
                         'original_class':r['original_class'],'predicted_class':predict(im),
                         'misclassified':predict(im)!=r['original_class'],
                         'ssim_to_cloak':float(ssim(np.array(protected),np.array(im),channel_axis=2))})
with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print(f'Wrote {len(rows)} rows to {OUT}')
