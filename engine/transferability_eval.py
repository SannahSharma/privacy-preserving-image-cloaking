"""Evaluate whether ResNet-50 cloaks transfer to MobileNetV3 and ViT.
Uses the existing ResNet-generated cloaks; no new attack is performed."""
import csv
from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image
from cloak_engine import cloak_image

INPUT=Path('../datasets/test_images'); OUT=Path('../datasets/transferability_results.csv')
prep=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
models_to_test={
 'mobilenetv3':models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).eval(),
 'vit':models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).eval(),
}
rows=[]
for path in sorted(INPUT.glob('*.jpeg')):
    image=Image.open(path).convert('RGB'); r=cloak_image(image,.5,'fgsm')
    x=prep(r['cloaked_image']).unsqueeze(0)
    for name,m in models_to_test.items():
        with torch.no_grad(): pred=m(x).argmax(1).item()
        rows.append({'filename':path.name,'source_model':'resnet50','target_model':name,'resnet_original_class':r['original_class'],'target_cloaked_class':pred})
with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print(f'Wrote {len(rows)} rows to {OUT}')
