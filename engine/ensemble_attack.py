"""Generate a transfer-oriented cloak using an ensemble of ImageNet models.
The perturbation maximizes the average cross-entropy loss across the selected
models, then applies the same resolution-decoupling used by the project.
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MODELS = {
    'resnet50': lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    'mobilenetv3': lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
    'vit': lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
}
PREP=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
STD=torch.tensor([.229,.224,.225]).view(3,1,1)

def ensemble_cloak(image, names, epsilon=.06, steps=10):
    models_=[MODELS[n]().eval() for n in names]
    x=PREP(image).unsqueeze(0)
    with torch.no_grad():
        probs=[torch.softmax(m(x),1) for m in models_]
    label=torch.stack([p.argmax(1) for p in probs]).mode(0).values
    original=x.detach().clone(); adv=original.clone(); alpha=epsilon/4
    for _ in range(steps):
        adv.requires_grad_(True)
        loss=sum(F.cross_entropy(m(adv),label) for m in models_)/len(models_)
        for m in models_: m.zero_grad(set_to_none=True)
        loss.backward()
        adv=(adv.detach()+alpha*adv.grad.sign())
        delta=torch.clamp(adv-original,-epsilon,epsilon)
        adv=(original+delta).detach()
    delta=(adv-original)*STD
    full=transforms.ToTensor()(image).unsqueeze(0)
    delta=F.interpolate(delta,size=(image.height,image.width),mode='bilinear',align_corners=False)
    out=torch.clamp(full+delta,0,1)[0].permute(1,2,0).numpy()*255
    return Image.fromarray(out.astype('uint8'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--models',default='resnet50,mobilenetv3'); ap.add_argument('--epsilon',type=float,default=.06); ap.add_argument('--out',default='ensemble_cloaked.png'); a=ap.parse_args()
    im=Image.open(a.image).convert('RGB'); ensemble_cloak(im,[x.strip() for x in a.models.split(',')],a.epsilon).save(a.out); print('saved',a.out)
