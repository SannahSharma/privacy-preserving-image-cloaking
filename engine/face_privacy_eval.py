"""Optional face-privacy evaluation scaffold.

Put face images under datasets/face_images/<identity>/*.jpg and install
facenet-pytorch separately. The script compares cosine similarity before and
after cloaking; lower similarity means the face embedding moved farther away.
This is an evaluation layer, not a claim that ImageNet classification equals
face recognition.
"""
from pathlib import Path
import sys
try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    from PIL import Image
except ImportError:
    raise SystemExit('Install optional dependency first: pip install facenet-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloak_engine import cloak_image

ROOT=Path('../datasets/face_images')
model=InceptionResnetV1(pretrained='vggface2').eval()

def emb(im):
    x=im.resize((160,160))
    x=torch.tensor(__import__('numpy').array(x),dtype=torch.float32).permute(2,0,1)/255
    x=(x-.5)/.5
    with torch.no_grad(): return model(x.unsqueeze(0)).squeeze(0)

def cosine(a,b): return torch.nn.functional.cosine_similarity(a,b,dim=0).item()

if not ROOT.exists(): raise SystemExit(f'Create {ROOT} with identity subfolders first.')
for person in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for path in sorted(person.glob('*')):
        if path.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        im=Image.open(path).convert('RGB'); r=cloak_image(im,.5,'fgsm')
        before=emb(im); after=emb(r['cloaked_image'])
        print(path.name, person.name, 'cosine_before_after=', round(cosine(before,after),4))
