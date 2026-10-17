"""Create explicitly synthetic, reproducible fixtures; not a student dataset."""
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def create():
    out=ROOT/'dataset/synthetic'
    out.mkdir(parents=True,exist_ok=True)
    im=np.full((480,800,3),42,np.uint8)
    # Simple road, vehicle body, bright rectangular plate and separated dark glyphs.
    cv2.rectangle(im,(90,140),(710,370),(68,63,60),-1)
    cv2.rectangle(im,(155,165),(645,248),(22,26,30),-1)
    cv2.circle(im,(200,385),44,(13,13,13),-1)
    cv2.circle(im,(600,385),44,(13,13,13),-1)
    cv2.rectangle(im,(270,285),(535,352),(180,180,180),-1)
    cv2.putText(im,'CAB 1234',(281,333),cv2.FONT_HERSHEY_SIMPLEX,1.25,(15,15,15),3,cv2.LINE_AA)
    rng=np.random.default_rng(6)
    noisy=np.uint8(np.clip(im.astype(np.float32)+rng.normal(0,2,im.shape),0,255))
    cv2.imwrite(str(out/'night_plate.png'),noisy)
    cv2.imwrite(str(out/'blank.png'),np.full_like(im,35))
    changed=np.full_like(im,35)
    cv2.rectangle(changed,(300,150),(390,340),(130,130,130),-1)
    cv2.imwrite(str(out/'change.png'),changed)
    return noisy

if __name__=='__main__':
    create()
    print('Synthetic fixtures created. These do not satisfy the 30-image dataset requirement.')
