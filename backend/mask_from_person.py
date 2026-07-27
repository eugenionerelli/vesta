"""Genera la maschera dell'area-capo dalla foto della persona, SENZA detectron2/DensePose.

Usa il segmentatore di vestiti `mattmdjaga/segformer_b2_clothes` (modello SegFormer,
~100MB, gira su CPU). La maschera e' bianca (255) sulla zona da rigenerare col try-on,
nera (0) sul resto, e ha la stessa dimensione della foto persona.
"""
import os

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf-cache")
os.environ.setdefault("HF_HOME", CACHE)

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

_MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# id delle classi del modello -> raggruppate per tipo di capo
# 0 Background 1 Hat 2 Hair 3 Sunglasses 4 Upper-clothes 5 Skirt 6 Pants
# 7 Dress 8 Belt 9 Left-shoe 10 Right-shoe 11 Face 12 Left-leg 13 Right-leg
# 14 Left-arm 15 Right-arm 16 Bag 17 Scarf
CATEGORY_CLASSES = {
    "upper": {4, 7},          # magliette, giacche, abiti
    "lower": {5, 6},          # gonne, pantaloni
    "overall": {4, 5, 6, 7},  # tutto il vestito
}

_proc = None
_model = None


def _load(device: str = "cpu"):
    global _proc, _model
    if _model is None:
        _proc = AutoImageProcessor.from_pretrained(_MODEL_ID)
        _model = AutoModelForSemanticSegmentation.from_pretrained(_MODEL_ID).to(device).eval()
    return _proc, _model


@torch.no_grad()
def garment_mask(person: Image.Image, category: str = "upper", dilate: int = 2, device: str = "cpu") -> Image.Image:
    person = person.convert("RGB")
    proc, model = _load(device)
    inputs = proc(images=person, return_tensors="pt").to(device)
    logits = model(**inputs).logits  # (1, num_classi, h/4, w/4)
    up = torch.nn.functional.interpolate(
        logits, size=person.size[::-1], mode="bilinear", align_corners=False
    )
    seg = up.argmax(1)[0].cpu().numpy()  # (H, W) con gli id di classe
    wanted = CATEGORY_CLASSES[category]
    mask = (np.isin(seg, list(wanted)).astype(np.uint8) * 255)
    m = Image.fromarray(mask, mode="L")
    # dilatazione: allarga un po' la maschera cosi' i bordi del vecchio capo non restano
    for _ in range(dilate):
        m = m.filter(ImageFilter.MaxFilter(9))
    return m


@torch.no_grad()
def person_anchors(img: Image.Image) -> dict:
    """Riquadri normalizzati dove appoggiare i capi nel montaggio.

    Ricavati dalla segmentazione reale della persona, non da costanti: una foto
    seduta o inquadrata stretta ha proporzioni molto diverse da una in piedi.
    """
    proc, model = _load("cpu")
    img = img.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    up = torch.nn.functional.interpolate(model(**inputs).logits, size=img.size[::-1],
                                         mode="bilinear", align_corners=False)
    seg = up.argmax(1)[0].cpu().numpy()
    h, w = seg.shape

    def box(classes, grow=0.0):
        m = np.isin(seg, classes)
        if m.sum() < (h * w) * 0.002:
            return None
        ys, xs = np.where(m)
        l, r = xs.min() / w, xs.max() / w
        t, b = ys.min() / h, ys.max() / h
        if grow:
            dx, dy = (r - l) * grow, (b - t) * grow
            l, r, t, b = max(0.0, l - dx), min(1.0, r + dx), max(0.0, t - dy), min(1.0, b + dy)
        return [round(float(v), 4) for v in (l, t, r, b)]

    upper = box([4, 7, 14, 15], 0.03)          # maglia/abito piu' le braccia
    lower = box([5, 6, 12, 13], 0.02)          # gonna/pantaloni/gambe
    shoes = box([9, 10], 0.05)
    person = box([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])

    out = {}
    if upper:
        out["upper"] = upper
        out["outerwear"] = [max(0.0, upper[0] - 0.04), upper[1], min(1.0, upper[2] + 0.04), upper[3]]
    if lower:
        out["lower"] = lower
    if shoes:
        out["shoes"] = shoes
    if upper and lower:
        out["overall"] = [min(upper[0], lower[0]), upper[1], max(upper[2], lower[2]), lower[3]]
    elif upper:
        out["overall"] = [upper[0], upper[1], upper[2], min(1.0, upper[3] + 0.3)]
    return {"anchors": out, "person": person, "found": sorted(out)}


@torch.no_grad()
def classify_garment(img: Image.Image) -> str:
    """Indovina la categoria di un capo (anche flat) via segformer: upper/lower/overall."""
    proc, model = _load("cpu")
    img = img.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    up = torch.nn.functional.interpolate(model(**inputs).logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    seg = up.argmax(1)[0].cpu().numpy()
    scores = [("upper", int((seg == 4).sum())),
              ("overall", int((seg == 7).sum())),
              ("lower", int(np.isin(seg, [5, 6]).sum()))]
    best = max(scores, key=lambda x: x[1])
    return best[0] if best[1] > 0 else "upper"


if __name__ == "__main__":
    import sys

    person_path = sys.argv[1]
    out_path = sys.argv[2]
    category = sys.argv[3] if len(sys.argv) > 3 else "upper"
    garment_mask(Image.open(person_path), category).save(out_path)
    print("maschera salvata:", out_path)
