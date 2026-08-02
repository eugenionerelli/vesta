"""Color-analysis dalla foto: sottotono pelle (ITA in CIELAB), capelli, contrasto -> stagione + palette.

Riusa il segmentatore segformer gia' caricato da mask_from_person per isolare viso/capelli.
I valori sono principi consolidati (ITA, CIELAB) + euristiche trasparenti per il sistema
delle 4 stagioni: una base solida, affinabile.
"""
import math
import os

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf-cache")
os.environ.setdefault("HF_HOME", CACHE)

import numpy as np
import torch
from PIL import Image

from mask_from_person import _load  # condivide lo stesso modello segformer

# id classi segformer
FACE, HAIR, L_ARM, R_ARM = 11, 2, 14, 15

# palette stagionali (hex), set curato di colori tipici per stagione
SEASON_PALETTES = {
    "Primavera": ["#F28E7B", "#F6C18B", "#F2D06B", "#8FBF6B", "#46C7C7", "#F5EFE0", "#C89B6B"],
    "Estate":    ["#E7A7B5", "#B7A7D6", "#9FB8D6", "#A9C2B0", "#6E7E8F", "#EDEDEA", "#C9A8C4"],
    "Autunno":   ["#B5532A", "#7A7A2E", "#C9912E", "#A8512F", "#3E5B3A", "#EFE3C8", "#1F6E5A"],
    "Inverno":   ["#C81D3B", "#1F4FA8", "#0E7C5A", "#B0297A", "#16181C", "#F2F4F7", "#7FB2E6"],
}
SEASON_ADVICE = {
    "Primavera": "Toni caldi e luminosi: corallo, pesca, giallo dorato, verde mela, turchese.",
    "Estate":    "Toni freddi e tenui: rosa polvere, lavanda, blu cipria, salvia, grigio-azzurro.",
    "Autunno":   "Toni caldi e profondi: ruggine, oliva, senape, terracotta, verde bosco.",
    "Inverno":   "Toni freddi e intensi: rosso vero, blu reale, smeraldo, magenta, bianco ottico.",
}


def _srgb_to_lab(rgb: np.ndarray):
    """rgb in [0,1] shape (N,3) -> (L*, a*, b*) arrays."""
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])  # D65
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a_ = 500 * (f[:, 0] - f[:, 1])
    b_ = 200 * (f[:, 1] - f[:, 2])
    return L, a_, b_


def _hex(rgb01: np.ndarray) -> str:
    r, g, b = (np.clip(rgb01, 0, 1) * 255).round().astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


@torch.no_grad()
def analyze(person: Image.Image) -> dict:
    person = person.convert("RGB")
    proc, model = _load("cpu")
    inputs = proc(images=person, return_tensors="pt")
    logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(logits, size=person.size[::-1], mode="bilinear", align_corners=False)
    seg = up.argmax(1)[0].cpu().numpy()
    arr = np.asarray(person).astype(np.float32) / 255.0

    def region_stats(mask):
        px = arr[mask]
        if len(px) < 30:
            return None
        med = np.median(px, axis=0)
        L, a, b = _srgb_to_lab(med[None, :])
        return dict(L=float(L[0]), a=float(a[0]), b=float(b[0]), hex=_hex(med))

    skin = region_stats(seg == FACE) or region_stats(np.isin(seg, [FACE, L_ARM, R_ARM]))
    hair = region_stats(seg == HAIR)
    if skin is None:
        return {"error": "viso non rilevato nella foto"}

    # ITA (depth) e tonalita' (undertone)
    ita = math.degrees(math.atan2(skin["L"] - 50, skin["b"]))
    hue = math.degrees(math.atan2(skin["b"], skin["a"]))  # piu' alto = piu' giallo/caldo

    if ita > 41:
        depth = "chiaro"
    elif ita > 19:
        depth = "medio"
    else:
        depth = "scuro"

    if hue >= 55:
        undertone = "caldo"
    elif hue <= 47:
        undertone = "freddo"
    else:
        undertone = "neutro"

    # contrasto pelle/capelli
    if hair is not None:
        dL = skin["L"] - hair["L"]
        contrast = "alto" if dL > 45 else ("medio" if dL > 22 else "basso")
    else:
        contrast = "medio"

    # stagione (4 stagioni) da sottotono x profondita'
    warm = undertone == "caldo" or (undertone == "neutro" and skin["b"] >= 18)
    light = depth == "chiaro" or (depth == "medio" and contrast != "alto")
    if warm and light:
        season = "Primavera"
    elif warm and not light:
        season = "Autunno"
    elif (not warm) and light:
        season = "Estate"
    else:
        season = "Inverno"

    palette = SEASON_PALETTES[season]
    return {
        "skin_hex": skin["hex"],
        "hair_hex": hair["hex"] if hair else None,
        "ita": round(ita, 1),
        "hue": round(hue, 1),
        "undertone": undertone,
        "depth": depth,
        "contrast": contrast,
        "season": season,
        "palette": palette,
        "palette_main": palette[:5],
        "advice": SEASON_ADVICE[season],
    }


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "web", "person_sample.jpg")
    print(json.dumps(analyze(Image.open(path)), indent=2, ensure_ascii=False))
