"""Costruisce il guardaroba dimostrativo ritagliando capi reali dalle foto studio.

Usa la segmentazione (segformer) per isolare pantaloni, scarpe e capi sopra dalle
foto di esempio: sono ritagli fotografici veri, non disegni, e coprono tutte le
categorie dello Specchio. Serve anche a mostrare cosa fa l'app senza chiavi API.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND)
WEB = os.path.join(ROOT, "web")
GARM = os.path.join(WEB, "garments")
PERSON = os.path.join(BACKEND, "CatVTON/resource/demo/example/person")

from chroma import trim_to_content, qa_cutout  # noqa: E402
from mask_from_person import _load  # noqa: E402
import torch  # noqa: E402

# classi segformer -> categoria dell'app
GROUPS = {
    "lower": ([6], "Pantaloni"),
    "shoes": ([9, 10], "Scarpe"),
    "upper": ([4], "Maglia"),
    "outerwear": ([7], "Capospalla"),
}

SOURCES = [
    ("men/Yifeng_0.png", ["lower", "shoes"], {"lower": "Pantaloni chino", "shoes": "Sneaker bianche"}),
    ("men/Simon_1.png", ["lower", "upper"], {"lower": "Jeans scuri", "upper": "Camicia azzurra"}),
    ("men/model_5.png", ["lower", "shoes"], {"lower": "Pantaloni neri", "shoes": "Stivaletti"}),
    ("men/model_7.png", ["upper"], {"upper": "Maglione blu"}),
]


@torch.no_grad()
def segment(img: Image.Image) -> np.ndarray:
    proc, model = _load("cpu")
    inputs = proc(images=img.convert("RGB"), return_tensors="pt")
    up = torch.nn.functional.interpolate(model(**inputs).logits, size=img.size[::-1],
                                         mode="bilinear", align_corners=False)
    return up.argmax(1)[0].cpu().numpy()


def cut(img: Image.Image, seg: np.ndarray, classes) -> Image.Image | None:
    m = np.isin(seg, classes)
    if m.sum() < seg.size * 0.01:
        return None
    alpha = Image.fromarray((m * 255).astype(np.uint8), "L")
    # bordo morbido: evita il taglio a scaletta della maschera
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2))
    out = img.convert("RGBA")
    out.putalpha(alpha)
    return trim_to_content(out, pad_frac=0.07)


def dominant(img: Image.Image) -> str:
    a = np.asarray(img.convert("RGBA"))
    px = a[a[..., 3] > 200][:, :3]
    if len(px) < 30:
        return "#999999"
    med = np.median(px, axis=0).astype(int)
    return "#%02X%02X%02X" % tuple(med)


def main() -> None:
    items = json.load(open(os.path.join(WEB, "garments.json")))
    items = [i for i in items if not i.get("kit")]   # rigenerabile
    added = 0
    for rel, wanted, labels in SOURCES:
        path = os.path.join(PERSON, rel)
        if not os.path.exists(path):
            print("manca", rel)
            continue
        img = Image.open(path).convert("RGB")
        seg = segment(img)
        for cat in wanted:
            classes, _ = GROUPS[cat]
            piece = cut(img, seg, classes)
            if piece is None:
                print(f"  {rel}: nessun {cat}")
                continue
            slug = f"kit_{cat}_{added+1}"
            fn = f"garments/{slug}.png"
            piece.save(os.path.join(WEB, f"{slug}.png".join(["garments/", ""])) if False else os.path.join(GARM, slug + ".png"))
            rep = qa_cutout(piece)
            items.append({
                "id": slug, "label": labels.get(cat, GROUPS[cat][1]), "category": cat,
                "file": fn, "color": dominant(piece), "demo": True, "kit": True,
                "material": "", "silhouette": "", "construction": "",
                "color_name": "", "note": "ritagliato da una foto di esempio",
            })
            added += 1
            print(f"  {rel} -> {slug} ({cat}) {dominant(piece)} qa_ok={rep['ok']}")
    json.dump(items, open(os.path.join(WEB, "garments.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n{added} capi aggiunti; totale {len(items)}")


if __name__ == "__main__":
    main()
