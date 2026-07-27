"""QA del motore chroma: casi sintetici difficili, senza chiamate di rete."""
import sys

import numpy as np
from PIL import Image, ImageDraw

from chroma import chroma_to_cutout, pick_key_for_color, remove_chroma

fails: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name + (("  " + extra) if extra else ""))
    if not cond:
        fails.append(name)


def make_scene(garment_rgb, key_hex="#00ff00", logo_rgb=None, size=900):
    """Capo (rettangolo arrotondato) su fondo chroma, con bordi morbidi."""
    key = tuple(int(key_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    img = Image.new("RGB", (size, size), key)
    d = ImageDraw.Draw(img)
    m = size // 6
    d.rounded_rectangle([m, m, size - m, size - m], radius=size // 12, fill=garment_rgb)
    if logo_rgb:
        c = size // 2
        d.ellipse([c - size // 10, c - size // 10, c + size // 10, c + size // 10], fill=logo_rgb)
    return img.filter(__import__("PIL.ImageFilter", fromlist=["ImageFilter"]).GaussianBlur(1.2))


def opaque_center_color(cut: Image.Image):
    a = np.asarray(cut)[..., 3]
    rgb = np.asarray(cut)[..., :3].astype(float)
    h, w = a.shape
    box = rgb[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
    boxa = a[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
    return box.reshape(-1, 3).mean(axis=0), float(boxa.mean()) / 255.0


CASES = [
    ("grigio medio", (128, 128, 128), "#00ff00"),
    ("bianco", (245, 245, 245), "#00ff00"),
    ("nero", (18, 18, 18), "#00ff00"),
    ("blu navy", (30, 45, 90), "#00ff00"),
    ("rosso", (190, 40, 40), "#00ff00"),
    ("beige", (232, 216, 190), "#00ff00"),
    ("verde menta su magenta", (140, 220, 160), "#ff00ff"),
    ("verde bosco su magenta", (30, 90, 50), "#ff00ff"),
]

for name, color, key in CASES:
    scene = make_scene(color, key)
    cut, report = chroma_to_cutout(scene, key)
    mean_rgb, mean_alpha = opaque_center_color(cut)
    dev = float(np.abs(mean_rgb - np.array(color, float)).max())
    check(f"{name}: capo opaco", mean_alpha > 0.98, f"alpha={mean_alpha:.3f}")
    check(f"{name}: colore preservato", dev < 12, f"scarto={dev:.1f}")
    check(f"{name}: QA ok", report["ok"], str(report["problems"]))
    check(f"{name}: angoli trasparenti", "angoli non trasparenti" not in report["problems"])

# dettaglio dello stesso colore della chiave DENTRO al capo: non deve bucarsi
scene = make_scene((128, 128, 128), "#00ff00", logo_rgb=(0, 255, 0))
cut, report = chroma_to_cutout(scene, "#00ff00")
a = np.asarray(cut)[..., 3]
h, w = a.shape
center_alpha = float(a[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8].mean()) / 255.0
check("logo verde interno non buca il capo", center_alpha > 0.9, f"alpha centro={center_alpha:.3f}")

# chiave scelta in automatico in base al colore del capo
check("chiave per capo verde -> magenta", pick_key_for_color("#3aa35a") == "#ff00ff")
check("chiave per capo rosso -> verde/blu", pick_key_for_color("#c02020") in ("#00ff00", "#0000ff"))

# fondo non uniforme (leggero gradiente): deve comunque sparire
grad = np.zeros((700, 700, 3), np.uint8)
grad[..., 1] = np.linspace(238, 255, 700).astype(np.uint8)[None, :]
bg = Image.fromarray(grad)
d = ImageDraw.Draw(bg)
d.rounded_rectangle([120, 120, 580, 580], radius=60, fill=(160, 120, 200))
cut, report = chroma_to_cutout(bg, "#00ff00")
check("fondo con gradiente rimosso", report["ok"], str(report["problems"]))

# senza chiave dichiarata: si ricava dal bordo
cut2 = remove_chroma(make_scene((200, 170, 120), "#00ff00"))
a2 = np.asarray(cut2)[..., 3]
check("chiave dedotta dal bordo", a2[5, 5] == 0 and a2[a2.shape[0] // 2, a2.shape[1] // 2] > 250)

print("\n" + ("TUTTO OK" if not fails else f"FALLITI: {fails}"))
sys.exit(1 if fails else 0)
