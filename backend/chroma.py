"""Dal render su fondo chroma al PNG trasparente da catalogo.

Il capo viene ricostruito dal modello su un fondo di colore pieno; qui quel fondo
diventa trasparenza vera. I punti che fanno la differenza, tutti verificati su
scene sintetiche difficili (qa_chroma.py):

1. la chiave si MISURA dal bordo, non si assume: i modelli rendono spesso un verde
   approssimato (#22cc44 invece di #00ff00) e una chiave dichiarata fallirebbe;
2. l'alpha combina due criteri con un massimo: distanza dal colore chiave e
   cromaticita'. Con il solo criterio di distanza un grigio medio (che dista 221
   dal verde puro) finirebbe semitrasparente;
3. diventa fondo solo cio' che e' collegato al bordo: un dettaglio verde dentro al
   capo resta al suo posto;
4. sui pixel di bordo il colore viene ricostruito togliendo il contributo del fondo
   (unpremultiply), che e' cio' che elimina davvero l'alone;
5. il despill agisce solo in una fascia stretta lungo il contorno: applicarlo ovunque
   spegnerebbe i dettagli del capo che hanno il colore della chiave.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

T_TRANSPARENT = 12.0   # distanza RGB sotto la quale e' fondo puro
T_OPAQUE = 220.0       # distanza sopra la quale e' capo pieno
K_LO, K_HI = 0.25, 0.80  # fascia di cromaticita' in cui l'alpha sfuma
BORDER_BAND = 12       # larghezza delle fasce da cui si misura la chiave
MIN_KEYNESS = 30.0     # sotto questa soglia il fondo non e' un vero chroma
SPILL_PX = 8           # fascia interna in cui si corregge l'alone

KEY_GREEN = (0, 255, 0)
KEY_MAGENTA = (255, 0, 255)
KEY_BLUE = (0, 0, 255)
KEYS = {"#00ff00": KEY_GREEN, "#ff00ff": KEY_MAGENTA, "#0000ff": KEY_BLUE}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = (h or "").lstrip("#")
    if len(h) != 6:
        return KEY_GREEN
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return KEY_GREEN


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb[:3])


def pick_key_for_color(garment_hex: str | None) -> str:
    """Chiave cromatica piu' lontana dal colore del capo."""
    if not garment_hex:
        return "#00ff00"
    g = np.array(hex_to_rgb(garment_hex), float)
    return max(KEYS, key=lambda hx: float(np.linalg.norm(np.array(KEYS[hx], float) - g)))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _axis(key: np.ndarray) -> tuple[list[int], list[int]]:
    """Canali dominanti della chiave (POS) e restanti (NEG)."""
    thr = max(float(key.max()) * 0.5, 40.0)
    pos = [i for i in range(3) if key[i] >= thr]
    neg = [i for i in range(3) if i not in pos]
    return (pos, neg) if pos and neg else ([1], [0, 2])


def resolve_key(rgb: np.ndarray, declared_hex: str | None = None, band: int = BORDER_BAND):
    """Chiave misurata sul bordo; ripiega su quella dichiarata se il fondo non e' chroma."""
    h, w = rgb.shape[:2]
    b = max(2, min(band, h // 8, w // 8))
    edges = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                            rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    med = np.median(edges, axis=0).astype(np.float32)
    declared = np.array(hex_to_rgb(declared_hex), np.float32) if declared_hex else None
    pos, neg = _axis(declared if declared is not None else med)
    keyness = float(med[pos].min() - med[neg].max())
    if keyness >= MIN_KEYNESS:
        return med, pos, neg, True, keyness
    return (declared if declared is not None else med), pos, neg, False, keyness


def _border_connected(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    keep = [int(v) for v in np.unique(border) if v]
    return np.isin(lab, keep) if keep else np.zeros_like(mask, dtype=bool)


def strip_chroma(img: Image.Image, key_hex: str | None = None,
                 spill_px: int = SPILL_PX) -> tuple[Image.Image, dict]:
    """Fondo chroma -> RGBA con alpha morbida. Restituisce (immagine, diagnostica)."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    key, pos, neg, empirical, keyness = resolve_key(rgb, key_hex)

    # 1) distanza euclidea dal colore chiave
    dist = np.linalg.norm(rgb - key[None, None, :], axis=-1)
    a_dist = _smoothstep((dist - T_TRANSPARENT) / (T_OPAQUE - T_TRANSPARENT))

    # 2) cromaticita': indipendente dalla luminosita', salva grigi/bianchi/neri
    score = rgb[..., pos].min(axis=-1) - rgb[..., neg].max(axis=-1)
    key_score = max(float(key[pos].min() - key[neg].max()), 1.0)
    ratio = score / key_score
    a_key = 1.0 - _smoothstep((ratio - K_LO) / (K_HI - K_LO))

    alpha = np.maximum(a_dist, a_key)

    # 3) e' fondo solo cio' che tocca il bordo
    transparent = alpha < 0.5
    bg = _border_connected(transparent)
    alpha = np.where(bg | ~transparent, alpha, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    out = rgb.copy()

    # 4) ricostruzione del colore dove l'alpha e' parziale (via il contributo del fondo)
    partial = (alpha > 0.15) & (alpha < 0.995)
    if partial.any():
        a3 = alpha[..., None]
        rec = (out - (1.0 - a3) * key[None, None, :]) / np.maximum(a3, 1e-3)
        out = np.where(partial[..., None], np.clip(rec, 0, 255), out)

    # 5) despill solo lungo il contorno, per non toccare i dettagli interni
    edge_band = partial.copy()
    try:
        from scipy import ndimage
        if spill_px > 0:
            grown = ndimage.binary_dilation(transparent, iterations=int(spill_px))
            edge_band |= grown & (alpha > 0.5)
    except Exception:
        pass
    if edge_band.any():
        cap = out[..., neg].max(axis=-1)
        for c in pos:
            out[..., c] = np.where(edge_band, np.minimum(out[..., c], cap), out[..., c])

    a8 = (alpha * 255.0).round().astype(np.uint8)
    rgb8 = np.clip(out.round(), 0, 255).astype(np.uint8)
    rgb8[a8 == 0] = 0
    meta = {"key": rgb_to_hex(key), "key_measured": bool(empirical),
            "border_keyness": round(keyness, 1),
            "removed_ratio": round(float((alpha < 0.5).mean()), 4)}
    return Image.fromarray(np.dstack([rgb8, a8]), mode="RGBA"), meta


# compatibilita' con il nome precedente
def remove_chroma(img: Image.Image, key_hex: str | None = None, **kw) -> Image.Image:
    return strip_chroma(img, key_hex)[0]


def trim_to_content(img: Image.Image, pad_frac: float = 0.06, square: bool = True) -> Image.Image:
    """Ritaglia sul contenuto opaco lasciando un margine costante."""
    a = np.asarray(img.convert("RGBA"))[..., 3]
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if square:
        side = max(w, h)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, y0, w, h = cx - side // 2, cy - side // 2, side, side
    pad = int(round(max(w, h) * pad_frac))
    box = (x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad)
    out = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (-box[0], -box[1]))
    return out


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(rgb, 0, 255) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def qa_cutout(img: Image.Image, key_hex: str | None = None, meta: dict | None = None,
              expect_color: str | None = None, category: str | None = None) -> dict:
    """Controlli sul ritaglio finale: problemi duri (si rigenera) e morbidi (si accetta)."""
    rgba = np.asarray(img.convert("RGBA"))
    a = rgba[..., 3].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    h, w = a.shape
    hard: list[str] = []
    soft: list[str] = []
    meta = meta or {}

    corner = max(2, min(h, w) // 50)
    if max(float(c.max()) for c in (a[:corner, :corner], a[:corner, -corner:],
                                    a[-corner:, :corner], a[-corner:, -corner:])) > 0.15:
        hard.append("angoli non trasparenti")

    band = max(2, min(h, w) // 50)
    border = np.concatenate([a[:band].ravel(), a[-band:].ravel(), a[:, :band].ravel(), a[:, -band:].ravel()])
    if float((border > 0.5).mean()) > 0.02:
        hard.append("il capo tocca il bordo")

    content = float((a > 0.5).mean())
    if content < 0.05:
        hard.append("contenuto quasi vuoto")
    if content > 0.90:
        hard.append("fondo non rimosso")
    if meta.get("removed_ratio") is not None and meta["removed_ratio"] < 0.05:
        if "fondo non rimosso" not in hard:
            hard.append("fondo non rimosso")
    if meta.get("border_keyness") is not None and meta["border_keyness"] < MIN_KEYNESS:
        hard.append("il fondo generato non e' un colore pieno")

    residual = 0.0
    key_ref = meta.get("key") or key_hex
    if key_ref:
        key = np.array(hex_to_rgb(key_ref), np.float32)
        pos, neg = _axis(key)
        score = rgb[..., pos].min(axis=-1) - rgb[..., neg].max(axis=-1)
        ratio = score / max(float(key[pos].min() - key[neg].max()), 1.0)
        visible = a > 0.5
        if visible.any():
            residual = float(((ratio > 0.45) & visible).mean())
            if residual > 0.01:
                hard.append("residui del fondo sul capo")
            # alone lungo il contorno
            try:
                from scipy import ndimage
                edge = ndimage.binary_dilation(a < 0.5, iterations=6) & visible
                if edge.any() and float(np.median(score[edge])) > 25:
                    soft.append("alone di colore sul bordo")
            except Exception:
                pass

    delta_e = None
    if expect_color:
        solid = a > 0.9
        if solid.sum() > 50:
            med = np.median(rgb[solid], axis=0)
            delta_e = float(np.linalg.norm(_srgb_to_lab(med) - _srgb_to_lab(np.array(hex_to_rgb(expect_color), float))))
            if delta_e > 25:
                hard.append("colore diverso da quello osservato nella foto")

    if category != "shoes":
        try:
            from scipy import ndimage
            lab, n = ndimage.label(a > 0.5)
            if n > 1:
                sizes = np.bincount(lab.ravel())[1:]
                big = sizes[sizes > 0.05 * sizes.max()]
                if len(big) > 1:
                    soft.append("piu' oggetti nel ritaglio")
        except Exception:
            pass

    grade = "a" if not hard and not soft else ("b" if not hard else "f")
    return {
        "ok": not hard, "grade": grade, "problems": hard, "warnings": soft,
        "content_ratio": round(content, 4), "residual_key": round(residual, 4),
        "delta_e": None if delta_e is None else round(delta_e, 1),
        "size": [int(w), int(h)], **{k: v for k, v in meta.items() if k != "removed_ratio"},
        "removed_ratio": meta.get("removed_ratio"),
    }


def chroma_to_cutout(img: Image.Image, key_hex: str | None = None, pad_frac: float = 0.06,
                     expect_color: str | None = None,
                     category: str | None = None) -> tuple[Image.Image, dict]:
    """Percorso completo: fondo via, ritaglio, controlli."""
    cut, meta = strip_chroma(img, key_hex)
    trimmed = trim_to_content(cut, pad_frac=pad_frac)
    return trimmed, qa_cutout(trimmed, key_hex, meta, expect_color, category)


def rgba_to_cutout(img: Image.Image, pad_frac: float = 0.06, expect_color: str | None = None,
                   category: str | None = None) -> tuple[Image.Image, dict]:
    """Per i modelli che producono gia' la trasparenza (OpenAI): niente chroma."""
    rgba = img.convert("RGBA")
    a = np.asarray(rgba)[..., 3]
    meta = {"key": None, "key_measured": False, "border_keyness": 999.0,
            "removed_ratio": round(float((a < 128).mean()), 4), "engine": "alpha nativa"}
    trimmed = trim_to_content(rgba, pad_frac=pad_frac)
    report = qa_cutout(trimmed, None, meta, expect_color, category)
    report.pop("il fondo generato non e' un colore pieno", None)
    return trimmed, report


if __name__ == "__main__":
    import sys

    out, report = chroma_to_cutout(Image.open(sys.argv[1]), sys.argv[3] if len(sys.argv) > 3 else None)
    out.save(sys.argv[2])
    print(sys.argv[2], report)
