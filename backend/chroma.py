"""Rimozione del fondo chroma da un'immagine generata, con matte morbido.

Il capo viene ricostruito dal modello generativo su un fondo di colore pieno
(verde o magenta). Qui quel fondo diventa trasparenza vera:

1. il colore chiave si ricava dai bordi dell'immagine (mediana), non si assume;
2. per ogni pixel si misura quanto "assomiglia" alla chiave con una differenza
   fra canali dominanti e non dominanti: e' insensibile alla luminosita', quindi
   grigi, bianchi e neri del capo restano opachi;
3. l'alpha e' morbido (smoothstep) sulla fascia di transizione, cosi' i bordi non
   risultano seghettati;
4. solo il fondo collegato al bordo diventa trasparente: un dettaglio verde
   *dentro* al capo (una scritta, una riga) resta al suo posto;
5. despill: sui pixel di bordo si abbassa il canale della chiave, per togliere
   l'alone colorato;
6. l'immagine viene ritagliata sul contenuto con un margine costante.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# chiavi disponibili: si sceglie quella piu' lontana dal colore del capo
KEY_GREEN = (0, 255, 0)
KEY_MAGENTA = (255, 0, 255)
KEY_BLUE = (0, 0, 255)
KEYS = {"#00ff00": KEY_GREEN, "#ff00ff": KEY_MAGENTA, "#0000ff": KEY_BLUE}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = (h or "").lstrip("#")
    if len(h) != 6:
        return KEY_GREEN
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb[:3])


def pick_key_for_color(garment_hex: str | None) -> str:
    """Sceglie la chiave cromatica piu' lontana dal colore dominante del capo."""
    if not garment_hex:
        return "#00ff00"
    r, g, b = hex_to_rgb(garment_hex)
    best, best_d = "#00ff00", -1.0
    for hx, key in KEYS.items():
        d = float(np.linalg.norm(np.array(key, float) - np.array((r, g, b), float)))
        if d > best_d:
            best, best_d = hx, d
    return best


def auto_key_from_border(arr: np.ndarray, band: int = 10) -> np.ndarray:
    """Colore della chiave stimato dalle quattro fasce di bordo (mediana)."""
    h, w = arr.shape[:2]
    band = max(2, min(band, h // 8, w // 8))
    edges = np.concatenate([
        arr[:band].reshape(-1, 3),
        arr[-band:].reshape(-1, 3),
        arr[:, :band].reshape(-1, 3),
        arr[:, -band:].reshape(-1, 3),
    ])
    return np.median(edges.astype(np.float32), axis=0)


def _key_channels(key: np.ndarray) -> tuple[list[int], list[int]]:
    """Canali dominanti della chiave (POS) e i restanti (NEG).

    Verde -> POS=[G], NEG=[R,B]; magenta -> POS=[R,B], NEG=[G].
    """
    thr = max(float(key.max()) * 0.5, 40.0)
    pos = [i for i in range(3) if key[i] >= thr]
    neg = [i for i in range(3) if i not in pos]
    if not pos or not neg:  # chiave degenere (grigia): ripiego sul verde
        return [1], [0, 2]
    return pos, neg


def _keyness(arr: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, float]:
    """Quanto ogni pixel somiglia alla chiave: min(POS) - max(NEG), 0 = per nulla."""
    pos, neg = _key_channels(key)
    score = arr[..., pos].min(axis=-1) - arr[..., neg].max(axis=-1)
    key_score = float(key[pos].min() - key[neg].max())
    return score, max(key_score, 1.0)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """Solo le regioni di `mask` che toccano il bordo (fondo vero, non buchi interni)."""
    try:
        from scipy import ndimage
    except Exception:
        return mask  # senza scipy: nessun filtro di connessione
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    keep = set(int(v) for v in np.unique(border) if v != 0)
    if not keep:
        return np.zeros_like(mask)
    return np.isin(lab, list(keep))


def remove_chroma(
    img: Image.Image,
    key_hex: str | None = None,
    soft_lo: float = 0.35,
    soft_hi: float = 0.80,
    despill: bool = True,
) -> Image.Image:
    """Toglie il fondo chroma e restituisce un'immagine RGBA.

    soft_lo/soft_hi: frazioni della "somiglianza alla chiave" fra cui l'alpha sfuma
    (sotto soft_lo = capo pieno, sopra soft_hi = fondo).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    key = hex_to_rgb(key_hex) if key_hex else None
    key_arr = np.array(key, dtype=np.float32) if key else auto_key_from_border(rgb)

    score, key_score = _keyness(rgb, key_arr)
    ratio = score / key_score  # 1 = fondo puro, <=0 = sicuramente capo

    # alpha morbido: 1 (opaco) sotto soft_lo, 0 (trasparente) sopra soft_hi
    alpha = 1.0 - _smoothstep((ratio - soft_lo) / max(soft_hi - soft_lo, 1e-6))

    # il fondo e' solo quello collegato al bordo: i dettagli interni restano
    bg = _border_connected(alpha < 0.5)
    alpha = np.where(bg, alpha, np.maximum(alpha, 1.0))
    alpha = np.clip(alpha, 0.0, 1.0)

    out = rgb.copy()
    if despill:
        pos, neg = _key_channels(key_arr)
        contaminated = (alpha < 0.995) & (ratio > 0.0)
        cap = out[..., neg].max(axis=-1)
        for c in pos:
            ch = out[..., c]
            out[..., c] = np.where(contaminated, np.minimum(ch, cap), ch)

    a8 = (alpha * 255.0).round().astype(np.uint8)
    rgb8 = out.round().clip(0, 255).astype(np.uint8)
    rgb8[a8 == 0] = 0
    return Image.fromarray(np.dstack([rgb8, a8]), mode="RGBA")


def trim_to_content(img: Image.Image, pad_frac: float = 0.06, square: bool = True) -> Image.Image:
    """Ritaglia sul contenuto opaco lasciando un margine, opzionalmente in quadrato."""
    a = np.asarray(img.convert("RGBA"))[..., 3]
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if square:
        side = max(w, h)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, y0 = cx - side // 2, cy - side // 2
        w = h = side
    pad = int(round(max(w, h) * pad_frac))
    box = (x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad)
    out = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    src = img.convert("RGBA")
    out.paste(src, (-box[0], -box[1]))
    return out


def qa_cutout(img: Image.Image, key_hex: str | None = None) -> dict:
    """Controlli numerici sul ritaglio finale: dice se e' accettabile e perche' no."""
    rgba = np.asarray(img.convert("RGBA"))
    a = rgba[..., 3].astype(np.float32) / 255.0
    h, w = a.shape
    problems: list[str] = []

    corner = max(2, min(h, w) // 50)
    corners = [a[:corner, :corner], a[:corner, -corner:], a[-corner:, :corner], a[-corner:, -corner:]]
    if max(float(c.max()) for c in corners) > 0.15:
        problems.append("angoli non trasparenti")

    band = max(2, min(h, w) // 50)
    border = np.concatenate([a[:band].ravel(), a[-band:].ravel(), a[:, :band].ravel(), a[:, -band:].ravel()])
    border_opaque = float((border > 0.5).mean())
    if border_opaque > 0.02:
        problems.append("il capo tocca il bordo (probabile taglio)")

    content = float((a > 0.5).mean())
    if content < 0.02:
        problems.append("contenuto quasi vuoto")
    if content > 0.92:
        problems.append("fondo non rimosso")

    residual = 0.0
    if key_hex:
        rgb = rgba[..., :3].astype(np.float32)
        score, key_score = _keyness(rgb, np.array(hex_to_rgb(key_hex), dtype=np.float32))
        visible = a > 0.5
        if visible.any():
            residual = float(((score / key_score > 0.45) & visible).mean())
            if residual > 0.01:
                problems.append("residui del fondo colorato sul capo")

    return {
        "ok": not problems,
        "problems": problems,
        "content_ratio": round(content, 4),
        "border_opaque": round(border_opaque, 4),
        "residual_key": round(residual, 4),
        "size": [int(w), int(h)],
    }


def qa_background_removed(cut: Image.Image) -> dict:
    """Controlli sull'immagine PRIMA del ritaglio.

    Serve perche' il ritaglio aggiunge un margine trasparente: senza questo passaggio
    un'immagine in cui il fondo non e' stato tolto affatto supererebbe i controlli.
    """
    a = np.asarray(cut.convert("RGBA"))[..., 3].astype(np.float32) / 255.0
    problems: list[str] = []
    removed = float((a < 0.5).mean())
    if removed < 0.05:
        problems.append("fondo non rimosso")

    corner = max(2, min(a.shape) // 50)
    corners = [a[:corner, :corner], a[:corner, -corner:], a[-corner:, :corner], a[-corner:, -corner:]]
    if max(float(c.mean()) for c in corners) > 0.2:
        problems.append("angoli non trasparenti")
    return {"removed_ratio": round(removed, 4), "problems": problems}


def chroma_to_cutout(img: Image.Image, key_hex: str | None = None,
                     pad_frac: float = 0.06) -> tuple[Image.Image, dict]:
    """Percorso completo: fondo via, controlli sul fondo, ritaglio, controlli finali."""
    cut = remove_chroma(img, key_hex)
    pre = qa_background_removed(cut)
    trimmed = trim_to_content(cut, pad_frac=pad_frac)
    report = qa_cutout(trimmed, key_hex)
    report["removed_ratio"] = pre["removed_ratio"]
    for p in pre["problems"]:
        if p not in report["problems"]:
            report["problems"].append(p)
    report["ok"] = not report["problems"]
    return trimmed, report


if __name__ == "__main__":
    import sys

    src, dst = sys.argv[1], sys.argv[2]
    key = sys.argv[3] if len(sys.argv) > 3 else None
    out, report = chroma_to_cutout(Image.open(src), key)
    out.save(dst)
    print(dst, report)
