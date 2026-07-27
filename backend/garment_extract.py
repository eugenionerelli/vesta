"""Estrazione dei capi da una foto: da persona vestita a schede di catalogo.

Non e' una rimozione dello sfondo: il capo viene *ricostruito* come foto prodotto
su un fondo chroma e poi reso trasparente. Il vantaggio e' che spariscono corpo,
pieghe da indosso e occlusioni, e restano il taglio e i dettagli reali.

Passi:
  1. inventario  - un modello vision elenca i capi con riquadro e attributi osservati
  2. ritaglio    - riquadro + 12% di margine su tela quadrata neutra
  3. ricostruzione - prompt basato solo su cio' che si vede, fondo chroma uniforme
  4. scontorno   - chroma.py: alpha morbido, despill, ritaglio sul contenuto
  5. verifica    - controlli numerici; se falliscono si ritenta con un'altra chiave

Senza chiave premium si ripiega su rembg (rimozione sfondo): qualita' inferiore
ma funziona offline.
"""
from __future__ import annotations

import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from chroma import chroma_to_cutout, pick_key_for_color
from premium_tryon import generate_image, resolve_provider, vision_json

MAX_ITEMS = int(os.environ.get("VESTA_MAX_ITEMS_PER_PHOTO", "6"))
CANVAS = 1200
PAD = 0.12

CATEGORIES = ("upper", "lower", "overall", "outerwear", "shoes", "accessory")

CATEGORY_IT = {
    "upper": "Maglia", "lower": "Pantaloni", "overall": "Abito",
    "outerwear": "Giacca", "shoes": "Scarpe", "accessory": "Accessorio",
}

_ITEM_PROPS = {
    "slug": {"type": "string"},
    "label": {"type": "string"},
    "category": {"type": "string", "enum": list(CATEGORIES)},
    "bbox": {"type": "array", "items": {"type": "number"}},
    "color_name": {"type": "string"},
    "color_hex": {"type": "string"},
    "material": {"type": "string"},
    "silhouette": {"type": "string"},
    "construction": {"type": "string"},
    "pattern": {"type": "string"},
    "graphic_policy": {"type": "string", "enum": ["exact", "mark-only", "omit"]},
    "graphic_text": {"type": "string"},
    "unknowns": {"type": "string"},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "description": {"type": "string"},
}

INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _ITEM_PROPS,
                "required": list(_ITEM_PROPS),
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

INVENTORY_PROMPT = f"""You are cataloguing the garments visible in this photograph for a wardrobe app.

List every deliberately worn or displayed garment: tops, bottoms, dresses, outerwear, shoes,
and notable accessories (belts, hats, bags, scarves). Ignore anything that is not clothing.
List at most {MAX_ITEMS} items, most prominent first. If a garment is mostly hidden or you
cannot tell what type it is, leave it out entirely.

For every item report ONLY what is actually visible in this photograph:
- slug: short lowercase-hyphenated english identifier, e.g. "navy-wool-cardigan"
- label: short human label in ITALIAN, max 3 words, e.g. "Cardigan blu"
- category: one of upper, lower, overall, outerwear, shoes, accessory
  (upper = top/shirt/sweater, lower = trousers/skirt/shorts, overall = dress/jumpsuit)
- bbox: [left, top, right, bottom] as floats 0..1 of the image, tight around the garment
  including the parts hidden behind arms, but not the whole person
- color_name: plain english colour description, e.g. "washed indigo"
- color_hex: dominant colour of the fabric as #rrggbb
- material: observed fabric and weight, e.g. "ribbed cotton jersey, mid weight"
- silhouette: cut and fit, e.g. "boxy, cropped, straight sleeves"
- construction: collar/neckline, cuffs, waistband, fastening, pockets you can actually see
- pattern: pattern or "solid"
- graphic_policy: "exact" if text/graphic is fully legible, "mark-only" if a graphic is
  visible but unreadable, "omit" if uncertain or none
- graphic_text: the exact legible text, otherwise empty string
- unknowns: attributes you cannot see (e.g. "back of the garment, hem shape"), empty if none
- confidence: high, medium or low
- description: one factual sentence

Never guess brands, logos, pockets or fasteners that are not clearly visible: prefer omission
over invention. Return JSON only."""


def _slugify(text: str, fallback: str = "capo") -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or fallback


def _clean_item(raw: dict, idx: int) -> dict | None:
    """Normalizza una voce dell'inventario; None se inutilizzabile."""
    cat = (raw.get("category") or "").strip().lower()
    if cat not in CATEGORIES:
        cat = "upper"
    bbox = raw.get("bbox") or []
    if len(bbox) != 4:
        return None
    try:
        l, t, r, b = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t
    l, t = max(0.0, min(l, 1.0)), max(0.0, min(t, 1.0))
    r, b = max(0.0, min(r, 1.0)), max(0.0, min(b, 1.0))
    if (r - l) < 0.04 or (b - t) < 0.04:  # riquadro degenere
        return None
    label = (raw.get("label") or "").strip() or CATEGORY_IT.get(cat, "Capo")
    return {
        "slug": _slugify(raw.get("slug") or label, f"capo-{idx}"),
        "label": label[:40],
        "category": cat,
        "bbox": [l, t, r, b],
        "color_name": (raw.get("color_name") or "").strip(),
        "color": (raw.get("color_hex") or "").strip() or None,
        "material": (raw.get("material") or "").strip(),
        "silhouette": (raw.get("silhouette") or "").strip(),
        "construction": (raw.get("construction") or "").strip(),
        "pattern": (raw.get("pattern") or "").strip(),
        "graphic_policy": (raw.get("graphic_policy") or "omit").strip(),
        "graphic_text": (raw.get("graphic_text") or "").strip(),
        "unknowns": (raw.get("unknowns") or "").strip(),
        "confidence": (raw.get("confidence") or "medium").strip(),
        "description": (raw.get("description") or "").strip(),
    }


def inventory(image: Image.Image, provider: str | None = None) -> list[dict]:
    """Elenca i capi visibili nella foto."""
    data = vision_json(INVENTORY_PROMPT, [image], INVENTORY_SCHEMA, provider)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    seen = set()
    for i, raw in enumerate(items[:MAX_ITEMS]):
        item = _clean_item(raw if isinstance(raw, dict) else {}, i)
        if not item:
            continue
        slug = item["slug"]
        n = 2
        while item["slug"] in seen:
            item["slug"] = f"{slug}-{n}"
            n += 1
        seen.add(item["slug"])
        out.append(item)
    return out


def crop_for_item(image: Image.Image, bbox, pad: float = PAD, canvas: int = CANVAS) -> Image.Image:
    """Ritaglio del capo con margine, centrato su tela quadrata neutra."""
    w, h = image.size
    l, t, r, b = bbox
    px, py = (r - l) * w * pad, (b - t) * h * pad
    box = (max(0, int(l * w - px)), max(0, int(t * h - py)),
           min(w, int(r * w + px)), min(h, int(b * h + py)))
    crop = image.convert("RGB").crop(box)
    side = max(crop.size)
    scale = min(1.0, canvas / side)
    if scale < 1.0:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    out = Image.new("RGB", (canvas, canvas), (245, 245, 245))
    out.paste(crop, ((canvas - crop.width) // 2, (canvas - crop.height) // 2))
    return out


_FRAMING = {
    "upper": "front view showing the neck opening, both complete sleeves, cuffs and the full hem",
    "outerwear": "front view showing collar, both complete sleeves, fastening and the full hem",
    "lower": "portrait view showing the waistband, both full legs and complete hems",
    "overall": "front view showing the neckline, the full length and the complete hem",
    "shoes": "matched pair, slightly elevated three-quarter view, both shoes complete",
    "accessory": "the complete item with both ends visible, long axis aligned to the canvas",
}


def build_prompt(item: dict, chroma_hex: str) -> str:
    """Prompt di ricostruzione: solo attributi osservati, nessuna invenzione."""
    name = item.get("color_name") and f"{item['color_name']} {item['label']}" or item["label"]
    chroma_name = {"#00ff00": "pure green", "#ff00ff": "pure magenta", "#0000ff": "pure blue"}.get(chroma_hex, "pure green")

    fidelity = [f"colour: {item['color_name'] or item.get('color') or 'as in the source'}"]
    for field, tag in (("material", "material"), ("silhouette", "silhouette"),
                       ("construction", "construction"), ("pattern", "pattern")):
        if item.get(field):
            fidelity.append(f"{tag}: {item[field]}")

    if item.get("graphic_policy") == "exact" and item.get("graphic_text"):
        graphic = f'Reproduce the visible graphic exactly as it appears, including the text "{item["graphic_text"]}".'
    elif item.get("graphic_policy") == "mark-only":
        graphic = ("A graphic is visible but not legible: render it as an abstract mark with the same "
                   "shape, placement and colours, and add no lettering.")
    else:
        graphic = "Omit all logos, lettering and branding: none is clearly readable in the source."

    unknowns = (f"Not visible in the source: {item['unknowns']}. Resolve these in the plainest possible way "
                "and add no detail.") if item.get("unknowns") else ""

    pair = " (a matched pair counts as one item)" if item["category"] == "shoes" else ""

    return f"""Use case: background-extraction
Asset type: transparent ecommerce clothing catalog cutout, generated first on a removable chroma key

Input image: the reference photograph shows the exact same {name} worn by a person. Use it only to
identify and reconstruct that single item. Do not mix in details from any other clothing in the frame.

Primary request: Reconstruct ONLY the complete empty {item['label']} ({item['category']}) as a clean
ecommerce catalog product photograph: {_FRAMING.get(item['category'], _FRAMING['upper'])}. Remove the
wearer, body, skin, hair, every other garment and the whole scene. Show the complete unoccluded item,
naturally and symmetrically arranged, as if laid flat and steamed, with no person, mannequin or hanger.

Item fidelity: preserve exactly what the source supports - {'; '.join(fidelity)}. {graphic} {unknowns}
Do not invent any other logo, lettering, label, pocket, seam, fastener, hardware, colour or decoration.

Composition: square canvas, item centred and complete inside the frame with generous even padding on
every side; nothing cropped or touching an edge.

Background: perfectly flat, absolutely uniform solid {chroma_name} ({chroma_hex}) from edge to edge.
Exactly one colour: no shadow, gradient, texture, vignette, floor, horizon, reflection or lighting variation.

Lighting: neutral diffuse high-end product lighting on the item only; no cast shadow, contact shadow,
reflection, prop, watermark, caption or border.

Critical: use no {chroma_name} anywhere on the item itself; keep a crisp separable outer silhouette;
output exactly one item{pair}."""


def reconstruct(image: Image.Image, item: dict, provider: str | None = None,
                attempts: int = 2) -> tuple[Image.Image, dict]:
    """Ricostruisce il capo e lo scontorna; ritenta con un'altra chiave se serve."""
    crop = crop_for_item(image, item["bbox"])
    keys = [pick_key_for_color(item.get("color"))]
    for alt in ("#ff00ff", "#00ff00", "#0000ff"):
        if alt not in keys:
            keys.append(alt)

    last: tuple[Image.Image, dict] | None = None
    for i in range(max(1, attempts)):
        key = keys[i % len(keys)]
        raw = generate_image(build_prompt(item, key), [crop], provider, size="1024x1024")
        cut, report = chroma_to_cutout(raw, key)
        report["chroma_key"] = key
        report["attempt"] = i + 1
        if report["ok"]:
            return cut, report
        last = (cut, report)
    return last  # type: ignore[return-value]


def _fallback_cutout(image: Image.Image, item: dict) -> tuple[Image.Image, dict]:
    """Senza chiave premium: ritaglio del riquadro e rimozione sfondo con rembg."""
    from garment_cutout import cutout_rgba

    crop = crop_for_item(image, item["bbox"], pad=0.06, canvas=900)
    cut = cutout_rgba(crop)
    from chroma import qa_cutout, trim_to_content

    cut = trim_to_content(cut)
    report = qa_cutout(cut)
    report["engine"] = "rembg"
    return cut, report


def extract_garments(image: Image.Image, provider: str | None = None,
                     progress=None, workers: int = 3) -> list[dict]:
    """Da una foto alla lista di capi pronti per il guardaroba."""
    def emit(**kw):
        if progress:
            try:
                progress(kw)
            except Exception:
                pass

    premium = resolve_provider(provider) is not None
    emit(stage="inventory", message="Cerco i capi nella foto…")

    if premium:
        items = inventory(image, provider)
    else:
        from mask_from_person import classify_garment
        items = [{
            "slug": "capo-1", "label": CATEGORY_IT.get(classify_garment(image), "Capo"),
            "category": classify_garment(image), "bbox": [0.0, 0.0, 1.0, 1.0],
            "color_name": "", "color": None, "material": "", "silhouette": "",
            "construction": "", "pattern": "", "graphic_policy": "omit", "graphic_text": "",
            "unknowns": "", "confidence": "low", "description": "",
        }]

    if not items:
        emit(stage="done", message="Nessun capo riconosciuto in questa foto.")
        return []

    emit(stage="found", total=len(items),
         items=[{"label": i["label"], "category": i["category"], "color": i.get("color")} for i in items],
         message=f"{len(items)} capi trovati, li ricostruisco…")

    done = 0

    def work(item):
        nonlocal done
        try:
            cut, report = reconstruct(image, item, provider) if premium else _fallback_cutout(image, item)
            item["image"] = cut
            item["qa"] = report
            item["engine"] = report.get("engine", "generativo")
        except Exception as exc:
            item["error"] = str(exc)
        done += 1
        emit(stage="item", index=done, total=len(items), label=item["label"],
             ok="error" not in item, error=item.get("error"))
        return item

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, items))

    ok = [r for r in results if r.get("image") is not None]
    emit(stage="done", total=len(ok), message=f"{len(ok)} capi pronti.")
    return ok
