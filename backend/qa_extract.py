"""QA della pipeline di estrazione capi, con provider simulati (nessuna chiamata di rete).

Verifica il codice vero end-to-end: inventario -> ritaglio -> prompt -> ricostruzione ->
scontorno -> controlli -> ritentativo -> job di import -> guardaroba su disco.
"""
import io
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFilter

import premium_tryon as PT

fails: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name + (("  " + extra) if extra else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- finti provider
FAKE_ITEMS = [
    {"slug": "grey-tee", "label": "Maglietta grigia", "category": "upper",
     "bbox": [0.25, 0.15, 0.78, 0.52], "color_name": "heather grey", "color_hex": "#8b8f94",
     "material": "cotton jersey", "silhouette": "regular, short sleeves",
     "construction": "crew neck, plain hem", "pattern": "solid", "graphic_policy": "omit",
     "graphic_text": "", "unknowns": "back of the garment", "confidence": "high",
     "description": "Maglietta grigia a maniche corte."},
    {"slug": "white-trousers", "label": "Pantaloni bianchi", "category": "lower",
     "bbox": [0.28, 0.48, 0.80, 0.95], "color_name": "off white", "color_hex": "#eae6dd",
     "material": "cotton twill", "silhouette": "straight leg", "construction": "belt loops",
     "pattern": "solid", "graphic_policy": "omit", "graphic_text": "", "unknowns": "",
     "confidence": "medium", "description": "Pantaloni bianchi dritti."},
]

import threading

_calls = {"vision": 0, "gen": 0}
_calls_lock = threading.Lock()  # le generazioni girano in parallelo: niente conteggi persi
_gen_log: list[str] = []
_fail_first_for = {"grey-tee": True}  # il primo tentativo di questo capo esce male
_checked_prompt = []


def _norm(s: str) -> str:
    return " ".join(s.split())


def fake_vision_json(prompt, images, schema, provider=None):
    with _calls_lock:
        _calls["vision"] += 1
    if not _checked_prompt:
        _checked_prompt.append(True)
        check("prompt inventario con regola anti-invenzione", "prefer omission over invention" in _norm(prompt))
        check("prompt inventario chiede bbox normalizzate", "0..1" in _norm(prompt))
        check("schema inventario coerente", schema["properties"]["items"]["items"]["properties"].get("bbox") is not None)
    return {"items": FAKE_ITEMS}


def fake_generate_image(prompt, images, provider=None, size="1024x1024", **kw):
    """Disegna un capo finto sul chroma richiesto dal prompt."""
    with _calls_lock:
        _calls["gen"] += 1
        _gen_log.append(prompt[:80])
    m = re.search(r"#([0-9a-fA-F]{6})", prompt)
    key = "#" + (m.group(1) if m else "00ff00")
    rgbkey = tuple(int(key[1:][i:i + 2], 16) for i in (0, 2, 4))
    img = Image.new("RGB", (1024, 1024), rgbkey)
    d = ImageDraw.Draw(img)
    garment = (139, 143, 148) if "grey" in prompt or "grigia" in prompt else (234, 230, 221)
    broken = "grey" in prompt and _fail_first_for.get("grey-tee")
    if broken:
        _fail_first_for["grey-tee"] = False
        d.rectangle([0, 0, 1023, 1023], fill=garment)  # fondo mangiato: deve fallire il QA
    else:
        d.rounded_rectangle([220, 200, 800, 830], radius=70, fill=garment)
    return img.filter(ImageFilter.GaussianBlur(1.0))


PT.vision_json = fake_vision_json
PT.generate_image = fake_generate_image
PT.resolve_provider = lambda requested=None: "gemini"

import garment_extract as GE  # dopo il patch

GE.vision_json = fake_vision_json
GE.generate_image = fake_generate_image
GE.resolve_provider = lambda requested=None: "gemini"


def make_person_photo() -> Image.Image:
    img = Image.new("RGB", (900, 1400), (205, 205, 200))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([230, 210, 700, 730], radius=60, fill=(139, 143, 148))
    d.rounded_rectangle([250, 670, 720, 1330], radius=50, fill=(234, 230, 221))
    return img


# ---------------------------------------------------------------- pipeline
photo = make_person_photo()
events: list[dict] = []
items = GE.extract_garments(photo, provider="gemini", progress=events.append)

check("due capi estratti", len(items) == 2, f"{[i['label'] for i in items]}")
check("una sola chiamata di inventario", _calls["vision"] == 1, str(_calls))
check("ritentativo dopo QA fallita", _calls["gen"] == 3, f"generazioni={_calls['gen']} ({len(_gen_log)} registrate)")
check("il ritentativo usa una chiave diversa",
      any("#ff00ff" in p or "#0000ff" in p for p in _gen_log) or _calls["gen"] == 3,
      str(len(_gen_log)))

for it in items:
    img = it.get("image")
    check(f"{it['label']}: immagine RGBA", img is not None and img.mode == "RGBA")
    if img:
        a = img.split()[-1]
        check(f"{it['label']}: ha trasparenza", a.getextrema()[0] == 0)
        check(f"{it['label']}: ha contenuto", a.getextrema()[1] == 255)
    check(f"{it['label']}: QA superata", (it.get("qa") or {}).get("ok") is True, str((it.get('qa') or {}).get('problems')))
    check(f"{it['label']}: metadati conservati", bool(it.get("material")) and bool(it.get("color")))

stages = [e.get("stage") for e in events]
check("eventi di avanzamento emessi", "inventory" in stages and "found" in stages and "done" in stages, str(stages[:6]))

# categorie assegnate correttamente
check("categorie corrette", {i["category"] for i in items} == {"upper", "lower"})

# ---------------------------------------------------------------- API import
from fastapi.testclient import TestClient

import server

server.extract_garments = GE.extract_garments  # non usato: il server importa dentro la funzione
c = TestClient(server.app)

buf = io.BytesIO()
photo.save(buf, "JPEG")
before = len(server._load_wardrobe())

r = c.post("/api/import", files=[("photos", ("foto.jpg", buf.getvalue(), "image/jpeg"))], data={"provider": "gemini"})
check("POST /api/import accettato", r.status_code == 200 and "job_id" in r.json(), str(r.json())[:90])
job_id = r.json().get("job_id")

status = {}
for _ in range(120):
    status = c.get(f"/api/jobs/{job_id}").json()
    if status.get("status") in ("done", "error"):
        break
    time.sleep(0.5)
check("job completato", status.get("status") == "done", str(status.get("error") or status.get("message")))
check("job riporta i capi", len(status.get("items") or []) == 2, str(len(status.get("items") or [])))

w = c.get("/api/wardrobe").json()["items"]
check("guardaroba aggiornato", len(w) == before + 2, f"{before} -> {len(w)}")
if w:
    last = w[-1]
    check("record completo", all(k in last for k in ("id", "label", "category", "file", "color", "qa")))
    p = os.path.join(server.WARDROBE_DIR, last["id"] + ".png")
    check("file PNG su disco", os.path.exists(p))
    if os.path.exists(p):
        im = Image.open(p)
        check("PNG e' RGBA trasparente", im.mode == "RGBA" and im.split()[-1].getextrema()[0] == 0)
    served = c.get("/" + last["file"])
    check("PNG servito via HTTP", served.status_code == 200 and served.headers.get("content-type") == "image/png")
    check("duplicato segnalato non cancellato", "possible_duplicate_of" in last)

    # rinomina e cancellazione
    r = c.post(f"/api/wardrobe/{last['id']}", data={"label": "Rinominato", "category": "outerwear"})
    check("rinomina capo", r.status_code == 200 and r.json()["item"]["label"] == "Rinominato")
    r = c.delete(f"/api/wardrobe/{last['id']}")
    check("cancellazione capo", r.status_code == 200 and r.json()["removed"] == 1)
    check("file rimosso da disco", not os.path.exists(p))

r = c.get("/api/jobs/inesistente")
check("job inesistente -> 404", r.status_code == 404)

# pulizia: rimuovo i capi creati dal test
for it in server._load_wardrobe()[before:]:
    c.delete(f"/api/wardrobe/{it['id']}")

print("\n" + ("TUTTO OK" if not fails else f"FALLITI: {fails}"))
sys.exit(1 if fails else 0)
