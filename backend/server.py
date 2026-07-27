"""Server di inferenza try-on (Vesta).

Carica la pipeline CatVTON UNA volta all'avvio (su MPS) e la riusa per ogni richiesta,
cosi' ogni chiamata paga solo il tempo di inferenza. Il client web lo chiama in rete locale.

Avvio:
  .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

BACKEND = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BACKEND, ".hf-cache")
CATVTON_DIR = os.path.join(BACKEND, "CatVTON")
os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, CATVTON_DIR)
sys.path.insert(0, BACKEND)

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from model.pipeline import CatVTONPipeline
from mask_from_person import garment_mask
from color_analysis import analyze as analyze_colors
from cloud_tryon import cloud_tryon
from premium_tryon import premium_tryon, resolve_provider, save_key, configured as premium_configured


def _read_paths() -> dict:
    paths = {}
    with open(os.path.join(BACKEND, "weights_paths.txt")) as fh:
        for line in fh:
            key, val = line.strip().split("=", 1)
            paths[key] = val
    return paths


# preset risoluzione/step: piu' alto = piu' bello ma piu' lento
QUALITY = {
    "fast": dict(width=384, height=512, steps=30),
    "balanced": dict(width=576, height=768, steps=35),
    "high": dict(width=768, height=1024, steps=45),
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bf16 di default: dimezza la memoria (cruciale su 16GB) ed e' piu' veloce su MPS;
# numericamente sicuro per il VAE (a differenza di fp16). Override con GIAMMI_DTYPE.
_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
    os.environ.get("VESTA_DTYPE", os.environ.get("GIAMMI_DTYPE", "bf16"))
]
_PATHS = _read_paths()

print(f"[vesta] carico la pipeline su {DEVICE} ({_DTYPE}) ...")
_t0 = time.perf_counter()
PIPE = CatVTONPipeline(
    base_ckpt=_PATHS["BASE"],
    attn_ckpt=_PATHS["MIX"],
    attn_ckpt_version="mix",
    weight_dtype=_DTYPE,
    device=DEVICE,
    skip_safety_check=True,
    use_tf32=True,
)
print(f"[vesta] pipeline pronta in {time.perf_counter() - _t0:.1f}s")

# una sola inferenza per volta: due diffusioni in parallelo farebbero esaurire la GPU
_LOCK = threading.Lock()

CACHE_DIR = os.path.join(BACKEND, "outputs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(title="Vesta try-on")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache(request, call_next):
    # evita che il browser serva una versione vecchia di index.html / garments.json
    response = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".html", ".json")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def _flatten_on_white(raw: bytes) -> bytes:
    """I capi del guardaroba sono PNG trasparenti: su bianco, mai su nero."""
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return raw
    if img.mode not in ("RGBA", "LA", "P"):
        return raw
    img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    buf = io.BytesIO()
    bg.save(buf, "JPEG", quality=94)
    return buf.getvalue()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE, "quality": list(QUALITY)}


@app.post("/tryon")
def tryon(
    person: UploadFile = File(...),
    cloth: UploadFile = File(...),
    category: str = Form("upper"),
    quality: str = Form("fast"),
    mode: str = Form("local"),
    provider: str = Form(""),
    garment_id: str = Form(""),
):
    q = QUALITY.get(quality, QUALITY["fast"])
    cat = category if category in ("upper", "lower", "overall") else "upper"
    person_bytes = _flatten_on_white(person.file.read())
    cloth_bytes = _flatten_on_white(cloth.file.read())

    mode_key = mode
    if mode == "premium":
        prov = resolve_provider(provider or None)
        if prov is None:
            return JSONResponse(status_code=400, headers={"Cache-Control": "no-store"},
                                content={"error": "Nessuna API key configurata: aggiungila in Profilo > Modelli premium."})
        mode_key = f"premium:{prov}"

    # cache su disco: stessa persona+capo+impostazioni -> ritorno immediato (anche pre-generato)
    key = hashlib.sha1(person_bytes + cloth_bytes + f"{cat}|{quality}|{mode_key}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, key + ".png")
    if os.path.exists(cache_path):
        return StreamingResponse(open(cache_path, "rb"), media_type="image/png",
                                 headers={"X-Cache": "hit", "X-Mode": mode_key})

    t0 = time.perf_counter()
    result = None
    used = mode_key
    if mode == "premium":
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        cloth_img = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
        meta = next((i for i in _load_wardrobe() if i.get("id") == garment_id), None) if garment_id else None
        try:
            result = premium_tryon(person_img, cloth_img, cat, provider or None, item=meta)
        except Exception as exc:
            print(f"[vesta] premium fallito: {exc}")
            return JSONResponse(status_code=502, headers={"Cache-Control": "no-store"},
                                content={"error": f"Generazione premium non riuscita. {exc}"})
    if mode == "cloud":
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as pf:
                pf.write(person_bytes); ppath = pf.name
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as cf:
                cf.write(cloth_bytes); cpath = cf.name
            result = cloud_tryon(ppath, cpath, cat)
        except Exception as exc:
            print(f"[vesta] cloud non disponibile ({exc}); fallback locale")
            used = "local-fallback"
            result = None

    if result is None:
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        cloth_img = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
        mask = garment_mask(person_img, cat)
        generator = torch.Generator(device="cpu").manual_seed(42)
        with _LOCK:
            result = PIPE(
                person_img, cloth_img, mask,
                num_inference_steps=q["steps"], guidance_scale=2.5,
                height=q["height"], width=q["width"], generator=generator,
            )[0]
        try:
            torch.mps.empty_cache()  # libera la cache MPS: evita la crescita a molti GB
        except Exception:
            pass

    dt = time.perf_counter() - t0
    result.save(cache_path, format="PNG")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"X-Inference-Seconds": f"{dt:.1f}", "X-Quality": quality, "X-Mode": used, "X-Cache": "miss"},
    )


@app.post("/analyze")
def analyze_endpoint(person: UploadFile = File(...)) -> dict:
    img = Image.open(io.BytesIO(person.file.read())).convert("RGB")
    return analyze_colors(img)


@app.post("/cutout")
def cutout_endpoint(image: UploadFile = File(...)) -> StreamingResponse:
    from garment_cutout import cutout_to_white  # import pigro: rembg pesante
    img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    out = cutout_to_white(img)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------- guardaroba
WEB_DIR_EARLY = os.path.join(os.path.dirname(BACKEND), "web")
WARDROBE_DIR = os.path.join(WEB_DIR_EARLY, "wardrobe")
DATA_DIR = os.path.join(BACKEND, "data")
WARDROBE_DB = os.path.join(DATA_DIR, "wardrobe.json")
os.makedirs(WARDROBE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_IMPORT_POOL = ThreadPoolExecutor(max_workers=2)


def _load_wardrobe() -> list[dict]:
    try:
        with open(WARDROBE_DB) as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_wardrobe(items: list[dict]) -> None:
    tmp = WARDROBE_DB + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, WARDROBE_DB)


def _color_distance(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 999.0
    try:
        pa = tuple(int(a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        pb = tuple(int(b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return 999.0
    return sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5


def _run_import(job_id: str, photos: list[bytes], provider: str | None) -> None:
    def emit(ev: dict) -> None:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["events"].append(ev)
                if ev.get("message"):
                    job["message"] = ev["message"]

    from garment_extract import extract_garments

    added: list[dict] = []
    try:
        for n, raw in enumerate(photos, 1):
            emit({"stage": "photo", "index": n, "total": len(photos),
                  "message": f"Foto {n} di {len(photos)}"})
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            for item in extract_garments(image, provider, progress=emit):
                img = item.pop("image", None)
                if img is None:
                    continue
                item_id = f"{item['slug']}-{uuid.uuid4().hex[:6]}"
                path = os.path.join(WARDROBE_DIR, item_id + ".png")
                img.save(path)
                existing = _load_wardrobe()
                dup = next((e for e in existing
                            if e.get("category") == item["category"]
                            and _color_distance(e.get("color"), item.get("color")) < 26), None)
                record = {
                    "id": item_id,
                    "label": item["label"],
                    "category": item["category"],
                    "file": f"wardrobe/{item_id}.png",
                    "color": item.get("color"),
                    "color_name": item.get("color_name"),
                    "material": item.get("material"),
                    "silhouette": item.get("silhouette"),
                    "construction": item.get("construction"),
                    "pattern": item.get("pattern"),
                    "description": item.get("description"),
                    "confidence": item.get("confidence"),
                    "engine": item.get("engine"),
                    "qa": item.get("qa"),
                    "created": time.time(),
                    "possible_duplicate_of": dup["id"] if dup else None,
                }
                _save_wardrobe(existing + [record])
                added.append(record)
                emit({"stage": "saved", "item": record})
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="done", items=added,
                                 message=f"{len(added)} capi aggiunti al guardaroba.")
    except Exception as exc:
        print(f"[vesta] import fallito: {exc}")
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error", error=str(exc), items=added,
                                 message=f"Import non riuscito. {exc}")


@app.post("/api/import")
def api_import(photos: list[UploadFile] = File(...), provider: str = Form("")):
    """Avvia l'estrazione dei capi dalle foto: risponde subito con l'id del lavoro."""
    raws = [p.file.read() for p in photos][:12]
    if not raws:
        return JSONResponse(status_code=400, content={"error": "nessuna foto ricevuta"})
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "events": [], "items": [],
                         "message": "Avvio…", "photos": len(raws), "created": time.time()}
    _IMPORT_POOL.submit(_run_import, job_id, raws, provider or None)
    return {"job_id": job_id, "photos": len(raws),
            "premium": resolve_provider(provider or None) is not None}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "lavoro non trovato"})
        return {k: v for k, v in job.items() if k != "events"} | {"events": job["events"][-40:]}


@app.get("/api/wardrobe")
def api_wardrobe() -> dict:
    return {"items": _load_wardrobe()}


@app.delete("/api/wardrobe/{item_id}")
def api_wardrobe_delete(item_id: str) -> dict:
    items = _load_wardrobe()
    keep = [i for i in items if i.get("id") != item_id]
    _save_wardrobe(keep)
    path = os.path.join(WARDROBE_DIR, item_id + ".png")
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True, "removed": len(items) - len(keep)}


@app.post("/api/wardrobe/{item_id}")
def api_wardrobe_update(item_id: str, label: str = Form(""), category: str = Form("")) -> dict:
    items = _load_wardrobe()
    for i in items:
        if i.get("id") == item_id:
            if label.strip():
                i["label"] = label.strip()[:40]
            if category.strip() in ("upper", "lower", "overall", "outerwear", "shoes", "accessory"):
                i["category"] = category.strip()
            _save_wardrobe(items)
            return {"ok": True, "item": i}
    return JSONResponse(status_code=404, content={"error": "capo non trovato"})


@app.get("/settings")
def settings_get() -> dict:
    # solo flag booleani: le chiavi non escono mai dal server
    return {"premium": premium_configured()}


@app.post("/settings")
def settings_post(provider: str = Form(...), key: str = Form("")):
    if provider not in ("openai", "gemini"):
        return JSONResponse(status_code=400, content={"error": "provider non valido"})
    save_key(provider, key)
    return {"ok": True, "premium": premium_configured()}


@app.post("/classify")
def classify_endpoint(image: UploadFile = File(...)) -> dict:
    from mask_from_person import classify_garment
    img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    return {"category": classify_garment(img)}


# serve il client web (index.html, guardaroba, ...) sullo stesso origine delle API.
# montato DOPO le route cosi' /health e /tryon hanno precedenza.
WEB_DIR = os.path.join(os.path.dirname(BACKEND), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    print(f"[vesta] client web servito da {WEB_DIR}")
