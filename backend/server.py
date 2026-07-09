"""Server di inferenza try-on (Vesta).

Carica la pipeline CatVTON UNA volta all'avvio (su MPS) e la riusa per ogni richiesta,
cosi' ogni chiamata paga solo il tempo di inferenza. Il client web lo chiama in rete locale.

Avvio:
  .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""
import hashlib
import io
import os
import sys
import tempfile
import threading
import time

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
):
    q = QUALITY.get(quality, QUALITY["fast"])
    cat = category if category in ("upper", "lower", "overall") else "upper"
    person_bytes = person.file.read()
    cloth_bytes = cloth.file.read()

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
        try:
            result = premium_tryon(person_img, cloth_img, cat, provider or None)
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
