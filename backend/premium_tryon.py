"""Try-on con modelli premium a pagamento (API key dell'operatore).

Provider supportati:
- openai -> gpt-image-1 (edit multi-immagine, input_fidelity=high per preservare il viso)
- gemini -> gemini-2.5-flash-image (ottima consistenza del soggetto)

Le chiavi si configurano dal Profilo dell'app (salvate in backend/.keys.json, fuori
da git, chmod 600) oppure via variabili d'ambiente OPENAI_API_KEY / GEMINI_API_KEY.
"""
import base64
import io
import json
import os

import requests
from PIL import Image

BACKEND = os.path.dirname(os.path.abspath(__file__))
KEYS_PATH = os.path.join(BACKEND, ".keys.json")

PROVIDERS = ("openai", "gemini")
_ENV_VAR = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}

CATEGORY_EN = {"upper": "top", "lower": "pair of pants", "overall": "full outfit/dress"}


def _load_keys() -> dict:
    try:
        with open(KEYS_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_key(provider: str, key: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"provider sconosciuto: {provider}")
    keys = _load_keys()
    key = (key or "").strip()
    if key:
        keys[provider] = key
    else:
        keys.pop(provider, None)
    with open(KEYS_PATH, "w") as fh:
        json.dump(keys, fh)
    os.chmod(KEYS_PATH, 0o600)


def get_key(provider: str) -> str | None:
    return _load_keys().get(provider) or os.environ.get(_ENV_VAR[provider]) or None


def configured() -> dict:
    return {p: bool(get_key(p)) for p in PROVIDERS}


def resolve_provider(requested: str | None = None) -> str | None:
    if requested in PROVIDERS:
        return requested if get_key(requested) else None
    for p in PROVIDERS:
        if get_key(p):
            return p
    return None


def _prompt(category: str) -> str:
    what = CATEGORY_EN.get(category, "piece of clothing")
    return (
        "Photorealistic virtual try-on. Dress the person from the first image in the garment "
        f"shown in the second image (a {what}). Keep the person's face, hair, skin tone, pose, "
        "body shape and the background exactly the same. Replace only the corresponding clothing "
        "item. Natural fabric drape and folds, consistent lighting and shadows, high detail."
    )


def _jpeg_bytes(img: Image.Image, max_side: int = 1280) -> bytes:
    img = img.convert("RGB")
    s = min(1.0, max_side / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _err(provider: str, r: requests.Response) -> RuntimeError:
    try:
        j = r.json()
        msg = (j.get("error") or {}).get("message") or str(j)[:300]
    except Exception:
        msg = r.text[:300]
    # errori comuni tradotti in azioni concrete
    if r.status_code == 429 and provider == "gemini" and "free_tier" in msg:
        return RuntimeError(
            "Gemini: la chiave gratuita non include i modelli di generazione immagini "
            "(limite free tier = 0). Attiva la fatturazione del progetto su aistudio.google.com "
            "(Get API key > progetto > Set up billing): ogni immagine costa ~0,04 $."
        )
    if r.status_code == 429:
        return RuntimeError(f"{provider}: limite di richieste raggiunto, riprova tra poco. ({msg[:160]})")
    if r.status_code in (401, 403):
        return RuntimeError(f"{provider}: chiave non valida o senza permessi. Controllala in Profilo > Modelli premium. ({msg[:160]})")
    if r.status_code == 400 and "billing" in msg.lower():
        return RuntimeError(f"{provider}: il piano dell'account non copre questo modello: attiva la fatturazione. ({msg[:160]})")
    return RuntimeError(f"{provider}: HTTP {r.status_code} - {msg}")


def _openai(person: Image.Image, cloth: Image.Image, category: str, key: str) -> Image.Image:
    model = os.environ.get("VESTA_OPENAI_IMAGE_MODEL", os.environ.get("GIAMMI_OPENAI_IMAGE_MODEL", "gpt-image-1"))
    r = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {key}"},
        files=[
            ("image[]", ("person.jpg", _jpeg_bytes(person), "image/jpeg")),
            ("image[]", ("garment.jpg", _jpeg_bytes(cloth), "image/jpeg")),
        ],
        data={
            "model": model,
            "prompt": _prompt(category),
            "size": "1024x1536",
            "quality": "high",
            "input_fidelity": "high",
            "n": "1",
        },
        timeout=240,
    )
    if r.status_code != 200:
        raise _err("openai", r)
    b64 = r.json()["data"][0]["b64_json"]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _gemini(person: Image.Image, cloth: Image.Image, category: str, key: str) -> Image.Image:
    model = os.environ.get("VESTA_GEMINI_IMAGE_MODEL", os.environ.get("GIAMMI_GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"))
    body = {
        "contents": [{
            "parts": [
                {"text": _prompt(category)},
                {"inlineData": {"mimeType": "image/jpeg",
                                "data": base64.b64encode(_jpeg_bytes(person)).decode()}},
                {"inlineData": {"mimeType": "image/jpeg",
                                "data": base64.b64encode(_jpeg_bytes(cloth)).decode()}},
            ],
        }],
    }
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=body,
        timeout=240,
    )
    if r.status_code != 200:
        raise _err("gemini", r)
    for cand in r.json().get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                return Image.open(io.BytesIO(base64.b64decode(blob["data"]))).convert("RGB")
    raise RuntimeError("gemini: nessuna immagine nella risposta (possibile blocco safety)")


def premium_tryon(person: Image.Image, cloth: Image.Image, category: str = "upper",
                  provider: str | None = None) -> Image.Image:
    prov = resolve_provider(provider)
    if prov is None:
        raise RuntimeError("nessuna API key configurata: aggiungila in Profilo > Modelli premium")
    if prov == "openai":
        return _openai(person, cloth, category, get_key("openai"))
    return _gemini(person, cloth, category, get_key("gemini"))
