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


def _prompt(category: str, item: dict | None = None) -> str:
    """Prompt del try-on. Con i dati del capo dal guardaroba diventa molto piu' preciso."""
    what = CATEGORY_EN.get(category, "piece of clothing")
    base = (
        "Photorealistic virtual try-on. Dress the person from the first image in the garment "
        f"shown in the second image (a {what}). Keep the person's face, hair, skin tone, pose, "
        "body shape and the background exactly the same. Replace only the corresponding clothing "
        "item. Natural fabric drape and folds, consistent lighting and shadows, high detail."
    )
    if not item:
        return base

    facts = []
    for field, tag in (("color_name", "colour"), ("material", "fabric"),
                       ("silhouette", "cut and fit"), ("construction", "construction"),
                       ("pattern", "pattern")):
        if item.get(field):
            facts.append(f"{tag}: {item[field]}")
    detail = f" The garment is shown as a catalog cutout on a plain background; reproduce it faithfully - {'; '.join(facts)}." if facts else ""
    return base + detail + (
        " Do not invent logos, lettering, pockets or hardware that are not visible in the garment image."
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
        if provider == "gemini":
            return RuntimeError(
                "Gemini: la chiave non viene accettata (401). Le chiavi che iniziano con 'AQ.' sono di "
                "Vertex AI express mode e spesso non sono abilitate: creane una che inizia con 'AIza' su "
                "aistudio.google.com > Get API key, con la fatturazione attiva sul progetto.")
        return RuntimeError(f"{provider}: chiave non valida o senza permessi. Controllala in Profilo > Modelli premium. ({msg[:160]})")
    if r.status_code == 400 and "billing" in msg.lower():
        return RuntimeError(f"{provider}: il piano dell'account non copre questo modello: attiva la fatturazione. ({msg[:160]})")
    return RuntimeError(f"{provider}: HTTP {r.status_code} - {msg}")


def gemini_base(key: str) -> str:
    """Le chiavi 'AQ.' sono di Vertex AI express mode: host e percorso diversi
    da quelle 'AIza' di AI Studio. Sceglie l'endpoint giusto per la chiave."""
    if (key or "").startswith("AQ."):
        return "https://aiplatform.googleapis.com/v1/publishers/google/models"
    return "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_post(model: str, body: dict, key: str, timeout: int = 240) -> requests.Response:
    return requests.post(f"{gemini_base(key)}/{model}:generateContent",
                         headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                         json=body, timeout=timeout)


def _to_gemini_schema(node):
    """Traduce uno schema in stile OpenAI nel dialetto OpenAPI di Gemini."""
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "additionalProperties":
            continue
        if k == "type" and isinstance(v, str):
            out["type"] = v.upper()
        elif k == "properties":
            out["properties"] = {pk: _to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out["items"] = _to_gemini_schema(v)
        else:
            out[k] = v
    return out


def generate_image(prompt: str, images: list[Image.Image], provider: str | None = None,
                   size: str = "1024x1024", quality: str = "high",
                   input_fidelity: str | None = "high") -> Image.Image:
    """Genera/modifica un'immagine col provider premium disponibile."""
    prov = resolve_provider(provider)
    if prov is None:
        raise RuntimeError("nessuna API key configurata: aggiungila in Profilo > Modelli premium")
    key = get_key(prov)

    if prov == "openai":
        model = os.environ.get("VESTA_OPENAI_IMAGE_MODEL", os.environ.get("GIAMMI_OPENAI_IMAGE_MODEL", "gpt-image-1"))
        files = [("image[]", (f"img{i}.jpg", _jpeg_bytes(im), "image/jpeg")) for i, im in enumerate(images)]
        data = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": "1"}
        if input_fidelity:
            data["input_fidelity"] = input_fidelity
        r = requests.post("https://api.openai.com/v1/images/edits",
                          headers={"Authorization": f"Bearer {key}"}, files=files, data=data, timeout=300)
        if r.status_code != 200:
            raise _err("openai", r)
        return Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB")

    model = os.environ.get("VESTA_GEMINI_IMAGE_MODEL", os.environ.get("GIAMMI_GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"))
    parts = [{"text": prompt}]
    for im in images:
        parts.append({"inlineData": {"mimeType": "image/jpeg",
                                     "data": base64.b64encode(_jpeg_bytes(im)).decode()}})
    r = _gemini_post(model, {"contents": [{"parts": parts}]}, key, timeout=300)
    if r.status_code != 200:
        raise _err("gemini", r)
    for cand in r.json().get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                return Image.open(io.BytesIO(base64.b64decode(blob["data"]))).convert("RGB")
    raise RuntimeError("gemini: nessuna immagine nella risposta (possibile blocco safety)")


def vision_json(prompt: str, images: list[Image.Image], schema: dict,
                provider: str | None = None) -> dict:
    """Interroga un modello vision e ottiene JSON conforme allo schema."""
    prov = resolve_provider(provider)
    if prov is None:
        raise RuntimeError("nessuna API key configurata: aggiungila in Profilo > Modelli premium")
    key = get_key(prov)

    if prov == "openai":
        model = os.environ.get("VESTA_OPENAI_VISION_MODEL", "gpt-4.1-mini")
        content = [{"type": "text", "text": prompt}]
        for im in images:
            b64 = base64.b64encode(_jpeg_bytes(im)).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": content}],
                  "response_format": {"type": "json_schema",
                                      "json_schema": {"name": "inventory", "strict": True, "schema": schema}}},
            timeout=180)
        if r.status_code != 200:
            raise _err("openai", r)
        return json.loads(r.json()["choices"][0]["message"]["content"])

    model = os.environ.get("VESTA_GEMINI_VISION_MODEL", "gemini-2.5-flash")
    parts = [{"text": prompt}]
    for im in images:
        parts.append({"inlineData": {"mimeType": "image/jpeg",
                                     "data": base64.b64encode(_jpeg_bytes(im)).decode()}})
    r = _gemini_post(model, {"contents": [{"parts": parts}],
                             "generationConfig": {"responseMimeType": "application/json",
                                                  "responseSchema": _to_gemini_schema(schema)}},
                     key, timeout=180)
    if r.status_code != 200:
        raise _err("gemini", r)
    txt = ""
    for cand in r.json().get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            txt += part.get("text") or ""
    if not txt.strip():
        raise RuntimeError("gemini: risposta vuota dall'analisi della foto")
    return json.loads(txt)


def premium_tryon(person: Image.Image, cloth: Image.Image, category: str = "upper",
                  provider: str | None = None, item: dict | None = None) -> Image.Image:
    prov = resolve_provider(provider)
    if prov is None:
        raise RuntimeError("nessuna API key configurata: aggiungila in Profilo > Modelli premium")
    prompt = _prompt(category, item)
    return generate_image(prompt, [person, cloth], prov, size="1024x1536",
                          quality="high", input_fidelity="high")
