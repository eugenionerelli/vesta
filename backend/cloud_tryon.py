"""Try-on via API gratuita: lo Space pubblico IDM-VTON su GPU Hugging Face.

Piu' veloce del Mac (~25s vs ~90s) e alta risoluzione. Usato come modalita' "cloud";
il server fa fallback al locale se questo fallisce (Space addormentato / quota GPU).
"""
import os
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf-cache"))

from gradio_client import Client, handle_file
from PIL import Image

SPACE = os.environ.get("VESTA_CLOUD_SPACE", os.environ.get("GIAMMI_CLOUD_SPACE", "yisol/IDM-VTON"))
_DESC = {"upper": "a shirt", "lower": "a pair of pants", "overall": "an outfit"}
_client = None


def _get_client():
    global _client
    if _client is None:
        tok = os.environ.get("HF_TOKEN")
        for kw in (([{"hf_token": tok}] if tok else []) + [{}]):
            try:
                _client = Client(SPACE, verbose=False, **kw)
                return _client
            except TypeError:
                continue
        _client = Client(SPACE, verbose=False)
    return _client


def cloud_tryon(person_path: str, cloth_path: str, category: str = "upper", steps: int = 30, seed: int = 42) -> Image.Image:
    client = _get_client()
    res = client.predict(
        {"background": handle_file(person_path), "layers": [], "composite": None},
        handle_file(cloth_path),
        _DESC.get(category, "a piece of clothing"),
        True,   # auto-mask
        False,  # auto-crop
        int(steps),
        int(seed),
        api_name="/tryon",
    )
    first = res[0] if isinstance(res, (list, tuple)) else res
    path = first["path"] if isinstance(first, dict) else first
    return Image.open(path).convert("RGB")


if __name__ == "__main__":
    import sys

    person = sys.argv[1] if len(sys.argv) > 1 else "/Users/eugenionerelli/dev/app-giammi/web/person_sample.jpg"
    cloth = sys.argv[2] if len(sys.argv) > 2 else "/Users/eugenionerelli/dev/app-giammi/web/garments/top_3.jpg"
    t = time.time()
    img = cloud_tryon(person, cloth, "upper")
    out = "/Users/eugenionerelli/dev/app-giammi/backend/outputs/cloud_test.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print(f"OK cloud in {time.time()-t:.1f}s -> {out} ({img.size})")
