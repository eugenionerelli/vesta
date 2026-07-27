"""Ritaglio del capo fotografato: rimozione sfondo con rembg (alternativa pratica a SAM2),
poi composizione su bianco -> input pulito per il try-on."""
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf-cache"))

from PIL import Image

_session = None


def cutout_rgba(img: Image.Image) -> Image.Image:
    """Rimozione sfondo: restituisce RGBA con trasparenza."""
    from rembg import remove, new_session
    global _session
    if _session is None:
        _session = new_session("u2netp")  # modello leggero
    return remove(img.convert("RGB"), session=_session).convert("RGBA")


def cutout_to_white(img: Image.Image) -> Image.Image:
    rgba = cutout_rgba(img)
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


if __name__ == "__main__":
    import sys
    import time

    default = "/Users/eugenionerelli/dev/app-giammi/backend/CatVTON/resource/demo/example/condition/person/mison407622250d_1719258948458_2-0._QL90_UX564_V12524t6_.jpg"
    src = sys.argv[1] if len(sys.argv) > 1 else default
    t = time.time()
    out = cutout_to_white(Image.open(src))
    p = "/Users/eugenionerelli/dev/app-giammi/backend/outputs/cutout_test.png"
    out.save(p)
    print(f"OK cutout in {time.time()-t:.1f}s -> {p} {out.size}")
