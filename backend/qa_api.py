"""QA headless dell'API e degli asset statici (TestClient in-process)."""
import io
import os
import sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND)

from fastapi.testclient import TestClient
from PIL import Image

import server

c = TestClient(server.app)
fails = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (("  " + extra) if extra else ""))
    if not cond:
        fails.append(name)


r = c.get("/health")
check("health 200", r.status_code == 200, str(r.json()))

# statici principali
for path, ctype in [
    ("/", "text/html"),
    ("/manifest.json", "application/json"),
    ("/icon-192.png", "image/png"),
    ("/icon-512.png", "image/png"),
    ("/garments.json", "application/json"),
    ("/person_sample.jpg", "image/jpeg"),
]:
    r = c.get(path)
    check(f"GET {path}", r.status_code == 200 and ctype in r.headers.get("content-type", ""),
          f"{r.status_code} {r.headers.get('content-type','')}")

idx = c.get("/").text
for frag in ["manifest.json", "apple-touch-icon", "theme-color", "Attendi: sto già creando un look",
             "Scatta con la webcam", "prefers-color-scheme", "<title>Vesta</title>", "vesta.person"]:
    check(f"index contiene '{frag[:34]}'", frag in idx)
check("nessun 'giammi.' residuo nell'index", "giammi." not in idx.replace("localStorage.getItem('giammi.", "OKMIGRAZIONE"))

mj = c.get("/manifest.json").json()
check("manifest name Vesta", mj.get("name") == "Vesta" and mj.get("short_name") == "Vesta")
check("favicon servito", c.get("/favicon.png").status_code == 200)

# no-cache sugli html/json
r = c.get("/")
check("no-cache su /", "no-cache" in r.headers.get("cache-control", ""))

person_path = os.path.join(os.path.dirname(BACKEND), "web", "person_sample.jpg")
cloth_path = os.path.join(os.path.dirname(BACKEND), "web", "garments", "top_3.jpg")

# /analyze
with open(person_path, "rb") as f:
    r = c.post("/analyze", files={"person": ("p.jpg", f, "image/jpeg")})
j = r.json() if r.status_code == 200 else {}
check("/analyze 200 + season", r.status_code == 200 and "season" in j, str(j.get("season")))

# /classify
with open(cloth_path, "rb") as f:
    r = c.post("/classify", files={"image": ("c.jpg", f, "image/jpeg")})
check("/classify 200 upper", r.status_code == 200 and r.json().get("category") == "upper", str(r.json()))

# /cutout
with open(person_path, "rb") as f:
    r = c.post("/cutout", files={"image": ("p.jpg", f, "image/jpeg")})
ok = r.status_code == 200 and r.headers.get("content-type") == "image/png"
if ok:
    Image.open(io.BytesIO(r.content)).verify()
check("/cutout 200 png valido", ok, f"{len(r.content)//1024} KB")

# /tryon mode=local: prima chiamata potenzialmente lenta -> uso la cache se esiste,
# altrimenti genero davvero (30 step) per validare il percorso completo.
with open(person_path, "rb") as fp, open(cloth_path, "rb") as fc:
    r = c.post("/tryon", files={"person": ("p.jpg", fp, "image/jpeg"), "cloth": ("c.jpg", fc, "image/jpeg")},
               data={"category": "upper", "quality": "fast", "mode": "local"})
ok = r.status_code == 200 and r.headers.get("content-type") == "image/png"
if ok:
    Image.open(io.BytesIO(r.content)).verify()
check("/tryon local 200 png", ok, f"cache={r.headers.get('x-cache')} {r.headers.get('x-inference-seconds','')}s")

# seconda chiamata identica: deve essere cache hit immediata
with open(person_path, "rb") as fp, open(cloth_path, "rb") as fc:
    r = c.post("/tryon", files={"person": ("p.jpg", fp, "image/jpeg"), "cloth": ("c.jpg", fc, "image/jpeg")},
               data={"category": "upper", "quality": "fast", "mode": "local"})
check("/tryon cache hit", r.status_code == 200 and r.headers.get("x-cache") == "hit")

# ---- premium: settings roundtrip e gestione errori (nessuna chiave reale) ----
import premium_tryon as PT

r = c.get("/settings")
check("/settings GET", r.status_code == 200 and "premium" in r.json(), str(r.json()))
had_key = any(r.json()["premium"].values())

if not had_key:
    r = c.post("/tryon", files={"person": ("p.jpg", open(person_path, "rb"), "image/jpeg"),
                                "cloth": ("c.jpg", open(cloth_path, "rb"), "image/jpeg")},
               data={"category": "upper", "quality": "fast", "mode": "premium"})
    check("/tryon premium senza chiave -> 400 + errore", r.status_code == 400 and "error" in r.json(), str(r.json())[:90])

    r = c.post("/settings", data={"provider": "openai", "key": "sk-test-fake"})
    check("/settings POST salva", r.status_code == 200 and r.json()["premium"]["openai"] is True)
    check(".keys.json creato 600", os.path.exists(PT.KEYS_PATH) and oct(os.stat(PT.KEYS_PATH).st_mode)[-3:] == "600")

    import subprocess
    ign = subprocess.run(["git", "check-ignore", PT.KEYS_PATH],
                         cwd=os.path.dirname(BACKEND), capture_output=True)
    check(".keys.json ignorato da git", ign.returncode == 0)

    r = c.post("/tryon", files={"person": ("p.jpg", open(person_path, "rb"), "image/jpeg"),
                                "cloth": ("c.jpg", open(cloth_path, "rb"), "image/jpeg")},
               data={"category": "upper", "quality": "fast", "mode": "premium"})
    check("/tryon premium chiave finta -> 502 + msg provider",
          r.status_code == 502 and "openai" in r.json().get("error", ""), str(r.json())[:110])

    r = c.post("/settings", data={"provider": "openai", "key": ""})
    check("/settings POST rimuove", r.status_code == 200 and r.json()["premium"]["openai"] is False)
else:
    print("SKIP  test premium distruttivi (chiave reale gia' configurata)")

r = c.post("/settings", data={"provider": "boh", "key": "x"})
check("/settings provider invalido -> 400", r.status_code == 400)

print("\n" + ("TUTTO OK" if not fails else f"FALLITI: {fails}"))
sys.exit(1 if fails else 0)
