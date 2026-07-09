# Vesta — try-on + color analysis (locale, Apple Silicon)

App mobile-web che gira **in locale** su Mac M4 (GPU Metal / MPS). Il Mac fa da
**server di inferenza** e serve anche il client web: un solo comando, un solo indirizzo.

Due pilastri:
1. **Color analysis** dalla foto → palette personale (stagione) + consigli colore.
2. **Virtual try-on** → ti vedi indossare i capi del guardaroba.

## Stack
- Try-on: **CatVTON** (SD1.5) su MPS, in **bfloat16** (metà memoria, sicuro sul VAE).
- Maschera capo: **segformer_b2_clothes** (niente detectron2/DensePose).
- Color analysis: ITA° in **CIELAB** + euristiche per le 4 stagioni (`color_analysis.py`).
- API + client: **FastAPI** (serve anche `web/` come app Preact, senza build).
- Modalità **cloud** (opz.): Space gratuito IDM-VTON via `gradio_client`. Ritaglio capi fotografati: `rembg` (u2netp).
- Pesi in `./.hf-cache` (~4 GB) e render in `outputs/cache` (cancellabili).

> ⚠️ **Licenza**: CatVTON è **non commerciale (ricerca)**. Ottimo per prototipare;
> prima di pubblicare va sostituito il motore con uno a licenza commerciale.
> L'architettura (`/tryon`, `/analyze`) resta identica.

## Avvio (macOS)
Semplice: doppio clic su **`Vesta.app`** (nella root del repo). Avvia il server e apre il browser.
Prima volta: tasto destro → **Apri** (Gatekeeper: app non firmata). Se compare "consentire a Python
connessioni in entrata?" → **Consenti** (serve per l'iPhone).

Equivalente da terminale (porta **8770**; la 8000 è di "AI Director"):
```bash
cd <repo>/backend
.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8770
```
Sul Mac apri `http://127.0.0.1:8770/`.

### Da iPhone (stessa Wi-Fi)
1. Avvia con `Vesta.app` (gira **nativo**, non in sandbox → raggiungibile in rete).
2. Sul telefono apri **`http://192.168.1.129:8770/`** (IP del Mac: `ipconfig getifaddr en0`).
3. Se non carica: Impostazioni di Sistema → Rete → Firewall → consenti Python; stessa Wi-Fi.
4. Foto su iPhone: usa "Carica una foto" → "Scatta foto" (la webcam web via http è bloccata da iOS).

App iOS nativa (WKWebView) in `ios/`. Override dtype: `VESTA_DTYPE=fp32 …`. Cloud: `HF_TOKEN` per più quota.

## Endpoint
- `GET /health` → stato + device.
- `POST /tryon` (multipart: `person`, `cloth`, `category` upper|lower|overall, `quality` fast|balanced|high, `mode` local|cloud) → PNG. Header `X-Inference-Seconds`, `X-Mode`, `X-Cache`. Cache su disco in `outputs/cache`.
- `POST /analyze` (multipart: `person`) → JSON: `season`, `undertone`, `depth`, `contrast`, `ita`, `palette`, `advice`.
- `POST /cutout` (multipart: `image`) → PNG del capo scontornato su bianco (rembg).

Modalità **cloud**: try-on via Space gratuito IDM-VTON (~25s, alta qualità); fallback automatico al locale se non disponibile. Imposta `HF_TOKEN` per quota più alta.

## App (web/index.html)
- Onboarding: carica la tua foto a figura intera (o "usa foto di esempio"). Resta in locale.
- Home: tu in primo piano, **palette personale + stagione + consiglio**, **Look del giorno**,
  inventario a slot (tap = indossa), selettore qualità, shuffle, confronto "originale".
- "Aggiungi"/fotocamera: carichi la foto di un capo e te lo provi addosso.
- Cache per capo (ri-selezione istantanea); stato di caricamento con timer per i ~90s.

## Tempi (M4 base 16 GB, bf16)
| preset | risoluzione | step | tempo ~ |
|---|---|---|---|
| fast | 384×512 | 30 | ~90s |
| balanced | 576×768 | 35 | ~3 min |
| high | 768×1024 | 45 | ~5 min |

Con bf16 una generazione usa ~1,9 GB di GPU. ⚠️ Tieni libera la RAM: avere **due** server
Vesta attivi insieme (o altri modelli grossi) manda il Mac in swap e rallenta tutto di 15-20×.

## File
- `server.py` — API + serve il client (carica la pipeline una volta, bf16, no-cache).
- `color_analysis.py` — analisi colore (ITA°/CIELAB → stagione + palette).
- `tryon.py` — try-on da riga di comando. `mask_from_person.py` — maschera capo.
- `download_weights.py` — scarica solo i pesi necessari. `check_env.py` — sanity MPS.
- `../web/` — client (index.html, garments/, person_sample.jpg, garments.json).
- `CatVTON/` — repo del modello (terze parti).
