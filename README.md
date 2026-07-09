# Giammi

Armadio virtuale per provare i vestiti addosso. Carichi una foto a figura intera, scegli un capo dal guardaroba (o lo fotografi) e un modello di virtual try-on te lo fa indossare. Stima anche i tuoi colori e propone una palette.

Vibe coded da Eugenio Nerelli e Gianmattia Barone.

## Funzioni
- Try-on a partire da una foto dell'utente.
- Analisi colore: sottotono della pelle (ITA in CIELAB), stagione e palette.
- Guardaroba filtrabile per tipo (maglie, pantaloni, abiti).
- Aggiunta di capi da foto, con rimozione dello sfondo e categoria assegnate in automatico.
- Generazione in locale sul Mac, su GPU cloud gratuita, oppure in modalità Premium con modelli a pagamento (OpenAI gpt-image-1 o Google gemini-2.5-flash-image) inserendo la propria API key dal Profilo.
- Interfaccia responsive (desktop e smartphone), tema chiaro/scuro automatico.

## Stack
Frontend in Preact servito dal backend, senza step di build. Backend in FastAPI su Apple Silicon (Metal/MPS). Try-on con CatVTON in locale e IDM-VTON via cloud. Segmentazione con segformer e rembg.

## Requisiti
Mac Apple Silicon e Python 3.11 per la generazione locale. Su altri sistemi resta disponibile solo la modalità cloud.

## Avvio
```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r ../requirements.txt
git clone https://github.com/Zheng-Chong/CatVTON
.venv/bin/python download_weights.py
.venv/bin/python prep_wardrobe.py
.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8770
```
Apri http://127.0.0.1:8770, oppure http://IP-DEL-MAC:8770 dallo smartphone sulla stessa rete.

## App iOS
Scaffold SwiftUI (WKWebView verso il server) nella cartella `ios/`.

## Modalità Premium
Dal Profilo si inserisce una API key OpenAI o Google Gemini: la generazione passa ai modelli a pagamento, con qualità e tempi da prodotto. Le chiavi restano in `backend/.keys.json` (escluso da git) o nelle variabili d'ambiente `OPENAI_API_KEY` / `GEMINI_API_KEY`, e non vengono mai esposte dalle API. Il piano completo per App Store e abbonamenti è in [PIANO.md](PIANO.md).

## Note
La modalità cloud usa una GPU gratuita di Hugging Face con quota limitata: a quota esaurita l'app torna alla generazione locale. CatVTON e IDM-VTON hanno licenza non commerciale; la modalità Premium usa API con termini commerciali.
