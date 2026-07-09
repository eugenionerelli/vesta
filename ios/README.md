# Vesta — app iOS (SwiftUI, base)

Versione iOS come "contenitore" nativo del web-client servito dal Mac (WKWebView).
È la base richiesta: l'app web (responsive) resta il cuore; questa la incapsula su iPhone.

## Come crearla in Xcode (5 minuti)
1. Xcode → New Project → iOS → App. Interface: **SwiftUI**, Language: **Swift**. Nome: `Vesta`.
2. Sostituisci il file `VestaApp.swift` generato con quello in questa cartella.
3. Imposta l'IP del Mac in `SERVER_URL` (sul Mac: `ipconfig getifaddr en0` → ora `192.168.1.129`).
4. In `Info` (target → Info / Custom iOS Target Properties) aggiungi:
   - **App Transport Security Settings** → `NSAllowsLocalNetworking` = YES (permette http sulla rete locale).
   - **Privacy - Local Network Usage Description** = "Per collegarsi al try-on sul Mac".
   - **Privacy - Camera Usage Description** = "Per scattare la foto del modello/capo".
   - **Privacy - Photo Library Usage Description** = "Per scegliere una foto".
5. Avvia il server sul Mac (apri `Vesta.app` o `uvicorn`), poi Run su iPhone (stessa Wi-Fi).

## Note
- La fotocamera: nel WKWebView su http la `getUserMedia` ("Scatta con la webcam") è bloccata,
  ma il pulsante **"Carica una foto"** apre il selettore iOS con **"Scatta foto"/"Libreria"** → la
  fotocamera funziona comunque per la foto modello e per i capi.
- Evoluzione: bridge nativo fotocamera/share-extension, o impacchettare il backend (per ora gira sul Mac).
