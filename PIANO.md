# Piano di azione: da prototipo a prodotto App Store

Obiettivo: portare Giammi a un livello pubblicabile sull'App Store iOS e piacevole da usare,
con generazione basata su modelli a pagamento state-of-the-art e un modello di business in
abbonamento sostenibile per chi gestisce l'app.

## 1. Stato attuale

Funziona già: onboarding foto (upload/webcam) con segmentazione, analisi colore (stagione e
palette), guardaroba con filtri e auto-catalogazione dei capi fotografati, try-on con tre motori
(locale CatVTON, cloud gratuito IDM-VTON, premium a pagamento), cache dei render, PWA
installabile, tema automatico, layout desktop e smartphone.

Limiti per la produzione: i motori locale e cloud gratuito hanno licenza non commerciale e
qualità/tempi non da prodotto; il backend gira sul Mac dell'operatore; non esistono account,
pagamenti, moderazione.

## 2. Motore di generazione a pagamento (implementato)

La modalità Premium usa modelli con licenza commerciale via API, selezionabili con una chiave:

| Provider | Modello | Costo per immagine (indicativo) | Note |
|---|---|---|---|
| OpenAI | gpt-image-1 (quality high, 1024x1536, input_fidelity high) | ~0,17–0,25 $ | qualità massima, ottima aderenza al viso |
| OpenAI | gpt-image-1 (quality medium) | ~0,06–0,08 $ | buon compromesso |
| Google | gemini-2.5-flash-image | ~0,04 $ | ottima consistenza del soggetto, veloce |

Implementazione già nel repo: `backend/premium_tryon.py` (astrazione provider), endpoint
`/settings` per configurare le chiavi dal Profilo (salvate in `backend/.keys.json`, escluso da
git, mai esposte dalle API), `mode=premium` in `/tryon` con cache per provider ed errori
leggibili in interfaccia. Per provare: Profilo → Modelli premium → incolla la chiave → seleziona
Premium → tocca un capo.

Entrambi i provider applicano moderazione lato modello (nudità/contenuti vietati), utile anche
per i requisiti App Store.

## 3. Architettura di produzione

Con la generazione via API non serve alcuna GPU: il backend diventa un servizio web ordinario.

- Backend FastAPI su un VPS o PaaS (Hetzner/Fly.io/Railway): 10–20 €/mese.
- Postgres gestito (utenti, crediti, guardaroba, storico): 5–15 €/mese.
- Storage S3-compatibile per le immagini (Cloudflare R2): ~0–5 €/mese all'inizio.
- Autenticazione: Sign in with Apple + email; sessioni JWT.
- Contabilità crediti a registro (ledger) con idempotenza sui rinnovi.
- Abbonamenti: StoreKit 2 su iOS (obbligatorio per beni digitali) con App Store Server
  Notifications v2; Stripe Billing sulla versione web.
- Telemetria ed errori: Sentry + log strutturati; rate limiting per IP/utente.

Le modalità locale e cloud gratuito restano nella versione open-source self-hosted; il prodotto
commerciale usa solo Premium.

## 4. Economia dell'abbonamento

Ipotesi prezzi Italia, costo medio ponderato per render ~0,05 $ (mix Gemini + OpenAI medium,
retry inclusi).

| Tier | Prezzo | Crediti/mese | Costo API stimato | Margine lordo dopo Apple 15%* |
|---|---|---|---|---|
| Free | 0 € | 3 render (una tantum + 1/mese) | ~0,15 $ | acquisizione |
| Pro | 7,99 €/mese | 100 render | ~5 $ | ~1,8 € |
| Pro annuale | 59,99 €/anno | 100/mese | ~60 $/anno | ~9 € |
| Boost (consumabile) | 4,99 € | 50 render extra | ~2,5 $ | ~1,7 € |

*Small Business Program (<1 M$ di ricavi): commissione Apple 15%. Sul web con Stripe la
commissione scende a ~2%, quindi il web conviene come canale di vendita parallelo.

Un render gpt-image-1 high costa ~4–5 crediti; Gemini 1 credito. Il costo variabile resta
proporzionale all'uso: nessun rischio di costi fissi GPU. Break-even dell'infrastruttura
(~30 €/mese) con ~15–20 abbonati Pro.

Contromisure abuso free tier: limite per device + Sign in required, rate limiting, DeviceCheck.

## 5. Requisiti App Store (checklist)

- **4.2 Minimum functionality**: niente wrapper WKWebView; interfaccia nativa SwiftUI che
  consuma le stesse API del backend.
- **3.1.1 In-App Purchase**: abbonamenti solo via StoreKit 2; nessun link a pagamenti esterni.
- **4.8 Sign in with Apple** accanto ad altri login.
- **5.1.1(v)**: cancellazione account e dati dall'app.
- **Privacy**: App Privacy labels, privacy policy, consenso esplicito all'uso delle foto,
  conservazione minima (render eliminabili, foto non usate per training).
- **Sicurezza contenuti (1.1)**: solo foto proprie (dichiarazione), moderazione provider,
  blocco minori (rating 12+), meccanismo di segnalazione.
- **2.1 Completeness**: account demo per la review, nessuna feature rotta, gestione offline.
- Permessi con testi chiari (camera, foto), icona/screenshots/descrizione, EULA.

## 6. Piano UX iOS (app nativa SwiftUI)

1. Onboarding in tre passi: promessa ("provati i vestiti senza provarli") → scatto guidato con
   sagoma e controllo luce → reveal della palette personale come primo momento di valore.
2. Cattura con Vision on-device per la segmentazione della persona (istantanea, privata).
3. Home "specchio": figura grande, outfit del giorno, azioni rapide; haptics sulle azioni.
4. Guardaroba: griglia con categorie auto, aggiunta da foto/rullino e da share extension
   (aggiungi un capo da Safari/Instagram con Condividi → Giammi).
5. Try-on in coda con notifica push al termine; storico dei look; confronto prima/dopo con
   slider; salvataggio e condivisione.
6. Paywall dopo il primo render gratuito (momento wow), trial 3 giorni, ripristino acquisti.
7. Idee: outfit del giorno legato a meteo e stagione, spiegazione del perché un capo funziona
   con la palette.
8. Rifiniture: SF Symbols, Dynamic Type, VoiceOver, localizzazione IT/EN, dark/light.

## 7. Piano UX web

1. Landing con esempi pre-generati e prova senza registrazione (1 render demo).
2. App (evoluzione della PWA attuale): drag & drop dei capi, incolla immagine dagli appunti,
   upload multiplo, galleria dei look con confronto a slider, condivisione di un look via link,
   scorciatoie tastiera; su desktop layout a tre colonne (guardaroba | specchio | dettagli).
3. Stripe Checkout + Customer Portal per gli abbonamenti web.
4. Accessibilità: focus visibili, aria-label, contrasto AA.

## 8. Roadmap

| Fase | Contenuto | Stima |
|---|---|---|
| F1 | Backend multi-utente hosted: auth, crediti, Stripe web, moderazione, storage | 1–2 settimane |
| F2 | App iOS nativa MVP: onboarding, cattura, guardaroba, try-on premium, paywall StoreKit | 2–3 settimane |
| F3 | TestFlight beta, telemetria, iterazione su prompt/provider e qualità | 1 settimana |
| F4 | Submission: privacy labels, review notes, account demo, lancio | 1 settimana |

In parallelo: privacy policy e ToS (template + revisione), landing page, materiale store.

## 9. Rischi principali

- Costi API da abuso del free tier → limiti per device, login obbligatorio oltre il demo.
- Qualità non uniforme tra provider → routing per caso d'uso, retry automatico sull'altro
  provider, valutazione periodica dei prompt.
- Review Apple sui contenuti con persone → policy "solo foto proprie", moderazione, rating 12+.
- GDPR → cancellazione completa, data minimization, DPA con i provider.
