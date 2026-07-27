# Vesta su iPhone

L'app iOS mostra l'interfaccia di Vesta servita dal Mac. Non e' un semplice contenitore:
gestisce il collegamento, gli stati di errore e la configurazione, cosi' non serve toccare
il codice quando cambia la rete.

Cosa fa
- schermata di avvio con il marchio, mentre si collega;
- se il Mac non risponde, spiega perche' e offre `Riprova` e `Cambia indirizzo`;
- l'indirizzo del server si imposta dall'app e resta salvato (di base il nome Bonjour del
  Mac, che non cambia quando cambia l'IP);
- **scuoti l'iPhone** per riaprire le impostazioni di collegamento;
- segue il tema chiaro/scuro di sistema, niente rimbalzo dello scroll, ritorno tattile al
  caricamento.

## Come crearla in Xcode
1. Xcode → New Project → iOS → App. Interface **SwiftUI**, Language **Swift**, nome `Vesta`.
2. Sostituisci il file `VestaApp.swift` generato con quello in questa cartella.
3. In target → Info aggiungi:
   - **App Transport Security Settings** → `NSAllowsLocalNetworking` = YES (http in rete locale);
   - **Privacy - Local Network Usage Description**: "Per collegarsi a Vesta sul Mac";
   - **Privacy - Camera Usage Description**: "Per scattare la foto del modello o di un capo";
   - **Privacy - Photo Library Usage Description**: "Per scegliere una foto dal rullino".
4. Avvia Vesta sul Mac, poi Run su iPhone (stessa Wi-Fi).

## Note
- La fotocamera dal web (`getUserMedia`) resta bloccata su http: il pulsante "Scegli una foto"
  apre comunque il selettore iOS con **Scatta foto** e **Libreria**, quindi fotocamera e rullino
  funzionano sia per la propria figura sia per i capi.
- Per l'App Store servira' un'interfaccia nativa che consuma le stesse API (linea guida 4.2):
  il percorso e' descritto in [PIANO.md](../PIANO.md).
