import SwiftUI
import WebKit
import UIKit

// L'app iOS e' il camerino portatile: mostra l'interfaccia di Vesta servita dal Mac.
// L'indirizzo non e' piu' scritto nel codice: si imposta dall'app e resta salvato.

private let kServerKey = "vesta.server"
// nome Bonjour del Mac: non cambia quando cambia l'IP
private let kDefaultServer = "http://MacBook-Pro-di-Eugenio.local:8770"

final class Settings: ObservableObject {
    @Published var server: String {
        didSet { UserDefaults.standard.set(server, forKey: kServerKey) }
    }
    init() { server = UserDefaults.standard.string(forKey: kServerKey) ?? kDefaultServer }
    var url: URL? { URL(string: server.trimmingCharacters(in: .whitespaces)) }
}

enum LoadState: Equatable { case loading, ready, failed(String) }

// MARK: - WebView

struct WebView: UIViewRepresentable {
    let url: URL
    @Binding var state: LoadState
    let reloadToken: Int

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.allowsInlineMediaPlayback = true
        cfg.mediaTypesRequiringUserActionForPlayback = []

        let wv = WKWebView(frame: .zero, configuration: cfg)
        wv.navigationDelegate = context.coordinator
        wv.scrollView.bounces = false                    // e' un'app, non una pagina
        wv.scrollView.contentInsetAdjustmentBehavior = .never
        wv.allowsBackForwardNavigationGestures = false
        wv.isOpaque = false
        wv.backgroundColor = .clear
        wv.scrollView.backgroundColor = .clear
        wv.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
        return wv
    }

    func updateUIView(_ wv: WKWebView, context: Context) {
        guard context.coordinator.lastToken != reloadToken else { return }
        context.coordinator.lastToken = reloadToken
        DispatchQueue.main.async { state = .loading }
        wv.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        let parent: WebView
        var lastToken = 0
        init(_ parent: WebView) { self.parent = parent }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            parent.state = .ready
            UIImpactFeedbackGenerator(style: .soft).impactOccurred()
        }
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            parent.state = .failed(error.localizedDescription)
        }
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            parent.state = .failed(error.localizedDescription)
        }
    }
}

// MARK: - Schermate di servizio

struct Splash: View {
    var body: some View {
        VStack(spacing: 16) {
            Text("Vesta").font(.custom("Didot", size: 42)).foregroundStyle(.primary)
            ProgressView().tint(.secondary)
            Text("Mi collego al tuo camerino").font(.footnote).foregroundStyle(.secondary)
        }
    }
}

struct Offline: View {
    let message: String
    let server: String
    var retry: () -> Void
    var openSettings: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Text("Vesta").font(.custom("Didot", size: 38))
            VStack(spacing: 8) {
                Text("Non riesco a raggiungere il Mac").font(.headline)
                Text("Controlla che Vesta sia aperta sul Mac e che iPhone e Mac siano sulla stessa rete Wi-Fi.")
                    .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
                Text(server).font(.system(.caption, design: .monospaced)).foregroundStyle(.tertiary)
                Text(message).font(.caption2).foregroundStyle(.tertiary).multilineTextAlignment(.center)
            }
            HStack(spacing: 12) {
                Button("Riprova", action: retry).buttonStyle(.borderedProminent)
                Button("Cambia indirizzo", action: openSettings).buttonStyle(.bordered)
            }
        }
        .padding(28)
    }
}

struct ServerSheet: View {
    @ObservedObject var settings: Settings
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Indirizzo del Mac") {
                    TextField("http://192.168.1.10:8770", text: $draft)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    Text("Sul Mac, in Terminale: ipconfig getifaddr en0")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section { Button("Usa il nome del Mac") { draft = kDefaultServer } }
            }
            .navigationTitle("Collegamento")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Annulla") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Salva") { settings.server = draft; dismiss() }
                }
            }
            .onAppear { draft = settings.server }
        }
    }
}

// MARK: - App

struct RootView: View {
    @StateObject private var settings = Settings()
    @State private var state: LoadState = .loading
    @State private var reloadToken = 0
    @State private var showSettings = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()          // il fondo del camerino, anche dietro le safe area

            if let url = settings.url {
                WebView(url: url, state: $state, reloadToken: reloadToken)
                    .ignoresSafeArea()
                    .opacity(state == .ready ? 1 : 0)
                    .animation(.easeOut(duration: 0.35), value: state)
            }

            switch state {
            case .loading: Splash()
            case .failed(let msg):
                Offline(message: msg, server: settings.server,
                        retry: { reloadToken += 1 },
                        openSettings: { showSettings = true })
            case .ready: EmptyView()
            }
        }
        .preferredColorScheme(nil)                 // segue il tema di sistema
        .sheet(isPresented: $showSettings) { ServerSheet(settings: settings) }
        .onChange(of: settings.server) { _ in reloadToken += 1 }
        .onReceive(NotificationCenter.default.publisher(for: .vestaShake)) { _ in showSettings = true }
    }
}

@main
struct VestaApp: App {
    var body: some Scene { WindowGroup { RootView() } }
}

// scuoti l'iPhone per cambiare indirizzo: serve quando l'IP del Mac cambia
extension Notification.Name { static let vestaShake = Notification.Name("vestaShake") }
extension UIWindow {
    open override func motionEnded(_ motion: UIEvent.EventSubtype, with event: UIEvent?) {
        if motion == .motionShake { NotificationCenter.default.post(name: .vestaShake, object: nil) }
    }
}
