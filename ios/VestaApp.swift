import SwiftUI
import WebKit

// IP del Mac sulla rete locale (cambialo col tuo: `ipconfig getifaddr en0`).
// Per ora: 192.168.1.129
let SERVER_URL = "http://192.168.1.129:8770/"

struct WebView: UIViewRepresentable {
    let url: URL
    func makeUIView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.allowsInlineMediaPlayback = true
        cfg.mediaTypesRequiringUserActionForPlayback = []
        let wv = WKWebView(frame: .zero, configuration: cfg)
        wv.scrollView.bounces = false
        wv.allowsBackForwardNavigationGestures = false
        wv.load(URLRequest(url: url))
        return wv
    }
    func updateUIView(_ wv: WKWebView, context: Context) {}
}

@main
struct VestaApp: App {
    var body: some Scene {
        WindowGroup {
            WebView(url: URL(string: SERVER_URL)!)
                .ignoresSafeArea()
                .preferredColorScheme(nil) // segue il tema di sistema
        }
    }
}
