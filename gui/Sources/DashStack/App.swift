import SwiftUI

@main
struct DashStackApp: App {
    @StateObject private var session = JobSession()

    var body: some Scene {
        WindowGroup("DashStack") {
            MainView()
                .environmentObject(session)
                .frame(minWidth: 980, minHeight: 680)
                .background(WindowBackground())
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

/// Translucent window background that blends with the desktop.
private struct WindowBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = .underWindowBackground
        view.blendingMode = .behindWindow
        view.state = .active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}
