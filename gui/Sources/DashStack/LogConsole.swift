import SwiftUI

/// Auto-scrolling, monospaced log view used by every job-running screen.
struct LogConsole: View {
    let lines: [LogLine]
    @State private var autoScroll: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "terminal.fill")
                    .foregroundStyle(Theme.accent)
                Text("Output")
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                Spacer()
                Toggle(isOn: $autoScroll) {
                    Text("Auto-scroll")
                        .font(.caption)
                }
                .toggleStyle(.switch)
                .controlSize(.mini)
                .tint(Theme.accent)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(lines) { line in
                            Text(line.text)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(line.color)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(line.id)
                        }
                    }
                    .padding(8)
                }
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.black.opacity(0.04))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Theme.cardStroke, lineWidth: 1)
                )
                .onChange(of: lines.count) { _, _ in
                    guard autoScroll, let last = lines.last else { return }
                    withAnimation(.easeOut(duration: 0.12)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }
}
