import SwiftUI

enum Mode: String, CaseIterable, Identifiable {
    case stack
    case dedup
    case upload
    case download

    var id: String { rawValue }

    var title: String {
        switch self {
        case .stack:    return "Stack"
        case .dedup:    return "Dedup"
        case .upload:   return "Upload"
        case .download: return "Download"
        }
    }

    var subtitle: String {
        switch self {
        case .stack:    return "Front + rear → one file"
        case .dedup:    return "Remove duplicate segments"
        case .upload:   return "rsync to remote"
        case .download: return "rsync from remote"
        }
    }

    var symbol: String {
        switch self {
        case .stack:    return "rectangle.stack.fill"
        case .dedup:    return "scissors"
        case .upload:   return "arrow.up.circle.fill"
        case .download: return "arrow.down.circle.fill"
        }
    }
}

struct MainView: View {
    @State private var mode: Mode = .stack
    @EnvironmentObject private var session: JobSession

    var body: some View {
        NavigationSplitView {
            Sidebar(selection: $mode)
                .navigationSplitViewColumnWidth(min: 220, ideal: 240, max: 280)
        } detail: {
            ZStack {
                switch mode {
                case .stack:    StackView()
                case .dedup:    DedupView()
                case .upload:   TransferView(direction: .upload)
                case .download: TransferView(direction: .download)
                }
            }
            .navigationSplitViewColumnWidth(min: 700, ideal: 800)
            .toolbar { ToolbarItem(placement: .principal) { Text(" ") } }
        }
        .navigationSplitViewStyle(.balanced)
        .tint(Theme.accent)
    }
}

private struct Sidebar: View {
    @Binding var selection: Mode
    @EnvironmentObject private var session: JobSession

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Brand
            HStack(spacing: 10) {
                Image(systemName: "rectangle.stack.badge.play.fill")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Theme.accent, Theme.accent.opacity(0.6)],
                            startPoint: .top, endPoint: .bottom
                        )
                    )
                VStack(alignment: .leading, spacing: 0) {
                    Text("DashStack")
                        .font(.system(.title3, design: .rounded).weight(.bold))
                    Text("Dashcam toolkit")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 18)
            .padding(.top, 24)
            .padding(.bottom, 22)

            // Nav items
            VStack(spacing: 4) {
                ForEach(Mode.allCases) { item in
                    SidebarRow(
                        item: item,
                        isSelected: selection == item,
                        action: { selection = item }
                    )
                }
            }
            .padding(.horizontal, 10)

            Spacer()

            // Job status footer
            JobStatusFooter()
                .padding(14)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.ultraThinMaterial)
    }
}

private struct SidebarRow: View {
    let item: Mode
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: item.symbol)
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 22)
                    .foregroundStyle(isSelected ? Color.white : Theme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.title)
                        .font(.system(.body, design: .rounded).weight(.semibold))
                    Text(item.subtitle)
                        .font(.caption2)
                        .foregroundStyle(isSelected ? Color.white.opacity(0.8) : .secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(
                        isSelected
                            ? AnyShapeStyle(
                                LinearGradient(
                                    colors: [Theme.accent, Theme.accent.opacity(0.78)],
                                    startPoint: .topLeading, endPoint: .bottomTrailing
                                )
                            )
                            : AnyShapeStyle(hovering ? Color.primary.opacity(0.06) : Color.clear)
                    )
            )
            .foregroundStyle(isSelected ? Color.white : Color.primary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

private struct JobStatusFooter: View {
    @EnvironmentObject private var session: JobSession

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if session.isRunning, session.progress > 0 {
                ProgressView(value: session.progress)
                    .tint(Theme.accent)
                Text(session.etaText)
                    .font(Theme.monoCaption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Theme.cardStroke, lineWidth: 1)
        )
    }

    private var statusColor: Color {
        if session.isRunning { return .green }
        if session.lastExitCode == 0 && session.lastFinishedCommand != nil { return .blue }
        if (session.lastExitCode ?? 0) != 0 { return .red }
        return .secondary
    }

    private var statusText: String {
        if session.isRunning { return "Running \(session.currentCommand ?? "job")…" }
        if let cmd = session.lastFinishedCommand {
            if session.lastExitCode == 0 { return "Last: \(cmd) succeeded" }
            return "Last: \(cmd) failed (exit \(session.lastExitCode ?? -1))"
        }
        return "Idle"
    }
}
