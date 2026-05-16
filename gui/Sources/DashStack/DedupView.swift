import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct DedupView: View {
    @EnvironmentObject private var session: JobSession
    @State private var files: [URL] = []
    @State private var maxOverlap: Double = 30
    @State private var dryRun: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Find duplicate segments")
                        .font(Theme.titleFont)
                    Text("Detects and removes overlapping footage in already-merged videos.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                Card {
                    VStack(alignment: .leading, spacing: 12) {
                        SectionLabel("Files", systemImage: "film.stack")

                        if files.isEmpty {
                            HStack(spacing: 10) {
                                Image(systemName: "tray")
                                    .foregroundStyle(.secondary)
                                Text("No files added yet.")
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            .padding(.vertical, 4)
                        } else {
                            VStack(spacing: 0) {
                                ForEach(Array(files.enumerated()), id: \.offset) { idx, url in
                                    HStack {
                                        Image(systemName: "video.fill")
                                            .foregroundStyle(Theme.accent)
                                        Text(url.lastPathComponent)
                                            .font(.system(.callout, design: .rounded))
                                        Spacer()
                                        Text(url.deletingLastPathComponent().path)
                                            .font(Theme.monoCaption)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                        Button {
                                            files.remove(at: idx)
                                        } label: {
                                            Image(systemName: "xmark.circle.fill")
                                                .foregroundStyle(.secondary)
                                        }
                                        .buttonStyle(.plain)
                                    }
                                    .padding(.vertical, 6)
                                    if idx < files.count - 1 {
                                        Divider().opacity(0.4)
                                    }
                                }
                            }
                        }

                        HStack {
                            Button("Add files…") { addFiles() }
                                .buttonStyle(GhostButtonStyle())
                            if !files.isEmpty {
                                Button("Clear") { files.removeAll() }
                                    .buttonStyle(GhostButtonStyle())
                            }
                            Spacer()
                        }
                    }
                }

                Card {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionLabel("Options", systemImage: "slider.horizontal.3")
                        HStack(spacing: 30) {
                            VStack(alignment: .leading) {
                                Text("Max overlap: \(Int(maxOverlap))s")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Slider(value: $maxOverlap, in: 5...120, step: 5)
                                    .frame(width: 260)
                            }
                            Toggle("Dry run (show only)", isOn: $dryRun)
                                .toggleStyle(.switch)
                                .tint(Theme.accent)
                            Spacer()
                        }
                    }
                }

                HStack {
                    Spacer()
                    if session.isRunning {
                        Button(role: .destructive) { session.cancel() } label: {
                            Label("Cancel", systemImage: "stop.fill")
                        }
                        .buttonStyle(GhostButtonStyle())
                    } else {
                        Button {
                            runDedup()
                        } label: {
                            Label(dryRun ? "Preview duplicates" : "Dedup files",
                                  systemImage: dryRun ? "eye.fill" : "scissors")
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(files.isEmpty)
                    }
                }

                if !session.logLines.isEmpty {
                    Card(padding: 12) {
                        LogConsole(lines: session.logLines)
                            .frame(minHeight: 220, maxHeight: 340)
                    }
                }
            }
            .padding(28)
        }
    }

    private func addFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [UTType.movie, UTType.mpeg4Movie, UTType.quickTimeMovie]
        if panel.runModal() == .OK {
            for u in panel.urls where !files.contains(u) {
                files.append(u)
            }
        }
    }

    private func runDedup() {
        var args: [String] = ["dedup"]
        args += files.map { $0.path }
        args += ["--max-overlap", String(Int(maxOverlap))]
        if dryRun { args.append("--dry-run") }
        session.start(args: args, commandLabel: "dashstack " + args.joined(separator: " "))
    }
}
