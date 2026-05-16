import SwiftUI
import AppKit

struct TransferView: View {
    enum Direction { case upload, download }

    let direction: Direction
    @EnvironmentObject private var session: JobSession

    // Upload: a list of local file paths + a remote destination
    @State private var uploadFiles: [URL] = []
    @State private var uploadDest: String = ""

    // Download: a remote source + a local destination directory
    @State private var downloadSource: String = ""
    @State private var downloadDest: URL? = nil

    @State private var deleteAfter: Bool = false
    @State private var dryRun: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(direction == .upload ? "Upload to remote" : "Download from remote")
                        .font(Theme.titleFont)
                    Text(direction == .upload
                         ? "Transfer local files to a remote host via rsync."
                         : "Pull files from a remote host into a local folder via rsync.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                Card {
                    if direction == .upload {
                        uploadForm
                    } else {
                        downloadForm
                    }
                }

                Card {
                    HStack(spacing: 20) {
                        Toggle("Delete source after transfer", isOn: $deleteAfter)
                        Toggle("Dry run", isOn: $dryRun)
                        Spacer()
                    }
                    .toggleStyle(.switch)
                    .tint(Theme.accent)
                    .font(.callout)
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
                            run()
                        } label: {
                            Label(direction == .upload ? "Start upload" : "Start download",
                                  systemImage: direction == .upload ? "arrow.up.circle.fill" : "arrow.down.circle.fill")
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(!canRun)
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

    // MARK: - Upload form

    private var uploadForm: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel("Files", systemImage: "doc.on.doc.fill")

            if uploadFiles.isEmpty {
                Text("Add one or more local files to upload.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(uploadFiles.enumerated()), id: \.offset) { idx, url in
                        HStack {
                            Image(systemName: "doc.fill").foregroundStyle(Theme.accent)
                            Text(url.lastPathComponent)
                                .font(.system(.callout, design: .rounded))
                            Spacer()
                            Text(url.deletingLastPathComponent().path)
                                .font(Theme.monoCaption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Button {
                                uploadFiles.remove(at: idx)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.vertical, 6)
                        if idx < uploadFiles.count - 1 { Divider().opacity(0.4) }
                    }
                }
            }
            HStack {
                Button("Add files…") { pickUploadFiles() }
                    .buttonStyle(GhostButtonStyle())
                if !uploadFiles.isEmpty {
                    Button("Clear") { uploadFiles.removeAll() }
                        .buttonStyle(GhostButtonStyle())
                }
                Spacer()
            }

            Divider().padding(.vertical, 6)
            SectionLabel("Remote destination", systemImage: "server.rack")
            TextField("user@host:/path/", text: $uploadDest)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
        }
    }

    // MARK: - Download form

    private var downloadForm: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel("Remote source", systemImage: "server.rack")
            TextField("user@host:/path/*.MP4", text: $downloadSource)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))

            Divider().padding(.vertical, 6)
            SectionLabel("Local destination", systemImage: "folder.fill")
            HStack(spacing: 12) {
                Image(systemName: "folder.fill")
                    .foregroundStyle(Theme.accent)
                Text(downloadDest?.path ?? "Choose folder…")
                    .foregroundStyle(downloadDest == nil ? .secondary : .primary)
                Spacer()
                Button("Choose…") { pickDownloadDest() }
                    .buttonStyle(GhostButtonStyle())
            }
        }
    }

    private var canRun: Bool {
        guard session.dashstackResolution != .missing else { return false }
        switch direction {
        case .upload:   return !uploadFiles.isEmpty && !uploadDest.isEmpty
        case .download: return !downloadSource.isEmpty && downloadDest != nil
        }
    }

    private func pickUploadFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            for u in panel.urls where !uploadFiles.contains(u) {
                uploadFiles.append(u)
            }
        }
    }

    private func pickDownloadDest() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK { downloadDest = panel.url }
    }

    private func run() {
        var args: [String]
        switch direction {
        case .upload:
            args = ["upload"]
            args += uploadFiles.map { $0.path }
            args.append(uploadDest)
        case .download:
            args = ["download", downloadSource]
            if let d = downloadDest { args.append(d.path) }
        }
        if deleteAfter { args.append("--delete") }
        if dryRun { args.append("--dry-run") }
        session.start(args: args, commandLabel: "dashstack " + args.joined(separator: " "))
    }
}
