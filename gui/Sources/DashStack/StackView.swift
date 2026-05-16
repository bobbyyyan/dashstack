import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct StackView: View {
    @EnvironmentObject private var session: JobSession
    @State private var inputDir: URL? = nil
    @State private var outputMode: OutputMode = .split
    @State private var outputFile: URL? = nil
    @State private var scan: ClipScanner.ScanResult? = nil
    @State private var showAdvanced: Bool = false

    // Settings
    @State private var audioSource: String = "front"
    @State private var videoCodec: String = "auto"
    @State private var videoBitrate: String = "16M"
    @State private var crf: Double = 20
    @State private var preset: String = "veryfast"
    @State private var gapThreshold: Double = 5.0
    @State private var workers: Double = 0           // 0 = auto
    @State private var clean: Bool = false
    @State private var overwrite: Bool = false
    @State private var dryRun: Bool = false

    enum OutputMode: String, CaseIterable, Identifiable {
        case split, single
        var id: String { rawValue }
        var label: String { self == .split ? "Split runs" : "Single file" }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header

                Card {
                    inputSection
                }

                if let scan {
                    Card {
                        scanSummary(scan)
                    }
                }

                Card {
                    outputSection
                }

                Card {
                    advancedDisclosure
                }

                runRow

                if !session.logLines.isEmpty {
                    Card(padding: 12) {
                        LogConsole(lines: session.logLines)
                            .frame(minHeight: 220, maxHeight: 320)
                    }
                }
            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    // MARK: - Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Stack dashcam clips")
                .font(Theme.titleFont)
            Text("Combine front and rear cameras into a single vertical video, in chronological order.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel("Input folder", systemImage: "folder.fill")

            HStack(spacing: 12) {
                Image(systemName: "tray.full.fill")
                    .font(.system(size: 22))
                    .foregroundStyle(Theme.accent)
                    .frame(width: 44, height: 44)
                    .background(Theme.accentSoft, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(inputDir?.path ?? "No folder selected")
                        .font(.system(.body, design: .rounded).weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(inputDir == nil ? .secondary : .primary)
                    if let inputDir {
                        Text(inputDir.deletingLastPathComponent().path)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.head)
                    } else {
                        Text("Choose a directory containing _F.mp4 / _R.mp4 pairs.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Button("Choose…") { pickFolder() }
                    .buttonStyle(GhostButtonStyle())
                if scan != nil {
                    Button("Rescan") { rescan() }
                        .buttonStyle(GhostButtonStyle())
                }
            }

            DropZone(isActive: inputDir == nil) { url in
                inputDir = url
                rescan()
            }
        }
    }

    private func scanSummary(_ s: ClipScanner.ScanResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel("Discovered clips", systemImage: "magnifyingglass")

            HStack(spacing: 10) {
                StatPill(label: "pairs", value: "\(s.pairs.count)", systemImage: "rectangle.on.rectangle")
                StatPill(label: "merged", value: "\(s.merged.count)", systemImage: "rectangle.stack")
                if !s.unmatched.isEmpty {
                    StatPill(label: "unmatched", value: "\(s.unmatched.count)", systemImage: "exclamationmark.triangle.fill")
                }
                if s.totalSourceBytes > 0 {
                    StatPill(label: "input",
                             value: ClipScanner.formatBytes(s.totalSourceBytes),
                             systemImage: "internaldrive")
                }
                Spacer()
            }

            if !s.pairs.isEmpty {
                ClipList(pairs: Array(s.pairs.prefix(50)),
                         hasMore: s.pairs.count > 50,
                         remaining: max(0, s.pairs.count - 50))
            } else if s.merged.isEmpty {
                Text("No dashcam clips detected. Expected filenames like `…YYYYMMDD_HHMMSS_F.mp4` and `…_R.mp4`.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            if !s.unmatched.isEmpty {
                DisclosureGroup("\(s.unmatched.count) unmatched") {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(s.unmatched) { u in
                            Text("\(u.timestamp) — only \(u.present): \(u.url.lastPathComponent)")
                                .font(Theme.monoCaption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.top, 6)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var outputSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel("Output", systemImage: "square.and.arrow.down.fill")

            Picker("", selection: $outputMode) {
                ForEach(OutputMode.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if outputMode == .split {
                HStack(spacing: 8) {
                    Image(systemName: "info.circle")
                        .foregroundStyle(Theme.accent)
                    Text("Creates one `_FR.mp4` file per continuous run in the input folder. Runs are split when clips are more than \(Int(gapThreshold))s apart.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 12) {
                    Image(systemName: "doc.fill")
                        .foregroundStyle(Theme.accent)
                    Text(outputFile?.lastPathComponent ?? "Choose output file…")
                        .font(.system(.body, design: .rounded))
                        .foregroundStyle(outputFile == nil ? .secondary : .primary)
                    Spacer()
                    Button("Choose…") { pickOutputFile() }
                        .buttonStyle(GhostButtonStyle())
                }
            }

            HStack(spacing: 20) {
                Toggle("Overwrite existing", isOn: $overwrite)
                Toggle("Clean source clips after", isOn: $clean)
                Toggle("Dry run", isOn: $dryRun)
                Spacer()
            }
            .toggleStyle(.switch)
            .tint(Theme.accent)
            .font(.callout)
        }
    }

    private var advancedDisclosure: some View {
        DisclosureGroup(isExpanded: $showAdvanced) {
            VStack(alignment: .leading, spacing: 14) {
                Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 12) {
                    GridRow {
                        labeledField("Audio source") {
                            Picker("", selection: $audioSource) {
                                Text("Front").tag("front")
                                Text("Rear").tag("rear")
                                Text("None (silent)").tag("none")
                            }
                            .labelsHidden()
                            .pickerStyle(.menu)
                        }
                        labeledField("Video codec") {
                            TextField("auto", text: $videoCodec)
                                .textFieldStyle(.roundedBorder)
                        }
                    }
                    GridRow {
                        labeledField("Video bitrate") {
                            TextField("16M", text: $videoBitrate)
                                .textFieldStyle(.roundedBorder)
                        }
                        labeledField("Preset (x264)") {
                            TextField("veryfast", text: $preset)
                                .textFieldStyle(.roundedBorder)
                        }
                    }
                    GridRow {
                        labeledField("CRF \(Int(crf))") {
                            Slider(value: $crf, in: 12...32, step: 1)
                        }
                        labeledField("Gap threshold \(String(format: "%.1f", gapThreshold))s") {
                            Slider(value: $gapThreshold, in: 1...30, step: 0.5)
                        }
                    }
                    GridRow {
                        labeledField(workers == 0 ? "Workers — auto" : "Workers — \(Int(workers))") {
                            Slider(value: $workers, in: 0...12, step: 1)
                        }
                        Color.clear.frame(height: 1)
                    }
                }
            }
            .padding(.top, 12)
        } label: {
            SectionLabel("Advanced", systemImage: "slider.horizontal.3")
        }
        .tint(Theme.accent)
    }

    private func labeledField<Content: View>(_ title: String,
                                             @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            content()
        }
    }

    private var runRow: some View {
        HStack(spacing: 12) {
            if session.isRunning {
                Button(role: .destructive) {
                    session.cancel()
                } label: {
                    Label("Cancel", systemImage: "stop.fill")
                }
                .buttonStyle(GhostButtonStyle())
                ProgressView(value: session.progress)
                    .progressViewStyle(.linear)
                    .tint(Theme.accent)
                Text(session.etaText)
                    .font(Theme.monoCaption)
                    .foregroundStyle(.secondary)
            } else {
                Spacer()
                Button {
                    runStack()
                } label: {
                    Label(dryRun ? "Preview command" : "Stack videos",
                          systemImage: dryRun ? "eye.fill" : "play.fill")
                }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(!canRun)
                .help(canRun ? "" : "Pick an input folder first.")
            }
        }
    }

    private var canRun: Bool {
        guard let _ = inputDir else { return false }
        if outputMode == .single && outputFile == nil { return false }
        if session.dashstackResolution == .missing { return false }
        return true
    }

    // MARK: - Actions

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose folder"
        if panel.runModal() == .OK, let url = panel.url {
            inputDir = url
            rescan()
        }
    }

    private func pickOutputFile() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.movie, UTType.mpeg4Movie]
        panel.nameFieldStringValue = "dashstack.mp4"
        if panel.runModal() == .OK {
            outputFile = panel.url
        }
    }

    private func rescan() {
        guard let dir = inputDir else { return }
        Task.detached(priority: .userInitiated) {
            let result = ClipScanner.scan(directory: dir)
            await MainActor.run { self.scan = result }
        }
    }

    private func runStack() {
        guard let dir = inputDir else { return }
        var args: [String] = [dir.path]

        if outputMode == .single, let out = outputFile {
            args += ["--output", out.path]
        }
        if audioSource != "front" { args += ["--audio-source", audioSource] }
        if videoCodec != "auto" { args += ["--video-codec", videoCodec] }
        if videoBitrate != "16M" { args += ["--video-bitrate", videoBitrate] }
        if Int(crf) != 20 { args += ["--crf", String(Int(crf))] }
        if preset != "veryfast" { args += ["--preset", preset] }
        if abs(gapThreshold - 5.0) > 0.01 { args += ["--gap-threshold", String(format: "%.1f", gapThreshold)] }
        if workers > 0 { args += ["--workers", String(Int(workers))] }
        if clean { args.append("--clean") }
        if overwrite { args.append("--overwrite") }
        if dryRun { args.append("--dry-run") }

        let label = "dashstack " + args.joined(separator: " ")
        session.start(args: args, commandLabel: label)
    }
}

// MARK: - Pair list

private struct ClipList: View {
    let pairs: [ClipScanner.Pair]
    let hasMore: Bool
    let remaining: Int

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array(pairs.enumerated()), id: \.element.id) { idx, pair in
                HStack(spacing: 12) {
                    Text("\(idx + 1)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 24, alignment: .trailing)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(ClipScanner.prettyTimestamp(pair.timestamp))
                            .font(.system(.callout, design: .rounded).weight(.medium))
                        Text("\(pair.frontURL.lastPathComponent)  •  \(pair.rearURL.lastPathComponent)")
                            .font(Theme.monoCaption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer()
                    Text(ClipScanner.formatBytes(pair.sizeBytes))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 4)
                if idx < pairs.count - 1 {
                    Divider().opacity(0.5)
                }
            }
            if hasMore {
                Text("… and \(remaining) more")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
    }
}

// MARK: - Drop zone

private struct DropZone: View {
    let isActive: Bool
    let onDrop: (URL) -> Void
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "arrow.down.doc")
                .foregroundStyle(.secondary)
            Text(isActive ? "or drop a folder here" : "Drop a different folder to switch")
                .font(.callout)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(
                    Theme.accent.opacity(hovering ? 0.65 : 0.25),
                    style: StrokeStyle(lineWidth: 1.5, dash: [5, 4])
                )
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(hovering ? Theme.accentSoft : Color.clear)
                )
        )
        .onDrop(of: [.fileURL], isTargeted: $hovering) { providers in
            guard let p = providers.first else { return false }
            _ = p.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                    DispatchQueue.main.async { onDrop(url) }
                }
            }
            return true
        }
    }
}
