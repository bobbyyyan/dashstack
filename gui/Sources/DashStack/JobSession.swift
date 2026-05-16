import Foundation
import SwiftUI

/// Shared state for the currently running (or last completed) dashstack job.
@MainActor
final class JobSession: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var progress: Double = 0           // 0…1, from CLI progress bar
    @Published var etaText: String = ""
    @Published var elapsedText: String = ""
    @Published var logLines: [LogLine] = []
    @Published var currentCommand: String? = nil
    @Published var lastFinishedCommand: String? = nil
    @Published var lastExitCode: Int? = nil

    /// Path to the dashstack binary. Resolved lazily.
    @Published var dashstackPath: String? = nil
    @Published var dashstackResolution: ResolutionStatus = .unknown

    enum ResolutionStatus { case unknown, found, missing }

    private var runner: Runner?

    func resolveBinary() {
        let candidates = [
            "/usr/local/bin/dashstack",
            "/opt/homebrew/bin/dashstack",
            "\(NSHomeDirectory())/.local/bin/dashstack",
        ]
        for c in candidates where FileManager.default.isExecutableFile(atPath: c) {
            dashstackPath = c
            dashstackResolution = .found
            return
        }
        // Fall back to `which dashstack`
        if let found = which("dashstack") {
            dashstackPath = found
            dashstackResolution = .found
            return
        }
        dashstackResolution = .missing
    }

    func start(args: [String], commandLabel: String) {
        guard !isRunning else { return }
        if dashstackPath == nil { resolveBinary() }
        guard let bin = dashstackPath else {
            appendError("Could not find `dashstack` on PATH. Install with `pip install -e .` in the project root.")
            return
        }
        logLines.removeAll(keepingCapacity: true)
        progress = 0
        etaText = ""
        elapsedText = ""
        currentCommand = commandLabel
        isRunning = true
        appendInfo("$ \(commandLabel)")

        let runner = Runner(
            binary: bin,
            arguments: args,
            onLine: { [weak self] line, stream in
                self?.handleLine(line, stream: stream)
            },
            onProgress: { [weak self] pct, elapsed, eta in
                self?.progress = pct
                self?.elapsedText = elapsed
                self?.etaText = "ETA \(eta)"
            },
            onExit: { [weak self] code in
                self?.handleExit(code: code, label: commandLabel)
            }
        )
        self.runner = runner
        runner.start()
    }

    func cancel() {
        runner?.cancel()
    }

    private func handleLine(_ line: String, stream: Runner.Stream) {
        let kind: LogLine.Kind = stream == .stderr ? .stderr : .stdout
        logLines.append(LogLine(text: line, kind: kind))
        // Cap retained lines to avoid unbounded memory.
        if logLines.count > 5000 {
            logLines.removeFirst(logLines.count - 5000)
        }
    }

    private func handleExit(code: Int, label: String) {
        isRunning = false
        lastExitCode = code
        lastFinishedCommand = label
        currentCommand = nil
        if code == 0 {
            progress = 1.0
            appendInfo("✓ Finished in \(elapsedText.isEmpty ? "—" : elapsedText)")
        } else {
            appendError("✗ Exited with code \(code)")
        }
    }

    private func appendInfo(_ text: String) {
        logLines.append(LogLine(text: text, kind: .info))
    }

    private func appendError(_ text: String) {
        logLines.append(LogLine(text: text, kind: .error))
    }
}

struct LogLine: Identifiable {
    enum Kind { case stdout, stderr, info, error }

    let id = UUID()
    let text: String
    let kind: Kind

    var color: Color {
        switch kind {
        case .stdout: return .primary
        case .stderr: return Color.primary.opacity(0.85)
        case .info:   return Theme.accent
        case .error:  return .red
        }
    }
}

private func which(_ name: String) -> String? {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    proc.arguments = ["which", name]
    let pipe = Pipe()
    proc.standardOutput = pipe
    proc.standardError = Pipe()
    // Inherit a reasonable PATH for GUI apps.
    var env = ProcessInfo.processInfo.environment
    let existing = env["PATH"] ?? ""
    let extra = "/usr/local/bin:/opt/homebrew/bin:\(NSHomeDirectory())/.local/bin"
    env["PATH"] = existing.isEmpty ? extra : "\(existing):\(extra)"
    proc.environment = env
    do {
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (path?.isEmpty == false) ? path : nil
    } catch {
        return nil
    }
}
