import Foundation

/// Launches `dashstack` and streams its output. The CLI writes a progress bar
/// on stderr using carriage returns (`\r  [...] 33%  0:05 elapsed  ETA 0:10`).
/// We split on either `\n` or `\r` so each progress redraw becomes its own
/// "line" we can parse without polluting the log view.
final class Runner {
    enum Stream { case stdout, stderr }

    private let binary: String
    private let arguments: [String]
    private let onLine: (String, Stream) -> Void
    private let onProgress: (Double, String, String) -> Void
    private let onExit: (Int) -> Void

    private var process: Process?

    /// Captures lines like `  [████████░░░] 33%  0:05 elapsed  ETA 0:10`.
    private static let progressRegex: NSRegularExpression = {
        // The bar uses block characters (█ / ░) — match any non-bracket chars
        // between [ and ], then capture pct / elapsed / eta.
        let pattern = #"\[[^\]]+\]\s+(\d+)%\s+([\d:]+)\s+elapsed(?:\s+ETA\s+([\d:\-]+))?"#
        return try! NSRegularExpression(pattern: pattern)
    }()

    init(
        binary: String,
        arguments: [String],
        onLine: @escaping (String, Stream) -> Void,
        onProgress: @escaping (Double, String, String) -> Void,
        onExit: @escaping (Int) -> Void
    ) {
        self.binary = binary
        self.arguments = arguments
        self.onLine = onLine
        self.onProgress = onProgress
        self.onExit = onExit
    }

    func start() {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binary)
        proc.arguments = arguments

        // Augment PATH so child processes (ffmpeg, ffprobe, rsync) are visible
        // when launched from a GUI bundle where PATH is minimal.
        var env = ProcessInfo.processInfo.environment
        let extras = ["/usr/local/bin", "/opt/homebrew/bin", "\(NSHomeDirectory())/.local/bin"]
        let existing = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        let merged = (existing + extras.filter { !existing.contains($0) }).joined(separator: ":")
        env["PATH"] = merged
        // Force unbuffered Python output so progress shows up live.
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        proc.standardInput = FileHandle.nullDevice

        attachReader(outPipe.fileHandleForReading, stream: .stdout)
        attachReader(errPipe.fileHandleForReading, stream: .stderr)

        proc.terminationHandler = { [weak self] p in
            guard let self else { return }
            let code = Int(p.terminationStatus)
            DispatchQueue.main.async { self.onExit(code) }
        }

        do {
            try proc.run()
            self.process = proc
        } catch {
            DispatchQueue.main.async {
                self.onLine("Failed to launch: \(error.localizedDescription)", .stderr)
                self.onExit(-1)
            }
        }
    }

    func cancel() {
        process?.terminate()
    }

    /// Reads from a pipe, splitting on `\n` or `\r` so carriage-return
    /// progress redraws become discrete events.
    private func attachReader(_ handle: FileHandle, stream: Stream) {
        var buffer = Data()
        handle.readabilityHandler = { [weak self] fh in
            let chunk = fh.availableData
            if chunk.isEmpty {
                fh.readabilityHandler = nil
                if !buffer.isEmpty, let leftover = String(data: buffer, encoding: .utf8) {
                    self?.dispatch(line: leftover, stream: stream)
                }
                return
            }
            buffer.append(chunk)
            while let idx = buffer.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
                let lineData = buffer[..<idx]
                buffer.removeSubrange(...idx)
                if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                    self?.dispatch(line: line, stream: stream)
                }
            }
        }
    }

    private func dispatch(line: String, stream: Stream) {
        // Strip any leftover CR.
        let cleaned = line.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
        guard !cleaned.isEmpty else { return }

        if let progress = Self.parseProgress(cleaned) {
            DispatchQueue.main.async {
                self.onProgress(progress.pct, progress.elapsed, progress.eta)
            }
            return  // don't spam log with redraw lines
        }
        DispatchQueue.main.async {
            self.onLine(cleaned, stream)
        }
    }

    static func parseProgress(_ line: String) -> (pct: Double, elapsed: String, eta: String)? {
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        guard let match = progressRegex.firstMatch(in: line, range: range) else { return nil }
        func group(_ i: Int) -> String? {
            guard match.numberOfRanges > i,
                  let r = Range(match.range(at: i), in: line) else { return nil }
            return String(line[r])
        }
        guard let pctStr = group(1), let elapsed = group(2),
              let pct = Double(pctStr) else { return nil }
        let eta = group(3) ?? "--:--"
        return (pct / 100.0, elapsed, eta)
    }
}
