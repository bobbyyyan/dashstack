import Foundation

/// Mirrors `discover_pairs` in cli.py so the GUI can preview what would be
/// processed without invoking the CLI.
enum ClipScanner {
    private static let fileRegex = try! NSRegularExpression(
        pattern: #"^(.*?)(\d{8}_\d{6})_([FR])\.(mp4|mov|m4v)$"#,
        options: [.caseInsensitive]
    )
    private static let frRegex = try! NSRegularExpression(
        pattern: #"^(.*?)(\d{8}_\d{6})(?:_(\d{8}_\d{6}))?_FR\.(mp4|mov|m4v)$"#,
        options: [.caseInsensitive]
    )

    struct ScanResult {
        var pairs: [Pair]              // matched F+R
        var unmatched: [Unmatched]     // one of F or R only
        var merged: [Merged]           // existing _FR files
        var suppressed: [String]       // source clips covered by an _FR
        var totalSourceBytes: Int64
    }

    struct Pair: Identifiable, Hashable {
        let id = UUID()
        let timestamp: String
        let frontURL: URL
        let rearURL: URL
        let sizeBytes: Int64
    }

    struct Unmatched: Identifiable, Hashable {
        let id = UUID()
        let timestamp: String
        let present: String   // "F" or "R"
        let url: URL
    }

    struct Merged: Identifiable, Hashable {
        let id = UUID()
        let startTS: String
        let endTS: String
        let url: URL
        let sizeBytes: Int64
    }

    static func scan(directory: URL) -> ScanResult {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: directory,
                                                     includingPropertiesForKeys: [.fileSizeKey],
                                                     options: [.skipsHiddenFiles])
        else {
            return ScanResult(pairs: [], unmatched: [], merged: [], suppressed: [], totalSourceBytes: 0)
        }
        let sorted = items.sorted { $0.lastPathComponent < $1.lastPathComponent }

        // Pass 1: existing _FR files
        var merged: [Merged] = []
        var frRanges: [(String, String)] = []
        for url in sorted {
            let name = url.lastPathComponent
            let range = NSRange(name.startIndex..<name.endIndex, in: name)
            guard let m = frRegex.firstMatch(in: name, range: range) else { continue }
            let start = stringAt(m, 2, in: name) ?? ""
            let end = stringAt(m, 3, in: name) ?? start
            frRanges.append((start, end))
            merged.append(Merged(
                startTS: start, endTS: end, url: url,
                sizeBytes: fileSize(url)
            ))
        }

        func isSuppressed(_ ts: String) -> Bool {
            for (s, e) in frRanges where s <= ts && ts <= e { return true }
            return false
        }

        // Pass 2: source files
        var grouped: [String: [String: URL]] = [:]   // ts → { "F": url, "R": url }
        var suppressed: [String] = []
        for url in sorted {
            let name = url.lastPathComponent
            let range = NSRange(name.startIndex..<name.endIndex, in: name)
            guard let m = fileRegex.firstMatch(in: name, range: range),
                  let ts = stringAt(m, 2, in: name),
                  let cam = stringAt(m, 3, in: name)?.uppercased() else { continue }
            if isSuppressed(ts) {
                suppressed.append(name)
                continue
            }
            grouped[ts, default: [:]][cam] = url
        }

        var pairs: [Pair] = []
        var unmatched: [Unmatched] = []
        var totalBytes: Int64 = 0
        for ts in grouped.keys.sorted() {
            let bucket = grouped[ts] ?? [:]
            if let f = bucket["F"], let r = bucket["R"] {
                let size = fileSize(f) + fileSize(r)
                totalBytes += size
                pairs.append(Pair(timestamp: ts, frontURL: f, rearURL: r, sizeBytes: size))
            } else if let only = bucket.first {
                unmatched.append(Unmatched(timestamp: ts, present: only.key, url: only.value))
            }
        }

        return ScanResult(
            pairs: pairs,
            unmatched: unmatched,
            merged: merged.sorted(by: { $0.startTS < $1.startTS }),
            suppressed: suppressed,
            totalSourceBytes: totalBytes
        )
    }

    // Format a 20240131_140530 timestamp as "Jan 31, 2024 14:05:30"
    static func prettyTimestamp(_ ts: String) -> String {
        guard ts.count == 15 else { return ts }
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        f.locale = Locale(identifier: "en_US_POSIX")
        guard let d = f.date(from: ts) else { return ts }
        let out = DateFormatter()
        out.dateFormat = "MMM d, yyyy  HH:mm:ss"
        out.locale = Locale(identifier: "en_US_POSIX")
        return out.string(from: d)
    }

    private static func stringAt(_ m: NSTextCheckingResult, _ i: Int, in s: String) -> String? {
        guard m.numberOfRanges > i, let r = Range(m.range(at: i), in: s) else { return nil }
        return String(s[r])
    }

    private static func fileSize(_ url: URL) -> Int64 {
        let v = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(v?.fileSize ?? 0)
    }

    static func formatBytes(_ bytes: Int64) -> String {
        let f = ByteCountFormatter()
        f.allowedUnits = [.useGB, .useMB]
        f.countStyle = .file
        return f.string(fromByteCount: bytes)
    }
}
