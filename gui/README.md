# DashStack — macOS GUI

A native SwiftUI app that wraps the `dashstack` Python CLI. Pick a folder of
dashcam clips, see what's there, tweak settings, and watch the encode progress
live — without touching the terminal.

![sidebar shell with Stack, Dedup, Upload, Download]

## Features

- **Stack** — folder picker (or drag-and-drop), live preview of discovered
  `_F` / `_R` pairs and existing `_FR` files, choice of *split runs* or
  *single file* output, plus the full set of advanced flags (codec, bitrate,
  CRF, preset, audio source, gap threshold, workers, clean/overwrite/dry-run).
- **Dedup** — multi-file picker with adjustable max-overlap window.
- **Upload / Download** — rsync-backed transfers with optional `--delete` and
  `--dry-run`.
- Streamed log console with auto-scroll.
- Live progress bar parsed from the CLI's own progress output.

## Requirements

- macOS 14 (Sonoma) or newer
- Xcode 15+ Command Line Tools (`xcode-select --install`)
- `dashstack` installed and on `PATH` — `pip install -e .` from the repo root
- `ffmpeg`, `ffprobe`, and (for transfers) `rsync` on `PATH`

The GUI augments `PATH` with `/usr/local/bin`, `/opt/homebrew/bin`, and
`~/.local/bin` when launching the CLI, so Homebrew installs and `pip --user`
installs are found automatically.

## Build & run

From this `gui/` directory:

```bash
swift run                  # build + launch
# or
swift build -c release     # optimized binary at .build/release/DashStack
open .build/release/DashStack
```

You can also open the folder in Xcode (`File → Open…` → select `gui/`) and
hit ⌘R.

## Project layout

```
gui/
├── Package.swift                       # SPM manifest, macOS 14+
└── Sources/DashStack/
    ├── App.swift                       # @main entry, window chrome
    ├── MainView.swift                  # NavigationSplitView + sidebar
    ├── Theme.swift                     # colors, fonts, Card, button styles
    ├── JobSession.swift                # @MainActor state shared across views
    ├── Runner.swift                    # Process wrapper, progress parser
    ├── ClipScanner.swift               # mirrors discover_pairs() for preview
    ├── LogConsole.swift                # auto-scrolling output view
    ├── StackView.swift                 # main stacking screen
    ├── DedupView.swift                 # dedup screen
    └── TransferView.swift              # upload/download screen
```

## How the CLI is invoked

The runner shells out to `dashstack <args>` and reads stdout/stderr in chunks,
splitting on both `\n` and `\r` so the CLI's carriage-return-based progress
redraws (`[████░░] 33%  0:05 elapsed  ETA 0:10`) are captured as discrete
events and converted into a `ProgressView`. Non-progress lines stream into the
log console.

If `dashstack` isn't found, the bottom-left status pill turns red and the run
button is disabled — install with `pip install -e .` from the project root and
relaunch.

## Packaging as a .app

`swift build` produces a plain Mach-O executable, not an `.app` bundle. For a
double-clickable app you can wrap the binary with a minimal bundle, e.g.:

```bash
swift build -c release
APP=DashStack.app
mkdir -p "$APP/Contents/MacOS"
cp .build/release/DashStack "$APP/Contents/MacOS/DashStack"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>DashStack</string>
  <key>CFBundleIdentifier</key><string>dev.dashstack.gui</string>
  <key>CFBundleExecutable</key><string>DashStack</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
open "$APP"
```
