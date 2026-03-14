# DashStack

DashStack stacks timestamp-matched dashcam clips vertically (front on top, rear on bottom) and concatenates them chronologically. It auto-detects the fastest encoder on your system — CUDA+NVENC GPU pipeline, multi-core CPU, or hardware encoder.

No external Python dependencies. Just `ffmpeg`.

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on PATH

## Installation

```bash
pip install .
```

Or editable for development:

```bash
pip install -e .
```

## Quick start

```bash
dashstack videos                  # split mode: one _FR file per continuous run
dashstack videos --dry-run        # preview the plan without encoding
dashstack videos --limit 3        # quick test with first 3 pairs
```

## How it works

DashStack scans a directory for files matching `*YYYYMMDD_HHMMSS_F.mp4` / `*_R.mp4`, pairs them by timestamp, and stacks each pair into a combined front+rear video.

### Split mode (default)

When no `--output` is given, DashStack detects gaps between clips (default >5s) and produces one `_FR` output file per continuous run, written to the input directory:

```
REC_20260313_135042_F.MP4  ┐
REC_20260313_135042_R.MP4  ┤→ REC_20260313_135042_20260313_135142_FR.MP4
REC_20260313_135142_F.MP4  ┤
REC_20260313_135142_R.MP4  ┘
```

On re-runs, existing `_FR` files are detected and reused — source clips already covered by an `_FR` file are skipped, so only new pairs get encoded.

### Single-file mode

Combine everything into one file:

```bash
dashstack videos --output combined.mp4
```

## Options

| Flag | Description |
|------|-------------|
| `input_dir` | Source directory (default: `.`) |
| `--output PATH` | Single combined output file (omit for split mode) |
| `--gap-threshold N` | Seconds of gap to split runs (default: `5.0`) |
| `--clean` | Delete source `_F`/`_R` files covered by `_FR` output |
| `--dry-run` | Preview the plan without encoding |
| `--limit N` | Process only the first N pairs |
| `--pipeline auto\|single-pass\|segment` | Processing strategy (default: `auto`) |
| `--target-width N` | Panel width for each camera stream |
| `--video-codec NAME` | Encoder (default: `auto`) |
| `--video-bitrate RATE` | Bitrate for HW codecs (default: `16M`) |
| `--crf N` | Quality for libx264 mode (default: `20`) |
| `--preset NAME` | Speed/quality for libx264 (default: `veryfast`) |
| `--audio-source front\|rear\|none` | Audio track to keep (default: `front`) |
| `--missing skip\|error` | Unmatched timestamp handling (default: `skip`) |
| `--workers N` | Parallel ffmpeg workers (default: half of CPU cores, max 4) |
| `--hwaccel METHOD` | HW decoding: `auto`, `cuda`, `vaapi`, `none`, etc. |
| `--work-dir PATH` | Temp directory (default: `.dashstack_work`) |
| `--keep-temp` | Retain intermediate segment files |
| `--overwrite` | Overwrite existing output without prompting |

## Filename pattern

Clips must contain a `YYYYMMDD_HHMMSS` timestamp followed by `_F` or `_R` before the extension. The prefix is flexible:

```
REC_20260312_105549_F.MP4
REC_20260312_105549_R.MP4
```

Unmatched timestamps (only F or only R) are skipped by default.
