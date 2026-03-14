# DashStack

DashStack is an open-source toolkit that combines timestamp-matched dashcam clips by placing:

- `F` (front) video on top
- `R` (rear) video on bottom

Then it concatenates all stacked segments in chronological order.

## Requirements

- `ffmpeg` and `ffprobe` available in your shell
- Python 3.9+

## Installation

```bash
pip install .
```

Or install in development/editable mode:

```bash
pip install -e .
```

After installing, the `dashstack` command is available globally.

## Usage

```bash
dashstack
```

By default, DashStack auto-detects the fastest hardware encoder and decoder on your system and writes `dashstack.mp4`.

Custom output path:

```bash
dashstack --output merged_drive.mp4 --overwrite
```

Quick test with first 2 clip pairs:

```bash
dashstack --limit 2 --output sample.mp4 --overwrite
```

Dry run (prints ffmpeg commands without executing):

```bash
dashstack --dry-run
```

## Useful options

- `--input-dir PATH`: source directory (default `.`)
- `--output PATH`: final output path (default `dashstack.mp4`)
- `--work-dir PATH`: temp files directory (default `.dashstack_work`)
- `--pipeline auto|single-pass|segment`: processing strategy (default `auto`)
- `--target-width N`: panel width for each camera stream
- `--video-codec NAME`: output codec for stacked segments (default: `auto`)
- `--video-bitrate RATE`: bitrate for non-`libx264` codecs (example: `16M`)
- `--missing skip|error`: how to handle unmatched timestamps (default `skip`)
- `--audio-source front|rear|none`: audio source per segment (default `front`)
- `--crf N`: quality for `libx264` mode (default `20`)
- `--preset NAME`: speed/quality for `libx264` mode (default `veryfast`)
- `--keep-temp`: retain intermediate segment files

## Notes

- Filenames are matched with pattern containing `YYYYMMDD_HHMMSS_F/R`, for example:
  `REC_20260312_105549_F.MP4` and `REC_20260312_105549_R.MP4`.
- Unmatched timestamps are skipped by default.
