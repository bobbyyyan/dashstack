# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DashStack is a single-file Python CLI tool (`dashstack.py`) that stacks front/rear dashcam clips vertically (front on top, rear on bottom) and concatenates them chronologically. It shells out to `ffmpeg`/`ffprobe` for all media processing.

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on PATH

## Running

```bash
# Basic usage (processes current directory, outputs dashstack.mp4)
./dashstack.py

# Quick test with limited pairs
./dashstack.py --limit 2 --output sample.mp4 --overwrite

# Dry run (prints ffmpeg commands without executing)
./dashstack.py --dry-run

# macOS hardware encoding + hardware decode acceleration
./dashstack.py --video-codec h264_videotoolbox --video-bitrate 16M --hwaccel videotoolbox --overwrite

# Parallel segment encoding (e.g. 4 workers)
./dashstack.py --pipeline segment --workers 4 --overwrite
```

## Architecture

The tool is a single script with no external Python dependencies. Key flow:

1. **Discovery** (`discover_pairs`): Scans `--input-dir` for files matching `*YYYYMMDD_HHMMSS_F.mp4` / `*_R.mp4` regex and groups them into `ClipPair`s by timestamp.
2. **Probing** (`ffprobe_clip`): Calls `ffprobe` to get resolution, fps, duration, and audio presence for each clip. Results are cached per-path.
3. **Pipeline selection** (`main`): `auto` mode chooses between two pipelines:
   - **single-pass**: Feeds two concat-demuxer lists (all fronts, all rears) into one ffmpeg invocation. Faster but fails if audio presence is mixed across clips.
   - **segment**: Stacks each pair individually (in parallel via `--workers`), then concatenates segments with stream copy. Handles mixed audio. Used as fallback if single-pass fails.
4. **Encoding**: Builds ffmpeg filter graphs (`vstack` with scaling/padding) and codec args. Supports `libx264` (CRF mode) and hardware codecs like `h264_videotoolbox` (bitrate mode).

The filename regex (`FILE_RE`) expects a timestamp group `YYYYMMDD_HHMMSS` followed by `_F` or `_R` before the extension. The prefix before the timestamp is flexible.
