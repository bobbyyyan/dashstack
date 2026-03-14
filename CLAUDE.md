# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DashStack is an installable Python CLI tool that stacks front/rear dashcam clips vertically (front on top, rear on bottom) and concatenates them chronologically. It shells out to `ffmpeg`/`ffprobe` for all media processing. No external Python dependencies.

## Project Structure

- `src/dashstack/cli.py` — all CLI logic (entry point: `main()`)
- `src/dashstack/__init__.py` — package version
- `pyproject.toml` — packaging config, defines `dashstack` console script

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on PATH

## Install & Run

```bash
pip install -e .   # editable install for development
dashstack          # run from anywhere
dashstack --help
```

## Architecture

Key flow in `src/dashstack/cli.py`:

1. **Discovery** (`discover_pairs`): Scans `--input-dir` for files matching `*YYYYMMDD_HHMMSS_F.mp4` / `*_R.mp4` regex and groups them into `ClipPair`s by timestamp.
2. **Probing** (`ffprobe_clip`): Calls `ffprobe` to get resolution, fps, duration, and audio presence for each clip. Results are cached per-path. Probing runs in parallel when multiple clips exist.
3. **Auto-detection** (`detect_video_codec`, `detect_hwaccel`): Queries ffmpeg for available hardware encoders and decoders. On multi-core systems (>=4 cores), auto mode prefers `libx264 ultrafast` with ABR (average bitrate) over hardware encoders, because multi-threaded CPU encoding with parallel segment workers outperforms single-threaded HW encoders at high resolutions.
4. **Pipeline selection** (`_main`): `auto` mode chooses between two pipelines:
   - **single-pass**: Feeds two concat-demuxer lists (all fronts, all rears) into one ffmpeg invocation. Used for HW encoders (parallelism doesn't help) or single-pair runs.
   - **segment**: Stacks each pair individually (in parallel via `--workers`), then concatenates segments with stream copy. Preferred for CPU-optimized encoding (parallel benefit), mixed audio, or as fallback if single-pass fails.
5. **Encoding**: Builds ffmpeg filter graphs (`vstack` with scaling/padding) and codec args. Supports `libx264` (CRF or ABR mode) and hardware codecs like `h264_videotoolbox` (bitrate mode). CPU-optimized mode allocates threads per worker based on available cores.

The filename regex (`FILE_RE`) expects a timestamp group `YYYYMMDD_HHMMSS` followed by `_F` or `_R` before the extension. The prefix before the timestamp is flexible.
