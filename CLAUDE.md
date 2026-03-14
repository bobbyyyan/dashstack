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
dashstack videos   # run on a directory
dashstack --help
```

## Architecture

Key flow in `src/dashstack/cli.py`:

1. **Discovery** (`discover_pairs`): Scans input dir for files matching `FILE_RE` (`*YYYYMMDD_HHMMSS_F.mp4` / `*_R.mp4`) and groups them into `ClipPair`s by timestamp. Also scans for existing `_FR` files via `FR_RE`; source clips whose timestamps fall within an `_FR` range are suppressed (skipped). Returns `(pairs, merged_clips, unmatched, overwritten)`.
2. **Probing** (`ffprobe_clip`): Calls `ffprobe` to get resolution, fps, duration, and audio presence for each clip. Results are cached per-path. Probing runs in parallel when multiple clips exist.
3. **Auto-detection** (`detect_video_codec`, `detect_hwaccel`): Queries ffmpeg for available hardware encoders and decoders. Priority: (1) CUDA + NVENC → full GPU pipeline with `scale_cuda`, (2) multi-core CPU (>=4 cores) → `libx264 ultrafast` CRF with parallel segment workers, (3) other HW encoders.
4. **Mode selection** (`_main`):
   - **Split mode** (default, no `--output`): `split_into_runs` groups clips + merged `_FR` files into continuous runs separated by gaps > `--gap-threshold`. Each run produces one `_FR` output file in the input directory via `run_split_pipeline`.
   - **Single-file mode** (`--output path.mp4`): combines everything into one file. `auto` pipeline chooses between single-pass and segment modes.
5. **Pipeline types**:
   - **single-pass**: Feeds two concat-demuxer lists (all fronts, all rears) into one ffmpeg invocation. Used for non-GPU HW encoders or single-pair runs. Only supports `ClipPair`s.
   - **segment** (`run_segment_pipeline`): Handles mixed `Segment` lists (`ClipPair` + `MergedClip`). Encodes each `ClipPair` individually (in parallel via `--workers`), passes `MergedClip` paths through to the concat list without re-encoding, then concatenates with stream copy.
6. **Encoding**: Builds ffmpeg filter graphs and codec args. When CUDA + NVENC are available, uses `scale_cuda` + `hwdownload` + CPU `vstack` to keep decode/scale on GPU. Otherwise uses CPU filters (`scale` + `pad` + `vstack`). Supports `libx264` (CRF mode) and hardware codecs.
7. **Filename fixing** (`_fix_fr_names`): On every run, automatically probes existing `_FR` files and renames any whose end timestamp is wrong or missing (computed as start + duration).
8. **Cleanup** (`--clean`): Deletes source `_F`/`_R` files whose timestamps are covered by `_FR` files, and also deletes older `_FR` files that are fully subsumed by a larger `_FR` file.

9. **Overlap trimming** (`_compute_overlaps`): When adjacent segments overlap in time, the earlier segment is truncated. ClipPairs use `-t` during encoding; MergedClips use concat demuxer `duration` directives.
10. **Dedup** (`dashstack dedup`): Post-hoc duplicate removal for already-merged videos. Extracts small grayscale thumbnails at 1fps, detects cut points (abrupt frame changes), then verifies if footage after each cut matches footage from before it. Removes duplicates via concat demuxer `inpoint`/`outpoint` with stream copy.

Key types: `ClipPair` (F+R source pair), `MergedClip` (existing `_FR` file with start/end timestamps), `Segment = Union[ClipPair, MergedClip]`.

## Guidelines

- **Keep README.md updated** when adding, changing, or removing user-facing features (new flags, subcommands, behavior changes). The README is the primary user documentation.
