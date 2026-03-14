#!/usr/bin/env python3
"""DashStack: stack front/rear dashcam clips and concatenate chronologically."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Union


FILE_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<timestamp>\d{8}_\d{6})_(?P<camera>[FR])\.(?P<ext>mp4|mov|m4v)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClipProbe:
    width: int
    height: int
    fps_expr: str
    duration: float
    has_audio: bool


FR_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<start_ts>\d{8}_\d{6})(?:_(?P<end_ts>\d{8}_\d{6}))?_FR\.(?P<ext>mp4|mov|m4v)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClipPair:
    timestamp: str
    front: Path
    rear: Path


@dataclass(frozen=True)
class MergedClip:
    start_ts: str
    end_ts: str
    path: Path


Segment = Union[ClipPair, MergedClip]


_SUBCOMMANDS = {"upload", "download", "dedup"}


def _add_stack_args(parser: argparse.ArgumentParser) -> None:
    """Add all stacking-related arguments to *parser*."""
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory containing dashcam clips (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for single combined output file. When omitted (default), split mode "
            "produces one _FR file per continuous run in the input directory."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".dashstack_work"),
        help="Temporary working directory for intermediate stacked segments.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["auto", "single-pass", "segment"],
        default="auto",
        help=(
            "Processing mode. 'auto' (default) picks the fastest compatible path: "
            "single-pass for multi-pair runs, segment mode when required for compatibility."
        ),
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="Output panel width for each camera view; default uses first pair min(F width, R width).",
    )
    parser.add_argument(
        "--video-codec",
        default="auto",
        help=(
            "Video encoder (default: auto). On multi-core systems (>=4 cores), auto "
            "selects libx264/ultrafast with parallel segment encoding for best speed. "
            "Otherwise detects fastest HW encoder: h264_videotoolbox (macOS), "
            "h264_nvenc (NVIDIA), h264_vaapi (Linux). Pass an explicit name to override."
        ),
    )
    parser.add_argument(
        "--video-bitrate",
        default="16M",
        help="Target video bitrate for HW encoders and auto-detected libx264 ABR mode (default: 16M).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="H.264 CRF quality for libx264 mode (default: 20).",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset for libx264 mode (default: veryfast).",
    )
    parser.add_argument(
        "--audio-source",
        choices=["front", "rear", "none"],
        default="front",
        help=(
            "Audio source per segment. If selected track is missing, silent audio is inserted. "
            "Default: front."
        ),
    )
    parser.add_argument(
        "--missing",
        choices=["skip", "error"],
        default="skip",
        help="Behavior when only one camera clip exists for a timestamp (default: skip).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N chronological pairs (useful for quick tests).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name or path.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable name or path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running ffmpeg.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated intermediate segments and concat list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if present.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel ffmpeg workers for segment mode "
            "(default: half of CPU cores, capped at 4)."
        ),
    )
    parser.add_argument(
        "--hwaccel",
        default="auto",
        help=(
            "Hardware-accelerated decoding (default: auto). Auto-detects best method: "
            "videotoolbox (macOS), cuda (NVIDIA), vaapi (Linux), or none (software fallback). "
            "Pass 'none' to force software decoding."
        ),
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=5.0,
        help="Seconds of gap between clips to split into separate output files (default: 5.0).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete source _F/_R files that are covered by _FR output files.",
    )


def _add_upload_args(parser: argparse.ArgumentParser) -> None:
    """Add upload-related arguments to *parser*."""
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files to upload followed by the rsync destination (e.g. host:/path).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what rsync would do without transferring.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete source files after successful upload.",
    )


def _add_download_args(parser: argparse.ArgumentParser) -> None:
    """Add download-related arguments to *parser*."""
    parser.add_argument(
        "source",
        help="rsync source, e.g. host:/path/ or user@host:/path/*.MP4",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default=".",
        help="Local destination directory (default: current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what rsync would do without transferring.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote source files after successful download.",
    )


def _add_dedup_args(parser: argparse.ArgumentParser) -> None:
    """Add dedup-related arguments to *parser*."""
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Video files to deduplicate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show duplicate regions without modifying files.",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=30,
        help="Maximum overlap to search for in seconds (default: 30).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name or path.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable name or path.",
    )


def parse_args() -> argparse.Namespace:
    # If the first positional arg isn't a known subcommand, treat the
    # invocation as the default "stack" command so that the existing
    # `dashstack videos` form keeps working.
    argv = sys.argv[1:]
    if not argv or argv[0] not in _SUBCOMMANDS and not argv[0].startswith("-"):
        # Could be `dashstack videos ...` or `dashstack --dry-run ...`
        # → default to stack.
        pass  # fall through to stack parser
    elif argv[0] in _SUBCOMMANDS:
        # Explicit subcommand — let the subparser dispatch handle it.
        parser = argparse.ArgumentParser(
            prog="dashstack",
            description="DashStack: dashcam clip stacking and utilities.",
        )
        sub = parser.add_subparsers(dest="command")
        upload_p = sub.add_parser(
            "upload",
            help="Upload files to a remote destination via rsync.",
        )
        _add_upload_args(upload_p)
        download_p = sub.add_parser(
            "download",
            help="Download files from a remote source via rsync.",
        )
        _add_download_args(download_p)
        dedup_p = sub.add_parser(
            "dedup",
            help="Find and remove duplicate portions in merged videos.",
        )
        _add_dedup_args(dedup_p)
        # Also register stack so --help shows it.
        stack_p = sub.add_parser(
            "stack",
            help="Stack front/rear dashcam clips (default command).",
        )
        _add_stack_args(stack_p)
        return parser.parse_args(argv)

    # Default: stack command.
    parser = argparse.ArgumentParser(
        description=(
            "DashStack: discover dashcam clips, stack front over rear for each timestamp, "
            "then concatenate in chronological order."
        )
    )
    _add_stack_args(parser)
    args = parser.parse_args(argv)
    args.command = None  # signals stack mode
    return args


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def run_command(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    printable = shlex.join(cmd)
    print(f"$ {printable}")
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _ProgressBar:
    """Thread-safe progress bar with ETA for ffmpeg encoding."""

    def __init__(self, total_seconds: float) -> None:
        self._total = max(total_seconds, 0.001)
        self._completed = 0.0
        self._active: dict[int, float] = {}
        self._lock = threading.Lock()
        self._start = time.monotonic()

    def update(self, worker_id: int, seconds: float) -> None:
        with self._lock:
            self._active[worker_id] = seconds
            self._draw()

    def done(self, worker_id: int, segment_seconds: float) -> None:
        with self._lock:
            self._active.pop(worker_id, None)
            self._completed += segment_seconds
            self._draw()

    def finish(self) -> None:
        elapsed = time.monotonic() - self._start
        print(
            f"\r  [{'█' * 30}] 100%  {_fmt_time(elapsed)} total{' ' * 20}",
            file=sys.stderr,
        )

    def _draw(self) -> None:
        current = self._completed + sum(self._active.values())
        pct = min(current / self._total, 1.0)
        elapsed = time.monotonic() - self._start
        filled = int(30 * pct)
        bar = "█" * filled + "░" * (30 - filled)
        if pct > 0.02 and elapsed > 1:
            eta = _fmt_time(elapsed / pct * (1 - pct))
        else:
            eta = "--:--"
        print(
            f"\r  [{bar}] {pct:4.0%}  {_fmt_time(elapsed)} elapsed  ETA {eta}  ",
            end="",
            file=sys.stderr,
            flush=True,
        )


def _run_with_progress(
    cmd: list[str],
    duration: float,
    bar: _ProgressBar,
    worker_id: int = 0,
) -> None:
    """Run ffmpeg with -progress pipe:1, feeding updates to the bar."""
    cmd = list(cmd)
    try:
        idx = cmd.index("-loglevel") + 2
    except ValueError:
        idx = 1
    cmd[idx:idx] = ["-progress", "pipe:1"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("out_time_us="):
            try:
                us = int(line.split("=", 1)[1])
                if us >= 0:
                    bar.update(worker_id, us / 1_000_000)
            except (ValueError, IndexError):
                pass
    proc.wait()
    bar.done(worker_id, duration)
    if proc.returncode != 0:
        stderr_text = proc.stderr.read() if proc.stderr else ""
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr_text)


def ffprobe_clip(path: Path, ffprobe_bin: str) -> ClipProbe:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found in {path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps_expr = video_stream.get("r_frame_rate", "30/1")

    duration = data.get("format", {}).get("duration")
    if duration is None:
        duration = video_stream.get("duration", "0")
    try:
        duration_float = float(duration)
    except (TypeError, ValueError):
        duration_float = 0.0

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return ClipProbe(
        width=width,
        height=height,
        fps_expr=fps_expr,
        duration=duration_float,
        has_audio=has_audio,
    )


def detect_hwaccel(ffmpeg_bin: str) -> str:
    """Auto-detect the best available hardware decoding method."""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10,
        )
        available = {
            line.strip().lower()
            for line in result.stdout.strip().splitlines()[1:]
            if line.strip()
        }
        for method in ["videotoolbox", "cuda", "vaapi", "qsv", "d3d11va", "dxva2"]:
            if method in available:
                return method
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "none"


def detect_video_codec(ffmpeg_bin: str) -> str:
    """Auto-detect the fastest available H.264 encoder."""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
        for encoder in [
            "h264_videotoolbox",
            "h264_nvenc",
            "h264_amf",
            "h264_vaapi",
            "h264_qsv",
        ]:
            if re.search(rf"^\s*V\S*\s+{re.escape(encoder)}\s", output, re.MULTILINE):
                return encoder
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "libx264"


def ensure_even(value: int) -> int:
    value = int(value)
    if value < 2:
        raise ValueError("Dimension must be >= 2")
    return value if value % 2 == 0 else value - 1


def panel_height(source_width: int, source_height: int, target_width: int) -> int:
    scaled = int(round(source_height * target_width / source_width))
    return ensure_even(scaled)


def escape_concat_path(path: Path) -> str:
    # concat demuxer single-quoted format: escape single quotes as '\''.
    return str(path).replace("'", "'\\''")


def _ts_seconds(ts: str) -> float:
    """Convert YYYYMMDD_HHMMSS timestamp to seconds since epoch."""
    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
    return dt.timestamp()


def _seconds_to_ts(seconds: float) -> str:
    """Convert seconds since epoch to YYYYMMDD_HHMMSS timestamp."""
    dt = datetime.fromtimestamp(seconds)
    return dt.strftime("%Y%m%d_%H%M%S")


def discover_pairs(
    input_dir: Path,
) -> tuple[list[ClipPair], list[MergedClip], list[str], list[str]]:
    grouped: dict[str, dict[str, Path]] = {}
    merged_clips: list[MergedClip] = []
    overwritten_notes: list[str] = []
    unmatched_notes: list[str] = []

    # First pass: find existing _FR files and build suppressed timestamp ranges.
    fr_ranges: list[tuple[str, str]] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FR_RE.match(path.name)
        if not match:
            continue
        start_ts = match.group("start_ts")
        end_ts = match.group("end_ts") or start_ts
        merged_clips.append(MergedClip(start_ts=start_ts, end_ts=end_ts, path=path))
        fr_ranges.append((start_ts, end_ts))

    def is_suppressed(ts: str) -> bool:
        for start, end in fr_ranges:
            if start <= ts <= end:
                return True
        return False

    # Second pass: find _F/_R files, skipping suppressed timestamps.
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FILE_RE.match(path.name)
        if not match:
            continue

        timestamp = match.group("timestamp")
        camera = match.group("camera").upper()

        if is_suppressed(timestamp):
            overwritten_notes.append(
                f"{timestamp}_{camera}: {path.name} suppressed by existing _FR file"
            )
            continue

        existing = grouped.setdefault(timestamp, {}).get(camera)
        if existing is not None:
            overwritten_notes.append(
                f"{timestamp}_{camera}: {existing.name} replaced by {path.name}"
            )
        grouped[timestamp][camera] = path

    pairs: list[ClipPair] = []
    for timestamp in sorted(grouped):
        bucket = grouped[timestamp]
        if "F" in bucket and "R" in bucket:
            pairs.append(ClipPair(timestamp=timestamp, front=bucket["F"], rear=bucket["R"]))
            continue

        present = "".join(sorted(bucket))
        missing = "F" if "F" not in bucket else "R"
        unmatched_notes.append(f"{timestamp}: present={present}, missing={missing}")

    merged_clips.sort(key=lambda mc: mc.start_ts)
    return pairs, merged_clips, unmatched_notes, overwritten_notes


def _seg_start_ts(seg: Segment) -> str:
    """Return the start timestamp of a Segment."""
    return seg.start_ts if isinstance(seg, MergedClip) else seg.timestamp


def split_into_runs(
    pairs: list[ClipPair],
    merged: list[MergedClip],
    probe: Callable[[Path], ClipProbe],
    gap_threshold: float,
) -> list[list[Segment]]:
    """Split clips into continuous runs separated by gaps > threshold."""
    timeline: list[Segment] = []
    timeline.extend(pairs)
    timeline.extend(merged)
    timeline.sort(key=_seg_start_ts)

    if not timeline:
        return []

    runs: list[list[Segment]] = [[timeline[0]]]
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]

        prev_start = _ts_seconds(_seg_start_ts(prev))
        if isinstance(prev, MergedClip):
            prev_dur = probe(prev.path).duration
        else:
            prev_dur = probe(prev.front).duration
        prev_end = prev_start + prev_dur

        curr_start = _ts_seconds(_seg_start_ts(curr))
        gap = curr_start - prev_end

        if gap > gap_threshold:
            runs.append([curr])
        else:
            runs[-1].append(curr)

    return runs


def _fr_output_path(
    run: list[Segment], input_dir: Path, probe: Callable[[Path], ClipProbe]
) -> Path:
    """Derive _FR output filename from a run's segments."""
    first = run[0]
    if isinstance(first, MergedClip):
        match = FR_RE.match(first.path.name)
        prefix = match.group("prefix") if match else ""
        ext = match.group("ext") if match else "mp4"
    else:
        match = FILE_RE.match(first.front.name)
        prefix = match.group("prefix") if match else ""
        ext = match.group("ext") if match else "mp4"

    start_ts = _seg_start_ts(run[0])

    # Compute actual end timestamp from last segment's start + duration.
    last = run[-1]
    last_start = _ts_seconds(_seg_start_ts(last))
    last_dur = _seg_duration(last, probe)
    end_ts = _seconds_to_ts(last_start + last_dur)

    if start_ts == end_ts:
        name = f"{prefix}{start_ts}_FR.{ext}"
    else:
        name = f"{prefix}{start_ts}_{end_ts}_FR.{ext}"

    return input_dir / name


def _seg_duration(seg: Segment, probe: Callable[[Path], ClipProbe]) -> float:
    """Return probed duration of a segment."""
    return probe(seg.path if isinstance(seg, MergedClip) else seg.front).duration


def _compute_overlaps(
    segments: list[Segment],
    probe: Callable[[Path], ClipProbe],
) -> list[float | None]:
    """Compute truncated durations for segments that overlap with the next.

    Returns one entry per segment: the truncated duration if the segment
    overlaps with the next one, or None if no truncation is needed.
    """
    result: list[float | None] = [None] * len(segments)
    for i in range(len(segments) - 1):
        curr_start = _ts_seconds(_seg_start_ts(segments[i]))
        curr_dur = _seg_duration(segments[i], probe)
        curr_end = curr_start + curr_dur

        next_start = _ts_seconds(_seg_start_ts(segments[i + 1]))
        overlap = curr_end - next_start
        if overlap > 0:
            result[i] = max(0.0, curr_dur - overlap)
    return result


def _print_split_plan(
    runs: list[list[Segment]],
    probe: Callable[[Path], ClipProbe],
    input_dir: Path,
) -> None:
    """Print a visual map of source files bracketed into _FR output files."""
    total_pairs = sum(sum(1 for s in r if isinstance(s, ClipPair)) for r in runs)
    total_merged = sum(sum(1 for s in r if isinstance(s, MergedClip)) for r in runs)

    entry_parts = []
    if total_pairs:
        entry_parts.append(f"{total_pairs} pair{'s' if total_pairs != 1 else ''}")
    if total_merged:
        entry_parts.append(f"{total_merged} existing")
    print(
        f"\n{' + '.join(entry_parts)}"
        f" → {len(runs)} output file{'s' if len(runs) != 1 else ''}\n"
    )

    collapse_threshold = 4  # segments
    show_head = 2
    show_tail = 1

    def _seg_lines(seg: Segment) -> list[str]:
        if isinstance(seg, MergedClip):
            return [seg.path.name]
        return [seg.front.name, seg.rear.name]

    for ri, run in enumerate(runs):
        fr_path = _fr_output_path(run, input_dir, probe)
        run_dur = sum(_seg_duration(s, probe) for s in run)
        n_pairs = sum(1 for s in run if isinstance(s, ClipPair))
        n_merged = sum(1 for s in run if isinstance(s, MergedClip))

        # Single MergedClip — already done.
        if len(run) == 1 and isinstance(run[0], MergedClip):
            print(f"  {run[0].path.name}  ── already merged")
        else:
            # Build display lines (one per file, with collapse for large runs).
            lines: list[str] = []
            collapse = len(run) > collapse_threshold
            if collapse:
                hidden = len(run) - show_head - show_tail
                for seg in run[:show_head]:
                    lines.extend(_seg_lines(seg))
                lines.append(f"⋮ ({hidden} more)")
                for seg in run[-show_tail:]:
                    lines.extend(_seg_lines(seg))
            else:
                for seg in run:
                    lines.extend(_seg_lines(seg))

            n_lines = len(lines)
            arrow_idx = (n_lines - 1) // 2
            max_w = max(len(l) for l in lines)

            parts = []
            if n_pairs:
                parts.append(f"{n_pairs} pair{'s' if n_pairs != 1 else ''}")
            if n_merged:
                parts.append(f"{n_merged} merged")
            label = f"→ {fr_path.name} ({' + '.join(parts)} · {_fmt_time(run_dur)})"

            for i, line in enumerate(lines):
                padded = line.ljust(max_w)
                if n_lines == 1:
                    print(f"  {padded}  ─{label}")
                elif i == arrow_idx:
                    bracket = "┐" if i == 0 else "┘" if i == n_lines - 1 else "┤"
                    print(f"  {padded}  {bracket}{label}")
                elif i == 0:
                    print(f"  {padded}  ┐")
                elif i == n_lines - 1:
                    print(f"  {padded}  ┘")
                else:
                    print(f"  {padded}  │")

        # Gap to next run
        if ri < len(runs) - 1:
            last_end = (
                _ts_seconds(_seg_start_ts(run[-1]))
                + _seg_duration(run[-1], probe)
            )
            next_start = _ts_seconds(_seg_start_ts(runs[ri + 1][0]))
            gap = next_start - last_end
            gap_str = _fmt_time(gap) if gap >= 60 else f"{gap:.0f}s"
            print(f"    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ {gap_str} gap ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")

    print()


def _use_cuda_filters(args: argparse.Namespace) -> bool:
    """Return True if full CUDA-accelerated filter graph should be used."""
    return args.hwaccel == "cuda" and "nvenc" in args.video_codec


def build_video_encode_args(args: argparse.Namespace, cuda_filters: bool = False) -> list[str]:
    if args.video_codec == "libx264":
        result = [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
        ]
        encode_threads = getattr(args, "encode_threads", 0)
        if encode_threads:
            result.extend(["-threads", str(encode_threads)])
        return result

    codec_args = [
        "-c:v",
        args.video_codec,
        "-b:v",
        args.video_bitrate,
    ]
    if not cuda_filters:
        codec_args.extend(["-pix_fmt", "yuv420p"])
    if "nvenc" in args.video_codec:
        codec_args.extend(["-preset", "p1"])
    if args.video_codec.endswith("videotoolbox"):
        codec_args.extend(["-allow_sw", "1"])
    if args.video_codec == "hevc_videotoolbox":
        # Improves compatibility with Apple players for HEVC.
        codec_args.extend(["-tag:v", "hvc1"])
    return codec_args


def _hwaccel_input(args: argparse.Namespace, path: str) -> list[str]:
    """Return ffmpeg flags for one input: optional hwaccel + -i <path>."""
    flags: list[str] = []
    if args.hwaccel != "none":
        flags.extend(["-hwaccel", args.hwaccel])
        if _use_cuda_filters(args):
            flags.extend(["-hwaccel_output_format", "cuda"])
    flags.extend(["-i", path])
    return flags


def write_concat_file(
    path: Path, clips: list[Path], durations: list[float | None] | None = None
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, clip in enumerate(clips):
            handle.write(f"file '{escape_concat_path(clip)}'\n")
            if durations and durations[i] is not None:
                handle.write(f"duration {durations[i]:.6f}\n")


def build_segment_command(
    *,
    args: argparse.Namespace,
    pair: ClipPair,
    segment_path: Path,
    target_width: int,
    front_panel_h: int,
    rear_panel_h: int,
    fps_expr: str,
    selected_has_audio: bool,
    max_duration: float | None = None,
) -> list[str]:
    cuda_filters = _use_cuda_filters(args)
    if cuda_filters:
        filter_complex = (
            f"[0:v]scale_cuda={target_width}:{front_panel_h}:format=yuv420p,"
            f"hwdownload,format=yuv420p,setsar=1[top];"
            f"[1:v]scale_cuda={target_width}:{rear_panel_h}:format=yuv420p,"
            f"hwdownload,format=yuv420p,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2:shortest=1,fps={fps_expr}[v]"
        )
    else:
        filter_complex = (
            f"[0:v]scale={target_width}:{front_panel_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{front_panel_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[top];"
            f"[1:v]scale={target_width}:{rear_panel_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{rear_panel_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2:shortest=1,fps={fps_expr},format=yuv420p[v]"
        )

    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    cmd.extend(_hwaccel_input(args, str(pair.front)))
    cmd.extend(_hwaccel_input(args, str(pair.rear)))
    cmd.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ])
    cmd.extend(build_video_encode_args(args, cuda_filters=cuda_filters))

    if args.audio_source == "none":
        cmd.extend(["-an"])
    else:
        audio_input_index = 0 if args.audio_source == "front" else 1
        if selected_has_audio:
            cmd.extend(["-map", f"{audio_input_index}:a:0"])
        else:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
            cmd.extend(["-map", "2:a:0"])
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ac",
                "2",
                "-ar",
                "48000",
            ]
        )

    if max_duration is not None:
        cmd.extend(["-t", f"{max_duration:.6f}"])
    cmd.extend(["-movflags", "+faststart", str(segment_path)])
    return cmd


def build_single_pass_command(
    *,
    args: argparse.Namespace,
    front_list_path: Path,
    rear_list_path: Path,
    output_path: Path,
    target_width: int,
    front_panel_h: int,
    rear_panel_h: int,
    fps_expr: str,
    selected_audio_any: bool,
    selected_audio_all: bool,
) -> list[str]:
    cuda_filters = _use_cuda_filters(args)
    if cuda_filters:
        filter_complex = (
            f"[0:v]scale_cuda={target_width}:{front_panel_h}:format=yuv420p,"
            f"hwdownload,format=yuv420p,setsar=1[top];"
            f"[1:v]scale_cuda={target_width}:{rear_panel_h}:format=yuv420p,"
            f"hwdownload,format=yuv420p,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2:shortest=1,fps={fps_expr}[v]"
        )
    else:
        filter_complex = (
            f"[0:v]scale={target_width}:{front_panel_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{front_panel_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[top];"
            f"[1:v]scale={target_width}:{rear_panel_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{rear_panel_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2:shortest=1,fps={fps_expr},format=yuv420p[v]"
        )

    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if args.hwaccel != "none":
        cmd.extend(["-hwaccel", args.hwaccel])
        if cuda_filters:
            cmd.extend(["-hwaccel_output_format", "cuda"])
    cmd.extend([
        "-f", "concat", "-safe", "0", "-i", str(front_list_path),
    ])
    if args.hwaccel != "none":
        cmd.extend(["-hwaccel", args.hwaccel])
        if cuda_filters:
            cmd.extend(["-hwaccel_output_format", "cuda"])
    cmd.extend([
        "-f", "concat", "-safe", "0", "-i", str(rear_list_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ])
    cmd.extend(build_video_encode_args(args, cuda_filters=cuda_filters))

    if args.audio_source == "none":
        cmd.extend(["-an"])
    else:
        if selected_audio_any and not selected_audio_all:
            raise RuntimeError(
                "single-pass cannot preserve mixed audio presence across selected clips; "
                "use segment mode."
            )
        audio_input_index = 0 if args.audio_source == "front" else 1
        if selected_audio_all:
            cmd.extend(["-map", f"{audio_input_index}:a:0"])
        else:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
            cmd.extend(["-map", "2:a:0"])
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ac",
                "2",
                "-ar",
                "48000",
            ]
        )

    cmd.extend(["-movflags", "+faststart", str(output_path)])
    return cmd


def run_single_pass_pipeline(
    *,
    args: argparse.Namespace,
    pairs: list[ClipPair],
    probe: Callable[[Path], ClipProbe],
    work_dir: Path,
    output_path: Path,
    target_width: int,
    front_panel_h: int,
    rear_panel_h: int,
    fps_expr: str,
    selected_audio_any: bool,
    selected_audio_all: bool,
    total_duration: float = 0.0,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    front_list_path = work_dir / "front_concat.txt"
    rear_list_path = work_dir / "rear_concat.txt"

    overlaps = _compute_overlaps(pairs, probe)
    overlap_count = sum(1 for d in overlaps if d is not None)
    if overlap_count:
        print(f"Trimming {overlap_count} overlapping clip(s) to remove duplicated portions.")

    write_concat_file(front_list_path, [pair.front for pair in pairs], overlaps)
    write_concat_file(rear_list_path, [pair.rear for pair in pairs], overlaps)

    cmd = build_single_pass_command(
        args=args,
        front_list_path=front_list_path,
        rear_list_path=rear_list_path,
        output_path=output_path,
        target_width=target_width,
        front_panel_h=front_panel_h,
        rear_panel_h=rear_panel_h,
        fps_expr=fps_expr,
        selected_audio_any=selected_audio_any,
        selected_audio_all=selected_audio_all,
    )

    print("Building final output with single-pass pipeline...")
    if args.dry_run:
        run_command(cmd, True)
    else:
        bar = _ProgressBar(total_duration)
        _run_with_progress(cmd, total_duration, bar)
        bar.finish()

    if not args.keep_temp and not args.dry_run:
        if front_list_path.exists():
            front_list_path.unlink()
        if rear_list_path.exists():
            rear_list_path.unlink()
        try:
            work_dir.rmdir()
        except OSError:
            pass


def run_segment_pipeline(
    *,
    args: argparse.Namespace,
    segments: list[Segment],
    probe: Callable[[Path], ClipProbe],
    work_dir: Path,
    output_path: Path,
    target_width: int,
    front_panel_h: int,
    rear_panel_h: int,
    fps_expr: str,
    workers: int = 1,
) -> None:
    segments_dir = work_dir / "segments"
    concat_list_path = work_dir / "segments.txt"

    segments_dir.mkdir(parents=True, exist_ok=True)
    for old_segment in segments_dir.glob("segment_*.mp4"):
        old_segment.unlink()
    if concat_list_path.exists():
        concat_list_path.unlink()

    # Detect overlapping segments and compute truncated durations.
    overlaps = _compute_overlaps(segments, probe)
    overlap_count = sum(1 for d in overlaps if d is not None)
    if overlap_count:
        print(f"Trimming {overlap_count} overlapping segment(s) to remove duplicated portions.")

    # Separate into ClipPairs that need encoding and MergedClips that pass through.
    concat_entries: list[Path] = []
    concat_durations: list[float | None] = []
    tasks: list[tuple[int, ClipPair, list[str]]] = []
    work_segment_paths: list[Path] = []
    encode_index = 0

    for i, seg in enumerate(segments):
        if isinstance(seg, MergedClip):
            concat_entries.append(seg.path)
            concat_durations.append(overlaps[i])  # duration directive for concat
        else:
            encode_index += 1
            segment_path = segments_dir / f"segment_{encode_index:05d}_{seg.timestamp}.mp4"
            concat_entries.append(segment_path)
            concat_durations.append(None)  # already truncated during encoding
            work_segment_paths.append(segment_path)

            selected_has_audio = False
            if args.audio_source != "none":
                selected_file = seg.front if args.audio_source == "front" else seg.rear
                selected_has_audio = probe(selected_file).has_audio

            cmd = build_segment_command(
                args=args,
                pair=seg,
                segment_path=segment_path,
                target_width=target_width,
                front_panel_h=front_panel_h,
                rear_panel_h=rear_panel_h,
                fps_expr=fps_expr,
                selected_has_audio=selected_has_audio,
                max_duration=overlaps[i],
            )
            tasks.append((encode_index, seg, cmd))

    num_encode = len(tasks)

    if num_encode > 0:
        effective_workers = min(workers, num_encode)
        if args.dry_run:
            for idx, pair, cmd in tasks:
                print(f"[{idx}/{num_encode}] Building stacked segment for {pair.timestamp}")
                run_command(cmd, True)
        else:
            total_dur = sum(probe(p.front).duration for _, p, _ in tasks)
            bar = _ProgressBar(total_dur)
            if effective_workers > 1:
                print(f"Encoding {num_encode} segments with {effective_workers} parallel workers...")
            else:
                print(f"Encoding {num_encode} segments...")

            def _encode(task: tuple[int, ClipPair, list[str]]) -> None:
                idx, pair, cmd = task
                seg_dur = probe(pair.front).duration
                _run_with_progress(cmd, seg_dur, bar, worker_id=idx)

            if effective_workers > 1:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    list(executor.map(_encode, tasks))
            else:
                for task in tasks:
                    _encode(task)
            bar.finish()

    # If only one entry and it's a MergedClip, nothing to concat.
    if len(concat_entries) == 1 and num_encode == 0:
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    write_concat_file(concat_list_path, concat_entries, concat_durations)
    concat_cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    print("Concatenating stacked segments...")
    run_command(concat_cmd, args.dry_run)

    if not args.keep_temp and not args.dry_run:
        if concat_list_path.exists():
            concat_list_path.unlink()
        for segment in work_segment_paths:
            if segment.exists():
                segment.unlink()
        try:
            segments_dir.rmdir()
        except OSError:
            pass
        try:
            work_dir.rmdir()
        except OSError:
            pass


def run_split_pipeline(
    *,
    args: argparse.Namespace,
    runs: list[list[Segment]],
    probe: Callable[[Path], ClipProbe],
    input_dir: Path,
    work_dir: Path,
    target_width: int,
    front_panel_h: int,
    rear_panel_h: int,
    fps_expr: str,
    workers: int = 1,
) -> None:
    """Process each continuous run into a separate _FR output file."""
    for run_index, run in enumerate(runs, start=1):
        output_path = _fr_output_path(run, input_dir, probe)

        # Skip if run is a single MergedClip (already done).
        if len(run) == 1 and isinstance(run[0], MergedClip):
            print(f"Run {run_index}/{len(runs)}: {output_path.name} (already merged, skipping)")
            continue

        if output_path.exists() and not args.overwrite:
            print(f"Run {run_index}/{len(runs)}: {output_path.name} already exists, skipping")
            continue

        print(f"Run {run_index}/{len(runs)}: building {output_path.name} ({len(run)} segments)")

        run_work_dir = work_dir / f"run_{run_index}"
        run_segment_pipeline(
            args=args,
            segments=run,
            probe=probe,
            work_dir=run_work_dir,
            output_path=output_path,
            target_width=target_width,
            front_panel_h=front_panel_h,
            rear_panel_h=rear_panel_h,
            fps_expr=fps_expr,
            workers=workers,
        )


def _fix_fr_names(input_dir: Path, ffprobe_bin: str) -> list[tuple[Path, Path]]:
    """Fix end timestamps in _FR filenames based on actual video duration.

    Returns list of (old_path, new_path) for files that were renamed.
    """
    renames: list[tuple[Path, Path]] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FR_RE.match(path.name)
        if not match:
            continue

        start_ts = match.group("start_ts")
        prefix = match.group("prefix")
        ext = match.group("ext")

        try:
            clip = ffprobe_clip(path, ffprobe_bin)
        except Exception:
            continue

        correct_end_ts = _seconds_to_ts(_ts_seconds(start_ts) + clip.duration)

        if start_ts == correct_end_ts:
            correct_name = f"{prefix}{start_ts}_FR.{ext}"
        else:
            correct_name = f"{prefix}{start_ts}_{correct_end_ts}_FR.{ext}"

        if path.name == correct_name:
            continue

        new_path = input_dir / correct_name
        if new_path.exists():
            continue

        path.rename(new_path)
        renames.append((path, new_path))

    return renames


def _clean_source_files(input_dir: Path, dry_run: bool) -> int:
    """Delete _F/_R files and superseded _FR files covered by _FR files."""
    fr_files: list[tuple[Path, str, str]] = []  # (path, start, end)
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FR_RE.match(path.name)
        if match:
            start = match.group("start_ts")
            end = match.group("end_ts") or start
            fr_files.append((path, start, end))

    if not fr_files:
        return 0

    to_delete: list[Path] = []

    # Find _F/_R source files covered by any _FR range.
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FILE_RE.match(path.name)
        if not match:
            continue
        ts = match.group("timestamp")
        for _, start, end in fr_files:
            if start <= ts <= end:
                to_delete.append(path)
                break

    # Find _FR files fully subsumed by a strictly larger _FR file.
    for fr_path, fr_start, fr_end in fr_files:
        for other_path, other_start, other_end in fr_files:
            if other_path == fr_path:
                continue
            if (other_start <= fr_start and fr_end <= other_end
                    and (other_start, other_end) != (fr_start, fr_end)):
                to_delete.append(fr_path)
                break

    if not to_delete:
        return 0

    print(f"\n{len(to_delete)} file(s) covered by _FR output:")
    for p in to_delete:
        print(f"  {p.name}")

    if dry_run:
        return len(to_delete)

    try:
        choice = input("Delete these files? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if choice != "y":
        return 0

    for p in to_delete:
        p.unlink()
    print(f"Deleted {len(to_delete)} source files.")
    return len(to_delete)


def _main(args: argparse.Namespace) -> int:

    input_dir = args.input_dir.resolve()
    split_mode = args.output is None
    output_path = None if split_mode else args.output.resolve()
    # Make work dir unique per invocation to avoid collisions when
    # multiple dashstack processes run concurrently.
    work_dir = args.work_dir.resolve() / f"run_{os.getpid()}_{int(time.time())}"

    workers = args.workers
    if workers is None:
        workers = max(1, min((os.cpu_count() or 4) // 2, 4))

    hwaccel_auto = args.hwaccel == "auto"
    if hwaccel_auto:
        args.hwaccel = detect_hwaccel(args.ffmpeg_bin)

    codec_auto = args.video_codec == "auto"
    cpu_optimized = False
    if codec_auto:
        cores = os.cpu_count() or 1
        hw_codec = detect_video_codec(args.ffmpeg_bin)
        if args.hwaccel == "cuda" and "nvenc" in hw_codec:
            args.video_codec = hw_codec
        elif cores >= 4 and hw_codec != "libx264":
            args.video_codec = "libx264"
            args.preset = "ultrafast"
            cpu_optimized = True
        else:
            args.video_codec = hw_codec

    if not input_dir.exists() or not input_dir.is_dir():
        eprint(f"Input directory does not exist or is not a directory: {input_dir}")
        return 1

    # Output-exists prompt only applies in single-file mode.
    if not split_mode and output_path.exists() and not args.overwrite:
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        counter = 1
        candidate = parent / f"{stem}_{counter}{suffix}"
        while candidate.exists():
            counter += 1
            candidate = parent / f"{stem}_{counter}{suffix}"

        print(f"Output file already exists: {output_path.name}")
        print(f"  [o] Overwrite")
        print(f"  [r] Rename to {candidate.name}")
        print(f"  [q] Quit")
        try:
            choice = input("Choice [o/r/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if choice == "o":
            pass
        elif choice == "r":
            output_path = candidate
        else:
            return 1

    renames = _fix_fr_names(input_dir, args.ffprobe_bin)
    if renames:
        print(f"Fixed {len(renames)} _FR filename(s):")
        for old, new in renames:
            print(f"  {old.name} → {new.name}")

    pairs, merged_clips, unmatched, overwritten = discover_pairs(input_dir)
    if overwritten:
        print("Warning: clip notes:")
        for note in overwritten:
            print(f"  - {note}")
    if unmatched:
        if args.missing == "error":
            eprint("Found timestamps with missing camera pairs:")
            for note in unmatched:
                eprint(f"  - {note}")
            return 1
        print("Skipping unmatched timestamps:")
        for note in unmatched:
            print(f"  - {note}")

    if args.limit is not None:
        if args.limit <= 0:
            eprint("--limit must be > 0")
            return 1
        pairs = pairs[: args.limit]

    if not pairs and not merged_clips:
        eprint("No clips found.")
        return 1

    if not pairs:
        if args.clean:
            print("All clips already merged into _FR files.")
            _clean_source_files(input_dir, args.dry_run)
            return 0
        print("All clips already merged into _FR files. Nothing to do.")
        return 0

    probe_cache: dict[Path, ClipProbe] = {}

    def probe(path: Path) -> ClipProbe:
        if path not in probe_cache:
            probe_cache[path] = ffprobe_clip(path, args.ffprobe_bin)
        return probe_cache[path]

    # Collect all unique files that need probing and probe them in parallel.
    files_to_probe: list[Path] = [pairs[0].rear]
    for pair in pairs:
        files_to_probe.append(pair.front)
    if args.audio_source == "rear":
        for pair in pairs:
            files_to_probe.append(pair.rear)
    for mc in merged_clips:
        files_to_probe.append(mc.path)
    unique_probe_files = list(dict.fromkeys(files_to_probe))

    if len(unique_probe_files) > 2 and workers > 1:
        print(f"Probing {len(unique_probe_files)} clips ({workers} parallel)...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(probe, unique_probe_files))
    else:
        for f in unique_probe_files:
            probe(f)

    # Compute runs early so we can use them in the visualization.
    if split_mode:
        runs = split_into_runs(pairs, merged_clips, probe, args.gap_threshold)
    else:
        # Non-split: show chronological clip order with gap analysis.
        timeline: list[Segment] = []
        timeline.extend(pairs)
        timeline.extend(merged_clips)
        timeline.sort(key=_seg_start_ts)

        print(f"\nChronological clip order ({len(timeline)} entries):")
        gap_count = 0
        for i, seg in enumerate(timeline, 1):
            ts = _seg_start_ts(seg)
            if isinstance(seg, MergedClip):
                dur = probe(seg.path).duration
                label = f"  {i:3d}. {ts} ({dur:.0f}s)  [merged] {seg.path.name}"
            else:
                dur = probe(seg.front).duration
                label = f"  {i:3d}. {ts} ({dur:.0f}s)  F={seg.front.name}  R={seg.rear.name}"
            if i == 1:
                gap_label = ""
            else:
                prev_seg = timeline[i - 2]
                prev_ts = _seg_start_ts(prev_seg)
                prev_dur = _seg_duration(prev_seg, probe)
                prev_end = _ts_seconds(prev_ts) + prev_dur
                gap = _ts_seconds(ts) - prev_end
                if gap > args.gap_threshold:
                    gap_label = f"  *** gap {gap:.0f}s ***"
                    gap_count += 1
                else:
                    gap_label = ""
            print(f"{label}{gap_label}")
        if gap_count:
            print(f"  ({gap_count} gap{'s' if gap_count != 1 else ''} detected)")
        print()

    first_pair = pairs[0]
    front_probe = probe(first_pair.front)
    rear_probe = probe(first_pair.rear)

    if args.target_width is None:
        target_width = ensure_even(min(front_probe.width, rear_probe.width))
    else:
        target_width = ensure_even(args.target_width)

    front_panel_h = panel_height(front_probe.width, front_probe.height, target_width)
    rear_panel_h = panel_height(rear_probe.width, rear_probe.height, target_width)
    fps_expr = front_probe.fps_expr

    selected_audio_presence: list[bool] = []
    if args.audio_source != "none":
        for pair in pairs:
            selected_file = pair.front if args.audio_source == "front" else pair.rear
            selected_audio_presence.append(probe(selected_file).has_audio)
    selected_audio_any = any(selected_audio_presence)
    selected_audio_all = all(selected_audio_presence) if selected_audio_presence else False

    # Configure thread allocation for CPU-optimized encoding.
    args.encode_threads = 0
    if cpu_optimized:
        effective_workers_est = min(workers, len(pairs))
        if effective_workers_est > 1:
            args.encode_threads = max(1, (os.cpu_count() or 4) // effective_workers_est)

    # Estimate total output duration for progress bar.
    total_duration = front_probe.duration * len(pairs)

    print(f"Input directory: {input_dir}")
    print(f"Pairs to encode: {len(pairs)}")
    if merged_clips:
        print(f"Existing _FR files: {len(merged_clips)}")
    print(f"Target panel width: {target_width}")
    print(f"Front panel: {target_width}x{front_panel_h}")
    print(f"Rear panel: {target_width}x{rear_panel_h}")
    print(f"Output FPS: {fps_expr}")
    print(f"Pipeline mode: {args.pipeline}")
    if cpu_optimized:
        codec_label = f"{args.video_codec}/{args.preset} CRF (cpu-optimized)"
    elif codec_auto:
        codec_label = f"{args.video_codec} (detected)"
    else:
        codec_label = args.video_codec
    print(f"Video codec: {codec_label}")
    if args.video_codec == "libx264":
        print(f"x264 preset/crf: {args.preset}/{args.crf}")
    else:
        print(f"Video bitrate: {args.video_bitrate}")
    print(f"Audio source: {args.audio_source}")
    if args.audio_source != "none":
        if selected_audio_all:
            print("Audio availability: present in all selected clips")
        elif not selected_audio_any:
            print("Audio availability: absent in all selected clips (silence will be generated)")
        else:
            print("Audio availability: mixed across selected clips")
    if args.hwaccel != "none":
        hwaccel_label = f"{args.hwaccel} (detected)" if hwaccel_auto else args.hwaccel
        print(f"HW decode: {hwaccel_label}")
    if _use_cuda_filters(args):
        print("GPU filters: CUDA decode + scale_cuda + NVENC encode")
    print(f"Workers: {workers}")
    print(f"Work directory: {work_dir}")
    if split_mode:
        print(f"Output mode: split ({len(runs)} run{'s' if len(runs) != 1 else ''})")
        _print_split_plan(runs, probe, input_dir)
    else:
        print(f"Output file: {output_path}")

    try:
        if split_mode:
            if args.dry_run:
                pass  # visualization above is the dry-run output
            else:
                run_split_pipeline(
                    args=args,
                    runs=runs,
                    probe=probe,
                    input_dir=input_dir,
                    work_dir=work_dir,
                    target_width=target_width,
                    front_panel_h=front_panel_h,
                    rear_panel_h=rear_panel_h,
                    fps_expr=fps_expr,
                    workers=workers,
                )
        else:
            # Single-file mode: build unified segment list.
            all_segments: list[Segment] = []
            all_segments.extend(pairs)
            all_segments.extend(merged_clips)
            all_segments.sort(key=_seg_start_ts)
            has_merged = len(merged_clips) > 0

            if args.pipeline == "segment" or has_merged:
                if has_merged:
                    print("Pipeline selected: segment (mixed pairs + merged clips)")
                else:
                    print("Pipeline selected: segment")
                run_segment_pipeline(
                    args=args,
                    segments=all_segments,
                    probe=probe,
                    work_dir=work_dir,
                    output_path=output_path,
                    target_width=target_width,
                    front_panel_h=front_panel_h,
                    rear_panel_h=rear_panel_h,
                    fps_expr=fps_expr,
                    workers=workers,
                )
            elif args.pipeline == "single-pass":
                print("Pipeline selected: single-pass")
                run_single_pass_pipeline(
                    args=args,
                    pairs=pairs,
                    probe=probe,
                    work_dir=work_dir,
                    output_path=output_path,
                    target_width=target_width,
                    front_panel_h=front_panel_h,
                    rear_panel_h=rear_panel_h,
                    fps_expr=fps_expr,
                    selected_audio_any=selected_audio_any,
                    selected_audio_all=selected_audio_all,
                    total_duration=total_duration,
                )
            else:
                if len(pairs) == 1:
                    print("Pipeline selected: segment (single clip pair)")
                    run_segment_pipeline(
                        args=args,
                        segments=all_segments,
                        probe=probe,
                        work_dir=work_dir,
                        output_path=output_path,
                        target_width=target_width,
                        front_panel_h=front_panel_h,
                        rear_panel_h=rear_panel_h,
                        fps_expr=fps_expr,
                        workers=workers,
                    )
                elif args.audio_source != "none" and selected_audio_any and not selected_audio_all:
                    print(
                        "Pipeline selected: segment "
                        "(single-pass cannot preserve mixed audio presence)"
                    )
                    run_segment_pipeline(
                        args=args,
                        segments=all_segments,
                        probe=probe,
                        work_dir=work_dir,
                        output_path=output_path,
                        target_width=target_width,
                        front_panel_h=front_panel_h,
                        rear_panel_h=rear_panel_h,
                        fps_expr=fps_expr,
                        workers=workers,
                    )
                elif _use_cuda_filters(args) and len(pairs) > 1 and workers > 1:
                    print("Pipeline selected: segment (parallel GPU encoding)")
                    run_segment_pipeline(
                        args=args,
                        segments=all_segments,
                        probe=probe,
                        work_dir=work_dir,
                        output_path=output_path,
                        target_width=target_width,
                        front_panel_h=front_panel_h,
                        rear_panel_h=rear_panel_h,
                        fps_expr=fps_expr,
                        workers=workers,
                    )
                elif cpu_optimized and len(pairs) > 1:
                    print("Pipeline selected: segment (parallel CPU encoding)")
                    run_segment_pipeline(
                        args=args,
                        segments=all_segments,
                        probe=probe,
                        work_dir=work_dir,
                        output_path=output_path,
                        target_width=target_width,
                        front_panel_h=front_panel_h,
                        rear_panel_h=rear_panel_h,
                        fps_expr=fps_expr,
                        workers=workers,
                    )
                else:
                    print("Pipeline selected: single-pass (fast default)")
                    try:
                        run_single_pass_pipeline(
                            args=args,
                            pairs=pairs,
                            probe=probe,
                            work_dir=work_dir,
                            output_path=output_path,
                            target_width=target_width,
                            front_panel_h=front_panel_h,
                            rear_panel_h=rear_panel_h,
                            fps_expr=fps_expr,
                            selected_audio_any=selected_audio_any,
                            selected_audio_all=selected_audio_all,
                            total_duration=total_duration,
                        )
                    except subprocess.CalledProcessError as exc:
                        if args.dry_run:
                            raise
                        eprint(
                            "Single-pass pipeline failed "
                            f"(exit {exc.returncode}); falling back to segment mode."
                        )
                        run_segment_pipeline(
                            args=args,
                            segments=all_segments,
                            probe=probe,
                            work_dir=work_dir,
                            output_path=output_path,
                            target_width=target_width,
                            front_panel_h=front_panel_h,
                            rear_panel_h=rear_panel_h,
                            fps_expr=fps_expr,
                            workers=workers,
                        )
    except subprocess.CalledProcessError as exc:
        eprint(f"Command failed with exit code {exc.returncode}")
        return exc.returncode
    except RuntimeError as exc:
        eprint(str(exc))
        return 1

    if args.clean and split_mode:
        _clean_source_files(input_dir, args.dry_run)

    print("Done.")
    return 0


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}PB"


# Matches rsync -P file-completion line, e.g.:
#     85983232 100%   31.19MB/s    0:00:02 (xfer#1, to-check=25/26)
_RSYNC_XFER_RE = re.compile(
    r"^\s+(?P<bytes>\d+)\s+100%\s+.*\(xfer#(?P<n>\d+),\s*to-check=(?P<remain>\d+)/(?P<total>\d+)\)"
)

# Matches rsync -P partial-progress line, e.g.:
#     2621440   3%   31.19MB/s    0:00:02
_RSYNC_PARTIAL_RE = re.compile(
    r"^\s+(?P<bytes>\d+)\s+(?P<pct>\d+)%\s+"
)


def _run_rsync_with_progress(cmd: list[str], total_bytes: int | None = None) -> int:
    """Run an rsync command and display an overall progress bar.

    Parses both per-file partial progress and file-completion lines
    from rsync's ``-P`` output to keep the bar moving smoothly.

    Returns the rsync exit code.
    """
    printable = shlex.join(cmd)
    print(f"$ {printable}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    assert proc.stdout is not None

    bytes_completed = 0   # bytes from fully transferred files
    bytes_partial = 0     # bytes from the file currently in progress
    file_pct = 0.0        # 0-1 progress within the current file
    total_files: int | None = None
    files_done = 0
    start = time.monotonic()

    for line in proc.stdout:
        m = _RSYNC_XFER_RE.match(line)
        if m:
            # File finished — add its size to completed, reset partial.
            bytes_completed += int(m.group("bytes"))
            bytes_partial = 0
            file_pct = 0.0
            total_files = int(m.group("total"))
            files_done = total_files - int(m.group("remain"))
        else:
            mp = _RSYNC_PARTIAL_RE.match(line)
            if mp:
                # Mid-file progress update.
                bytes_partial = int(mp.group("bytes"))
                file_pct = int(mp.group("pct")) / 100.0
            else:
                continue

        current = bytes_completed + bytes_partial
        if total_bytes:
            pct = min(current / total_bytes, 1.0)
        elif total_files:
            # Blend file count with partial progress for smoother bar.
            pct = min((files_done + file_pct) / total_files, 1.0)
        else:
            pct = 0.0

        elapsed = time.monotonic() - start
        filled = int(30 * pct)
        bar = "█" * filled + "░" * (30 - filled)
        if pct > 0.02 and elapsed > 1:
            eta = _fmt_time(elapsed / pct * (1 - pct))
        else:
            eta = "--:--"

        size_info = (
            f"{_fmt_bytes(current)}/{_fmt_bytes(total_bytes)}"
            if total_bytes
            else f"{_fmt_bytes(current)}  {files_done}/{total_files or '?'} files"
        )
        print(
            f"\r  [{bar}] {pct:4.0%}  {size_info}"
            f"  {_fmt_time(elapsed)} elapsed  ETA {eta}  ",
            end="",
            file=sys.stderr,
            flush=True,
        )

    proc.wait()

    if bytes_completed > 0:
        elapsed = time.monotonic() - start
        total_display = _fmt_bytes(total_bytes) if total_bytes else _fmt_bytes(bytes_completed)
        print(
            f"\r  [{'█' * 30}] 100%  {total_display}"
            f"  {_fmt_time(elapsed)} total{' ' * 30}",
            file=sys.stderr,
        )

    if proc.returncode != 0:
        stderr_text = proc.stderr.read() if proc.stderr else ""
        eprint(f"\nrsync failed with exit code {proc.returncode}")
        if stderr_text.strip():
            eprint(stderr_text.strip())

    return proc.returncode


def _frame_mad(a: bytes, b: bytes, max_mad: float = 256.0) -> float:
    """Mean absolute difference between two grayscale frame byte arrays.

    Bails out early if the running total exceeds *max_mad* per pixel.
    """
    total = 0
    n = len(a)
    cutoff = int(max_mad * n)
    for i in range(n):
        total += abs(a[i] - b[i])
        if total > cutoff:
            return 256.0
    return total / n


def _find_duplicate_regions(
    frames: list[bytes],
    fps: int,
    max_overlap_secs: int,
) -> list[tuple[float, float]]:
    """Find duplicate regions using cut-point detection and verification.

    Real duplicates occur at clip boundaries: there's an abrupt frame
    change (a cut), and the footage after the cut matches footage from
    a few seconds before it.  This avoids false positives from
    naturally similar dashcam footage.

    Returns list of (start_seconds, duration_seconds) for each duplicate.
    """
    if len(frames) < 3:
        return []

    # Step 1: Compute consecutive frame differences.
    diffs = [_frame_mad(frames[t], frames[t - 1]) for t in range(1, len(frames))]

    # Step 2: Find cut points — frames where the difference from the
    # previous frame is significantly higher than the local norm.
    sorted_diffs = sorted(diffs)
    median_diff = sorted_diffs[len(sorted_diffs) // 2]
    cut_threshold = max(median_diff * 3, 15.0)

    cut_points = [t + 1 for t, d in enumerate(diffs) if d > cut_threshold]

    # Step 3: At each cut point, look for a matching offset where the
    # footage after the cut repeats footage from before it.
    max_offset = max_overlap_secs * fps
    match_threshold = 8.0  # strict — actual dups are near-identical
    min_match = max(2, fps)  # at least ~1 second of sustained match

    regions: list[tuple[float, float]] = []
    skip_until = 0

    for cp in cut_points:
        if cp < skip_until:
            continue

        best_len = 0

        for d in range(fps, min(max_offset + 1, cp + 1)):
            if _frame_mad(frames[cp], frames[cp - d], match_threshold) > match_threshold:
                continue

            # Count how many consecutive frames match at this offset.
            match_len = 1
            while (
                cp + match_len < len(frames)
                and cp - d + match_len < cp
                and _frame_mad(
                    frames[cp + match_len],
                    frames[cp - d + match_len],
                    match_threshold,
                )
                <= match_threshold
            ):
                match_len += 1

            if match_len >= min_match and match_len > best_len:
                best_len = match_len

        if best_len > 0:
            regions.append((cp / fps, best_len / fps))
            skip_until = cp + best_len

    return regions


def _dedup(args: argparse.Namespace) -> int:
    """Find and remove duplicate portions in merged videos."""
    rc = 0
    for video_path in args.files:
        if not video_path.exists():
            eprint(f"{video_path}: not found")
            rc = 1
            continue

        print(f"\nAnalyzing {video_path.name}...")

        clip = ffprobe_clip(video_path, args.ffprobe_bin)

        # Extract small grayscale thumbnails at 1fps for comparison.
        # Use CUDA decoding when available for faster extraction.
        thumb_w, thumb_h = 32, 24
        frame_size = thumb_w * thumb_h
        fps = 1
        hwaccel = detect_hwaccel(args.ffmpeg_bin)
        cmd = [args.ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if hwaccel != "none":
            cmd.extend(["-hwaccel", hwaccel])
        cmd.extend([
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps},scale={thumb_w}:{thumb_h},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ])
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            eprint(f"  Failed to extract frames: {result.stderr.decode().strip()}")
            rc = 1
            continue

        raw = result.stdout
        frames: list[bytes] = []
        for i in range(0, len(raw) - frame_size + 1, frame_size):
            frames.append(raw[i : i + frame_size])

        print(f"  Extracted {len(frames)} frames ({_fmt_time(clip.duration)} video)")

        regions = _find_duplicate_regions(frames, fps, args.max_overlap)

        if not regions:
            print("  No duplicate regions found.")
            continue

        total_dup = sum(dur for _, dur in regions)
        print(f"  Found {len(regions)} duplicate region(s) ({total_dup:.1f}s total):")
        for start, dur in regions:
            print(f"    {_fmt_time(start)} ({dur:.1f}s)")

        if args.dry_run:
            continue

        # Build good (non-duplicate) sections.
        good_sections: list[tuple[float, float]] = []
        pos = 0.0
        for start, dur in regions:
            if start > pos:
                good_sections.append((pos, start))
            pos = start + dur
        if pos < clip.duration:
            good_sections.append((pos, clip.duration))

        # Write concat file with inpoint/outpoint to skip duplicates.
        work_dir = video_path.parent / ".dashstack_work" / f"dedup_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)
        concat_path = work_dir / "dedup_concat.txt"

        with concat_path.open("w", encoding="utf-8") as f:
            for sect_start, sect_end in good_sections:
                f.write(f"file '{escape_concat_path(video_path.resolve())}'\n")
                f.write(f"inpoint {sect_start:.6f}\n")
                f.write(f"outpoint {sect_end:.6f}\n")

        temp_output = work_dir / video_path.name
        concat_cmd = [
            args.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(temp_output),
        ]
        print("  Removing duplicate regions...")
        try:
            subprocess.run(concat_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            eprint(f"  ffmpeg failed (exit {exc.returncode})")
            rc = 1
            continue

        # Replace original: rename original out, move new in, delete original.
        backup = video_path.with_suffix(f".dup{video_path.suffix}")
        video_path.rename(backup)
        temp_output.rename(video_path)
        backup.unlink()

        # Cleanup temp files.
        concat_path.unlink(missing_ok=True)
        try:
            work_dir.rmdir()
            work_dir.parent.rmdir()
        except OSError:
            pass

        new_clip = ffprobe_clip(video_path, args.ffprobe_bin)
        saved = clip.duration - new_clip.duration
        print(f"  Done. Removed {saved:.1f}s of duplicate footage.")

    return rc


def _upload(args: argparse.Namespace) -> int:
    """Upload files to a remote destination via rsync over SSH."""
    if len(args.paths) < 2:
        eprint("Need at least one file and a destination.")
        eprint("Usage: dashstack upload [--dry-run] [--delete] <files...> <destination>")
        return 1

    destination = args.paths[-1]
    file_args = args.paths[:-1]

    files: list[Path] = []
    for f in file_args:
        p = Path(f)
        if p.exists():
            files.append(p)
        else:
            eprint(f"Warning: {f} does not exist, skipping.")

    if not files:
        eprint("No valid files to upload.")
        return 1

    rsync = shutil.which("rsync")
    if rsync is None:
        eprint("rsync not found on PATH.")
        return 1

    total_bytes = sum(f.stat().st_size for f in files)
    cmd = [rsync, "-avP"]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += [str(f) for f in files]
    cmd.append(destination)

    rc = _run_rsync_with_progress(cmd, total_bytes=total_bytes)
    if rc != 0:
        return rc

    if args.delete and not args.dry_run:
        print(f"\nDeleting {len(files)} source file(s)...")
        for f in files:
            f.unlink()
            print(f"  Deleted {f}")

    print("Upload complete.")
    return 0


def _download(args: argparse.Namespace) -> int:
    """Download files from a remote source via rsync over SSH."""
    rsync = shutil.which("rsync")
    if rsync is None:
        eprint("rsync not found on PATH.")
        return 1

    cmd = [rsync, "-avP"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.delete:
        cmd.append("--remove-source-files")
    cmd.append(args.source)
    cmd.append(args.destination)

    rc = _run_rsync_with_progress(cmd)
    if rc != 0:
        return rc

    print("Download complete.")
    return 0


def main() -> int:
    """Entry point that wraps _main and translates return code to sys.exit."""
    args = parse_args()
    if args.command == "upload":
        sys.exit(_upload(args))
    elif args.command == "download":
        sys.exit(_download(args))
    elif args.command == "dedup":
        sys.exit(_dedup(args))
    else:
        sys.exit(_main(args))


if __name__ == "__main__":
    main()
