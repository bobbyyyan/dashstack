#!/usr/bin/env python3
"""DashStack: stack front/rear dashcam clips and concatenate chronologically."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DashStack: discover dashcam clips, stack front over rear for each timestamp, "
            "then concatenate in chronological order."
        )
    )
    parser.add_argument(
        "--input-dir",
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
    return parser.parse_args()


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def run_command(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    printable = shlex.join(cmd)
    print(f"$ {printable}")
    subprocess.run(cmd, check=True)


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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def _fr_output_path(run: list[Segment], input_dir: Path) -> Path:
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
    end_ts = _seg_start_ts(run[-1])

    if start_ts == end_ts:
        name = f"{prefix}{start_ts}_FR.{ext}"
    else:
        name = f"{prefix}{start_ts}_{end_ts}_FR.{ext}"

    return input_dir / name


def _seg_duration(seg: Segment, probe: Callable[[Path], ClipProbe]) -> float:
    """Return probed duration of a segment."""
    return probe(seg.path if isinstance(seg, MergedClip) else seg.front).duration


def _print_split_plan(
    runs: list[list[Segment]],
    probe: Callable[[Path], ClipProbe],
    input_dir: Path,
) -> None:
    """Print a visual map of clips grouped into _FR output files."""
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

    clip_num = 0
    collapse_threshold = 6
    show_head = 3
    show_tail = 1

    for ri, run in enumerate(runs):
        fr_path = _fr_output_path(run, input_dir)

        run_dur = sum(_seg_duration(s, probe) for s in run)
        n_pairs = sum(1 for s in run if isinstance(s, ClipPair))
        n_merged = sum(1 for s in run if isinstance(s, MergedClip))

        # Header
        print(f"  ┌ {fr_path.name}")

        # Clips (collapse large runs)
        collapse = len(run) > collapse_threshold
        for si, seg in enumerate(run):
            clip_num += 1
            ts = _seg_start_ts(seg)
            dur = _seg_duration(seg, probe)

            if collapse and show_head <= si < len(run) - show_tail:
                if si == show_head:
                    hidden = len(run) - show_head - show_tail
                    print(f"  │       ⋮ ({hidden} more)")
                continue

            suffix = ""
            if isinstance(seg, MergedClip):
                suffix = f"  ← {seg.path.name}"

            print(f"  │  {clip_num:3d}. {ts}  {dur:3.0f}s{suffix}")

        # Footer
        parts = []
        if n_pairs:
            parts.append(f"{n_pairs} pair{'s' if n_pairs != 1 else ''}")
        if n_merged:
            parts.append(f"{n_merged} merged")
        print(f"  └ {' + '.join(parts)} · {_fmt_time(run_dur)}")

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


def write_concat_file(path: Path, clips: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(f"file '{escape_concat_path(clip)}'\n")


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

    write_concat_file(front_list_path, [pair.front for pair in pairs])
    write_concat_file(rear_list_path, [pair.rear for pair in pairs])

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

    # Separate into ClipPairs that need encoding and MergedClips that pass through.
    concat_entries: list[Path] = []
    tasks: list[tuple[int, ClipPair, list[str]]] = []
    work_segment_paths: list[Path] = []
    encode_index = 0

    for seg in segments:
        if isinstance(seg, MergedClip):
            concat_entries.append(seg.path)
        else:
            encode_index += 1
            segment_path = segments_dir / f"segment_{encode_index:05d}_{seg.timestamp}.mp4"
            concat_entries.append(segment_path)
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
    write_concat_file(concat_list_path, concat_entries)
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
        output_path = _fr_output_path(run, input_dir)

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


def _main() -> int:
    args = parse_args()

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

    print("Done.")
    return 0


def main() -> int:
    """Entry point that wraps _main and translates return code to sys.exit."""
    sys.exit(_main())


if __name__ == "__main__":
    main()
