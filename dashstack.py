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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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


@dataclass(frozen=True)
class ClipPair:
    timestamp: str
    front: Path
    rear: Path


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
        default=Path("dashstack.mp4"),
        help="Path for final combined output.",
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
            "Video encoder (default: auto). Auto-detects fastest available encoder: "
            "h264_videotoolbox (macOS), h264_nvenc (NVIDIA), h264_vaapi (Linux), "
            "or libx264 software fallback. Pass an explicit name to override."
        ),
    )
    parser.add_argument(
        "--video-bitrate",
        default="16M",
        help="Target video bitrate when --video-codec is not libx264 (default: 16M).",
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
    return parser.parse_args()


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def run_command(cmd: list[str], dry_run: bool) -> None:
    printable = shlex.join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


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


def discover_pairs(input_dir: Path) -> tuple[list[ClipPair], list[str]]:
    grouped: dict[str, dict[str, Path]] = {}
    unmatched_notes: list[str] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = FILE_RE.match(path.name)
        if not match:
            continue

        timestamp = match.group("timestamp")
        camera = match.group("camera").upper()
        grouped.setdefault(timestamp, {})[camera] = path

    pairs: list[ClipPair] = []
    for timestamp in sorted(grouped):
        bucket = grouped[timestamp]
        if "F" in bucket and "R" in bucket:
            pairs.append(ClipPair(timestamp=timestamp, front=bucket["F"], rear=bucket["R"]))
            continue

        present = "".join(sorted(bucket))
        missing = "F" if "F" not in bucket else "R"
        unmatched_notes.append(f"{timestamp}: present={present}, missing={missing}")

    return pairs, unmatched_notes


def build_video_encode_args(args: argparse.Namespace) -> list[str]:
    if args.video_codec == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
        ]

    codec_args = [
        "-c:v",
        args.video_codec,
        "-b:v",
        args.video_bitrate,
        "-pix_fmt",
        "yuv420p",
    ]
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
    cmd.extend(build_video_encode_args(args))

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

    cmd.extend(["-movflags", "+faststart", "-shortest", str(segment_path)])
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
    cmd.extend([
        "-f", "concat", "-safe", "0", "-i", str(front_list_path),
    ])
    if args.hwaccel != "none":
        cmd.extend(["-hwaccel", args.hwaccel])
    cmd.extend([
        "-f", "concat", "-safe", "0", "-i", str(rear_list_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ])
    cmd.extend(build_video_encode_args(args))

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

    cmd.extend(["-movflags", "+faststart", "-shortest", str(output_path)])
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
    run_command(cmd, args.dry_run)

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
    pairs: list[ClipPair],
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

    # Pre-build all segment commands so we can run them in parallel.
    segment_paths: list[Path] = []
    tasks: list[tuple[int, ClipPair, list[str]]] = []
    total = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        segment_path = segments_dir / f"segment_{index:05d}_{pair.timestamp}.mp4"
        segment_paths.append(segment_path)

        selected_has_audio = False
        if args.audio_source != "none":
            selected_file = pair.front if args.audio_source == "front" else pair.rear
            selected_has_audio = probe(selected_file).has_audio

        cmd = build_segment_command(
            args=args,
            pair=pair,
            segment_path=segment_path,
            target_width=target_width,
            front_panel_h=front_panel_h,
            rear_panel_h=rear_panel_h,
            fps_expr=fps_expr,
            selected_has_audio=selected_has_audio,
        )
        tasks.append((index, pair, cmd))

    def _encode_segment(task: tuple[int, ClipPair, list[str]]) -> None:
        idx, p, c = task
        print(f"[{idx}/{total}] Building stacked segment for {p.timestamp}")
        run_command(c, args.dry_run)

    effective_workers = min(workers, total)
    if effective_workers > 1 and not args.dry_run:
        print(f"Encoding {total} segments with {effective_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            list(executor.map(_encode_segment, tasks))
    else:
        for task in tasks:
            _encode_segment(task)

    write_concat_file(concat_list_path, segment_paths)
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
        for segment in segment_paths:
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


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    work_dir = args.work_dir.resolve()

    workers = args.workers
    if workers is None:
        workers = max(1, min((os.cpu_count() or 4) // 2, 4))

    codec_auto = args.video_codec == "auto"
    if codec_auto:
        args.video_codec = detect_video_codec(args.ffmpeg_bin)

    hwaccel_auto = args.hwaccel == "auto"
    if hwaccel_auto:
        args.hwaccel = detect_hwaccel(args.ffmpeg_bin)

    if not input_dir.exists() or not input_dir.is_dir():
        eprint(f"Input directory does not exist or is not a directory: {input_dir}")
        return 1

    if output_path.exists() and not args.overwrite:
        # Find the next available filename for the rename option.
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
            pass  # proceed with existing output_path
        elif choice == "r":
            output_path = candidate
        else:
            return 1

    pairs, unmatched = discover_pairs(input_dir)
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

    if not pairs:
        eprint("No complete F/R clip pairs found.")
        return 1

    probe_cache: dict[Path, ClipProbe] = {}

    def probe(path: Path) -> ClipProbe:
        if path not in probe_cache:
            probe_cache[path] = ffprobe_clip(path, args.ffprobe_bin)
        return probe_cache[path]

    # Collect all unique files that need probing and probe them in parallel.
    files_to_probe: list[Path] = [pairs[0].front, pairs[0].rear]
    if args.audio_source != "none":
        for pair in pairs:
            files_to_probe.append(pair.front if args.audio_source == "front" else pair.rear)
    unique_probe_files = list(dict.fromkeys(files_to_probe))

    if len(unique_probe_files) > 2 and workers > 1:
        print(f"Probing {len(unique_probe_files)} clips ({workers} parallel)...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(probe, unique_probe_files))
    else:
        for f in unique_probe_files:
            probe(f)

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

    print(f"Input directory: {input_dir}")
    print(f"Pairs selected: {len(pairs)}")
    print(f"Target panel width: {target_width}")
    print(f"Front panel: {target_width}x{front_panel_h}")
    print(f"Rear panel: {target_width}x{rear_panel_h}")
    print(f"Output FPS: {fps_expr}")
    print(f"Pipeline mode: {args.pipeline}")
    codec_label = f"{args.video_codec} (detected)" if codec_auto else args.video_codec
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
    print(f"Workers: {workers}")
    print(f"Work directory: {work_dir}")
    print(f"Output file: {output_path}")

    try:
        if args.pipeline == "segment":
            print("Pipeline selected: segment")
            run_segment_pipeline(
                args=args,
                pairs=pairs,
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
            )
        else:
            if len(pairs) == 1:
                print("Pipeline selected: segment (single clip pair)")
                run_segment_pipeline(
                    args=args,
                    pairs=pairs,
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
                    pairs=pairs,
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
                        pairs=pairs,
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


if __name__ == "__main__":
    sys.exit(main())
