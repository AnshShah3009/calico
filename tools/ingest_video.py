#!/usr/bin/env python3
"""Ingest multi-camera video files into CALICO dataset format.

Reads synchronized video streams (e.g., MP4, AVI, MKV) for each camera,
extracts synchronized frames, and structures them into:
  <output_dir>/data/<cam_name>/external/%05d.png
  <output_dir>/network_specification_file.yaml
  <output_dir>/pattern_square_mm0.txt ...

Usage:
  python3 tools/ingest_video.py -o /path/to/calico_in \
    --videos cam0=vids/cam0.mp4 cam1=vids/cam1.mp4 cam2=vids/cam2.mp4 \
    --fps 5 --square-mm 40.0 --pattern april
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def parse_video_mappings(video_args: List[str]) -> Dict[str, Path]:
    """Parse list of 'cam_name=path/to/video.ext' mappings."""
    mappings: Dict[str, Path] = {}
    for item in video_args:
        if "=" not in item:
            raise ValueError(f"Invalid video mapping '{item}'. Expected format: cam_name=path/to/video")
        name, path_str = item.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        mappings[name] = path
    return mappings


def discover_videos_in_dir(video_dir: Path, extensions: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")) -> Dict[str, Path]:
    """Discover video files in a directory sorted by name."""
    mappings: Dict[str, Path] = {}
    files = sorted([f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions])
    for f in files:
        mappings[f.stem] = f.resolve()
    return mappings


def inspect_videos(mappings: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    """Open videos and read metadata (FPS, frame count, width, height)."""
    meta: Dict[str, Dict[str, float]] = {}
    for name, path in mappings.items():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        meta[name] = {"fps": fps, "frames": n_frames, "width": w, "height": h}
    return meta


def extract_synchronized_frames(
    mappings: Dict[str, Path],
    meta: Dict[str, Dict[str, float]],
    out_dir: Path,
    target_fps: Optional[float] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> int:
    """Extract synchronized frames from all video streams and save as PNGs."""
    data_dir = out_dir / "data"
    cam_dirs: Dict[str, Path] = {}
    caps: Dict[str, cv2.VideoCapture] = {}

    for name, path in mappings.items():
        c_dir = data_dir / name / "external"
        c_dir.mkdir(parents=True, exist_ok=True)
        cam_dirs[name] = c_dir
        cap = cv2.VideoCapture(str(path))
        caps[name] = cap

    # Determine frame stepping
    first_name = list(mappings.keys())[0]
    base_fps = meta[first_name]["fps"]
    if target_fps is not None and target_fps > 0:
        step = max(1, int(round(base_fps / target_fps)))
    else:
        step = max(1, stride)

    saved_count = 0
    frame_idx = 0

    try:
        while True:
            # Check bounds
            current_sec = frame_idx / base_fps if base_fps > 0 else 0.0
            if end_sec is not None and current_sec > end_sec:
                break
            if max_frames is not None and saved_count >= max_frames:
                break

            # Read frame from each camera
            frames_bgr = {}
            all_valid = True
            for name, cap in caps.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    all_valid = False
                    break
                frames_bgr[name] = frame

            if not all_valid:
                # One of the videos ended
                break

            if current_sec >= start_sec and (frame_idx % step == 0):
                for name, frame in frames_bgr.items():
                    out_path = cam_dirs[name] / f"{saved_count:05d}.png"
                    cv2.imwrite(str(out_path), frame)
                saved_count += 1
                if saved_count % 20 == 0:
                    print(f"Extracted {saved_count} synchronized frames...", flush=True)

            frame_idx += 1
    finally:
        for cap in caps.values():
            cap.release()

    return saved_count


def setup_calico_specifications(
    out_dir: Path,
    pattern: str = "april",
    square_mm: float = 40.0,
    number_boards: int = 2,
    root_dir: Optional[Path] = None,
) -> None:
    """Generate or copy network_specification_file.yaml and pattern_square_mm*.txt."""
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[1]

    spec_name = "april-grid.yaml" if pattern.lower() == "april" else "charuco-grid.yaml"
    src_spec = root_dir / "configs" / spec_name
    dest_spec = out_dir / "network_specification_file.yaml"

    if src_spec.is_file():
        shutil.copy(src_spec, dest_spec)
    else:
        # Fallback minimal template
        dest_spec.write_text(
            f"%YAML:1.0\ntype: {pattern.lower()}\nsquaresX: 6\nsquaresY: 6\n"
            f"squareLength: 80\nmargins: 20\ntagSpace: 16\nnumberBoards: {number_boards}\n"
            f"april_family: tag36h11\n",
            encoding="utf-8",
        )

    for b in range(number_boards):
        sq_file = out_dir / f"pattern_square_mm{b}.txt"
        sq_file.write_text(f"squareLength_mm  {square_mm:.1f}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest synchronized video streams into CALICO dataset format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--videos",
        nargs="+",
        help="List of camera mappings, e.g. cam0=vid0.mp4 cam1=vid1.mp4 ...",
    )
    group.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing video files named by camera (e.g. cam0.mp4, cam1.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Target CALICO dataset directory",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target extraction frame rate (e.g. 5.0 for 5 fps)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame extraction step / stride if --fps is not specified (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum synchronized frames to extract",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time offset in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="End time offset in seconds",
    )
    parser.add_argument(
        "--pattern",
        choices=["april", "charuco"],
        default="april",
        help="Pattern type: april or charuco (default: april)",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=40.0,
        help="Physical square size in millimeters (default: 40.0)",
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=2,
        help="Number of boards in calibration pattern rig (default: 2)",
    )

    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.videos:
        mappings = parse_video_mappings(args.videos)
    else:
        mappings = discover_videos_in_dir(args.video_dir.resolve())

    if len(mappings) < 2:
        print(f"Warning: Only {len(mappings)} camera found. CALICO requires multi-camera inputs.", file=sys.stderr)

    print("Inspecting video streams:")
    meta = inspect_videos(mappings)
    for name, info in meta.items():
        print(f"  {name}: {mappings[name].name} ({info['width']}x{info['height']} @ {info['fps']:.2f} fps, {int(info['frames'])} frames)")

    print(f"\nExtracting synchronized frames to {out_dir}/data/ ...")
    count = extract_synchronized_frames(
        mappings=mappings,
        meta=meta,
        out_dir=out_dir,
        target_fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    print(f"Extraction finished. Extracted {count} synchronized frames across {len(mappings)} cameras.")

    print("Configuring CALICO pattern specifications...")
    setup_calico_specifications(
        out_dir=out_dir,
        pattern=args.pattern,
        square_mm=args.square_mm,
        number_boards=args.boards,
    )
    print(f"Ready for calibration. Dataset prepared at: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
