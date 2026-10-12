#!/usr/bin/env python3
"""Pre-flight dataset validator for CALICO multi-camera calibration.

Inspects dataset structure, image resolutions, frame synchronization,
network YAML specifications, dictionary marker capacity limits, and
pattern metric size files before calibration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

# Known dictionary marker capacity limits
ARUCO_CAPACITIES = {
    0: ("DICT_4X4_50", 50),
    1: ("DICT_4X4_100", 100),
    2: ("DICT_4X4_250", 250),
    3: ("DICT_4X4_1000", 1000),
    4: ("DICT_5X5_50", 50),
    5: ("DICT_5X5_100", 100),
    6: ("DICT_5X5_250", 250),
    7: ("DICT_5X5_1000", 1000),
    8: ("DICT_6X6_50", 50),
    9: ("DICT_6X6_100", 100),
    10: ("DICT_6X6_250", 250),
    11: ("DICT_6X6_1000", 1000),
    12: ("DICT_7X7_50", 50),
    13: ("DICT_7X7_100", 100),
    14: ("DICT_7X7_250", 250),
    15: ("DICT_7X7_1000", 1000),
    16: ("DICT_ARUCO_ORIGINAL", 1024),
    17: ("DICT_APRILTAG_16h5", 30),
    18: ("DICT_APRILTAG_25h9", 35),
    19: ("DICT_APRILTAG_36h10", 2320),
    20: ("DICT_APRILTAG_36h11", 587),
}

APRIL_CAPACITIES = {
    "tag16h5": 30,
    "tag25h9": 35,
    "tag36h11": 587,
    "tag36h10": 2320,
    "tag25h7": 242,
    "tag36h9": 486,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse key: value lines from YAML without requiring external PyYAML."""
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%YAML"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.split("#")[0].strip()
            # Try int, float, or str
            if re.match(r"^-?\d+$", v):
                data[k] = int(v)
            elif re.match(r"^-?\d+\.\d+$", v):
                data[k] = float(v)
            else:
                data[k] = v
    return data


def validate_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Validate a CALICO dataset directory."""
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {
        "dataset_path": str(dataset_dir.resolve()),
        "cameras": {},
        "pattern": {},
        "boards": [],
    }

    if not dataset_dir.is_dir():
        return {
            "valid": False,
            "errors": [f"Dataset path does not exist or is not a directory: {dataset_dir}"],
            "warnings": [],
            "summary": summary,
        }

    # 1. Check data directory
    data_dir = dataset_dir / "data"
    if not data_dir.is_dir():
        # Check if cameras are directly under dataset_dir
        candidate_cams = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if any(list(d.glob("*.png")) + list(d.glob("external/*.png")) for d in candidate_cams):
            data_dir = dataset_dir
        else:
            errors.append(f"Missing 'data/' directory inside {dataset_dir}")

    cam_dirs: List[Path] = []
    if data_dir.is_dir():
        cam_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        if len(cam_dirs) < 2:
            errors.append(
                f"Multi-camera calibration requires at least 2 cameras; found {len(cam_dirs)} in {data_dir}"
            )

    # Inspect each camera
    frame_counts: Dict[str, int] = {}
    resolutions: Dict[str, Tuple[int, int]] = {}

    for cam in cam_dirs:
        ext_dir = cam / "external"
        target_dir = ext_dir if ext_dir.is_dir() else cam
        img_files = sorted([f for f in target_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        frame_counts[cam.name] = len(img_files)

        if len(img_files) == 0:
            errors.append(f"Camera '{cam.name}' contains no readable images in {target_dir}")
            continue

        # Inspect first image
        sample_img = cv2.imread(str(img_files[0]))
        if sample_img is None:
            errors.append(f"Could not decode image: {img_files[0]}")
            continue

        h, w = sample_img.shape[:2]
        resolutions[cam.name] = (w, h)

        # Inspect last image resolution consistency
        if len(img_files) > 1:
            last_img = cv2.imread(str(img_files[-1]))
            if last_img is not None:
                lh, lw = last_img.shape[:2]
                if (lw, lh) != (w, h):
                    errors.append(
                        f"Resolution mismatch in camera '{cam.name}': {img_files[0].name} is {w}x{h}, but {img_files[-1].name} is {lw}x{lh}"
                    )

        summary["cameras"][cam.name] = {
            "frames": len(img_files),
            "width": w,
            "height": h,
            "folder": str(target_dir.relative_to(dataset_dir)),
        }

    # Check frame synchronization
    if len(frame_counts) >= 2:
        distinct_counts = set(frame_counts.values())
        if len(distinct_counts) > 1:
            warnings.append(
                f"Frame count variance detected across cameras: {frame_counts}. CALICO matches frames pairwise by index; ensure images are synchronized."
            )

    # 2. Check network specification YAML
    yaml_path = dataset_dir / "network_specification_file.yaml"
    if not yaml_path.exists():
        errors.append(f"Missing specification file: {yaml_path.name}")
    else:
        try:
            yaml_data = parse_simple_yaml(yaml_path.read_text(encoding="utf-8"))
            pat_type = str(yaml_data.get("type", "charuco")).lower()
            num_boards = yaml_data.get("numberBoards", 1)
            sq_x = yaml_data.get("squaresX", 0)
            sq_y = yaml_data.get("squaresY", 0)
            sq_len = yaml_data.get("squareLength", 0)
            marker_len = yaml_data.get("markerLength", 0)

            summary["pattern"] = {
                "type": pat_type,
                "numberBoards": num_boards,
                "squaresX": sq_x,
                "squaresY": sq_y,
                "squareLength": sq_len,
            }

            if sq_x <= 0 or sq_y <= 0:
                errors.append(f"Invalid board dimensions in {yaml_path.name}: squaresX={sq_x}, squaresY={sq_y}")

            if pat_type == "charuco":
                summary["pattern"]["markerLength"] = marker_len
                if marker_len >= sq_len and sq_len > 0:
                    errors.append(
                        f"Invalid ChArUco dimensions: markerLength ({marker_len}) must be strictly less than squareLength ({sq_len})"
                    )

                arc_code = yaml_data.get("arcCode", 10)
                summary["pattern"]["arcCode"] = arc_code
                dict_name, capacity = ARUCO_CAPACITIES.get(arc_code, (f"Custom-{arc_code}", 250))
                summary["pattern"]["dictionary"] = dict_name
                summary["pattern"]["dict_capacity"] = capacity

                markers_per_board = (sq_x * sq_y) // 2
                total_needed = markers_per_board * num_boards
                if total_needed > capacity:
                    errors.append(
                        f"Dictionary marker overflow! {num_boards} board(s) of {sq_x}x{sq_y} requires {total_needed} markers, but {dict_name} capacity is only {capacity}."
                    )
            elif pat_type == "april":
                fam = str(yaml_data.get("april_family", "tag36h11")).lower()
                summary["pattern"]["april_family"] = fam
                capacity = APRIL_CAPACITIES.get(fam, 587)
                summary["pattern"]["dict_capacity"] = capacity
                total_needed = sq_x * sq_y * num_boards
                if total_needed > capacity:
                    errors.append(
                        f"AprilTag overflow! {num_boards} grid(s) of {sq_x}x{sq_y} requires {total_needed} tags, but family {fam} capacity is only {capacity}."
                    )
            else:
                warnings.append(f"Unrecognized pattern type '{pat_type}'; expected 'charuco' or 'april'")

            # 3. Check pattern_square_mm<N>.txt files
            for i in range(num_boards):
                txt_path = dataset_dir / f"pattern_square_mm{i}.txt"
                if not txt_path.exists():
                    errors.append(f"Missing required metric board size file: {txt_path.name}")
                else:
                    content = txt_path.read_text(encoding="utf-8").strip()
                    m = re.search(r"[-+]?\d*\.?\d+", content)
                    if m:
                        val = float(m.group(0))
                        if val <= 0:
                            errors.append(f"{txt_path.name}: squareLength_mm must be positive, got {val}")
                        summary["boards"].append({"board": i, "square_mm": val})
                    else:
                        errors.append(f"Could not parse numeric square size in {txt_path.name}")
        except Exception as e:
            errors.append(f"Error parsing {yaml_path.name}: {e}")

    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CALICO calibration dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to CALICO dataset directory")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    res = validate_dataset(args.dataset_dir)

    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    print("=" * 60)
    print(f"CALICO Pre-Flight Dataset Validation: {args.dataset_dir}")
    print("=" * 60)

    summary = res["summary"]
    cams = summary.get("cameras", {})
    print(f"\n[Cameras]: {len(cams)} detected")
    for name, info in cams.items():
        print(f"  • {name}: {info['frames']} frame(s), resolution: {info['width']}x{info['height']}")

    pat = summary.get("pattern", {})
    if pat:
        print(f"\n[Pattern Specifications]:")
        print(f"  • Type: {pat.get('type')}")
        print(f"  • Dimensions: {pat.get('squaresX')} x {pat.get('squaresY')} squares")
        print(f"  • Boards: {pat.get('numberBoards')}")
        if "dictionary" in pat:
            print(f"  • ArUco Dictionary: {pat['dictionary']} (Max {pat['dict_capacity']} markers)")
        if "april_family" in pat:
            print(f"  • AprilTag Family: {pat['april_family']} (Max {pat['dict_capacity']} tags)")

    boards = summary.get("boards", [])
    if boards:
        print(f"\n[Metric Board Scales]:")
        for b in boards:
            print(f"  • Board {b['board']}: {b['square_mm']} mm / square")

    if res["warnings"]:
        print(f"\n⚠️  [WARNINGS] ({len(res['warnings'])}):")
        for w in res["warnings"]:
            print(f"  ! {w}")

    if res["errors"]:
        print(f"\n❌ [ERRORS] ({len(res['errors'])}):")
        for e in res["errors"]:
            print(f"  ✗ {e}")
        print("\nResult: Validation FAILED. Please resolve the errors above.")
        return 1

    print("\n✅ Result: Dataset is VALID and ready for calibration!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
