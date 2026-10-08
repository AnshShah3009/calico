#!/usr/bin/env python3
"""Export CALICO calibration results to industry-standard formats.

Supports:
- ROS / ROS2 CameraInfo YAML (`--format ros`)
- NeRF / Nerfstudio transforms.json (`--format nerfstudio`)
- COLMAP cameras.txt and images.txt (`--format colmap`)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_calibration_data(cali_dir: Path) -> Dict[str, Any]:
    """Parse calibration.json from CALICO output directory."""
    json_path = cali_dir / "calibration.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def rot_mat_to_quat_wxyz(R: List[List[float]]) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]."""
    m00, m01, m02 = R[0][0], R[0][1], R[0][2]
    m10, m11, m12 = R[1][0], R[1][1], R[1][2]
    m20, m21, m22 = R[2][0], R[2][1], R[2][2]

    tr = m00 + m11 + m20
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m20):
        S = math.sqrt(1.0 + m00 - m11 - m20) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m20:
        S = math.sqrt(1.0 + m11 - m00 - m20) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m20 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    # Normalize
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 1e-9:
        qw /= n
        qx /= n
        qy /= n
        qz /= n
    return [qw, qx, qy, qz]


def export_ros(data: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Export ROS / ROS2 CameraInfo YAML files (one per camera)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for cam in data.get("cameras", []):
        name = cam.get("name", "camera")
        fx = cam.get("fx", 800.0)
        fy = cam.get("fy", 800.0)
        cx = cam.get("cx", 480.0)
        cy = cam.get("cy", 360.0)
        k1 = cam.get("k1", 0.0)
        k2 = cam.get("k2", 0.0)
        p1 = cam.get("p1", 0.0)
        p2 = cam.get("p2", 0.0)
        k3 = cam.get("k3", 0.0)

        width = int(cx * 2) if cx > 0 else 960
        height = int(cy * 2) if cy > 0 else 720

        # Camera matrix K (3x3)
        K = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]
        D = [k1, k2, p1, p2, k3]
        R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix P (3x4)
        P = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        yaml_content = f"""# ROS / ROS2 CameraInfo for {name}
# Generated from CALICO multi-camera calibration
image_width: {width}
image_height: {height}
camera_name: {name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{', '.join(f'{x:.8f}' for x in K)}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{', '.join(f'{x:.8f}' for x in D)}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [{', '.join(f'{x:.8f}' for x in R)}]
projection_matrix:
  rows: 3
  cols: 4
  data: [{', '.join(f'{x:.8f}' for x in P)}]
"""
        yaml_path = out_dir / f"{name}_camera_info.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        generated.append(yaml_path)

    return generated


def export_nerfstudio(data: Dict[str, Any], out_path: Path) -> Path:
    """Export NeRF / Nerfstudio format (transforms.json)."""
    cameras = data.get("cameras", [])
    if not cameras:
        raise ValueError("No cameras found in calibration data.")

    first = cameras[0]
    fx = first.get("fx", 800.0)
    fy = first.get("fy", 800.0)
    cx = first.get("cx", 480.0)
    cy = first.get("cy", 360.0)
    w = int(cx * 2) if cx > 0 else 960
    h = int(cy * 2) if cy > 0 else 720

    frames = []
    for cam in cameras:
        T_wc = cam.get("T_world_camera")
        if not T_wc or len(T_wc) != 4:
            continue

        # Convert OpenCV camera convention (X right, Y down, Z forward)
        # to OpenGL/NeRF convention (X right, Y up, Z backward)
        # In OpenGL/NeRF: Y_nerf = -Y_cv, Z_nerf = -Z_cv
        # Translation: mm to meters
        c2w = [
            [float(T_wc[0][0]), -float(T_wc[0][1]), -float(T_wc[0][2]), float(T_wc[0][3]) / 1000.0],
            [float(T_wc[1][0]), -float(T_wc[1][1]), -float(T_wc[1][2]), float(T_wc[1][3]) / 1000.0],
            [float(T_wc[2][0]), -float(T_wc[2][1]), -float(T_wc[2][2]), float(T_wc[2][3]) / 1000.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        frame_entry = {
            "file_path": f"images/{cam.get('name', 'camera')}.png",
            "transform_matrix": c2w,
            "fl_x": cam.get("fx", fx),
            "fl_y": cam.get("fy", fy),
            "cx": cam.get("cx", cx),
            "cy": cam.get("cy", cy),
            "k1": cam.get("k1", 0.0),
            "k2": cam.get("k2", 0.0),
            "p1": cam.get("p1", 0.0),
            "p2": cam.get("p2", 0.0),
        }
        frames.append(frame_entry)

    nerf_data = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_model": "OPENCV",
        "frames": frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nerf_data, indent=2), encoding="utf-8")
    return out_path


def export_colmap(data: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Export COLMAP cameras.txt and images.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = data.get("cameras", [])

    cameras_txt = out_dir / "cameras.txt"
    images_txt = out_dir / "images.txt"

    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]

    for idx, cam in enumerate(cameras, start=1):
        name = cam.get("name", f"cam{idx}")
        fx = cam.get("fx", 800.0)
        fy = cam.get("fy", 800.0)
        cx = cam.get("cx", 480.0)
        cy = cam.get("cy", 360.0)
        k1 = cam.get("k1", 0.0)
        k2 = cam.get("k2", 0.0)
        p1 = cam.get("p1", 0.0)
        p2 = cam.get("p2", 0.0)
        w = int(cx * 2) if cx > 0 else 960
        h = int(cy * 2) if cy > 0 else 720

        # COLMAP OPENCV model params: fx, fy, cx, cy, k1, k2, p1, p2
        cam_lines.append(
            f"{idx} OPENCV {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}"
        )

        T_wc = cam.get("T_world_camera")
        if T_wc and len(T_wc) == 4:
            # In COLMAP, pose is world-to-camera (T_cw = inv(T_wc)):
            # R_cw = R_wc^T, t_cw = -R_wc^T * t_wc (in meters)
            R_wc = [[T_wc[r][c] for c in range(3)] for r in range(3)]
            t_wc = [T_wc[r][3] / 1000.0 for r in range(3)]  # convert mm to meters

            # Transpose R_wc
            R_cw = [[R_wc[c][r] for c in range(3)] for r in range(3)]
            t_cw = [
                -(R_cw[r][0] * t_wc[0] + R_cw[r][1] * t_wc[1] + R_cw[r][2] * t_wc[2])
                for r in range(3)
            ]

            qw, qx, qy, qz = rot_mat_to_quat_wxyz(R_cw)
            img_lines.append(
                f"{idx} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f} {idx} {name}.png"
            )
            img_lines.append("")  # Empty line for feature points

    cameras_txt.write_text("\n".join(cam_lines) + "\n", encoding="utf-8")
    images_txt.write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    return [cameras_txt, images_txt]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export CALICO calibration data to ROS, NeRF, or COLMAP."
    )
    parser.add_argument(
        "cali_dir",
        type=Path,
        help="Path to CALICO output directory containing calibration.json",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["all", "ros", "nerfstudio", "colmap"],
        default="all",
        help="Target export format (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: <cali_dir>/exports)",
    )

    args = parser.parse_args()
    cali_dir = args.cali_dir.resolve()
    if not cali_dir.is_dir():
        print(f"Error: Directory {cali_dir} does not exist.", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve() if args.output_dir else cali_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading calibration from: {cali_dir}")
    data = load_calibration_data(cali_dir)
    num_cams = len(data.get("cameras", []))
    print(f"Found {num_cams} camera(s).")

    if args.format in ("all", "ros"):
        ros_dir = out_dir / "ros"
        files = export_ros(data, ros_dir)
        print(f"[ROS] Exported {len(files)} CameraInfo YAML file(s) to {ros_dir}")

    if args.format in ("all", "nerfstudio"):
        nerf_file = out_dir / "nerfstudio" / "transforms.json"
        export_nerfstudio(data, nerf_file)
        print(f"[NeRF] Exported transforms.json to {nerf_file}")

    if args.format in ("all", "colmap"):
        colmap_dir = out_dir / "colmap"
        files = export_colmap(data, colmap_dir)
        print(f"[COLMAP] Exported cameras.txt and images.txt to {colmap_dir}")

    print("Export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
