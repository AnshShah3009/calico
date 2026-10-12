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


def safe_export_name(name: str) -> str:
    """Strip path components so camera names cannot escape the export directory."""
    return Path(str(name)).name or "camera"


def image_size_from_cam(cam: Dict[str, Any], cali_dir: Optional[Path] = None) -> tuple:
    """Prefer explicit width/height; else probe a source image; last resort cx/cy*2."""
    w = cam.get("width") or cam.get("image_width")
    h = cam.get("height") or cam.get("image_height")
    if w and h and int(w) > 0 and int(h) > 0:
        return int(w), int(h)

    name = safe_export_name(cam.get("name", "camera"))
    if cali_dir is not None:
        cam_dir = Path(cali_dir) / "data" / name
        if cam_dir.is_dir():
            for pattern in ("**/*.png", "**/*.jpg", "**/*.jpeg"):
                for img_path in sorted(cam_dir.glob(pattern)):
                    try:
                        import cv2  # optional dependency for legacy JSON
                        im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                        if im is not None and im.shape[0] > 0 and im.shape[1] > 0:
                            return int(im.shape[1]), int(im.shape[0])
                    except Exception:
                        break
                    break

    cx = float(cam.get("cx", 0.0) or 0.0)
    cy = float(cam.get("cy", 0.0) or 0.0)
    # Legacy calibration.json lacked width/height; approximate when possible.
    if cx > 0:
        w_approx = int(cx * 2)
        h_approx = int(cy * 2) if cy > 0 else 720
        sys.stderr.write(
            f"Warning: camera '{name}' missing width/height; "
            f"falling back to approx {w_approx}x{h_approx}.\n"
        )
        return w_approx, h_approx
    return 0, 0


def optical_center_mm(T: List[List[float]]) -> List[float]:
    """OpenCV world-to-camera T: optical center C = -R^T t (mm)."""
    R = [[float(T[r][c]) for c in range(3)] for r in range(3)]
    t = [float(T[r][3]) for r in range(3)]
    Rt = [[R[c][r] for c in range(3)] for r in range(3)]
    return [
        -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
        -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
        -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
    ]


def invert_rigid_mm(T: List[List[float]]) -> List[List[float]]:
    """Invert a rigid 4x4 (rotation + mm translation)."""
    R = [[float(T[r][c]) for c in range(3)] for r in range(3)]
    t = [float(T[r][3]) for r in range(3)]
    Rt = [[R[c][r] for c in range(3)] for r in range(3)]
    tinv = [
        -(Rt[r][0] * t[0] + Rt[r][1] * t[1] + Rt[r][2] * t[2])
        for r in range(3)
    ]
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for r in range(3):
        for c in range(3):
            out[r][c] = Rt[r][c]
        out[r][3] = tinv[r]
    out[3][3] = 1.0
    return out


def rot_mat_to_quat_wxyz(R: List[List[float]]) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz] using Shepperd's algorithm."""
    m00, m01, m02 = R[0][0], R[0][1], R[0][2]
    m10, m11, m12 = R[1][0], R[1][1], R[1][2]
    m20, m21, m22 = R[2][0], R[2][1], R[2][2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
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


def export_ros(data: Dict[str, Any], out_dir: Path, cali_dir: Optional[Path] = None) -> List[Path]:
    """Export ROS / ROS2 CameraInfo YAML files (one per camera)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for cam in data.get("cameras", []):
        name = safe_export_name(cam.get("name", "camera"))
        fx = cam.get("fx", 800.0)
        fy = cam.get("fy", 800.0)
        cx = cam.get("cx", 480.0)
        cy = cam.get("cy", 360.0)
        k1 = cam.get("k1", 0.0)
        k2 = cam.get("k2", 0.0)
        p1 = cam.get("p1", 0.0)
        p2 = cam.get("p2", 0.0)
        k3 = cam.get("k3", 0.0)

        width, height = image_size_from_cam(cam, cali_dir)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Camera '{name}' missing width/height and no source images found."
            )

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


def export_nerfstudio(data: Dict[str, Any], out_path: Path, cali_dir: Optional[Path] = None) -> Path:
    """Export NeRF / Nerfstudio format (transforms.json)."""
    cameras = data.get("cameras", [])
    if not cameras:
        raise ValueError("No cameras found in calibration data.")

    first = cameras[0]
    fx = first.get("fx", 800.0)
    fy = first.get("fy", 800.0)
    cx = first.get("cx", 480.0)
    cy = first.get("cy", 360.0)
    w, h = image_size_from_cam(first, cali_dir)
    if w <= 0 or h <= 0:
        raise ValueError(
            "calibration.json cameras lack width/height and no source images found."
        )

    frames = []
    for cam in cameras:
        T_w2c = cam.get("T_world_camera")
        if not T_w2c or len(T_w2c) != 4:
            continue

        # Stored T is OpenCV world-to-camera. Nerfstudio wants camera-to-world
        # in OpenGL axes (Y up, Z back): invert, then flip Y/Z columns/rows.
        c2w_cv = invert_rigid_mm(T_w2c)
        c2w = [
            [float(c2w_cv[0][0]), -float(c2w_cv[0][1]), -float(c2w_cv[0][2]), float(c2w_cv[0][3]) / 1000.0],
            [float(c2w_cv[1][0]), -float(c2w_cv[1][1]), -float(c2w_cv[1][2]), float(c2w_cv[1][3]) / 1000.0],
            [float(c2w_cv[2][0]), -float(c2w_cv[2][1]), -float(c2w_cv[2][2]), float(c2w_cv[2][3]) / 1000.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        frame_entry = {
            "file_path": f"images/{safe_export_name(cam.get('name', 'camera'))}.png",
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


def export_colmap(data: Dict[str, Any], out_dir: Path, cali_dir: Optional[Path] = None) -> List[Path]:
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
        name = safe_export_name(cam.get("name", f"cam{idx}"))
        fx = cam.get("fx", 800.0)
        fy = cam.get("fy", 800.0)
        cx = cam.get("cx", 480.0)
        cy = cam.get("cy", 360.0)
        k1 = cam.get("k1", 0.0)
        k2 = cam.get("k2", 0.0)
        p1 = cam.get("p1", 0.0)
        p2 = cam.get("p2", 0.0)
        w, h = image_size_from_cam(cam, cali_dir)
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Camera '{name}' missing width/height and no source images found."
            )

        # COLMAP OPENCV model params: fx, fy, cx, cy, k1, k2, p1, p2
        cam_lines.append(
            f"{idx} OPENCV {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}"
        )

        T_w2c = cam.get("T_world_camera")
        if T_w2c and len(T_w2c) == 4:
            # CALICO stores OpenCV world-to-camera; COLMAP wants the same. Do not invert.
            R_cw = [[float(T_w2c[r][c]) for c in range(3)] for r in range(3)]
            t_cw = [float(T_w2c[r][3]) / 1000.0 for r in range(3)]  # mm -> m

            qw, qx, qy, qz = rot_mat_to_quat_wxyz(R_cw)
            img_lines.append(
                f"{idx} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f} {idx} {name}.png"
            )
            img_lines.append("")  # Empty line for feature points

    cameras_txt.write_text("\n".join(cam_lines) + "\n", encoding="utf-8")
    images_txt.write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    return [cameras_txt, images_txt]


def export_opencv(data: Dict[str, Any], out_dir: Path, cali_dir: Optional[Path] = None) -> List[Path]:
    """Export OpenCV cv::FileStorage YAML files (one per camera)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for cam in data.get("cameras", []):
        name = safe_export_name(cam.get("name", "camera"))
        fx = cam.get("fx", 800.0)
        fy = cam.get("fy", 800.0)
        cx = cam.get("cx", 480.0)
        cy = cam.get("cy", 360.0)
        k1 = cam.get("k1", 0.0)
        k2 = cam.get("k2", 0.0)
        p1 = cam.get("p1", 0.0)
        p2 = cam.get("p2", 0.0)
        k3 = cam.get("k3", 0.0)
        rms = cam.get("rms", 0.0)

        width, height = image_size_from_cam(cam, cali_dir)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Camera '{name}' missing width/height and no source images found."
            )

        T_wc = cam.get("T_world_camera", [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        yaml_content = f"""%YAML:1.0
---
image_width: {width}
image_height: {height}
camera_name: "{name}"
reprojection_rms: {rms:.6f}
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {fx:.8f}, 0., {cx:.8f}, 0., {fy:.8f}, {cy:.8f}, 0., 0., 1. ]
distortion_coefficients: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ {k1:.8e}, {k2:.8e}, {p1:.8e}, {p2:.8e}, {k3:.8e} ]
transform_world_camera: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ {T_wc[0][0]:.8f}, {T_wc[0][1]:.8f}, {T_wc[0][2]:.8f}, {T_wc[0][3]:.8f},
           {T_wc[1][0]:.8f}, {T_wc[1][1]:.8f}, {T_wc[1][2]:.8f}, {T_wc[1][3]:.8f},
           {T_wc[2][0]:.8f}, {T_wc[2][1]:.8f}, {T_wc[2][2]:.8f}, {T_wc[2][3]:.8f},
           {T_wc[3][0]:.8f}, {T_wc[3][1]:.8f}, {T_wc[3][2]:.8f}, {T_wc[3][3]:.8f} ]
"""
        yaml_path = out_dir / f"{name}_opencv.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        generated.append(yaml_path)

    return generated


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
        choices=["all", "ros", "nerfstudio", "colmap", "opencv"],
        default="all",
        help="Export format target (default: all)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output destination directory (default: <cali_dir>/exports)",
    )

    args = parser.parse_args()

    if not args.cali_dir.exists():
        print(f"Error: Directory does not exist: {args.cali_dir}", file=sys.stderr)
        return 1

    try:
        data = load_calibration_data(args.cali_dir)
    except Exception as e:
        print(f"Error reading calibration data: {e}", file=sys.stderr)
        return 1

    out_base = args.out or (args.cali_dir / "exports")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Loading calibration from: {args.cali_dir}")
    print(f"Found {len(data.get('cameras', []))} camera(s).")

    if args.format in ("all", "ros"):
        ros_dir = out_base / "ros"
        res = export_ros(data, ros_dir, args.cali_dir)
        print(f"[ROS] Exported {len(res)} CameraInfo YAML file(s) to {ros_dir}")

    if args.format in ("all", "nerfstudio"):
        nerf_path = out_base / "nerfstudio" / "transforms.json"
        export_nerfstudio(data, nerf_path, args.cali_dir)
        print(f"[NeRF] Exported transforms.json to {nerf_path}")

    if args.format in ("all", "colmap"):
        colmap_dir = out_base / "colmap"
        res = export_colmap(data, colmap_dir, args.cali_dir)
        print(f"[COLMAP] Exported cameras.txt and images.txt to {colmap_dir}")

    if args.format in ("all", "opencv"):
        cv_dir = out_base / "opencv"
        res = export_opencv(data, cv_dir, args.cali_dir)
        print(f"[OpenCV] Exported {len(res)} cv::FileStorage YAML file(s) to {cv_dir}")

    print("Export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
