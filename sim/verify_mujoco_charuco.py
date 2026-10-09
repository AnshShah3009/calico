#!/usr/bin/env python3
"""Render a multi-camera ChArUco scene in MuJoCo, run CALICO, and verify vs Ground Truth."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.pop("PYOPENGL_PLATFORM", None)

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path.home() / "calico-mujoco-sim"
BOARD_SRC = SIM_ROOT / "charuco-pat" / "patterns"
OUT_IN = SIM_ROOT / "mujoco-charuco-in"
OUT_CALI = SIM_ROOT / "mujoco-charuco-out"
PREVIEWS = SIM_ROOT / "preview-charuco"
USE_CUDA = False

# Match configs/charuco-grid.yaml
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_PX = 80
MARKER_PX = 60
SQUARE_MM = 40.0
MARGINS_PX = 20
N_BOARDS = 2
N_CAMS = 3
N_FRAMES = 12
WIDTH, HEIGHT = 960, 720
FOVY_DEG = 48.46  # fy ≈ 800 px at 720p


def euler_zyx_to_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """MuJoCo wxyz quaternion from ZYX Euler angles in radians."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


def look_at_xyaxes(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> str:
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return f"{x[0]:.8f} {x[1]:.8f} {x[2]:.8f} {y[0]:.8f} {y[1]:.8f} {y[2]:.8f}"


def mj_cam_to_opencv_T(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 world-from-camera in OpenCV axes (X right, Y down, Z forward), meters."""
    z_mj = eye - target
    z_mj = z_mj / np.linalg.norm(z_mj)
    x = np.cross(up, z_mj)
    x = x / np.linalg.norm(x)
    y_mj = np.cross(z_mj, x)
    y_mj = y_mj / np.linalg.norm(y_mj)
    R = np.column_stack([x, -y_mj, -z_mj])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def rot_err_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a.T @ R_b
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def fy_from_fovy() -> float:
    return (HEIGHT * 0.5) / math.tan(math.radians(FOVY_DEG * 0.5))


def camera_specs() -> list[dict]:
    target = np.array([0.0, 0.0, 0.22])
    up = np.array([0.0, 0.0, 1.0])
    eyes = [
        np.array([0.00, -0.72, 0.42]),
        np.array([0.48, -0.62, 0.40]),
        np.array([-0.48, -0.62, 0.40]),
    ]
    cams = []
    for i, eye in enumerate(eyes):
        cams.append(
            {
                "name": f"cam{i}",
                "eye": eye,
                "target": target,
                "up": up,
                "xyaxes": look_at_xyaxes(eye, target, up),
                "T_m": mj_cam_to_opencv_T(eye, target, up),
            }
        )
    return cams


def generate_charuco_patterns() -> None:
    """Ensure ChArUco pattern images exist."""
    BOARD_SRC.mkdir(parents=True, exist_ok=True)
    p0 = BOARD_SRC / "Board0.png"
    p1 = BOARD_SRC / "Board1.png"
    if p0.exists() and p1.exists():
        return

    print("Generating ChArUco pattern boards using calico-cpu...")
    pat_in = SIM_ROOT / "charuco-pat-in"
    pat_in.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "charuco-grid.yaml", pat_in / "network_specification_file.yaml")

    cmd = [
        "docker",
        "compose",
        "-f",
        str(ROOT / "docker-compose.yml"),
        "run",
        "--rm",
        f"--user={os.getuid()}:{os.getgid()}",
        "--no-deps",
        "-v",
        f"{pat_in}:/docker_dir/sim-in",
        "-v",
        f"{BOARD_SRC.parent}:/docker_dir/sim-out",
        "--entrypoint",
        "calico-dec2023",
        "calico-cpu",
        "--charuco",
        "--create-patterns",
        "--input=/docker_dir/sim-in/",
        "--output=/docker_dir/sim-out/",
        "--src-dir=/src/",
    ]
    subprocess.run(cmd, check=True)
    if not (p0.exists() and p1.exists()):
        raise RuntimeError("Failed to generate ChArUco pattern images.")


def build_xml(cams: list[dict], half_w: float, half_h: float) -> str:
    cam_xml = "\n".join(
        f'    <camera name="{c["name"]}" pos="{c["eye"][0]:.4f} {c["eye"][1]:.4f} {c["eye"][2]:.4f}" '
        f'xyaxes="{c["xyaxes"]}" fovy="{FOVY_DEG:.2f}"/>'
        for c in cams
    )
    boards_xml = ""
    for b in range(N_BOARDS):
        boards_xml += f"""
    <body name="board{b}" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="{half_w:.5f} 0.001 {half_h:.5f}"
            rgba="1 1 1 1" contype="0" conaffinity="0"/>
    </body>"""
    return f"""<mujoco model="calico_charuco_verification">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <quality offsamples="4"/>
  </visual>
  <worldbody>
    <light pos="0 -0.5 1.5" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.6 -0.8 1.0" dir="-0.4 0.5 -0.7" diffuse="0.45 0.45 0.45"/>
    <light pos="-0.6 -0.8 1.0" dir="0.4 0.5 -0.7" diffuse="0.45 0.45 0.45"/>
{boards_xml}
{cam_xml}
  </worldbody>
</mujoco>"""


def board_pose(frame: int, board: int) -> tuple[np.ndarray, np.ndarray]:
    t = frame * 0.35
    side = -1.0 if board == 0 else 1.0
    pos = np.array(
        [
            side * 0.28 + 0.02 * math.sin(t + board),
            0.04 * board + 0.015 * math.cos(t * 0.7),
            0.20 + 0.02 * math.sin(t * 0.4 + board),
        ]
    )
    yaw = side * 0.25 + 0.06 * math.sin(t * 0.5)
    pitch = 0.04 * math.cos(t * 0.4 + board)
    roll = 0.03 * math.sin(t * 0.3 + board)
    quat = euler_zyx_to_quat(yaw, pitch, roll)
    return pos, quat


def set_board(model, data, board: int, pos, quat) -> None:
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"board{board}")
    adr = model.jnt_qposadr[jnt]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat


def camera_K() -> np.ndarray:
    fy = fy_from_fovy()
    return np.array([[fy, 0.0, WIDTH / 2.0], [0.0, fy, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def T_opencv_from_mj_camera(model, data, cam_name: str) -> np.ndarray:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    pos = data.cam_xpos[cam_id]
    mat = data.cam_xmat[cam_id].reshape(3, 3)
    x = mat[:, 0]
    y_mj = mat[:, 1]
    z_mj = mat[:, 2]
    R_cv = np.column_stack([x, -y_mj, -z_mj])
    T = np.eye(4)
    T[:3, :3] = R_cv
    T[:3, 3] = pos
    return T


def project_points(T_world_cam: np.ndarray, K: np.ndarray, pts_w: np.ndarray):
    T_cw = inv_T(T_world_cam)
    pts_h = np.column_stack([pts_w, np.ones(len(pts_w))])
    pts_c = (T_cw @ pts_h.T).T[:, :3]
    if np.any(pts_c[:, 2] <= 0.05):
        return None, None
    uv_h = (K @ pts_c.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    return uv.astype(np.float32), float(np.mean(pts_c[:, 2]))


def paste_board(canvas: np.ndarray, texture: np.ndarray, quad_uv: np.ndarray) -> None:
    th, tw = texture.shape[:2]
    src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, quad_uv)
    warped = cv2.warpPerspective(texture, H, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.ones((th, tw), dtype=np.uint8) * 255,
        H,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    idx = mask > 0
    canvas[idx] = warped[idx]


def render_pinhole(model, data, textures, half_w: float, half_h: float, cams: list[dict], K: np.ndarray):
    hy = 0.001
    # Corner 0: top-left, Corner 1: top-right, Corner 2: bottom-right, Corner 3: bottom-left
    corners_local = np.array(
        [
            [-half_w, -hy, half_h],
            [half_w, -hy, half_h],
            [half_w, -hy, -half_h],
            [-half_w, -hy, -half_h],
        ],
        dtype=np.float64,
    )
    row = []
    for c in cams:
        T_wc = T_opencv_from_mj_camera(model, data, c["name"])
        canvas = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
        layers = []
        for b in range(N_BOARDS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"board{b}")
            t = np.array(data.xpos[bid], dtype=np.float64)
            R = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            pts_w = (R @ corners_local.T).T + t
            uv, zmean = project_points(T_wc, K, pts_w)
            if uv is None:
                continue
            layers.append((zmean, uv, textures[b]))
        layers.sort(key=lambda x: -x[0])
        for _, uv, tex in layers:
            paste_board(canvas, tex, uv)
        row.append(canvas)
    return row


def write_dataset(frames_bgr, cams) -> None:
    clean_dir(OUT_IN)
    (OUT_IN / "data").mkdir(parents=True)
    for ci in range(N_CAMS):
        ext = OUT_IN / "data" / f"cam{ci}" / "external"
        ext.mkdir(parents=True)
        for fi, images in enumerate(frames_bgr):
            cv2.imwrite(str(ext / f"{fi:04d}.png"), images[ci])
    shutil.copy(ROOT / "configs" / "charuco-grid.yaml", OUT_IN / "network_specification_file.yaml")
    for b in range(N_BOARDS):
        (OUT_IN / f"pattern_square_mm{b}.txt").write_text(f"squareLength_mm  {SQUARE_MM:.1f}\n")
    fy = fy_from_fovy()
    gt = {
        "width": WIDTH,
        "height": HEIGHT,
        "fovy_deg": FOVY_DEG,
        "fx": fy,
        "fy": fy,
        "cx": WIDTH / 2.0,
        "cy": HEIGHT / 2.0,
        "units": "meters then T_mm = 1000*T_m",
        "cameras": [],
    }
    for c in cams:
        Tmm = c["T_m"].copy()
        Tmm[:3, 3] *= 1000.0
        gt["cameras"].append(
            {
                "name": c["name"],
                "T_world_camera_m": c["T_m"].tolist(),
                "T_world_camera_mm": Tmm.tolist(),
            }
        )
    (OUT_IN / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    ingest = SIM_ROOT / "intrinsics"
    if ingest.exists():
        shutil.rmtree(ingest)
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    for c in cams:
        d = ingest / c["name"]
        d.mkdir(parents=True)
        (d / "cali_results.txt").write_text(
            "internal_matrix\n"
            f"{fy} 0 {cx}\n"
            f"0 {fy} {cy}\n"
            "0 0 1\n"
            "distortion_size 5\n"
            "distortion_vector\n"
            "0 0 0 0 0\n"
        )


def clean_dir(p: Path) -> None:
    if not p.exists():
        return
    try:
        shutil.rmtree(p)
    except PermissionError:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{p.parent}:/target", "alpine:3.20", "rm", "-rf", f"/target/{p.name}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def render_scene() -> tuple:
    generate_charuco_patterns()
    scale = (SQUARE_MM / 1000.0) / SQUARE_PX
    w_px = SQUARES_X * SQUARE_PX + 2 * MARGINS_PX
    h_px = SQUARES_Y * SQUARE_PX + 2 * MARGINS_PX
    half_w = 0.5 * w_px * scale
    half_h = 0.5 * h_px * scale

    cams = camera_specs()
    xml = build_xml(cams, half_w, half_h)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    textures = []
    for b in range(N_BOARDS):
        p = BOARD_SRC / f"Board{b}.png"
        if not p.exists():
            raise FileNotFoundError(f"Need generated board {p}")
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        textures.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    K = camera_K()
    frames = []
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for fi in range(N_FRAMES):
        for b in range(N_BOARDS):
            pos, quat = board_pose(fi, b)
            set_board(model, data, b, pos, quat)
        mujoco.mj_forward(model, data)
        if fi == 0:
            for i, c in enumerate(cams):
                cams[i]["T_m"] = T_opencv_from_mj_camera(model, data, c["name"])
        row = render_pinhole(model, data, textures, half_w, half_h, cams, K)
        frames.append(row)
        if fi == 0:
            labeled_row = []
            for ci, c in enumerate(cams):
                labeled = row[ci].copy()
                cv2.putText(
                    labeled,
                    c["name"],
                    (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(PREVIEWS / f"{c['name']}_t0.png"), labeled)
                labeled_row.append(labeled)
            cv2.imwrite(str(PREVIEWS / "t0_all_cams.png"), np.hstack(labeled_row))
    write_dataset(frames, cams)
    return cams


def run_calico() -> None:
    clean_dir(OUT_CALI)
    OUT_CALI.mkdir(parents=True)
    service = "calico-cuda" if USE_CUDA else "calico-cpu"
    cmd = [
        "docker",
        "compose",
        "-f",
        str(ROOT / "docker-compose.yml"),
        "run",
        "--rm",
        f"--user={os.getuid()}:{os.getgid()}",
        "--no-deps",
        "-v",
        f"{OUT_IN}:/docker_dir/sim-in",
        "-v",
        f"{OUT_CALI}:/docker_dir/sim-out",
        "-v",
        f"{SIM_ROOT / 'intrinsics'}:/docker_dir/sim-k",
        "--entrypoint",
        "calico-dec2023",
        service,
        "--charuco",
        "--calibrate",
        "--input=/docker_dir/sim-in/",
        "--output=/docker_dir/sim-out/",
        "--src-dir=/src/",
        "--json",
        "--summary",
        "--ingest-intrinsics=/docker_dir/sim-k/",
    ]
    if USE_CUDA:
        cmd.append("--use-cuda")
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"CALICO exited {r.returncode}")


def mat4(list4) -> np.ndarray:
    return np.array(list4, dtype=np.float64)


def compare() -> int:
    gt = json.loads((OUT_IN / "ground_truth.json").read_text())
    cali_path = OUT_CALI / "calibration.json"
    if not cali_path.exists():
        print("Missing", cali_path)
        return 2
    est = json.loads(cali_path.read_text())
    fx_gt, fy_gt = gt["fx"], gt["fy"]
    cx_gt, cy_gt = gt["cx"], gt["cy"]
    print("\n=== Intrinsics (px) ===")
    print(f"{'cam':6} {'fx_gt':>8} {'fx':>8} {'dfx':>8} {'fy':>8} {'dfy':>8} {'cx':>8} {'dcx':>8} {'cy':>8} {'dcy':>8}")
    k_ok = True
    for cam in est["cameras"]:
        dfx = cam["fx"] - fx_gt
        dfy = cam["fy"] - fy_gt
        dcx = cam["cx"] - cx_gt
        dcy = cam["cy"] - cy_gt
        print(
            f"{cam['name']:6} {fx_gt:8.1f} {cam['fx']:8.2f} {dfx:8.2f} "
            f"{cam['fy']:8.2f} {dfy:8.2f} {cam['cx']:8.2f} {dcx:8.2f} "
            f"{cam['cy']:8.2f} {dcy:8.2f}"
        )
        if abs(dfx) > 2 or abs(dfy) > 2 or abs(dcx) > 2 or abs(dcy) > 2:
            k_ok = False

    print("\n=== Relative camera poses (OpenCV, mm) vs GT ===")
    Tgt = [mat4(c["T_world_camera_mm"]) for c in gt["cameras"]]
    Test = [mat4(c["T_world_camera"]) for c in est["cameras"]]
    Test_inv = [inv_T(T) for T in Test]

    def report(label, Test_use):
        worst_r, worst_t = 0.0, 0.0
        print(f"-- convention {label}")
        print(f"{'pair':10} {'rot_deg':>10} {'trans_mm':>10}")
        for i in range(len(Test_use)):
            for j in range(i + 1, len(Test_use)):
                rel_gt = inv_T(Tgt[i]) @ Tgt[j]
                rel_es = inv_T(Test_use[i]) @ Test_use[j]
                re = rot_err_deg(rel_gt[:3, :3], rel_es[:3, :3])
                te = float(np.linalg.norm(rel_gt[:3, 3] - rel_es[:3, 3]))
                print(f"cam{i}->cam{j} {re:10.3f} {te:10.2f}")
                worst_r = max(worst_r, re)
                worst_t = max(worst_t, te)
        return worst_r, worst_t

    r0, t0 = report("T as world-from-camera", Test)
    r1, t1 = report("T inverted (camera-from-world)", Test_inv)
    if r1 + t1 / 100.0 < r0 + t0 / 100.0:
        pose_ok = r1 <= 3.0 and t1 <= 25.0
        print(f"using inverted T (rot {r1:.3f} deg, trans {t1:.2f} mm)")
    else:
        pose_ok = r0 <= 3.0 and t0 <= 25.0
        print(f"using world-from-camera T (rot {r0:.3f} deg, trans {t0:.2f} mm)")

    rms_ok = True
    print("\n=== Per-camera OpenCV RMS ===")
    for i in range(N_CAMS):
        p = OUT_CALI / "data" / f"cam{i}" / "cali_results.txt"
        if not p.exists():
            print("missing", p)
            rms_ok = False
            continue
        rms = None
        for line in p.read_text().splitlines():
            if line.startswith("rms "):
                rms = float(line.split()[1])
        print(f"cam{i} rms={rms}")
        if rms is None or rms > 1.5:
            rms_ok = False

    det = OUT_CALI / "total_results.txt"
    if det.exists():
        print("\n=== total_results.txt (tail) ===")
        lines = det.read_text().strip().splitlines()
        print("\n".join(lines[-20:]))

    print("\n=== VERDICT ===")
    print("Intrinsics (fx, fy, cx, cy):", "PASS" if k_ok else "FAIL")
    print("Relative poses (rot <= 3 deg, trans <= 25 mm):", "PASS" if pose_ok else "FAIL")
    print("Reprojection RMS (<= 1.5 px):", "PASS" if rms_ok else "FAIL")
    passed = k_ok and pose_ok and rms_ok
    print("OVERALL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def main() -> int:
    global OUT_CALI, USE_CUDA
    preview_only = "--preview-only" in sys.argv
    USE_CUDA = "--use-cuda" in sys.argv or "--cuda" in sys.argv
    if USE_CUDA:
        OUT_CALI = SIM_ROOT / "mujoco-charuco-out-cuda"
    print("MuJoCo", mujoco.__version__, "GL", os.environ.get("MUJOCO_GL"))
    print("Rendering ChArUco scene...")
    render_scene()
    print("Wrote dataset to", OUT_IN)
    if preview_only:
        print("Preview rendered to", PREVIEWS)
        return 0
    print("Running CALICO via Docker...")
    run_calico()
    print("Comparing to Ground Truth...")
    return compare()


if __name__ == "__main__":
    sys.exit(main())
