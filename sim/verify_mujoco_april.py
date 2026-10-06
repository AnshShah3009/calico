#!/usr/bin/env python3
"""Render a multi-camera AprilTag scene in MuJoCo, run CALICO, compare to GT."""
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
BOARD_SRC = ROOT / "results" / "april-pat" / "patterns"
SIM_ROOT = Path.home() / "calico-mujoco-sim"
OUT_IN = SIM_ROOT / "mujoco-april-in"
OUT_CALI = SIM_ROOT / "mujoco-april-out-sep"
PREVIEWS = SIM_ROOT / "preview"
USE_CUDA = False

# Match configs/april-grid.yaml + pattern_square_mm
SQUARE_PX = 80
SQUARE_MM = 40.0
TAG_SPACE_PX = 16
MARGINS_PX = 20
SQUARES = 6
N_BOARDS = 2
N_CAMS = 3
N_FRAMES = 10
WIDTH, HEIGHT = 960, 720
FOVY_DEG = 48.46  # fy ≈ 800 px at 720p


def euler_zyx_to_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """MuJoCo wxyz from ZYX Euler (radians)."""
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
    # OpenCV: X=X_mj, Y=-Y_mj, Z=-Z_mj
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


def board_half_m() -> float:
    w_px = SQUARES * (SQUARE_PX + TAG_SPACE_PX) - TAG_SPACE_PX + 2 * MARGINS_PX
    return 0.5 * w_px * (SQUARE_MM / SQUARE_PX) / 1000.0


def camera_specs() -> list:
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


def board_pose(frame: int, board: int) -> tuple:
    # Boards are 0.3 m wide; keep a ~0.2 m gap so they never read as one sheet.
    t = frame * 0.35
    side = -1.0 if board == 0 else 1.0
    pos = np.array(
        [
            side * 0.38 + 0.02 * math.sin(t + board),
            0.04 * board + 0.015 * math.cos(t * 0.7),
            0.20 + 0.02 * math.sin(t * 0.4 + board),
        ]
    )
    yaw = side * 0.32 + 0.06 * math.sin(t * 0.5)
    pitch = 0.04 * math.cos(t * 0.4 + board)
    roll = 0.03 * math.sin(t * 0.3 + board)
    quat = euler_zyx_to_quat(yaw, pitch, roll)
    return pos, quat


def build_xml(cams, half: float) -> str:
    cam_xml = []
    for c in cams:
        e = c["eye"]
        cam_xml.append(
            f'    <camera name="{c["name"]}" pos="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}" '
            f'xyaxes="{c["xyaxes"]}" fovy="{FOVY_DEG}"/>'
        )
    hx = half
    boards = []
    for b in range(N_BOARDS):
        boards.append(
            f"""    <body name="board{b}" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="{half:.6f} 0.001 {half:.6f}" rgba="1 1 1 1"
            contype="0" conaffinity="0"/>
    </body>"""
        )
    return f"""<mujoco model="calico_april">
  <compiler autolimits="true"/>
  <option gravity="0 0 0" timestep="0.01">
    <flag contact="disable" gravity="disable"/>
  </option>
  <visual>
    <headlight ambient="0.9 0.9 0.9" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="120" elevation="-20" offwidth="{WIDTH}" offheight="{HEIGHT}"/>
  </visual>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.4 -1" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
{os.linesep.join(boards)}
{os.linesep.join(cam_xml)}
  </worldbody>
</mujoco>
"""


def fy_from_fovy() -> float:
    return (0.5 * HEIGHT) / math.tan(math.radians(FOVY_DEG) * 0.5)


def write_dataset(frames_bgr, cams) -> None:
    if OUT_IN.exists():
        shutil.rmtree(OUT_IN)
    (OUT_IN / "data").mkdir(parents=True)
    for ci in range(N_CAMS):
        ext = OUT_IN / "data" / f"cam{ci}" / "external"
        ext.mkdir(parents=True)
        for fi, images in enumerate(frames_bgr):
            cv2.imwrite(str(ext / f"{fi:04d}.png"), images[ci])
    shutil.copy(ROOT / "configs" / "april-grid.yaml", OUT_IN / "network_specification_file.yaml")
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


def set_board(model, data, board: int, pos, quat) -> None:
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"board{board}")
    adr = model.jnt_qposadr[jnt]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat


def camera_K() -> np.ndarray:
    fy = fy_from_fovy()
    return np.array([[fy, 0.0, WIDTH / 2.0], [0.0, fy, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def T_opencv_from_mj_camera(model, data, name: str) -> np.ndarray:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    pos = np.array(data.cam_xpos[cid], dtype=np.float64)
    Rmj = np.array(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    R = np.column_stack([Rmj[:, 0], -Rmj[:, 1], -Rmj[:, 2]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def paste_board(canvas: np.ndarray, tex: np.ndarray, dst: np.ndarray) -> None:
    th, tw = tex.shape[:2]
    src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(tex, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
    canvas[mask > 0] = warped[mask > 0]


def project_points(T_wc: np.ndarray, K: np.ndarray, pts_w: np.ndarray, require_all_in_front: bool = True):
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    p_cam = (R.T @ (pts_w - t).T).T
    if require_all_in_front and np.any(p_cam[:, 2] <= 1e-4):
        return None, None
    z = np.maximum(p_cam[:, 2], 1e-4)
    uvw = (K @ p_cam.T).T
    uv = uvw[:, :2] / z[:, None]
    in_front = p_cam[:, 2] > 1e-4
    if not np.any(in_front):
        return None, None
    zmean = float(p_cam[in_front, 2].mean())
    return uv, zmean


def draw_room(canvas: np.ndarray, T_wc: np.ndarray, K: np.ndarray) -> None:
    """Gray floor + back wall so the two boards sit in a room, not a void."""
    canvas[:] = (42, 40, 38)
    xs = np.linspace(-1.6, 1.6, 17)
    ys = np.linspace(-0.2, 1.4, 9)
    color_line = (118, 114, 110)
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            quad_w = np.array(
                [
                    [xs[j], ys[i], 0.0],
                    [xs[j + 1], ys[i], 0.0],
                    [xs[j + 1], ys[i + 1], 0.0],
                    [xs[j], ys[i + 1], 0.0],
                ],
                dtype=np.float64,
            )
            uv, _ = project_points(T_wc, K, quad_w, require_all_in_front=True)
            if uv is None:
                continue
            shade = (88, 84, 80) if (i + j) % 2 == 0 else (78, 74, 70)
            cv2.fillConvexPoly(canvas, np.round(uv).astype(np.int32), shade)
            cv2.polylines(canvas, [np.round(uv).astype(np.int32)], True, color_line, 1, cv2.LINE_AA)
    wall = np.array(
        [[-1.6, 1.35, 0.0], [1.6, 1.35, 0.0], [1.6, 1.35, 1.2], [-1.6, 1.35, 1.2]],
        dtype=np.float64,
    )
    uvw, _ = project_points(T_wc, K, wall, require_all_in_front=True)
    if uvw is not None:
        cv2.fillConvexPoly(canvas, np.round(uvw).astype(np.int32), (58, 56, 54))


def render_pinhole(model, data, textures, half: float, cams, K: np.ndarray):
    hy = 0.001
    corners_local = np.array(
        [
            [-half, -hy, half],
            [half, -hy, half],
            [half, -hy, -half],
            [-half, -hy, -half],
        ],
        dtype=np.float64,
    )
    row = []
    for c in cams:
        T_wc = T_opencv_from_mj_camera(model, data, c["name"])
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_room(canvas, T_wc, K)
        layers = []
        for b in range(N_BOARDS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"board{b}")
            R = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            t = np.array(data.xpos[bid], dtype=np.float64)
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


def render_scene() -> tuple:
    half = board_half_m()
    cams = camera_specs()
    xml = build_xml(cams, half)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    textures = []
    for b in range(N_BOARDS):
        p = BOARD_SRC / f"pattern{b}.png"
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
        row = render_pinhole(model, data, textures, half, cams, K)
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
        "--april",
        "--calibrate",
        "--input=/docker_dir/sim-in/",
        "--output=/docker_dir/sim-out/",
        "--src-dir=/opt/calico",
        "--json",
        "--detection-summary",
        "--no-debug-images",
        "--no-visualization",
        "--checkpoint",
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

    ok = k_ok and pose_ok and rms_ok
    print("\n=== VERDICT ===")
    print("intrinsics:", "PASS" if k_ok else "FAIL")
    print("relative poses:", "PASS" if pose_ok else "FAIL")
    print("reproj RMS:", "PASS" if rms_ok else "FAIL")
    print("overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    global OUT_CALI, USE_CUDA
    preview_only = "--preview-only" in sys.argv
    USE_CUDA = "--use-cuda" in sys.argv or "--cuda" in sys.argv
    if USE_CUDA:
        OUT_CALI = SIM_ROOT / "mujoco-april-out-cuda"
    print("MuJoCo", mujoco.__version__, "GL", os.environ.get("MUJOCO_GL"))
    print("Rendering AprilTag scene...")
    render_scene()
    print("Wrote", OUT_IN)
    print("Preview", PREVIEWS / "t0_all_cams.png")
    if preview_only:
        return 0
    run_calico()
    return compare()


if __name__ == "__main__":
    sys.exit(main())
