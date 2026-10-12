# AGENTS.md — calico (Ansh QoL + OpenCV 5 / CUDA)

> [!TIP]
> **OpenCV 5.0, CUDA Acceleration & CI/CD**:
> The `main` branch provides Ansh Shah's 21 Quality-of-Life (QoL) flags, structured JSON export, and checkpointing for OpenCV <= 4.3.
> For the modernized suite with OpenCV 5.0 compatibility, CUDA GPU-accelerated detection, Ceres robust loss functions, CTest unit testing, and Docker Hub CI/CD, see branch [`feat/setup-opencv5-cuda`](https://github.com/AnshShah3009/calico/tree/feat/setup-opencv5-cuda) or [PR #1](https://github.com/AnshShah3009/calico/pull/1).

Multi-camera calibration (C++17, CMake). Paper: https://arxiv.org/abs/1903.06811

This tree is **AnshShah3009/calico** (Amy Tabb’s CALICO plus QoL flags), extended with OpenCV 5-compatible ArUco, CUDA-assisted detection, working `--resume`, richer `--json`, and ingestable intrinsics.

## Build

CPU:

```bash
mkdir -p build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
make -j$(nproc)
```

CUDA (Ceres + GPU grayscale / optional NVIDIA cuAprilTags):

```bash
cmake ../src -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)
```

Optional NVIDIA cuAprilTags (tag36h11 only):

```bash
cmake ../src -DENABLE_CUDA=ON \
  -DCUAPRILTAGS_INCLUDE_DIR=/path/to/cuapriltags \
  -DCUAPRILTAGS_LIBRARY=/path/to/libcuapriltags.so
```

Produces `calico-dec2023` and `compute-dec2023`.

Docker:

```bash
docker compose build calico-cpu
# GPU machine:
docker compose build calico-cuda
```

## OpenCV

CMake tries **OpenCV 5**, then **4.7+**, then **4.3**. ArUco lives in `objdetect` on 4.7+/5 (`ArucoDetector` / `CharucoBoard` constructor). Legacy `CharucoBoard::create` is still compiled for 4.3.

Pinned Docker images use **OpenCV 5.0.0**. On 4.7+/5, CALICO uses:

- `CharucoBoard::setLegacyPattern(true)` so printed boards from OpenCV < 4.6 still decode
- `ArucoDetector::refineDetectedMarkers` per board (recovers markers from rejected quads)
- `CharucoDetector::detectBoard` for chessboard-corner interpolation
- ingested K/dist, when `--ingest-intrinsics` is set, for pose-reprojection interpolation (more accurate than homography)

There is no `cv::cuda::aruco` API.

## CUDA / AprilTag

`--use-cuda` (no-op if the binary was built without `ENABLE_CUDA`):

- GPU BGR→gray (`apriltag_cuda.cu`; OpenCV `cudaimgproc` only if you built OpenCV 5 with `opencv_contrib` `cudev`)
- AprilTag: NVIDIA **cuAprilTags** for `tag36h11` when headers/libs are passed into CMake; otherwise GPU gray + CPU Kaess/AprilRobotics (or OpenCV `DICT_APRILTAG_*`)
- Ceres dense CUDA when Ceres was built with `USE_CUDA`

The Docker CUDA image does **not** vendor cuAprilTags or OpenCV contrib; `--use-cuda --april` still GPU-converts frames, then detects on CPU.

ChArUco detection stays on CPU (OpenCV has no `cv::cuda::aruco`). `--use-cuda` still runs a GPU BGR→gray probe and, if Ceres was built with CUDA, the dense solver.

## AprilTag grids

`--april` (exclusive with `--charuco`) uses a grid of AprilTags (`squaresX` × `squaresY` tags, `tagSpace` gap, one or more `numberBoards`).

Supported families: `tag36h11` (default), `tag25h9`, `tag16h5`, `tag36h10` (OpenCV render + detect), plus Kaess `tag25h7` / `tag36h9` (detect-only; no pattern PNGs). Aliases such as `tagCodes36h11` are accepted.

Detection order: GPU **cuAprilTags** (`tag36h11` only, if linked) → Kaess CPU → OpenCV `DICT_APRILTAG_*`. `--use-cuda` always GPU-converts BGR→gray first.

Example specs: `configs/april-grid.yaml` (default 6×6 `tag36h11`) and `configs/april-variants/` (`tag16h5` / `tag25h9` / `tag36h10` / dense `tag36h11`). Generate with `--april --create-patterns`, then calibrate with `--april --calibrate`.

## Running

`--charuco` or `--april` is **mandatory** (exclusive or). `--calibrate` or `--create-patterns` also mandatory.

```bash
./calico-dec2023 --charuco --calibrate --input=<dir> --output=<dir> \
  --config=configs/calico.cfg --json --checkpoint
```

Resume a killed Stage 4 solve (poses only; detection is not replayed from the checkpoint file):

```bash
./calico-dec2023 --charuco --calibrate --input=<dir> --output=<dir> \
  --resume=<same-output-dir> --checkpoint
```

Use factory / previous `cali_results.txt`:

```
--ingest-intrinsics=/path/to/intrinsics
# expects <dir>/<camera_name>/cali_results.txt
```

### Quality-of-life flags

See `--help`. Config file format (`configs/calico.cfg`):

```
quiet: 1
summary: 1
use-cuda: 1
ingest-intrinsics: /data/intrinsics
```

CLI overrides config. Hyphenated and underscored keys both work.

`--json` writes `calibration.json` with intrinsics **and** 4×4 camera/board poses.

`--resume` loads `checkpoint.txt` (also written as `checkpoint_stage4.txt`).

## Input format

```
<input>/data/
  camera0/
  camera1/
<input>/network_specification_file.yaml
<input>/pattern_square_mm<N>.txt
```

## Verification & 3D Visualization

### Synthetic MuJoCo Ground-Truth Verification
An end-to-end multi-camera testbench is located in `sim/verify_mujoco_april.py`. It renders a multi-board scene across 3 cameras in MuJoCo, executes CALICO via Docker (`calico-cpu` or `calico-cuda`), and checks recovered intrinsics and camera poses against analytical ground truth:

```bash
# CPU mode:
MUJOCO_GL=osmesa python3 sim/verify_mujoco_april.py

# CUDA mode:
MUJOCO_GL=osmesa python3 sim/verify_mujoco_april.py --use-cuda
```

Verification thresholds (enforced by `sim/verify_mujoco_*.py`): $f_x, f_y, c_x, c_y \le 2.0\text{ px}$, relative rotation $\le 3.0^\circ$, relative translation $\le 25\text{ mm}$, reprojection $\text{RMS} \le 1.5\text{ px}$.


### Dataset Pre-Flight Validation
Verify dataset directory layout, image integrity, frame synchronicity, pattern geometry, and marker dictionary limits before calibration:

```bash
python3 tools/validate_dataset.py <dataset_dir>
# JSON output for automated pipelines:
python3 tools/validate_dataset.py <dataset_dir> --json
```

### CALICO Studio Web GUI
Launch the zero-dependency browser interface to configure calibrations, run pre-flight dataset checks, monitor real-time optimization progress, inspect 3D camera frustums, and export to ROS/NeRF/COLMAP:

```bash
./tools/calico_gui.py --port 8080
```
Then visit `http://localhost:8080`.

### Interactive 3D WebGL / HTML Visualizer
Generate an interactive Three.js 3D report from any CALICO output directory containing `calibration.json`:

```bash
python3 tools/visualize_calibration.py <output_dir> -o <output_dir>/report_3d.html
```

Renders 3D camera frustums, optical centers, baseline distances, boards, and an interactive properties/reprojection panel.

### Real-world datasets
Verify real capture rigs with Zenodo datasets: http://doi.org/10.5281/zenodo.3520866
