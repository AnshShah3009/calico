# Pull Request Description: OpenCV 5, CUDA-Assisted Calibration, Robust Optimization, CALICO Studio & Multi-Camera Verification Suite

## Summary

This pull request modernizes **CALICO** (Camera Calibration with Incremental Ceres Optimization) for modern computer vision and robotics pipelines. It introduces:
1. **OpenCV 5.0 Compatibility & Compatibility Shim** (`src/opencv-compat.hpp`).
2. **CUDA-Assisted Acceleration** (GPU BGR→gray, optional cuAprilTags for `tag36h11`, Ceres dense CUDA when built with it). ChArUco detection remains on CPU.
3. **Ceres Robust Loss Functions** (Huber, Cauchy, trivial) to handle outliers and partial occlusions.
4. **Stage 4/5 Checkpointing & Resume** (`--checkpoint`, `--resume=<dir>`).
5. **Pre-Flight Dataset Validator** (`tools/validate_dataset.py`) for frame synchronicity, resolution consistency, marker length constraints, and ArUco/AprilTag dictionary capacity limits.
6. **CALICO Studio Web Application** (`tools/calico_gui.py`):
   - Zero-dependency web UI (loopback bind + bearer token by default).
   - Calibration Wizard with configuration presets (ChArUco Standard, AprilTag Grid GPU, High-Robustness Cauchy, Quick Draft).
   - Live streaming solver terminal with progress bar, elapsed timer, copy/clear controls, and toast notifications.
   - Integrated 3D visualizer with interactive camera inspection, baseline measurements, and "Look Through Camera" projection view.
7. **Multi-Format Calibration Exporter** (`tools/export_calibration.py`):
   - ROS / ROS2 `CameraInfo` YAML.
   - Nerfstudio `transforms.json` format for NeRF and 3D Gaussian Splatting (3DGS).
   - COLMAP Structure-from-Motion format (`cameras.txt`, `images.txt`).
8. **Multi-Camera Video Ingestion & Frame Extraction** (`tools/ingest_video.py`).
9. **Ground-Truth Simulation Benchmarks** in MuJoCo for both AprilTag grids and ChArUco targets.
10. **Automated Testing & CI/CD**:
   - CTest C++ unit test suite in Docker (`calico_tests`).
   - Python unit test suite (`tools/test_tools.py`) with dual AprilTag and ChArUco coverage.
   - GitHub Actions CI verifying Python tools, and both `calico-cpu` and `calico-cuda` Docker builds.

---

## Visual Showcase: Rig Setup, Detection & Reconstruction

Full gallery (AprilTag + ChArUco MuJoCo runs, detection overlays, recovered 3D networks):

![PR verification gallery](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/pr_verification_gallery.png)

### 1. Multi-Camera Rig Setup in MuJoCo
Three-camera surround views of dual non-coplanar targets:

| AprilTag grid | ChArUco |
| :---: | :---: |
| ![AprilTag rig](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/rig_setup_april.png) | ![ChArUco rig](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/rig_setup_charuco.png) |

### 2. Pattern Boards & Detection
| Patterns | ChArUco multi-cam detection | AprilTag labeled boards |
| :---: | :---: | :---: |
| ![Boards](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/pattern_boards_charuco_april.png) | ![Detection](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/charuco_detection_multicam.png) | ![April labels](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/april_boards_labeled.png) |

### 3. Recovered 3D Camera Networks
Camera frustums, baselines (mm), and board poses from `calibration.json`:

| AprilTag MuJoCo | ChArUco MuJoCo |
| :---: | :---: |
| ![April network](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/calibrated_camera_network_3d.png) | ![ChArUco network](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/calibrated_camera_network_charuco_3d.png) |

### 4. AprilTag Grid Variants (families × sizes)
OpenCV render + detect across `tag16h5`, `tag25h9`, `tag36h10`, `tag36h11` and multiple grid densities (9/9 full detections). Sample YAML specs live in `configs/april-variants/`.

![AprilTag grid variants](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/april_grids_gallery.png)

### 5. ChArUco Board Variants (dictionaries × sizes)
OpenCV `CharucoBoard` generate + `CharucoDetector` across `4X4` / `5X5` / `6X6` / `7X7` and AprilTag-backed ChArUco (`arcCode: 20`), sparse → dense boards (9/9 full chessboard-corner detections). Sample YAML in `configs/charuco-variants/`.

![ChArUco board variants](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/charuco_boards_gallery.png)

---

## Ground-Truth Verification Thresholds

Enforced by `sim/verify_mujoco_april.py` / `sim/verify_mujoco_charuco.py` (no `--ingest-intrinsics` with harness GT K — that would tautologically pass intrinsics):

| Metric | Tolerance |
| :--- | :---: |
| Intrinsics $f_x, f_y, c_x, c_y$ | $\le 2.0$ px |
| Relative pose rotation | $\le 3.0^\circ$ |
| Relative pose translation | $\le 25$ mm |
| OpenCV reprojection RMS | $\le 1.5$ px |

Pose checks assert the `inv(T)` convention (calibration.json `T_world_camera` is camera-from-world) and do not min() over both conventions.

**Local checks for this update:** `python3 tools/test_tools.py` (AprilTag multi-family grids + ChArUco dict×size boards). Showcase images from MuJoCo runs + pattern-variant galleries.

---

## Detailed Changes

### 1. OpenCV 5.0 Modernization & CUDA Support
- Added `src/opencv-compat.hpp` to bridge API differences between OpenCV 3/4 and OpenCV 5 (`cv::aruco::CharucoBoard`, `cv::aruco::CharucoDetector`, detector parameters).
- Added `src/apriltag_cuda.cu` and `src/cuda-detect.cpp` (`CALICO_USE_CUDA` / `CALICO_HAVE_CUAPRILTAGS`) for GPU BGR→gray and optional cuAprilTags (`tag36h11`).
- ChArUco detection stays on CPU (no `cv::cuda::aruco`); `--use-cuda` still GPU-converts frames and can use Ceres CUDA when built with it.
- Ingested $K$/distortion preserved correctly on OpenCV 5 (`--ingest-intrinsics`).

### 2. Optimization Robustness & Checkpointing
- Ceres losses: `--loss=[trivial|huber|cauchy]` and `--loss-scale=[FLOAT]` (validated; CLI overrides config).
- Stage 4/5 checkpointing (`checkpoint.txt`, full double precision, atomic latest replace) and `--resume=<dir>`.

### 3. CALICO Studio Web GUI (`tools/calico_gui.py`)
- Loopback bind (`127.0.0.1`) + bearer token by default; no phantom `--stride`.
- Wizard presets, live solver terminal, 3D visualizer, export hub.

### 4. Pre-Flight Dataset Validator (`tools/validate_dataset.py`)
- Frame sync, resolution consistency, ChArUco/AprilTag capacity checks, `pattern_square_mm*.txt` presence.

### 5. Multi-Format Calibration Exporter (`tools/export_calibration.py`)
- ROS CameraInfo, Nerfstudio `transforms.json`, COLMAP, OpenCV FileStorage; safe camera names; prefers JSON `width`/`height`.

### 6. Multi-Camera Ingestion Tool (`tools/ingest_video.py`)
- Video → CALICO layout; `--boards` reflected in `network_specification_file.yaml`.

### 7. Testing & CI/CD
- Python suite (`tools/test_tools.py`) and C++ `calico_tests` in Docker.
- PR CI builds **both** `calico-cpu` and `calico-cuda`.

### 8. Review hardening (post-Greptile / deep review)
- Restored CUDA macro path; cuAprilTags `dev_ptr` + RGB upload; CUDA error checks.
- Resume residuals, ingested tangential distortion, per-board AprilTag **and** ChArUco mm scales.
- CLI-vs-config precedence for `num-pattern` / `num-threads` / `loss` / `loss-scale` / `max-images`.
- GUI auth + XSS escaping; visualizer `</script>` hardening.
- Sim harness: no self-fed ingest, fixed pose convention, camera count/name asserts, stale pattern regeneration.
- Docker jammy `libwebp7`; README/AGENTS phantom-flag cleanup.

---

## How to Test

### 1. Run Python Unit Test Suite
```bash
python3 tools/test_tools.py
```

### 2. Launch CALICO Studio
```bash
python3 tools/calico_gui.py --port 8080
# Open http://localhost:8080 in your browser
```

### 3. Run Pre-Flight Validation
```bash
# Validate ChArUco dataset
python3 tools/validate_dataset.py DATA/mult-3-base

# Validate AprilTag grid dataset
python3 tools/validate_dataset.py results/april-in
```

### 4. Run Multi-Format Export
```bash
# Export to OpenCV FileStorage, ROS, Nerfstudio, and COLMAP
python3 tools/export_calibration.py results/run5 --format all -o exports/run5_all
```

### 5. Run C++ Unit Tests (Inside Docker)
```bash
docker compose run --rm --entrypoint calico_tests calico-cpu
# CUDA image (CI also builds this on PRs):
docker compose run --rm --entrypoint calico_tests calico-cuda
```

### 6. MuJoCo Ground-Truth Verification
```bash
MUJOCO_GL=osmesa python3 sim/verify_mujoco_april.py
MUJOCO_GL=osmesa python3 sim/verify_mujoco_charuco.py
# optional GPU path:
MUJOCO_GL=osmesa python3 sim/verify_mujoco_april.py --use-cuda
```
