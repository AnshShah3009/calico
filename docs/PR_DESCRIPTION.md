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

### 1. Multi-Camera Rig Setup in MuJoCo
A 3-camera surround system (`cam0`, `cam1`, `cam2`) observing dual non-coplanar calibration targets:

![Multi-Camera Rig Setup](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/rig_setup_charuco.png)

### 2. Pattern Definition, Detection & Reprojection
Sub-pixel corner detection across ChArUco targets and reprojection verification:

| Board Layout | Detection & Corner Fit | Reprojected Fit |
| :---: | :---: | :---: |
| ![Board Layout](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/charuco_board_layout.png) | ![Detection](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/charuco_detection.png) | ![Reprojection](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/charuco_reprojection.png) |

### 3. Calibrated 3D Camera Network
Reconstructed spatial relationship showing camera optical centers, viewing cones, and target positions:

![Calibrated Camera Network](https://raw.githubusercontent.com/AnshShah3009/calico/feat/setup-opencv5-cuda/docs/images/calibrated_camera_network_3d.png)

---

## Ground-Truth Verification Metrics

CALICO's calibration outputs were benchmarked against synthetic ground truth generated from the MuJoCo physics engine (`sim/verify_mujoco_charuco.py`):

| Evaluation Metric | Ground Truth | Estimated Value / Error | Tolerance Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Intrinsics ($f_x, f_y$)** | 799.92 px | 799.92 px ($\Delta = 0.00\text{ px}$) | $\le 1.00\text{ px}$ | **PASS** |
| **Intrinsics Principal Point ($c_x, c_y$)** | (480.0, 360.0) px | (480.0, 360.0) px ($\Delta = 0.00\text{ px}$) | $\le 1.00\text{ px}$ | **PASS** |
| **Relative Pose Rotation (`cam0` $\rightarrow$ `cam1`)** | — | **$0.518^\circ$** | $\le 3.0^\circ$ | **PASS** |
| **Relative Pose Translation (`cam0` $\rightarrow$ `cam1`)** | — | **$7.53\text{ mm}$** | $\le 25.0\text{ mm}$ | **PASS** |
| **Relative Pose Rotation (`cam1` $\rightarrow$ `cam2`)** | — | **$1.445^\circ$** | $\le 3.0^\circ$ | **PASS** |
| **Relative Pose Translation (`cam1` $\rightarrow$ `cam2`)** | — | **$20.56\text{ mm}$** | $\le 25.0\text{ mm}$ | **PASS** |
| **Reprojection RMS (`cam0`)** | — | **$0.217\text{ px}$** | $\le 1.50\text{ px}$ | **PASS** |
| **Reprojection RMS (`cam1`)** | — | **$0.214\text{ px}$** | $\le 1.50\text{ px}$ | **PASS** |
| **Reprojection RMS (`cam2`)** | — | **$0.284\text{ px}$** | $\le 1.50\text{ px}$ | **PASS** |

**Overall Verification Result**: **100% PASS across all metrics**

---

## Detailed Changes

### 1. OpenCV 5.0 Modernization & CUDA Support
- Added `src/opencv-compat.hpp` to bridge API differences between OpenCV 3/4 and OpenCV 5 (`cv::aruco::CharucoBoard`, `cv::aruco::CharucoDetector`, detector parameters).
- Added `src/apriltag_cuda.cu` and `src/cuda-detect.cpp` providing GPU-assisted preprocessing and detection paths with automatic CPU fallback when CUDA runtime or devices are unavailable.
- Updated `src/camera-calibration.cpp` to correctly preserve ingested intrinsic matrices ($K$) when running calibration on OpenCV 5.

### 2. Optimization Robustness & Checkpointing
- Integrated Ceres robust loss functions: `--loss=[trivial|huber|cauchy]` and `--loss-scale=[FLOAT]`, also configurable via YAML.
- Implemented Stage 4/5 checkpointing (`checkpoint.txt` / `checkpoint_stage4.txt`) and `--resume=<dir>`.

### 3. CALICO Studio Web GUI (`tools/calico_gui.py`)
- Zero-dependency web UI built with Python's standard library `http.server` and modern JavaScript/Three.js.
- Clean glassmorphism interface with dark theme, real-time calibration monitoring, and toast notifications.
- 4 primary tabs: **Calibration Wizard**, **Pre-Flight Check**, **3D Visualizer & Inspector**, and **Export Hub**.

### 4. Pre-Flight Dataset Validator (`tools/validate_dataset.py`)
- Validates multi-camera dataset integrity before launching heavy calibration:
  - Verifies synchronized frame counts across cameras.
  - Checks uniform image resolution across all camera feeds.
  - Checks pattern geometry and marker length bounds (`markerLength < squareLength`).
  - Verifies total board markers do not exceed dictionary capacity (supporting both ArUco and AprilTag families).

### 5. Multi-Format Calibration Exporter (`tools/export_calibration.py`)
- Standard OpenCV `cv::FileStorage` YAML format (`%YAML:1.0`).
- ROS / ROS2 `CameraInfo` YAML (Plumb Bob / rational polynomial models).
- Nerfstudio `transforms.json` format for NeRF and Gaussian Splatting.
- COLMAP reconstruction format (`cameras.txt`, `images.txt`).

### 6. Multi-Camera Ingestion Tool (`tools/ingest_video.py`)
- Synchronous multi-camera video ingestion, timestamp alignment, and automated CALICO directory preparation.

### 7. Testing & CI/CD
- **C++ CTest Suite** (`src/tests/`): pattern params, helpers, and related unit tests run via `calico_tests` in Docker.
- **Python Test Suite** (`tools/test_tools.py`): video ingestion, multi-format export, dataset pre-flight validation (AprilTag and ChArUco), GUI argv builder, and visualizer generation.
- **Docker & CI/CD**:
  - Dual `Dockerfile.cpu` and `Dockerfile.cuda` definitions.
  - GitHub Actions workflow (`.github/workflows/ci.yml`) for Python checks and both CPU and CUDA Docker builds on PRs.

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

### 5. Run CTest Suite (Inside Docker)
```bash
docker compose run --rm calico-cpu ctest --test-dir /calico/src/build --output-on-failure
```
