# Release Notes — v2.0.0 (OpenCV 5, CUDA, Ceres Robust Losses, CALICO Studio)

Released on main after merging PR #1 ("OpenCV 5, CUDA Acceleration, Ceres Robust Losses,
CALICO Studio UI & Multi-Camera Verification Suite").

## Highlights

- **OpenCV 5.0 compatibility** with a compatibility shim and fallback support for 4.7+/4.3.
- **CUDA-accelerated** BGR→gray, optional cuAprilTags (`tag36h11`), and Ceres dense CUDA
  when Ceres is built with CUDA support. ChArUco detection remains on CPU.
- **Ceres robust loss functions**: trivial (default), Huber, Cauchy (configurable scale).
- **Stage 4/5 checkpointing & resume** (`--checkpoint`, `--resume=<dir>`).
- **Pre-flight dataset validator** (`tools/validate_dataset.py`).
- **CALICO Studio** zero-dependency web UI (`tools/calico_gui.py`), with live solver
  output, 3D visualization, presets, and multi-format export.
- **Multi-format exporters** (`tools/export_calibration.py`): ROS/ROS2 CameraInfo, Nerfstudio
  `transforms.json` (NeRF/3DGS), COLMAP `cameras.txt`/`images.txt`.
- **Video ingestion** (`tools/ingest_video.py`) and MuJoCo ground-truth verification harnesses
  for AprilTag and ChArUco grids.
- **ArUco GridBoard marker grids** (`--aruco`) with MIP 36h12 calibration.
- **Multi-family AprilTag grids** (`--april`) and multi-dictionary ChArUco boards (`--charuco`).
- **CI/CD**: GitHub Actions for Python tools, CPU and CUDA Docker builds, and Docker Hub
  publishing; CTest C++ unit-test suite in-container.

## Images

- `anshshah3009/calico:cpu-latest` (CPU)
- `anshshah3009/calico:cuda-latest` (CUDA)

## Docker quick start

```bash
docker pull anshshah3009/calico:cpu-latest
docker run --rm anshshah3009/calico:cpu-latest --help
docker run --rm --entrypoint calico_tests anshshah3009/calico:cpu-latest
```

## Links

- Full PR description: `docs/PR_DESCRIPTION.md`
- Calibration walkthrough: `docs/CALIBRATION_WALKTHROUGH.md`
- Source PR: https://github.com/AnshShah3009/calico/pull/1
