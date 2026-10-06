# AGENTS.md — calico (Ansh QoL + OpenCV 5 / CUDA)

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

- `CharucoBoard::setLegacyPattern(true)` so printed boards from OpenCV &lt; 4.6 still decode
- `ArucoDetector::refineDetectedMarkers` per board (recovers markers from rejected quads)
- `CharucoDetector::detectBoard` for chessboard-corner interpolation
- ingested K/dist, when `--ingest-intrinsics` is set, for pose-reprojection interpolation (more accurate than homography)

There is no `cv::cuda::aruco` API.

## CUDA / AprilTag

`--use-cuda` (no-op if the binary was built without `ENABLE_CUDA`):

- GPU BGR→gray (OpenCV `cudaimgproc` if present, otherwise `apriltag_cuda.cu`)
- AprilTag: NVIDIA **cuAprilTags** for `tag36h11` when headers/libs are passed into CMake; otherwise GPU gray + CPU Kaess/AprilRobotics detector
- Ceres dense CUDA when Ceres was built with `USE_CUDA`

ChArUco detection stays on CPU (OpenCV has no `cv::cuda::aruco`). `--use-cuda` still runs a GPU BGR→gray probe and, if Ceres was built with CUDA, the dense solver.

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

## No tests

Still no unit tests. Verify with a Zenodo dataset: http://doi.org/10.5281/zenodo.3520866
