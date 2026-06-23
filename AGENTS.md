# AGENTS.md — calico

Multi-camera calibration (C++17, CMake). Paper: https://arxiv.org/abs/1903.06811

## Build

```bash
mkdir build && cd build
cmake ../src
make -j$(nproc)
```

Produces `calico-dec2023` and `compute-dec2023`.

## Key constraints

- **OpenCV ≤ 4.3.0 only** (4.4+ incompatible). Pinned in `src/CMakeLists.txt:21`.
- Requires Eigen3, Ceres Solver, AprilTags (two libs: `apriltags` + `apriltag`), OpenMP (`libgomp1`).
- Dockerfile at repo root builds everything on Ubuntu 22.04.

## Running

`--charuco` or `--april` is **mandatory** (exclusive or). `--calibrate` or `--create-patterns` also mandatory.

```bash
./calico-dec2023 --charuco --calibrate --input=<dir> --output=<dir>
```

### Quality-of-life flags (21 total)

| Flag | Type | Description |
|------|------|-------------|
| `--version` | flag | Print version and exit |
| `--dry-run` | flag | Validate input without calibrating |
| `--quiet` | flag | Suppress all calibration stdout |
| `--summary` | flag | Print brief summary after calibration |
| `--progress` | flag | Show stage 4/5 progress percentage |
| `--no-progress` | flag | Disable progress even with `--progress` |
| `--timestamp` | flag | Append timestamp to output dir |
| `--no-overwrite` | flag | Refuse if output dir has results |
| `--force` | flag | Override `--no-overwrite` |
| `--checkpoint` | flag | Periodic solver state checkpointing |
| `--resume=DIR` | arg | Resume from checkpoint |
| `--auto-rename` | flag | Add input basename to output dir |
| `--json` | flag | Write `calibration.json` output |
| `--config=FILE` | arg | Read options from config file |
| `--camera-color=STR` | arg | Per-camera PLY colors (`R,G,B\|R,G,B\|...`) |
| `--pattern-color=STR` | arg | Per-pattern PLY colors |
| `--camera-names=STR` | arg | Comma-separated camera name overrides |
| `--exclude-camera=STR` | arg | Comma-separated cameras to exclude |
| `--focus-camera=STR` | arg | Only calibrate listed cameras (inverse of exclude) |
| `--max-images=INT` | arg | Unified image cap for all cameras |
| `--min-boards=INT` | arg | Auto-exclude cameras detecting fewer than N boards |
| `--num-threads=N` | arg | Thread count (default: OMP max) |
| `--no-visualization` | flag | Skip PLY meshes + equation PNGs (saves ~37 MB) |
| `--no-debug-images` | flag | Skip per-image detection PNGs (779 files) |
| `--detection-summary` | flag | Print board visibility table per camera |
| `--per-camera-mse` | flag | Append per-camera reprojection MSE to output |

Config file format (simple key: value):
```
quiet: 1
summary: 1
num-threads: 4
exclude-camera: camera_images0,camera_images1
no-visualization: 1
detection-summary: 1
```
CLI args override config values. Both `hyphenated-keys` and `underscored_keys` work.

### Output additions
- `total_results.txt` includes per-camera focal/distortion summary, plus per-camera MSE with `--per-camera-mse`
- `calibration.json` written with `--json` (machine-readable format)
- `--no-visualization` skips `cameras-incremental/` and `reconstructed-patterns/` (all PLY/PNG)
- `--no-debug-images` skips per-camera `initial_detect*.png` (779 files, ~37 MB)

See README for full flag reference. Test datasets: http://doi.org/10.5281/zenodo.3520866

## Input format

```
<input>/data/
  camera0/           # images (or internal/ + external/ subdirs)
  camera1/
  ...
<input>/network_specification_file.yaml   # pattern def
<input>/pattern_square_mm<N>.txt          # one per pattern
```

## Output format

```
<output>/
  cameras-incremental/   # calibration results + .ply visualizations
  data/                  # per-camera cali_results.txt
  patterns/              # generated pattern images
  reconstructed-patterns/
  total_results.txt
  camera_cali_incremental.txt   # final network calibration
  arguments-calico.txt
```

## No tests

No test framework, test directory, or CI config exists. The only verification path is to download a Zenodo dataset and run.

## Repo structure

- `src/` — all source, flat layout, no subdirectories
- `Dockerfile` — reproducible build environment
- No monorepo, no packages, no codegen, no linter/formatter config
