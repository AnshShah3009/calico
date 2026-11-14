#!/usr/bin/env bash
# Board type study on the simulated ~200 deg fisheye: one hand-held session rendered with each
# board type (AprilGrid with the Kaess and the AprilTag 3 detector, ChArUco, checkerboard),
# detection scored against the true corners, then a full intrinsics calibration per type.
#
#   tools/robocal_board_study.sh [OUT_DIR] [BIN_DIR] [--frames=N]
set -euo pipefail
OUT=${1:-board_study}
BIN=${2:-build}
shift $(( $# > 2 ? 2 : $# ))
"$BIN/calico-robocal-sim" boards --out="$OUT" "$@"
for d in "$OUT"/*/; do
  [ -f "$d/stage1.yaml" ] || continue
  name=$(basename "$d")
  set +e
  "$BIN/calico-robocal" --config="$d/stage1.yaml" --out="$d/out" > "$d/log.txt" 2>&1
  rc=$?
  set -e
  echo "== $name: calibration rc=$rc"
  sed -n '/== robocal result/,/gates/p' "$d/log.txt" | head -20
done
"$BIN/calico-robocal-sim" boards-report --sim="$OUT"
