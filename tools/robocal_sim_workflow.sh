#!/usr/bin/env bash
# End-to-end simulation of the calibration workflow, run exactly like a real calibration:
#   0. focus: calico-robocal-sim focus renders each lens's focus-ring sweep (board at the
#      harmonic mean of the camera's operating range); calico-focus picks the sharpest setting;
#      the lens is locked there and the near / far check is run on new frames
#   1. calico-robocal-sim generate: ground-truth cell + rendered sessions + nominal configs
#   2. stage 1 per camera: intrinsics from a hand-held session; stop if the gates fail
#   3. stage 2: robot calibration with the stage-1 intrinsics held constant
#   4. calico-robocal-sim evaluate: everything against the truth on unseen poses
#
#   tools/robocal_sim_workflow.sh [SIM_DIR] [BIN_DIR] [generate options...]
#   e.g. tools/robocal_sim_workflow.sh sim_hard build --scale=2 --joint-noise-deg=0.02
set -euo pipefail
SIM=${1:-robocal_sim}
BIN=${2:-build}
shift $(( $# > 2 ? 2 : $# ))

LENS=kb
for a in "$@"; do case "$a" in --lens=*) LENS=${a#--lens=};; esac; done
echo "=== step 0: focus ==="
"$BIN/calico-robocal-sim" focus --out="$SIM" --lens="$LENS"
for cam in left right center; do
  "$BIN/calico-focus" --config="$SIM/focus_$cam/focus.yaml" --images="$SIM/focus_$cam/sweep" \
    --out="$SIM/focus_$cam/out" > "$SIM/focus_$cam/sweep_log.txt"
  grep '^\[focus\] target' "$SIM/focus_$cam/sweep_log.txt" | sed "s/^/$cam /"
done

"$BIN/calico-robocal-sim" generate --out="$SIM" "$@"

for cam in left right center; do
  "$BIN/calico-focus" --config="$SIM/focus_$cam/focus.yaml" --images="$SIM/focus_$cam/check" \
    --out="$SIM/focus_$cam/check_out" > "$SIM/focus_$cam/check_log.txt"
  grep '^\[focus\] near' "$SIM/focus_$cam/check_log.txt" | sed "s/^/$cam check: /"
done

for cam in left right center; do
  echo "=== stage 1: $cam intrinsics ==="
  set +e
  "$BIN/calico-robocal" --config="$SIM/stage1_$cam/stage1.yaml" --out="$SIM/stage1_$cam/out" \
    > "$SIM/stage1_$cam/log.txt" 2>&1
  rc=$?
  set -e
  sed -n '/== robocal result/,$p' "$SIM/stage1_$cam/log.txt"
  if [ $rc -eq 3 ]; then echo "stage 1 $cam: intrinsics gates FAILED -- recapture before stage 2"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "stage 1 $cam failed (rc=$rc), see $SIM/stage1_$cam/log.txt"; exit $rc; fi
done

for cam in left right center; do
  if [ -f "$SIM/stage1b_$cam/stage1.yaml" ]; then
    echo "=== repeatability: $cam, second intrinsics session ==="
    "$BIN/calico-robocal" --config="$SIM/stage1b_$cam/stage1.yaml" --out="$SIM/stage1b_$cam/out" \
      > "$SIM/stage1b_$cam/log.txt" 2>&1 || true
    "$BIN/calico-robocal" --compare-intrinsics="$SIM/stage1_$cam/out/$cam.camera_info.yaml,$SIM/stage1b_$cam/out/$cam.camera_info.yaml" \
      --max-diff-px=0.15 || echo "  -> sessions disagree: recapture"
  fi
done

echo "=== stage 2: robot calibration (intrinsics constant) ==="
set +e
"$BIN/calico-robocal" --config="$SIM/stage2/stage2.yaml" --out="$SIM/stage2/out" > "$SIM/stage2/log.txt" 2>&1
rc=$?
set -e
if [ $rc -eq 4 ]; then echo "stage 2: robot solve FAILED its sanity gate, see $SIM/stage2/log.txt"; exit 4; fi
if [ $rc -ne 0 ]; then echo "stage 2 failed (rc=$rc), see $SIM/stage2/log.txt"; exit $rc; fi
sed -n '/== robocal result/,$p' "$SIM/stage2/log.txt"

echo "=== evaluation against the truth ==="
"$BIN/calico-robocal-sim" evaluate --sim="$SIM"
