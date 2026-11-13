#!/usr/bin/env bash
# One-command stage-1 (intrinsics) run for one camera session.
#
#   tools/robocal_intrinsics_session.sh <session_dir> <config_template> [calico-robocal binary]
#
# <session_dir>/images/*.png   the captured frames (one camera, still frames, any names)
# <config_template>            e.g. configs/robocal/sessions/wrist_intrinsics.yaml
#                              (frames: frames.yaml, detection.cache: detections.jsonl, one camera)
#
# Writes into <session_dir>:
#   frames.yaml, frames_{blkA,blkB,h1,h2}.yaml           (sorted image list; blkA/B = alternate blocks of
#                                                         BLOCK frames, h1/h2 = first / second half in time)
#   intr*.yaml                                            (configs; parts without hold-out / bootstrap; near /
#                                                         far = views below / above the median board distance)
#   out/        full solve: <camera>.camera_info.yaml (+ .equidistant.yaml), robot_calib.yaml,
#               residuals.csv, candidates/, viz/ (coverage, residual maps, examples)
#   out_{blkA,blkB,h1,h2,near,far}/                       split solves (in parallel); all three splits must pass
#   log_*.txt, summary.txt                                logs and the verdict
# Exit: 0 all PASS, 3 gates or repeatability FAIL, other = error.
#
# Run inside the dev container if calico is not installed on the host, e.g.
#   docker exec robocal-dev /repo/tools/robocal_intrinsics_session.sh /sim/real/<session> \
#       /repo/configs/robocal/sessions/wrist_intrinsics.yaml /build/calico-robocal
set -euo pipefail
SESSION=$(cd "$1" && pwd)
TEMPLATE=$(readlink -f "$2")
BIN=${3:-calico-robocal}
TOOLS=$(cd "$(dirname "$0")" && pwd)
MAX_DIFF_PX=${MAX_DIFF_PX:-0.30}     # split-half gate: rotation-aligned max lens difference (px) = 2x the full fit's error;
                                     # 0.30 from the stage-2 lens budget sim (EXPERIMENTS 13) ...
MAX_RMS_PX=${MAX_RMS_PX:-0.14}       # ... its rms
MAX_RAW_PX=${MAX_RAW_PX:-0.60}       # ... raw max (principal point included)
THETA_DEG=${THETA_DEG:-85}           # ... over rays out to this angle
BLOCK=${BLOCK:-40}                   # split-half block length (frames); for still-frame captures
                                     # with independent poses, BLOCK=1 is fine

cd "$SESSION"
[ -d images ] || { echo "no $SESSION/images directory" >&2; exit 2; }
CAM=$(sed -n '/^cameras:/,$p' "$TEMPLATE" | grep -m1 -oE 'name: *[A-Za-z0-9_]+' | sed 's/name: *//')

python3 - "$CAM" "$BLOCK" <<'EOF'
import glob, os, sys
cam, block = sys.argv[1], max(1, int(sys.argv[2]))
imgs = sorted(glob.glob("images/*.png"))
if len(imgs) < 20:
    sys.exit(f"only {len(imgs)} images in images/ (need >= 20; aim for 100-200 still frames)")
def write(name, sel):
    with open(name, "w") as f:
        f.write("frames:\n")
        for p in sel:
            f.write(f"  - {{camera: {cam}, image: {p}}}\n")
write("frames.yaml", imgs)
a = [p for i, p in enumerate(imgs) if (i // block) % 2 == 0]
b = [p for i, p in enumerate(imgs) if (i // block) % 2 == 1]
write("frames_blkA.yaml", a)
write("frames_blkB.yaml", b)
h = len(imgs) // 2
write("frames_h1.yaml", imgs[:h])
write("frames_h2.yaml", imgs[h:])
print(f"{len(imgs)} frames; splits: alternating blocks of {block} ({len(a)} / {len(b)}), first / second half "
      f"({h} / {len(imgs) - h}), near / far at the median board distance")
EOF

# Config for the full solve: the template with the frames list and the block length (hold-out and
# bootstrap use whole blocks of BLOCK frames).
python3 - "$TEMPLATE" "$BLOCK" > intr.yaml <<'PY'
import re, sys
t, block = sys.argv[1], sys.argv[2]
s = open(t).read()
s = re.sub(r"(?m)^frames: .*$", "frames: frames.yaml", s)
if "block_frames" not in s:
    if re.search(r"(?m)^solver: *\{", s):
        s = re.sub(r"(?m)^solver: *\{", f"solver: {{block_frames: {block}, ", s)
    elif re.search(r"(?m)^solver: *$", s):
        s = re.sub(r"(?m)^solver: *$", f"solver:\n  block_frames: {block}", s)
    else:
        s += f"\nsolver: {{block_frames: {block}}}\n"
# Rectified (virtual-view) tags: found once by the full solve, read by the splits (same images).
if "rectified_cache" not in s:
    if re.search(r"(?m)^detection: *\{", s):
        s = re.sub(r"(?m)^detection: *\{", "detection: {rectified_cache: rectified_tags.jsonl, ", s)
    elif re.search(r"(?m)^detection: *$", s):
        s = re.sub(r"(?m)^detection: *$", "detection:\n  rectified_cache: rectified_tags.jsonl", s)
    else:
        s += "\ndetection: {rectified_cache: rectified_tags.jsonl}\n"
print(s, end="")
PY
rm -f rectified_tags.jsonl

rc_all=0
run() {  # $1 = suffix
  local rc=0
  "$BIN" --config="intr$1.yaml" --out="out$1" > "log_out$1.txt" 2>&1 || rc=$?
  echo "out$1 rc=$rc"
  [ "$rc" = 0 ] || [ "$rc" = 3 ] || { tail -n 20 "log_out$1.txt" >&2; return "$rc"; }
  return 0
}
run "" || exit $?
# The splits' detections are the full set's, frame by frame (same detector, same images): filter the
# full cache instead of detecting again (byte-identical to a fresh detection).
python3 - <<'FILTER'
import json, os
def imgs(p):
    return [l.split("image: ", 1)[1].rstrip("}\n").strip() for l in open(p) if "image:" in l]
if os.path.exists("detections.jsonl"):
    full = imgs("frames.yaml")
    for part in ("_blkA", "_blkB", "_h1", "_h2"):
        out = f"detections{part}.jsonl"
        if os.path.exists(out) or not os.path.exists(f"frames{part}.yaml"):
            continue
        idx = {im: i for i, im in enumerate(imgs(f"frames{part}.yaml"))}
        with open("detections.jsonl") as f, open(out + ".tmp", "w") as o:
            for line in f:
                if not line.strip():
                    continue
                k = json.loads(line)["frame"]
                if full[k] in idx:
                    o.write(line.replace('{"frame": %d,' % k, '{"frame": %d,' % idx[full[k]], 1))
        os.replace(out + ".tmp", out)
FILTER
[ -s "out/$CAM.camera_info.yaml" ] || rc_all=3
# Splits: the model the full solve chose, no hold-out (each part trains on all of its frames), no
# bootstrap, gates off (the comparison is their test); the 6 solves run in parallel.
ci() { ls "out$1/$CAM.camera_info"*.yaml 2>/dev/null | grep -v '\.equidistant\.' | head -1; }
MODEL=$(grep -m1 -oE '^distortion_model: *[a-z_0-9]+' "$(ci "")" | sed 's/.*: *//')
MED=$(grep -m1 -oE 'board distance median [0-9.]+' log_out.txt | awk '{print $4}')
NPROC=$(nproc 2>/dev/null || echo 8)
THREADS=$(( NPROC / 6 > 1 ? NPROC / 6 : 1 ))
for part in _blkA _blkB _h1 _h2 _near _far; do
  python3 - intr.yaml "$part" "${MODEL:-equidistant}" "${MED:-0.25}" "$THREADS" > "intr${part}.yaml" <<'PY'
import re, sys
t, part, model, med, thr = sys.argv[1:]
s = open(t).read()
if part in ("_near", "_far"):   # same frames and detections, views filtered by board distance
    rng = f"[0.0, {med}]" if part == "_near" else f"[{med}, 100.0]"
    s = re.sub(r"(?m)^solver: *\{", f"solver: {{view_distance_m: {rng}, ", s)
else:
    s = re.sub(r"(?m)^frames: .*$", f"frames: frames{part}.yaml", s)
    s = s.replace("cache: detections.jsonl", f"cache: detections{part}.jsonl")
s = re.sub(r"model: *auto", f"model: {model}", s)
s = re.sub(r"holdout_every: *[0-9]+", "holdout_every: 0", s)
s = re.sub(r"bootstrap: *[0-9]+", "bootstrap: 0", s)
s = re.sub(r"threads: *[0-9]+", f"threads: {thr}", s)
s = re.sub(r", *dump_rounds: *[^,}]+", "", s)
s = re.sub(r"(?ms)^intrinsics_gates:.*?(?=^\S|\Z)", "", s)
s += "\nintrinsics_gates: {max_holdout_rms_px: 0, min_theta_deg: 0, min_coverage: 0, max_curve_sd_px: 0, max_mean_radial_px: 0}\n"
print(s, end="")
PY
done
pids=()
for part in _blkA _blkB _h1 _h2 _near _far; do run "$part" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || rc_all=3; done

compare() {  # $1 $2 suffixes, $3 label
  local rc=0
  "$BIN" --compare-intrinsics="$(ci "$1"),$(ci "$2")" \
         --max-diff-px="$MAX_DIFF_PX" --max-rms-px="$MAX_RMS_PX" --max-raw-px="$MAX_RAW_PX" \
         --theta-deg="$THETA_DEG" > "log_compare$3.txt" 2>&1 || rc=$?
  echo "${3#_}: $(grep -m1 'repeatability over rays' "log_compare$3.txt" | sed 's/.*deg: //')"
  return $rc
}
SPLITS=""
for c in "_blkA _blkB _blocks" "_h1 _h2 _halves" "_near _far _distance"; do
  set -- $c
  line=$(compare "$1" "$2" "$3") || rc_all=3
  SPLITS="$SPLITS$line"$'\n'
done
cp log_compare_blocks.txt log_compare.txt 2>/dev/null || true

# Visualisation needs numpy, opencv, matplotlib, pyyaml: plain python3 if it has them, else uv.
VIZ=(python3)
python3 -c 'import yaml, cv2, matplotlib, numpy' 2>/dev/null || \
  VIZ=(uv run -q --with pyyaml --with opencv-python-headless --with matplotlib --with numpy python)
"${VIZ[@]}" "$TOOLS/robocal_visualize.py" --out-dir=out --images-root="$SESSION" --camera="$CAM" > log_viz.txt 2>&1 \
  || echo "visualisation skipped / failed (log_viz.txt). On the host (outputs written in a container are" \
          "root-owned): tools/robocal_visualize.py --out-dir=<session>/out --images-root=<session>" \
          "--camera=$CAM --viz-dir=<writable dir>" >&2

{
  echo "session: $SESSION   camera: $CAM   config: $TEMPLATE   blocks: $BLOCK frames"
  for s in "" _blkA _blkB _h1 _h2 _near _far; do
    echo "--- out$s"
    grep -E "model .*hold-out|model auto|board distance median|  $CAM: train|lens $CAM:|gates:|^      - " "log_out$s.txt" || true
  done
  echo "--- splits (each: rotation-aligned max <= $MAX_DIFF_PX px, rms <= $MAX_RMS_PX px, raw max <= $MAX_RAW_PX px, rays to $THETA_DEG deg; all must pass)"
  printf "%s" "$SPLITS"
  echo "--- verdict: $([ $rc_all = 0 ] && echo PASS || echo FAIL)"
  echo "camera_info for stage 2: $SESSION/out/$CAM.camera_info.yaml"
} | tee summary.txt
exit $rc_all
