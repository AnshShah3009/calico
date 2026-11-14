# calico-robocal: robot + camera joint calibration

Offline calibration of a robot cell with one or more serial arms (any URDF chain), wrist
cameras on the arms and static cameras, from raw frames and joint readings. calico does the
detection, pose initialization, intrinsics and the joint solve itself; nothing else is needed
besides the images, the joints, the URDFs and the board layouts.

## Operator checklist

1. **Boards.** Kalibr AprilGrids (tag36h11, 2-bit border, spacer squares), matte, on rigid backing.
   Wrist cameras (13-30 cm): 2 x A4 7x10, 20 mm tags. Center camera (30-150 cm): 2 x A2 7x10, 35 mm
   tags for intrinsics at 0.3-0.8 m (100 mm tags for the 1.5 m focus check). Cell: A2 5x7 55 mm table
   (flat), 2 x A3 3x4 55 mm side boards, A2 4x6 60 mm back wall. Disjoint id blocks.
2. **Focus** each camera with `calico-focus` at the harmonic mean of its range; lock the lens; check
   near and far read the same blur.
3. **Intrinsics** per camera, off the robot (`intrinsics_session.yaml`): move the camera slowly, pause
   to capture; tilt the boards into every image corner (that is where 85-100 deg lives). Accept
   only `gates: PASS`; the `camera_info` is then fixed for good (refocus = redo).
4. **Robot run**: 30+ poses per arm, wrist roll / pitch on both sides of balance, some steep upper-arm
   poses. Capture image + joints only after >= 0.4 s still; log `still_s` and each joint's signed
   final travel (`travel_deg`) per frame.
5. **Solve** (`piper_bimanual.yaml`: intrinsics from step 3, `drive:` on, `joint_noise_deg`), then read
   the report: weak directions, PRIOR-LIMITED joints, drive table, hold-out errors.

## Workflow

0. **Focus, before anything is calibrated** (`calico-focus`). Fixed-focus M12 lenses are focused
   by screwing the lens in its thread, which also moves it away from the sensor and scales the
   focal length (~0.5 % across the useful range): focus is set and locked first, intrinsics after.
   The target is the harmonic mean of the camera's working range, `s = 2 near far / (near + far)`:
   defocus blur grows with `|1/d - 1/s|`, so this `s` gives the smallest worst-case blur, equal at
   both ends (wrist 0.13-0.30 m -> 0.181 m; center 0.30-1.50 m -> 0.50 m).

   ```bash
   calico-focus --config=intrinsics_session.yaml --device=0          # live
   calico-focus --config=intrinsics_session.yaml --images=<dir> --out=<dir>
   ```

   Board centred, camera and board on stands, at the printed target distance (the tool measures
   it with a rough PnP from `f_init`). Turn the lens through focus: the tool fits an edge-spread
   function across the tag borders and prints the blur sigma (px) per frame and the sharpest so
   far; go back to the minimum and lock the lens. Then show the board at `near` and at `far`:
   the summary says `balanced`, or which way the lens is off. Config: `focus: {near_m, far_m,
   max_off_axis_deg}` in the session file.
1. **Intrinsics, off the robot** (`intrinsics_session.yaml`): each wide-angle camera alone, with the
   camera moved over static boards (`mode: handheld`) or boards moved in front of it
   (`mode: free_views`). Output: `<out>/<camera>.camera_info.yaml` plus a lens report
   (residuals by off-axis angle, coverage, largest measured angle, bootstrap spread of the lens
   curve, hold-out error with poses refitted alone) and pass / fail `intrinsics_gates`.
2. **Robot / extrinsics** (`piper_bimanual.yaml`): the cameras load those `camera_info` files and
   their intrinsics stay constant; joint offsets, camera mounts, bases, static cameras, boards and
   the drive model are solved together. Intrinsics are never freed here: the robot run only covers
   the image where the arms put the boards, and a freed 200 deg lens fits that region and
   extrapolates the rest (simulation: wrist lenses 2x worse, the static camera's lens off by
   > 100 px beyond 45 deg, robot accuracy unchanged). calico refuses `optimize_intrinsics` in a
   robot solve unless `solver.allow_intrinsics_in_robot_solve: true`.

## What it estimates, in one problem

| Unknown | Per | Notes |
|---|---|---|
| `delta_q` | arm joint | `q_model = q_encoder + delta_q`; per joint free or held at a known value |
| `T_tip_cam` | wrist camera | tip link (e.g. `link6`) to the camera optical frame |
| `T_world_base` | arm | one arm base is fixed (the world gauge); the others are free, optional prior |
| `T_world_cam` | static camera | e.g. the center / head camera |
| `T_world_board` | board | boards are keyed by the samples' `session`; shared ids are one board |
| intrinsics | camera | optional prepass per camera, then refined in the joint solve (`optimize_intrinsics`) |
| board shape | board | optional (`fit_shape`): bow along x / y and twist (mm), `fit_scale`: x / y print scale |
| `T_link_board` | arm-mounted board | e.g. a gripper marker; free, or fixed (known mount: makes the last joint solvable) |
| frame poses | hand-held camera | one free pose per frame (intrinsics sessions without a robot) |
| geometry terms | arm joint | optional (`arms[].geometry`): 2 axis tilts + 2 shifts perpendicular to each joint axis |

Residuals are pixel reprojections of every board corner (Huber loss), plus priors:
`delta_q` around `delta_q_init_rad`, optional base priors, and an optional level prior that
holds a flat board's normal parallel to world z (`boards[].level`).

Lens models (`cameras[].model`):
- `equidistant`: Kannala-Brandt k1..k4, evaluated with `atan2`, so it is valid past 90 deg off axis
  (Kalibr / OpenCV write it as distortion on top of a pinhole and fail for points behind the
  image plane; calico does not).
- `kb_rt`: Kannala-Brandt + decentering p1, p2 on the distorted coordinates (lens elements off
  centre or tilted -- common in M12 lenses).
- `double_sphere` (Usenko et al. 2018): xi, alpha; closed-form inverse; valid past 90 deg.
- `auto`: fits each of `models` (default `[equidistant, kb_rt]`; `equidistant` first, the others
  start from it) and keeps the lowest BIC on the training residuals among the models whose hold-out
  error is within 5 % of the best; `equidistant` (what ROS loads) on a near tie. Hold-out error alone
  cannot choose: each held-out frame refits its pose, which absorbs smooth lens errors such as
  decentering (in simulation `kb_rt` was 4-5x closer to the true lens while its hold-out RMS was
  only 2 % lower). Every candidate is written to `<out>/candidates/<model>/`. When a richer model
  wins, `<cam>.camera_info.equidistant.yaml` is written next to it: the closest equidistant model,
  with its error over the measured rays in the header, for tools that only know ROS `equidistant`.
- `kb4_pupil`: `equidistant` plus an entrance pupil that slides along the optical axis with the
  field angle, z0(theta) = e1 (theta / 90 deg)^2 (metres, towards the scene); d = [k1..k4, e1].
  A non-central camera: very wide M12 lenses have e1 of 1-1.5 mm, and a central model then needs a
  different lens for near and far boards. Real AR0234 200 deg wrist cameras (full video, 1200 px):
  e1 = 1.16 mm (left), 1.24 mm (right). The stage 1 sessions use it (`configs/robocal/sessions/`).
  ROS has no such model, so `<cam>.camera_info.equidistant.yaml` is written next to it: the
  central KB4 closest to it at `export_distance_m` (wrist 0.25 m, centre 0.50 m), with its error
  over the measured rays in the header. Give the robot solve the full `kb4_pupil` file; tools that
  only know `equidistant` (OpenCV fisheye, the calibration_toolkit overlay) read k1..k4 and leave
  out the pupil shift (< 1 px beyond ~0.3 m, a few px at the image edge for very close parts).
  `--compare-intrinsics=a,b --distance-m=d` compares two such fits at a working distance.
- `plumb_bob`, `rational_polynomial` (pinhole, z > 0 only).

Real left wrist camera (loose sheets, 413 frames): hold-out `equidistant` 0.317 px, `kb_rt` 0.313
(BIC -826 k vs -821 k, so `auto` picks it), `double_sphere` 0.317 (its lens equals `equidistant` to
0.002 px over 95 deg). But on 40-frame block halves `kb_rt` repeats 3x worse (rotation-aligned
1.16 vs 0.35 px): its decentering trades with the principal point when the data are weak. Check a
`kb_rt` pick with the split-half compare (the session script does) and fall back to
`models: [equidistant]` if it does not repeat.

In simulation a typical 200 deg Double Sphere lens is reproduced by `equidistant` to 0.004 px;
what `equidistant` cannot represent is decentering: with ~0.15 px of it at the image edge,
`equidistant` lands at 0.06-0.11 px (p95) from the true lens, `kb_rt` at 0.015-0.021 px.

Repeatability: compare two independent fits of one camera -- two sessions, or two halves of one
session split into alternate blocks of ~40 frames (not alternate frames: consecutive frames of a
continuous recording are near-duplicates, their residuals correlate 0.9+):

    calico-robocal --compare-intrinsics=a.camera_info.yaml,b.camera_info.yaml \
        [--max-diff-px=0.30] [--max-rms-px=0.14] [--max-raw-px=0.60] [--theta-deg=85] [--align=rotation|none]

It prints, per 5 deg ring, the pixel difference of the two lenses over a ray fan: raw; after the best
rotation of b's rays onto a (a camera rotation is absorbed by the stage-2 mount / pose, so it is not a
lens difference); and with b's principal point set to a's (curve shape only, diagnostic). The gate
(exit 3) is the rotation-aligned max and rms plus the raw max. Keep `--theta-deg` where both halves have
data (>= ~400 corners per 5 deg ring each). For independent halves the full solve's error at the worst
ray is about half the difference.

Why 0.30 px: the split difference is about 2x the full solve's error, so 0.30 px means a delivered lens
good to ~0.15 px. In the stage-2 simulation, lens errors of the measured real shape at 0.22 px kept
every robot target (gripper, mounts, bases, centre camera); at 0.45 px wrist mounts and the centre
camera failed (gripper p95 hides it: the solve pushes lens error into the mounts). Fable's block
statistics give the same limit (rms 0.14 px). 0.15 px stays a quality bar for a very good capture.
`solver.view_distance_m: [min, max]` keeps only board views at that distance (board centre), e.g. to
compare a working-range fit with the full fit: on the real wrist video the fitted lens depends on
board distance (fit on views >= 23 cm differs from the full fit by fx +2 px, 1.3 px at 80-85 deg).

`tools/robocal_intrinsics_session.sh <session> configs/robocal/sessions/<wrist|center>_intrinsics.yaml`
does a whole stage-1 session in one command: frames list from `<session>/images/*.png`, full solve,
block split-half solves (model fixed to the full solve's, no hold-out), the comparison, the
visualisations (`tools/robocal_visualize.py`) and `summary.txt`; exit 0 only when everything passes.

## Joint zeros the images cannot see

With no marker on the robot, J1's zero is indistinguishable from a yaw of that arm's base (or,
for the gauge arm, of the whole world), and J6's zero from a roll of the wrist camera about the
J6 axis. Give them a tight prior from the mechanical zero (`delta_q_sigma_deg: [0.5, 20, 20, 20,
20, 0.5]`, all free; or hold them at the mechanical zero with `delta_q_free: [false, true, true,
true, true, false]`, which is also the default when `delta_q_free` is omitted): they stay at the prior, their true error goes into the base yaw / camera
roll, and everything relative (camera to boards, camera to camera, arm to arm, gripper
position) is unaffected. What remains is the gripper's roll about J6 and the arm's yaw against
the physical base housing, both bounded by the mechanical knowledge. A marker on the gripper
with a known mount (`boards[].mount`, `fixed: true`) makes J6 observable.

## Drive train: backlash and compliance under gravity

Joint encoders on the motor side see neither the reducer's backlash nor its torsional
compliance, and gravity loads both. Unmodelled, they do not just add noise: in simulation (PiPER
masses, typical small-planetary play 12-21 arcmin, stiffness 300-1500 N m/rad) they biased mounts,
base and static camera by 3-11 mm and the gripper by 8 mm. calico models them linearly:

    theta_j = q_j + dq_j + c_j (tau_j - tau_ref_j) + h_j (s_j - s_ref_j)

`tau_j` is the gravity torque from the URDF `<inertial>` masses (off-chain links such as the gripper
are lumped into their chain link), `s_j` the gear flank the link rests on, `c_j` compliance, `h_j`
half the backlash; `dq_j` is the zero at the dataset's working load. Entries without spread in
their regressor are held (J1 / J6 compliance; J2 / J3 backlash unless the poses flip their load).
Prefer `approach: against_gravity` in the capture: in simulation it took the right-base error from
1.04–1.08 mm (random approach, 30–50 poses) to 0.91–0.93 mm (30–40 poses); with random approach the
unobservable J2 / J3 backlash leaves a ~1.0–1.08 mm floor.

```yaml
drive: {model: true, approach: against_gravity, shared: true}
# approach: against_gravity - every capture move ends against gravity (fixed + where unloaded),
#                             so s = sign(tau): no extra logging
#           logged          - frames carry the signed final travel per joint:
#                             travel_deg: {left: [2, -2, 2, 2, -2, 2]}
#           none            - only strongly loaded joints are modelled
# friction_nm, tau_min_nm, backlash_prior_deg / backlash_sigma_deg, compliance_sigma, gravity
# backlash_joints / compliance_joints: [bool x n] masks, false = term off (held at 0)
# flank_select: true - each training frame takes the gear flank (+-1) that fits its images best
#               (two-state errors-in-variables; diagnostic: on the PiPER data it over-fits)
```

Per arm, `drive_fixed: {compliance_rad_per_nm: [n], half_backlash_rad: [n]}` (with `shared: false`)
holds the drive terms at given values, e.g. transferred from another session.

Real PiPER bimanual cell (teach-mode captures under gravity compensation, 76 captures, 5-split
cross-validation, held-out RMS both wrists): rigid offsets 5.56 px, `approach: against_gravity` with
the inertial URDF 4.77 px (J2 half backlash ~0.3 deg). `approach: logged` with travel reconstructed
from consecutive snapshots did not help (5.28 px): the hand's last motion into a pose is not the net
change between captures. Details and figures: `docs/robocal/README.md`.

Use it together with errors-in-variables (`joint_noise_deg`, scalar or per joint, ~0.03 deg with the
model). Frames whose flank is ambiguous get their encoder prior widened by the backlash. The
result (`robot_calib.yaml`, `arms.<arm>.drive`) has compliance, half backlash, the reference load
and the unloaded zero `delta_q_unloaded_rad`; a controller that applies the same terms gets the
compensated accuracy, one that ignores them the effective zero at the reference load.

## Arm geometry (`arms[].geometry`)

```yaml
arms:
  - name: left
    geometry: {sigma_mm: 1.0, sigma_deg: 0.3}   # or geometry: true (these defaults)
```

Minimal geometric calibration (after Lu et al. 2025 / Tobuschat et al. 2026): per joint, two tilts of
the axis and two shifts perpendicular to it, applied in the joint frame before the joint motion
(`UrdfChain::Forward`). The shift along / tilt about the axis itself are left out (they duplicate the
next joint or the zero offset). Gaussian priors at manufacturing tolerance keep the directions the
images cannot see (J1 vs the base, J6 vs the camera mount) at zero. Written to `robot_calib.yaml:
arms.<arm>.geometry_terms` (per joint: tilt 1 / 2 in deg, shift 1 / 2 in mm). Real PiPER cell: held-out
4.62 -> 3.91 px (with backlash, filter and fixed boards), terms <= 0.37 deg / 1.5 mm; the result is
insensitive to the prior between 0.3 and 1 mm. The exported URDF and `T_EC` do not include these terms:
an FK that should reproduce the solve must apply them.

## Solver

Defaults are the right choice; the options exist for diagnosis and were benchmarked in simulation
(stage 2, 80 frames, ~400 unknowns):

```yaml
solver:
  method: lm            # lm | dogleg | dogleg_subspace (trust region, 2-5 s, identical results)
                        # lbfgs | bfgs | nlcg (line search: 3-30x slower, nlcg stops short)
  linear_solver: auto   # dense_qr | dense_schur | sparse_schur | iterative_schur | sparse_normal_cholesky
  global: none          # multistart | basin_hopping (annealed restarts around the joint solve)
  mount_init: auto      # camera mount start: closed-form hand-eye from the data vs CAD, best kept
  max_rms_px: 1.5       # robot solve sanity gate (training RMS per camera)
  distance_sigma_mm: 0  # > 0: per-corner weight 1/sqrt(px^2 + (mm f / depth)^2), Tenhumberg 2022
  distance_sigma_px: 0.3  #     its pixel part (close corners weigh less; neutral on the PiPER data)
  view_weights: false   # true (2 rounds) | n: after pass 3, a view whose residual spread s_v
                        # exceeds the pixel noise s_0 is weighted by s_0 / s_v (blurred views, a
                        # sheet bending in one view): a few bad sequences cannot pull the lens.
                        # Real loose sheets: ~100 views carry ~60 % of the squared error, but
                        # block-split repeatability only improves 7 % (0.346 -> 0.323 px)
```

Every stage line in the log ends with `[<stage s>, t=<total s>]`. Real wrist session (KB4, 1306
views, 136 k points, 6 threads): detection + guided rounds 80 s, pass 3 162 s, pass 4 112 s; `model:
auto` runs the whole pipeline once per candidate and once more for the chosen model.

What the benchmark showed: from CAD with joint offsets up to 33 deg every method reaches the same
minimum; multistart / basin hopping (5-8x the time) found no other. The one real failure was
discrete -- a camera mount wrong by 180 deg in CAD -- which no continuous search escapes; the
data-driven mount start (`mount_init: auto`) fixes it, and the sanity gate catches anything
similar: a robot solve whose training RMS stays above `max_rms_px` is written as
`robot_calib.FAILED.yaml` / `calibrated_cell.FAILED.urdf` and exits with code 4 (code 3: intrinsics
gates failed). The default 1.5 px suits arms whose joints repeat to ~0.01 deg; hobby-grade arms
without a drive model (backlash, compliance) stay at several px: the real PiPER teach-mode runs
reach 3.4-4.7 px at the wrist cameras, so raise the gate for them instead of trusting a FAILED file.

Reported errors (RMS, p95, `residuals.csv`, lens reports, outlier and sanity tests) are the plain
image errors. `solver.robust: corner` (default) down-weights corners beyond `huber_px` (scaled by
the view's noise factor) during the solve only; those weights never enter what is reported.

## Capture timing

Joint states reach the recorder late. A frame whose image and joints were grabbed while the
arm was still moving (or ringing after a stop) pairs the image with stale joints. Record every
frame only after the reported joints have been stationary for a while, and log that time as
`still_s` in the frame entry; `capture: {min_still_s: 0.3}` drops frames below it.

## Inputs

- **Frames** (`frames:`): one entry per image, YAML list or JSON lines:
  `{camera: left, image: left_000.png, joints: {left: [q1..qn]}}` (or `q: {joint_name: value}`).
  Wrist cameras need their arm's encoder joints; static cameras need none.
- **Boards** (`boards:`): static boards, any number and any mix of types, layouts and sizes.
  - `type: aprilgrid` (default): Kalibr AprilGrid. Tag id = `first_id + row * cols + col`, row 0 at
    the bottom of the print, `tag_size` = black square (2-bit border tags), `tag_spacing` = gap /
    tag size. Every tag corner is an X-junction (tag + spacer square).
  - `type: charuco`: chessboard with markers in the white squares; `rows` / `cols` count squares,
    `square_size`, `marker_size`, `dictionary` (cv::aruco name, default `DICT_5X5_1000`),
    `first_id`, `legacy: true` for the OpenCV < 4.6 layout. The top-left square is black.
  - `type: checkerboard`: plain chessboard, `rows` / `cols` squares (odd x even, so the
    orientation is unambiguous), `square_size`. It is only detected when wholly visible and only
    one per run can be told apart.
  Ids must not overlap between boards of one family / dictionary (checked). `level: true` marks a
  board lying flat (any height): its normal is held parallel to world z with `level_sigma_deg`.
- **Detection**:
  - AprilGrid: the Kaess detector (`detection.apriltag: kaess`, default; tag36h11, tag25h9,
    tag16h5, tag36h9, tag25h7) or AprilTag 3 with a Kalibr-border family
    (`apriltag: apriltag3`; tag36h11, tag25h9, tag16h5). Then per-corner `cornerSubPix`, half window
    `clip(edge / 10, 3, 9)` px.
  - ChArUco: cv::aruco markers. Each chessboard corner is predicted from the homographies of its
    adjacent markers (local, so valid under strong lens distortion), predictions that disagree are
    dropped, then the corner is refined with `cornerSubPix`, its window kept short of the markers.
  - Checkerboard: `findChessboardCornersSB` (exhaustive, accuracy), oriented by the colour of the
    corner square.
  Corner refinement (`refine: xfit`, default): `cornerSubPix` as the start, then a fit of a
  blurred X-junction model (two erf edges at any angle, on gamma-linearised intensities, over an
  elliptical window sized in printed modules) with polarity, contrast, residual and saturation
  gates. It is unbiased under blur, gamma and exposure; a corner that fails is dropped, never kept
  raw.
  `detection.gamma` (default 2.2) is the encoding undone before the fit. Get it right for the stream:
  on the AR0234 1920 x 1200 mode (dark frames, paper white ~140 DN) 2.2 made the contrast / residual
  gates reject ~80 % of corners (238 views, max angle 75 deg, gates FAIL); `gamma: 1.0` kept 5x more
  (322 views, 95 deg, PASS). On bright frames it hardly matters. `ROBOCAL_XFIT_DEBUG=1` prints the
  reject reasons. `refine: subpix` is the plain `cornerSubPix`.
  Guided pass (`guided: true`, default, cameras with `intrinsics_prepass`): after the first lens
  fit every board corner is projected through the lens and its frame's pose and refined where the
  tags did not decode -- far off-axis, blurred or small tags still give clean saddles.
  The first round also re-detects in 17 rectified virtual pinhole views (`rectified: true`);
  rounds (fit -> add -> refit) repeat up to `guided_rounds` (3) or until < 1 % is new. Real wrist
  session: detector 139.5 k corners, round 1 +26.9 k (+86 beyond 90 deg, where the detector had 48),
  round 2 +1.7 k (whole boards whose tags never decoded), round 3 +47; about 80 s in total.
  `dump_rounds: <dir>` writes `<camera>_round<k>.csv` (+ `_lens.txt`) after detection and each
  round; `tools/robocal_rounds.py <dir>` renders density maps per round, corners per off-axis
  angle / board by round of origin, and annotated frames with zoom insets.
  A board view needs `min_tags_per_board` tags / markers and `min_points_per_board` points.
  Reliable tags for the **robot solve** (`max_theta_deg`, `min_tag_px`; 0 = off): only AprilGrid
  tags whose centre is within `max_theta_deg` of the optical axis (through each camera's own, fixed
  lens) and whose mean edge is at least `min_tag_px` are kept; a board view with fewer than
  `min_tags_per_board` tags left is removed. Applied after detection / the cache (the cache keeps
  every detection, so thresholds can be changed without re-detecting) and after each guided round;
  other board types get the angle test per point. With intrinsics fixed, off-axis and small-tag
  errors would go straight into the joint offsets and mounts. Real whole-system run (fullbody_01,
  76 captures, `max_theta_deg: 40`, `min_tag_px: 35`): kept 8.4 k / 27.7 k corners (left),
  7.3 k / 23.1 k (right), 1.2 k / 14.1 k (centre); training RMS 3.87 / 4.68 / 0.16 px
  (p95 7.1 / 8.3 / 0.27): a rigid arm model without backlash; set `solver.max_rms_px` above that.
- **Cameras** have a `mode`: `wrist` (on an arm's `camera:`), `static` (one pose for the run),
  `free_views` (one free pose per board view: boards moved independently) or
  `handheld` (one free pose per frame; with no arms the world frame is `reference_board`, else the
  level board, else the first board seen). Hand-held hold-out frames are scored after refitting
  their pose alone, so the hold-out error measures the lens and boards.
- **Arm-mounted boards**: `mount: {arm, link, fixed, T_link_board_init}`. Frames that see them
  need the mounting arm's joints (`joints: {left: [...], right: [...]}` may list several arms).
  A camera never scores a board on its own arm (the chain cancels).
- **Board shape**: `fit_shape: true` adds bow along x / y and twist (prior `shape_sigma_mm`, 3 mm).
  A bowed print biases the focal length by ~0.5 % per mm of sag if it is not modelled.
  `fit_points: true` adds a height per corner (prior `point_sigma_mm`, e.g. 0.3) for loose paper.
- **Print scale**: office printers scale each axis differently (real A4 sheets: y 0.4-1.6 % short,
  x within +-1 %, y/x 0.5 % below nominal; it varies sheet to sheet).
  - `print_scale_pct: [x, y]`: the measured scale of this sheet in % (outer tag-edge extent with a
    steel rule, (measured / nominal - 1) * 100). Used as the value (held) or, with `fit_scale`, as
    the prior centre. Default [0, 0].
  - `fit_scale: true`: x / y scale free (prior `scale_sigma_pct`, 1 %); turns on `fit_shape` too
    (the scale is part of the shape block). On the real loose-sheet
    session hold-out 0.317 -> 0.150 px and fx/fy 1.0047 -> 1.0001.
  - What reaches the lens is only the **anisotropy** (x vs y). In stage 1 (`free_views` /
    `handheld`) a sheet's overall size is absorbed by the view distances: holding only the fitted
    anisotropy gave the same lens to 0.023 px. The data fix the anisotropy to ~0.02 %, better than
    a ruler (0.1 mm on 180 mm = 0.06 %). So for intrinsics use `fit_scale: true` on printed paper;
    do not freeze a ruler value there.
  - In the **robot solve** board size is metric (1 % of 242 mm = 2.4 mm against a 1 mm target):
    enter the ruler-measured `print_scale_pct` (to +-0.25 mm on the long side), or better measure
    `tag_size` on the final rigid board.
  - Freeing or fixing the scale does **not** make a loose-sheet session repeatable (block split
    0.35 -> 0.52-0.56 px rotation-aligned): the split-half gap is principal-point sampling noise
    from too few independent wide-angle sequences, not scale. Rigid boards + more wide-angle passes.
- **Print warp** (`fit_warp: true`, `warp_sigma_mm` 0.3): 8 quadratic in-plane terms per board.
  Not recommended: on real data the per-board warps disagree between halves and repeatability does
  not improve (0.344 vs 0.346 px).
- **Detection cache**: `detection: {cache: detections.jsonl}` is written on the first run and read
  back afterwards (the frames must match).
- **URDF** per arm (revolute / continuous / prismatic / fixed joints).
- **Intrinsics** per camera: a ROS `camera_info` YAML, or `model` + `width` + `height` +
  `f_init` with `intrinsics_prepass: true` (a per-camera fit with free view poses before the
  joint solve) and `optimize_intrinsics: true` (refined again in the joint solve).
- Optional import of pre-detected corners: per-arm `samples:` in the calibration_toolkit
  `*_handeye_samples.jsonl` format, and `observations:` for static cameras.

Board poses for initialization come from calico's own PnP, which works at any field angle:
bearings are rotated onto the axis of a virtual pinhole aimed at the board, solved with IPPE
(both solutions), and refined through the real lens model.

See `piper_bimanual.yaml` for a commented config.

## Run

```bash
calico-robocal --config=configs/robocal/piper_bimanual.yaml --out=<out_dir>
```

Writes `<out_dir>/robot_calib.yaml`:

- `arms.<name>`: `delta_q_rad` / `delta_q_deg`, `base_to_base`, `T_tip_cam_optical`,
  `camera_link6` (body frame, when `optical_in_body_rpy` is set), `T_EC` (when `ee_link` is
  set) and per-camera `quality` (train / hold-out);
- `arms.<name>.joint_corrections_deg` (with `joint_noise_deg`): per frame the image, encoder joints
  and the fitted correction -- the joint error the cameras see, for backlash / gear-play diagnosis;
- `arms.<name>.geometry_terms` (with `geometry`);
- `boards`, `T_base_board` (the level board), `sensors.<static camera>.T_base_cam`;
- `diagnostics`: free parameter count, condition number, **weak directions** (combinations the
  images do not determine, with the parameters involved), parameters with no data, and the
  1-sigma of every free parameter (deg / mm / px, scaled by the residual variance).

All RMS values are per point, `sqrt(mean |e|^2)` in px (OpenCV's convention). The per-axis
value is that divided by sqrt(2). `residuals.csv` has one row per corner (pixel error, hold-out flag,
off-axis angle, `depth_m` = distance from the camera with the solved poses, for mm-scale reports).

Other per-arm / per-board options used for validation:
- `camera_mount_fixed: true` with `T_tip_cam_init`: hold the wrist camera mount (cross-session tests);
- `joint_noise_loss: cauchy`: robust per-joint errors-in-variables prior (over-fits on teach-mode data);
- boards: `shape_fixed_mm: [bow_x, bow_y, twist]` (+ `print_scale_pct`) holds a board's shape, e.g.
  from an image-only fit (camera pose free per frame), so arm error does not leak into the boards.

## Pipeline

1. Detect, then PnP every view. Optional per-camera intrinsics prepass.
2. Initialize: boards from the fixed-base arm's views (`FK(q + dq0) * T_tip_cam * T_cam_board`),
   free bases from the boards they share, static cameras from their views.
3. Pass 1: all poses, `delta_q` held. Pass 2: `delta_q` free.
4. Outliers: points above `max(outlier_floor_px, median + k * 1.4826 * MAD)` dropped, re-solve.
5. Optional pass 4: intrinsics of cameras with `optimize_intrinsics`.
6. Diagnostics: data-only Jacobian (priors excluded) for weak directions; covariance with priors.

Every `holdout_every`-th view per camera is held out of the fit and scored afterwards.

## Noise model

- **Pixel noise** is estimated after the first full pass (robust per-axis spread of the training
  residuals, or `solver.pixel_sigma_px`) and weights the images against the priors, which are in
  absolute units (encoder noise, level board, board shape, dq). The covariance is then honest.
- **Encoder noise**: `joint_noise_deg` per arm (the encoder / backlash spec, 1 sigma). Each frame
  gets its own joint errors with that prior (errors-in-variables). Without it, noisy encoders
  bias the offsets and the reported sigma is several times too small. The summary prints the
  fitted per-frame errors against the prior: they should match.
- **Hold-out** is by whole frames (every `holdout_every`-th frame per camera), never single board
  views; a board whose views would all be held out in a camera is kept for the fit.

## Lens report and gates (stage 1)

For every camera with `optimize_intrinsics` in `handheld` / `free_views` mode:

- residuals of the training corners by off-axis angle (0-15 ... 90-95 deg), radial and tangential
  about the principal point; a non-zero radial mean in a bin is a lens-model or board bias there;
- the largest off-axis angle measured, and coverage of the image (16 x 9 cells inside 95 deg);
- bootstrap (`solver.bootstrap` refits on resampled frames, boards held): sd of fx / fy / cx / cy
  and of the lens curve (image radius of a ray at each angle up to the measured one);
- hold-out error: every `holdout_every`-th frame is left out of the fit, its pose refitted alone.

`intrinsics_gates` turns these into PASS / FAIL (in the summary, `robot_calib.yaml` and the
`camera_info` header). Stage 2 should only load a `camera_info` that passed.

## End-to-end simulation

`calico-robocal-sim` builds a ground-truth cell and the nominal system calibration starts from,
renders every step to disk, and scores the outputs against the truth on unseen poses:

```bash
tools/robocal_sim_workflow.sh <sim_dir> <build_dir> [generate options]
#   --scale=S             J2-J5 offsets, mount and base errors x S
#   --joint-noise-deg=D   encoder noise on the recorded joints
#   --route-poses=N       stage-2 poses per arm (default 30)
#   --stage1-frames=N --stage1-edges=0   a poor intrinsics session (the gates must stop it)
```

The simulated hardware follows the real one: AR0234 sensors (1920 x 1200, 3 um, global shutter,
YUV output) behind ~200 deg M12 fisheyes (EFL ~2 mm, F2.2 assumed): the image circle is about the
sensor diagonal, ~97 deg in the corners. Images come from a ray tracer with a lamp, ambient
light, shadows of the arms, glossy (laminated) prints, vignetting, field- and depth-dependent
optical blur from the lens's actual focus, motion blur from the camera's velocity, a small
decentering distortion that Kannala-Brandt cannot represent, auto exposure, shot and read noise,
gamma and 8-bit output.

Prints are rendered like toner prints (`--print=toner`, default): toner black ~0.08 of the paper,
density banding along the feed, dot gain (~40 um), ragged edges, paper grain, toner specks on the
white, dropouts in the black, soft smudges and an occasional streak. `--lens=double_sphere` renders
with a Double Sphere truth lens instead of Kannala-Brandt (a model family calico may not pick).

Steps, as on the real cell:

0. focus: each lens arrives badly focused (left at infinity, right at ~1 m, center at ~17 cm).
   `calico-robocal-sim focus` renders the ring sweep with the board at the target distance,
   `calico-focus` picks the sharpest frame, `generate` locks the ring there (focal length moves
   with it) and renders the near / far check, which `calico-focus` scores;
1. three hand-held intrinsics sessions (A4 6x8 boards for the wrist cameras, A2 for the center
   camera; paused frames with hand tremor, one in five taken while moving);
2. the robot run: the arm arrives at each pose and rings, joint states lag 80 ms, the recorder
   captures after 0.4 s stationary (`still_s`); two captures per arm are early-trigger glitches
   that `capture.min_still_s` drops. No marker on the robot; J1 / J6 zeros are off by < 1 deg
   and held by their priors, J2-J5 off by 4.5-11 deg;
3. `calico-robocal-sim evaluate`: focus (worst defocus blur over the range, before / after),
   intrinsics over a fan of rays to 90 deg against the real lens (decentering included),
   offsets, mounts, base, center camera, bow -- with the J1 / J6 gauge removed -- and wrist
   camera pose, gripper position / roll and board reprojection on 300 unseen poses per arm, next
   to the nominal system. `calico-robocal-sim video --sim=<dir>` renders every step to PNG frames.

## Tests

`robocal_render_tests` renders 190 deg Kannala-Brandt images and runs the whole pipeline
(detection, PnP, intrinsics prepass from a 2 % focal error, joint solve), as the two stages:

- hand-held intrinsics session: one camera over four static boards (one bowed 2 mm), coverage to
  ~90 deg; checks intrinsics, board relative poses, the bow, the gates, the bootstrap spread and
  the hold-out error, and exports the `camera_info`;
- free views: static camera, boards moved independently; same checks;

- arms, with the exported `camera_info` held constant: two wrist cameras and a static camera, four static AprilGrids (24-40 mm tags, disjoint id
  blocks, one flat, one bowed 2 mm) and a marker on the left gripper with a known mount; checks
  joint offsets including the left J6, mounts, base, camera, boards, the bow, and that the
  detection cache reproduces the solve.

`robocal_tests` builds a synthetic two-arm cell (fisheye wrist cameras, a static camera, one
flat and two tilted boards, known joint offsets), writes it in the toolkit formats, solves it
through the config path, and checks the recovered offsets, mounts, base and camera, the RMS,
and that freeing J1 is reported as a weak direction.
