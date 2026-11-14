# Literature: calibrating low-cost arms with cameras, including backlash and compliance

Survey behind the modelling choices in calico-robocal (see [README.md](README.md) for the PiPER results).

**Context.** Two AgileX PiPER arms (planetary quasi-direct-drive joints, motor-side encoders), a 200°
fisheye camera on each wrist, one static camera and 13 static AprilGrids, solved as one Ceres bundle
adjustment (joint offsets, camera mounts, arm base, boards, optional backlash / compliance and
per-capture joint errors). With joint offsets only, the wrist cameras fit to about 4 px and predict
unseen captures to about 5 px (about 0.45°); the per-capture joint errors are 0.1-0.15° and the data
were captured by hand in teach mode under gravity compensation.

**Marks on each entry.** [FULL] full text read (equations, protocol and numbers taken from it);
[V] abstract or publisher page checked; [V-meta] citation confirmed only; [M] cited from memory.
Where no full text was available, the reason is given.

## Key takeaways

1. **Backlash is handled by the capture protocol first, then by a model with a known sign.**
   - Tobuschat et al. 2026 (KUKA KR30, 27 µm): every pose was "approached from the same direction in
     joint space to eliminate errors caused by backlash"; the missing backlash model is a limitation
     "whenever a unidirectional approach to a pose cannot be guaranteed".
   - Wampler, Hollerbach, Arai 1995: "backlash errors tend to be bimodal ... depending on the last
     direction of travel"; modelling the last direction may restore the Gaussian assumption.
   - KEBA white paper: a better backlash model switches on the sign of the gear torque, not of the motion.
2. **Hand-guided, gravity-compensated data fail on non-harmonic gearboxes** (MUKCa, Franzese et al. 2025).
   Harmonic-drive Franka and KUKA reached 0.15-0.2 mm; the planetary-gear Kinova only 4.7 -> 1.5 mm on
   training and 6.0 -> 2.7 mm on held-out data, attributed to backlash invisible to the encoders and over-fitting.
3. **Models that absorb non-repeatable error show large train / held-out gaps:** van Wyk (NIST, offsets
   only) about 0.3 mm vs 1.5 mm; Lu 2025 (configuration-dependent POE) 0.04 vs 0.42 mm. Under a controlled
   protocol the gap is small (Tobuschat: 24.5 vs 26.8 µm). Always validate on separate poses or sessions.
4. **Reusable formulas** (section "Reusable equations"): joint-noise propagation into pixels (Pradeep),
   chi-square goodness of fit and parameter covariance (Wampler), distance-dependent weighting
   (Tenhumberg), lumped-mass compliance and piecewise-linear joint maps (Tobuschat), minimal POE geometry
   (Lu), observability indices (Nahvi), Bouc-Wen-type hysteresis (Ruderman), backlash / lost-motion /
   stiffness definitions (Weigand), ISO 9283 repeatability RP = l̄ + 3·S_l.
5. **Quoted numbers come from the evaluation sets stated in each entry** (e.g. Moore 2025: 0.022 mm on the
   training grid, 0.085 mm on an offset grid; Daş & Burdick 2024 recover injected encoder biases only).

**Sources not read in full:**

| Paper | Reason |
|---|---|
| Alam & Ibaraki | no open copy |
| Sammons 2014 | ASME paywall |
| Ma 2018 | Elsevier paywall; dissertation copy not retrievable |
| Ruderman 2009 (IEEE TIE) | IEEE paywall; the 2015 arXiv paper and the 2012 chapter by the same author were read |
| ICARCV 2018 GP | IEEE paywall |
| Borm & Menq, Joubair & Bonev, Nguyen 2015, Nubiola, Slamani | paywalls |
| Sun & Hollerbach 2008 | repository not retrievable |
| Shirafuji 2024 | ASME paywall; lab summary only |
| Koide 2019 | author server unavailable |

PiPER gear ratios and encoder resolution are not published; the joints are planetary quasi-direct drives.

---

## 1. Kinematic calibration of serial robots with cameras / joint hand-eye

**1. Pradeep, Konolige, Berger. "Calibrating a Multi-arm Multi-sensor Robot: A Bundle Adjustment Approach." ISER 2010, Springer STAR 79, pp. 211-225.** [FULL] (Willow Garage PDF via web.archive.org)
- **Model:** ML bundle adjustment over robot parameters x and target points p_j: min Σ r_ijᵀ Σ_ij⁻¹ r_ij.
- **Joint noise:** for an actuated camera, joint noise is *propagated* into pixel covariance: z = proj(f(x, θ), p) with Σ_ij = J Σ_θ Jᵀ + Σ_c, where J = ∂h/∂θ. For an arm holding a target, Σ = J_f Σ_θ J_fᵀ in Cartesian space.
- **Estimated:** joint offsets, camera poses, laser pose, stereo baselines, tilting-laser gear ratio. Link lengths and axis angles were deliberately *not* estimated ("precision machined").
- **Data:** 30 checkerboard poses per hand plus 4 far boards.
- **Solver:** LM, adding parameters in stages (cameras/laser, then baselines, then gear ratio, then joint offsets) to avoid local minima.
- **Results:** head cameras 0.44-0.60 px RMS; forearm (arm-mounted) cameras 1.30 / 1.76 px; arm 6 mm.
- **Limitations noted:** unmodelled belt stretch. The plug-in task needed a *local* calibration offset near the task configuration.
- **Relevance:** Our architecture with simple Σ propagation in place of per-frame variables. Even on the PR2, wrist cameras were 3-4x worse than head cameras with offset-only kinematics.

**2. Stepanova, Pajdla, Hoffmann. "Robot self-calibration using multiple kinematic chains — a simulation study on the iCub humanoid robot." IEEE RA-L 4(2):1900-1907, 2019.** [FULL] DOI 10.1109/LRA.2019.2898320, arXiv:1805.07263
- **Setup:** simulation with DH parameters (up to 86), cameras at 5 px noise, touch at 5 mm, solved with LM.
- **Data:** 5055 configurations, 300 held-out test poses, training subsets of 10/20/50/1000.
- **Observability:** SVD of the identification Jacobian, O1 and O4.
- **Findings:**
  - Multi-chain closure (both eyes plus touch) gives the highest observability and lowest error.
  - A 10-pose single-camera chain was rank-deficient (24 parameters, 20 equations).
  - About 2 mm with 50 poses and about 1 mm with 100.
  - Residuals become Gaussian only with enough poses.
- **Relevance:** sizing guide. Use both wrists plus the static camera on shared boards, and ≥50-100 poses per arm.

**3. Rozlivek, Rustler, Stepanova, Hoffmann. "Multisensorial robot calibration framework and toolbox." IEEE-RAS Humanoids 2020 (held 2021), pp. 459-466.** [V-meta paper; README read] https://github.com/ctu-vras/multirobot-calibration
- **What the toolbox does:** Matlab; chains described by DH or 6-D vectors; a whitelist selects free parameters (`onlyOffsets` mode); relative bounds; LM or trust-region-reflective solver.
- **Evaluation:** built-in train/test split (default `splitPoint` 0.7).
- **Relevance:** reference design for parameter whitelisting and bounds.

**4. Tenhumberg, Winkelbauer, Burschka, Bäuml. "Self-Contained Calibration of an Elastic Humanoid Upper Body Using Only a Head-Mounted RGB Camera." Humanoids 2022.** [FULL] arXiv:2311.08338
- **Model:** elastic DH, ρ = ρ0 + κτ (Caenen & Angue), solved at static torque equilibrium. 129 parameters: DH, joint and lateral elasticities, camera intrinsics/extrinsics.
- **Objective:** MAP, Σ Δuᵀ C_m⁻¹ Δu + ΔΘᵀ C_p⁻¹ ΔΘ.
- **Key contribution, virtual Cartesian noise:** an imperfect model is handled by adding isotropic noise σ_x̃ to the 3D marker *in base coordinates*. Marginalising gives a per-observation pixel covariance:
  `C̃_u = σ_u² I + σ_x̃² J_U J_Uᵀ`, where `J_U = ∂u/∂x`. For a pinhole this is `(f²/x_z²)·[[1+x_x²/x_z², x_x x_y/x_z²],[x_x x_y/x_z², 1+x_y²/x_z²]]`.
  - **Settings used:** σ_x̃ = 1 cm, σ_u = 0.2 px.
- **Data:**
  - Collection: rejection-sampled configurations (more than 10 M sampled per 100 feasible), each viewed with 5 head positions; 50 configurations x 5 x 3 markers = 750 samples.
  - Split: 500 train / 250 test, a random split of the same session.
  - Timing: 32 minutes in total.
- **Results (Cartesian, external Vicon):**
  - Plain MAP: 4.77 mm mean / 11.75 mm max.
  - With virtual noise and intrinsics: 3.94 / 9.16 mm.
  - External-tracker calibration: 3.12 / 8.23 mm.
  - Pixel error *rose* slightly (1.05 to 1.15 px) while the Cartesian error fell more than 20%.
- **Limitations:** the torso is the remaining error source; a non-linear term is proposed.
- **Relevance:** our boards sit at very different depths, so plain pixel residuals over-weight near boards. A direct, cheap change for Ceres (see the equations section).

**4b. Tenhumberg, Bäuml. "Calibration of an Elastic Humanoid Upper Body and Efficient Compensation for Motion Planning." Humanoids 2021.** [FULL] arXiv:2311.08333
- **DH variant:** modified DH with extra β (Hayati) for parallel axes.
- **Compliance model:** κ acts on (α, β, θ). Torques from lumped masses ν_j = [m_j, w_j] via τ = ((T_j w_j − p_i) × m_j g)·e_axis, solved by damped fixed-point iteration ρ_n = (1−λ)ρ_{n−1} + λρ(ρ0, κ, τ(F(q, ρ_{n−1}))).
- **Priors:** σ 0.1 m (lengths), 0.2 rad (angles), 0.1 rad/kNm (elasticities); multistart.
- **Masses:** taken from CAD. Calibrating masses "did not improve the residual error".
- **Data:** 300 train / 200 test (Vicon); 10 s per sample, including a 1 s pause.
- **Ablation (test mean):**

  | Model | Test mean error |
  |---|---|
  | Frames only | 21.3 mm |
  | + θ offsets | 18.2 mm |
  | + d, r, α, β | 10.5 mm |
  | + κ_θ | 5.6 mm |
  | + κ_α, κ_β (full) | 3.1 mm |

  - **Leave-one-out:** removing κ_θ gives 9.1 mm; removing offsets θ gives 6.4 mm.
  - **Sample count:** the main effect appears at N = 50.
  - Joint elasticity also absorbs **position-controller** compliance.
- **Relevance:** compliance pays off when deflection is large and deterministic: a position-controlled arm with known flank state. It is not demonstrated for back-drivable teach mode.

**5. Vicente, Jamone, Bernardino. "Markerless Eye-Hand Kinematic Calibration on the iCub Humanoid Robot." Frontiers in Robotics and AI, 2018.** [V-meta]
- Joint offsets by sequential Monte Carlo.

**6. Koide, Menegatti. "General Hand-Eye Calibration Based on Reprojection Error Minimization." IEEE RA-L 4(2):1021-1028, 2019.** [V; full text not obtained: author server timed out]
- **Method:** pose-graph minimisation of reprojection error for hand-eye; robot kinematics fixed.
- **Code:** https://github.com/koide3/st_handeye_graph (README read; supports pinhole and X-ray models).

**7. Tabb, Ahmad Yousef. "Solving the robot-world hand-eye(s) calibration problem with iterative methods." Machine Vision and Applications 28:569-590, 2017.** [V] arXiv:1907.12425
- Multi-camera AX=ZB with reprojection-based cost functions.

**8. Franzese, Spahn, Kober, Della Santina. "MUKCa: Accurate and Affordable Cobot Calibration Without External Measurement Devices." arXiv:2503.12584, 2025.** [FULL]
- **Model:** URDF-style RPY + translation per link, 6 parameters per frame (over-parameterised), axis-angle joints.
- **Cost:** consistency of a ball in two 3D-printed sockets plus the known inter-socket distance.
- **Data:** robots in **gravity-compensated, back-drivable mode** are hand-positioned; at least 30 samples per socket.
- **Explicit assumption:** Δq ≈ 0. "Fair when using ... harmonic drive gearing ... negligible backlash".
- **Results (mean absolute error, train → test):**

  | Robot | Gearbox | Original → calibrated |
  |---|---|---|
  | Franka | Harmonic | about 8.3 mm → 0.17 mm |
  | KUKA iiwa 14 | Harmonic | 0.72 mm → 0.11 mm (≈ repeatability) |
  | Kinova Gen3lite | Planetary | train 4.72 → 1.52 mm; test 5.99 → 2.66 mm (repeatability about 1 mm) |

  - The authors attribute the Kinova result to backlash "not observable in the encoder measurements". Calibrating such robots "can result in overfitting", with error growing away from the training tool position.
- **Relevance:** the closest analogue to our teach-mode data on a non-harmonic arm. Same symptoms, same diagnosis.

**9. Smith, Van Hoorick, Guizilini, Wang. "Fiducial Exoskeletons: Image-Centric Robot State Estimation." arXiv:2601.08034, 2026.** [FULL]
- **Method:** a 3D-printed fiducial on each link of an SO-100 gives a per-link SE(3) pose from one image. Joint angles are solved by L-BFGS to match those link poses; encoders are only a warm start.
- **Calibration:** joint offset Δθ = θ* − θ_enc. A closed-loop "delta refinement" re-commands the target.
- **Results:** about 75% lower end-effector state-estimation error and about 45% lower control error than encoder forward kinematics on a low-cost arm.
- **Relevance:** the alternative to modelling backlash is to *measure* link angles. Our EIV joint variables already do this per frame. FidEx shows the same idea can run at deployment time (vision-in-the-loop correction).

**10. Hollerbach, Wampler. "The calibration index and taxonomy for robot kinematic calibration methods." IJRR 15(6), 1996.** [FULL, skimmed]
- Taxonomy of closed-loop and implicit-loop methods; input noise handled by the implicit-loop formulation.

**11. van Wyk, Falco, Cheok (NIST). "Efficiently Improving and Quantifying Robot Accuracy In Situ." arXiv:1908.07273, 2019.** [FULL]
- **Setup:** 7-DoF KUKA in stiff joint-position control. **Zero-offsets only**, estimated by Bayesian inference (MLE plus uncertainty).
- **Measurement:** low-cost mocap vs laser tracker.
- **Data:** 36 Cartesian poses for optimisation (173 joint configurations); a *separate* set of 32 poses (156 configurations) for validation.
- **Results:** Vernier remastering gave more than 3 mm mean error. Training absolute error was under 0.3 mm. **Validation** absolute error was 1.5-1.8 mm mean (95%: 0.7-2.95 mm), relative error about 0.5 mm.
- **Other notes:** the literature figure that more than 90% of error is zero-offsets is quoted. Mocap matched the laser tracker.
- **Relevance:** offset-only calibrations look much better on their own training set. Always report a separate pose set.

**Open-source tools (READMEs read):**
- **Fetch `robot_calibration` (M. Ferguson, ROS/ROS2).**
  - Free parameters: `free_params` (joint offsets, camera parameters) and `free_frames` (6-DoF transforms with selectable axes; link lengths go here).
  - Error blocks: `chain3d_to_chain3d` (reprojection in pixels), `chain3d_to_plane`, `chain3d_to_mesh`, `plane_to_plane`, and `outrageous` (a soft prior that penalises absurd parameter values).
  - Output: offsets go into URDF `<calibration rising=...>` tags. The driver must add them to readings and subtract them from commands.
- **ATOM (Oliveira et al., ESWA 207:118000, 2022; github.com/lardemua/atom).** Optimises "atomic" URDF transforms across multi-modal sensors; has hand-eye variants (Pedrosa et al., T-RO 2021). No joint non-geometric models.
- **ctu-vras/multirobot-calibration:** see entry 3.
- **Kalibrot (Cursi, Bai, Kormushev, IROS 2021, pp. 8852-8859):** MATLAB kinematic calibration package cited by Daş 2024 [V-meta].
- **PR2 `calibration_estimation`:** the code for entry 1.

---

## 2. Non-geometric errors: backlash, hysteresis, compliance, gravity, transmission error

**12. Ruderman. "Compensation of Nonlinear Torsion in Flexible Joint Robots: Comparison of Two Approaches." arXiv:1507.07067, 2015** [FULL]; **Ruderman. "Modeling of Elastic Robot Joints with Nonlinear Damping and Hysteresis", InTech chapter, 2012** [FULL]; **Ruderman, Hoffmann, Bertram, IEEE TIE 56(10), 2009** [V-meta; IEEE paywall].
- **Elastic-joint model:** H(q)q̈ + C + G = τ(Δ), Jθ̈ + τ = u − f(θ̇), with torsion Δ = θ − q.
- **Bouc-Wen-like torsion hysteresis:**
  - Static part: τ̄ = k1Δ + k3Δ³.
  - Dynamic part: τ̂ = k1∫ẋ + k3(∫ẋ)³, with ẋ = Δ̇ − ψ|Δ̇||x|^{η−1}x − ξΔ̇|x|^η.
  - Total: τ = wτ̄ + (1−w)τ̂.
  - Inverse map: Δ̃ = α⁻¹(W⁻¹(τ̃ − (I−W)β(Δ̃))), fed by a momentum-observer torque estimate.
- **Experiment (chapter):** 160:1 harmonic drive, 15-bit motor resolver, **20-bit output encoder**, torque cell. Torsion-torque loops span about ±1° at about ±40 Nm. Identification used a down-chirp from 30 to 1 Hz.
- **Simulation findings:**
  - At standstill a **non-zero residual torsion (lost motion) remains**.
  - Creep relaxation can last up to about 900 s when hysteresis and friction interact.
  - "Single gravity-related compensation ... can be insufficient".
- **Relevance:**
  - Hysteresis models need the *history* (Δ̇) and a link-side reference.
  - With motor-side encoders and a quasi-static capture, the settled torsion depends on the path taken.
  - A settle time matters, but full relaxation can be slow.

**13. Alam, Ibaraki, Fukuda, Morita, Usuki (+Otsuki, Yoshioka). "Identification of a kinematic model of a 6DOF industrial manipulator with angular positioning deviation 'error map' of rotary axes" (ISFA 2020) and "Inclusion of Bidirectional Angular Positioning Deviations in the Kinematic Model of a Six-DOF Articulated Robot for Static Volumetric Error Compensation" (IEEE/ASME Mechatronics-type venue, 2022).** [V; no open full text]
- **Model:** DH plus a **look-up table of angular positioning deviation per axis, as a function of commanded angle and rotation direction** (bidirectional, which captures backlash).
- **Identification:** laser tracker with each axis indexed at set angles ("circle point method").
- **Related:** Ibaraki & Usui (Precision Engineering 2021, SCARA, DOI 10.1016/j.precisioneng.2021.11.002) apply the same bidirectional map [V-meta].
- **Relevance:** a two-map model (one per direction) is the standard form. It requires the approach direction to be *known* at each pose.

**14. Sammons, Ma, Embry, Armstrong, Bristow. "Modeling and Compensation of Backlash and Harmonic Drive-Induced Errors in Robotic Manipulators." ASME MSEC 2014, V002T02A048.** [V; ASME paywall]
- FANUC LR Mate 200i, J1 moved randomly over ±130°, laser tracker. Periodic harmonic-drive error and the backlash value fitted jointly by maximum likelihood.
- **Results:** mean error 0.063 → 0.026 mm; max 0.297 → 0.095 mm.

**15. Ma, Bazzoli, Sammons, Landers, Bristow. "Modeling and calibration of high-order joint-dependent kinematic errors for industrial robots." RCIM 50:153-167, 2018, DOI 10.1016/j.rcim.2017.09.006.** [V; Elsevier paywall; the open dissertation copy blocked automated download]
- 250 laser-tracker poses. The constant model explains 79.4% of the error and the joint-dependent model 97.0%.
- The "3.614 → 0.131 mm on an independent set" figure belongs to a companion paper, not this one.

**16. Tobuschat, Duenser, Bambach, Aschwanden. "A Unified Calibration Framework for High-Accuracy Articulated Robot Kinematics." arXiv:2601.16638, 2026.** [FULL]
- **Model:** each joint becomes T_joint(q_i) · T^J · T^C · T^G · T^T (virtual joints).
  - **Geometry:** 5 parameters per joint (3 rotations + 2 translations ⊥ axis); 6 for the base; 4 per joint suffice without the thermal term.
  - **Compliance:** rotation k_i τ_i* about the joint axis. τ_i = r_i m_i t_i × g + Σ_{j>i} m_j (t_{i→j} + r_j t_j) × g with r = ½. Lumped masses m_i are *calibrated*.
    - Mass/compliance scale ambiguity: fix m6.
    - m1 and k1 are unidentifiable (J1 parallel to gravity).
    - Identified masses can be negative (non-physical) without hurting accuracy.
    - Initialise masses at 1, not 0, to avoid a saddle point.
  - **Thermal:** linear link elongation α_i(κ − κ0)t_i.
  - **Joint correction:** piecewise-linear function of the *commanded* angle with support density 80 rad⁻¹ (0.72° spacing). It requires **≥10 samples per segment**; data outside the covered range are discarded.
- **Solver:** regularised Gauss-Newton, analytic gradients; per-submodel gradient (Fisher) spectra checked for rank deficiency.
- **Protocol:**
  - Leica AT960 laser tracker; 4571 poses after filtering; KR30 with a dead-weight payload.
  - **Every pose approached from the same joint-space direction "to eliminate errors caused by backlash"**.
  - Robot warmed up (thermal steady state confirmed by thermal camera); tracker samples with σ above 6 µm discarded.
- **Validation:** **5-fold temporal cross-validation** (contiguous time chunks).
  - Repeatability per ISO 9283: RP = l̄ + 3S_l = 14.5 µm (l̄ = 6.3 µm) after drift removal, from a 14 h night run following a 5 h warm-up.
- **Results (train / validation mean, µm):**

  | Model | Train mean | Validation mean | Validation max |
  |---|---|---|---|
  | Geometry | 99.3 | 102.3 | — |
  | + compliance | 65.7 | 69.9 | — |
  | + thermal | 57.8 | 58.5 | — |
  | Full (+ joint correction) | 24.5 | 26.8 | 97.4 |

  - The joint-correction curves are only about ±0.2 arcmin.
  - With 100 training poses the joint maps vary across folds and shrink toward zero (regularisation).
- **Limitation (verbatim sense):** no backlash model, which "limits accuracy whenever a unidirectional approach to a pose cannot be guaranteed".
- **Relevance:** the best template. Copy the compliance formula, the joint map, the ≥10-samples rule and temporal CV. It succeeds *because* the protocol removed backlash.

**17. Klimchik, Pashkevich, et al. "Design of Calibration Experiments for Identification of Manipulator Elastostatic Parameters." arXiv:1211.6101.** [FULL, skimmed]
- Plans of experiments are *configurations plus external loads*. A "test-pose" criterion minimises compliance error at the task configuration and task load.
- **Relevance:** stiffness identification needs designed load variation. Our teach-mode data have none.

**18. Slamani, Joubair, Bonev (Robotica 2014); Nubiola & Bonev (RCIM 29:236-245, 2013).** [V-meta; paywalls]
- 30-parameter model: geometry plus 5 joint compliances, on an ABB IRB 1600.

**19. Ibaraki, Theissen, Archenti, Alam. "Evaluation of Kinematic and Compliance Calibration of Serial Articulated Industrial Manipulators." IJAT 15(5):567-580, 2021.** [V]
- Review paper.

**20. KEBA / Kemro X Robotics. "Robot accuracy" white paper, 2022.** [FULL]
- **On backlash:** "impossible to know at which position within the hysteresis window the joint will settle".
  - Basic models add hysteresis by direction of movement; an improved model adds it "depending on the sign of the gear torque".
  - Real behaviour is elasticity plus friction ("hysteresis loss"). The deformation does not relax to zero when the load is removed.
- **On dual encoders:** "gear effects can be eliminated by a dual encoder system".
- **On models:** local models can beat global ones, but unmodelled effects bias global parameters.

**21. Weigand et al. "Flatness Based Control of an Industrial Robot Joint Using Secondary Encoders." arXiv:2101.04992 (KUKA KR300 Ultra SE).** [FULL, model section]
- **Definitions:**
  - **Backlash:** output-angle difference at zero torque.
  - **Lost motion:** torsion angle at the midpoint of the hysteresis curve, measured within ±3% of nominal torque.
  - **Torsional rigidity:** measured between 50 and 100% of nominal torque.
- **Stiffness model:** τ_E = 0 inside the backlash band φ_B*; slope c_LM inside the lost-motion band φ_LM; slope c_TR beyond. Effective backlash φ_B = φ_B* + φ_LM.
- **Measurement:** secondary (link-side) encoders measure Δq directly.
- **Relevance:** use these definitions. Our 0.1-0.15° is lost motion plus elasticity at low torque, not pure backlash.

**22. Harmonic-drive transmission error:** Taghirad & Bélanger 1998 [M]; FFT/Fourier models with harmonics at 2x and 4x the input rotation [V-meta].
- Only relevant for strain-wave gears; the PiPER uses planetary gears.

---

## 3. Learning-based residual compensation

**23. Hwang, Thananjeyan, Paradis, Seita, Ichnowski, Fer, Low, Goldberg. "Efficiently Calibrating Cable-Driven Surgical Robots With RGBD Fiducial Sensing and Recurrent Neural Networks." IEEE RA-L 5(4), 2020.** [FULL] arXiv:2003.08520
- **Setup:** dVRK, Zivid RGBD, 3D-printed spheres giving "true" joint angles (sphere 0.32 mm RMS; joint angle 0.02° for q1/q2 up to 0.59° for q6).
- **Data:** the robot **stops between commands**. Drand: 4000 random samples in 66 min. Dpick: 1800 task-like samples in 31 min. Test sets are 10% of each, collected the same way in the same run.
- **Models compared (13):** linear (LASSO), feed-forward network, and LSTM, on command history of horizon H, predicting either absolute angles or the error (delta).
- **Findings:**
  - All models improve with H > 0 and converge at about H = 2-4. The best is RNN-Cmd-Delta with H = 4, an ensemble of 10.
  - LASSO coefficients show **joints 1-3 depend only on their own target** (arm-joint RMS 0.063° / 0.049° / 0.255 mm).
  - Wrist joints show hysteresis (dependence on previous commands) and coupling.
- **Results (unseen pick-and-place, mean):**
  - Uncalibrated 2.96 mm.
  - RNN-forward trained on random data: 2.05 mm.
  - Trained on task-like data: 0.65 mm.
  - Peg transfer success 39.4% → 69.2% (random training data) → 96.7% (task-like training data).
- **Limitations:** per-instrument data needed; **distribution match between training and test trajectories matters a lot**.
- **Relevance:** history-dependent error is learnable only with the *command history* as input. Even then, it generalises poorly when the motion distribution changes.

**24. Lu, He, Julius, Wen. "Configuration-Dependent Robot Kinematics Model and Calibration." arXiv:2510.19962, 2025.** [FULL]
- **Model:** minimal POE. Each joint axis is h_i = R(k1, θ_i)R(k2, φ_i)h̄_i and its origin is shifted in the plane ⊥ the nominal axis: p_{i−1,i} = p̄ + (v_i k1 + w_i k2) − (v_{i−1}k1 + w_{i−1}k2). That gives 4n parameters.
- **Cost:** ‖p1 − p2‖² + a‖R1R2ᵀ − I‖_F², with a = 1641 (1 mm ≡ 1°).
- **Configuration-dependent part:** local POE fits at L clusters (248 and 280 anchors x 7 samples, OptiTrack) are interpolated with a Fourier gravity basis β(q2, q3) = [1, s2, c2, s3, c3, s23, c23, sin2q2, cos2q2, sin2q3, cos2q3, sin2(q2+q3), cos2(q2+q3)] (13 terms, SVD-reduced to 7). A = CB⁺.
- **Test results (1500 poses, Yaskawa MA2010 / MA1440):**

  | Robot | Best global model, test mean / max | FBF/NN, test mean / max |
  |---|---|---|
  | MA2010 | NLS-0: 0.41 / 0.99 mm | 0.31 / 0.73-0.74 mm |
  | MA1440 | Base-only: 1.17 / 2.93 mm (nominal 1.08 / 3.07) | 0.42-0.43 / 0.77-0.99 mm |

  - The configuration-dependent model fits the **training** data at 0.04 mm on MA1440 but tests at 0.42 mm, a large gap.
- **Relevance:** the Fourier gravity basis is a ready-made low-dimensional regressor for "drive model" terms. The train/test gap is a caution.

**25. Moore, Burkinshaw, Sawyer. "Machine learning for robotic accuracy improvement in drilling operations." IEEE CASE 2025, pp. 2960-2966, DOI 10.1109/CASE58245.2025.11163955.** [FULL]
- **Setup:** KUKA drilling cell, one plane (z = 750 mm), fixed orientation, 232-point grid at 100 mm pitch.
- **Model:** MATLAB Regression Learner, GPR with exponential and Matérn 5/2 kernels, inputs (x, y), default hyperparameters.
- **Results:**
  - Training grid: x mean 0.993 → 0.022 mm.
  - **Offset grid (+50 mm, unseen): 0.085 mm (x) and 0.051 mm (y).**
  - Drilling relative to a datum: x 0.688 → 0.323 mm, but **y 0.261 → 0.378 mm (worse)**.
- **Relevance:** GP gains measured on training points overstate the benefit. Transfer to the real use condition can fail.

**26. Daş, Burdick. arXiv:2303.03658 (2023, simulation only)** [FULL]; **Daş, Touma, Burdick. "Bayesian Optimal Experimental Design for Robot Kinematic Calibration." arXiv:2409.10802 (v3, 2025)** [FULL]
- **Method:** GP residual with GP-UCB to pick poses. The 2024 version adds valid Riemannian SE/Matérn kernels on S³ x R³ and DH correction via a QP.
- **Hardware:** OWLAT (Barrett WAM, RealSense D415, two tag36h11 tags). It recovered **injected** encoder biases with about 20 poses. **No mm or degree accuracy is reported.**
- **Relevance:** pose-selection idea only.

**27. Nguyen, Zhou, Kang, Neurocomputing 151:996-1005, 2015 (EKF + ANN)** [V-meta; paywall]; **Jing et al., ICARCV 2018 (local POE + GP, up to −37.2%)** [V-meta; paywall].

---

## 4. Errors-in-variables, observability, pose selection

**28. Wampler, Hollerbach, Arai. "An implicit loop method for kinematic calibration and its application to closed-chain mechanisms." IEEE T-RA 11(5):710-724, 1995.** [FULL]
- **Model:** f(x_i + ẽ_i, p + p̃) = 0 exactly. All measurements, joints included, are noisy, and the motion variables x "may include ... backlashes or other small unknown displacements".
- **ML objective:** Σ ẽ_iᵀ E_e⁻¹ ẽ_i + p̃ᵀ E_p⁻¹ p̃. This is EIV / orthogonal distance regression with **a-priori parameter variances**. Omitting E_p is unsafe because weakly measured combinations get magnified.
- **Solver:** normalise y_i = E_e^{−½} ẽ_i, q = E_p^{−½} p̃. Gauss-Newton, eliminating Δy_i per sample by QR.
- **Parameter covariance:** C_p = E_p^{½}(D̃ᵀD̃)⁻¹E_p^{½}, with D̃ = [D; I]. For singular values s_i of D, the error remaining in direction i is 1/√(s_i² + 1).
- **Goodness of fit:** χ² = Σ y_iᵀy_i + qᵀq has expectation **N·m** (N poses, m loop equations) and σ = √(2Nm). A value outside ±3σ means wrong variances, an unmodelled factor, or outliers. Two calibrations can be compared with a χ² test on (p1 − p2) with n dof.
- **On backlash:** "Backlash errors tend to be bimodal ... depending on the last direction of travel. A model that accounts for the last direction of travel may restore validity of the Gaussian assumption."
- **Experiments:** RSI hand controller (N = 12 poses) and the MEL Stewart platform (20 poses). χ² values were close to expectation (e.g. 109 vs 108).
- **Relevance:** our per-frame joint variables are this method. Add the χ² test and the prior term, and treat teach-mode joint noise as bimodal, not Gaussian.

**29. Hollerbach & Nahvi (ICRA 1997, TLS); Ikits & Hollerbach (ICRA 1997, plane constraint + TLS)** [Ikits FULL, skimmed].

**30. Nahvi, Hollerbach. "The Noise Amplification Index for Optimal Pose Selection in Robot Calibration." ICRA 1996.** [FULL]
- **Indices on the scaled identification Jacobian, singular values σ1 ≥ ... ≥ σL:**

  | Index | Definition |
  |---|---|
  | O1 (Borm & Menq) | (σ1⋯σL)^{1/L}/√n |
  | O2 | σ_L/σ1 (inverse condition number) |
  | O3 | σ_L (minimum singular value) |
  | O4 (noise amplification) | σ_L²/σ1 |

- **Notes:** proper task-variable and parameter scaling is required first. Schroer's rule of thumb: condition number < 100.
- **Others** [V-meta; paywalls]: Borm & Menq (IJRR 1991), Sun & Hollerbach (ICRA 2008; maps O1 to D-optimality, O3 to E-optimality, O4 to noise amplification), Joubair & Bonev (MMT 70:254-265, 2013), Daney et al. (IJRR 2005).

**31. Shirafuji, Goto, Zhang, Okuhara, Takamura, Kagawa, Baba, Ota. "Visual-Biased Observability Index for Camera-Based Robot Calibration." ASME J. Mechanisms and Robotics 16(5), 2024, DOI 10.1115/1.4062801.** [V; ASME paywall; Japanese lab summary read]
- O1 extended to include camera pose-estimation error (Ov1).
- Follow-up: joint-compliance calibration with a **hand-eye camera and floor markers**. The lab summary notes joint offsets can account for up to 90% of RMS error.

---

## 5. Cheap arms, capture protocols, hysteresis models

**32. Conenna, Guo, Wedler. "Elasto-Kinematic Calibration of the Lunar Rover Mini 6 DOF Robotic Arm." ERF 2025, Springer SPAR 36, pp. 162-167, DOI 10.1007/978-3-031-89471-8_25.** [FULL]
- **Arm:** 3D-printed with RC servos (no feedback).
- **Three-level calibration:** PWM-to-angle tuning, kinematic offsets (21 parameters), joint elasticity (7 parameters); 28 in total.
- **Data:** single-servo elasticity measured first (2.4-3.4·10⁻² rad/Nm). 393 poses on a {−100°, 0°, 100°}⁶ grid, measured by ART mocap with and without a 66.3 g load, during "approaching and retrieving" phases.
- **Pose selection:** only **50 poses**, chosen to maximise Joubair's observability index (DETMAX).
- **Validation:** 343 held-out poses. 32.9 → 5.65 mm (elasto-kinematic), 10 mm (elastic only), 19 mm (kinematic only). Cross-load transfer improved by 81-83%.
- **Relevance:** on a truly cheap arm with large compliance, compliance dominated and *did* generalise, even across loads. Their deflection is large (rad/Nm-scale) relative to our 0.1°.

**33. He, Guo, Sun, IEEE Access 2024** (title not retrieved; gear-backlash hysteresis in a modified joint space, iterative feed-forward correction, average error −81% to 0.093 mm) [V-meta].

**34. Bouc-Wen / Preisach in robot joints:** see Ruderman (entry 12) for usable equations. Xie, Lai, Li (Sensors 2019) is an adaptive-control use [V-meta].
- No study was found that uses these models in static geometric calibration of a 6R arm with held-out validation.

**35. REPLAB (Yang et al., ICRA 2019, arXiv:1905.07447).** [FULL, skimmed]
- Low-cost WidowX-class arm with camera calibration. About 2 cm error over the workspace and about 1 cm in the centre; authors note control noise on cheap arms.

**36. ISO 9283:1998 practice.** [V via secondary sources; the standard itself is paywalled; the Hexagon notes returned 404]
- **AP and RP:** measured over 30 cycles on P1-P5 in the test cube, each pose approached the same way.
- **Multi-directional pose accuracy variation (vAP):** P1, P2, P4 visited from **three different approach directions**; vAP is the maximum distance between the mean attained poses.
- **Repeatability:** RP = l̄ + 3S_l (formula as used in Tobuschat).
- **ISO 230-2** (machine tools) separates unidirectional and bidirectional repeatability, and defines a reversal value B (the bidirectional difference), i.e. backlash measured per axis [V-meta].

**Low-cost arm search (PiPER, xArm, SO-100/LeRobot, myCobot, WidowX):** no peer-reviewed calibration-accuracy study for these specific arms was found, beyond FidEx (SO-100) and REPLAB (WidowX-class). PiPER spec pages list only ±0.1 mm repeatability (vendor claim, conditions unspecified).

---

## 6. Teach-mode / hand-guided data, dual encoders

- **Hand-guided calibration data:** the only primary study using it on cobots is MUKCa (entry 8). It works on harmonic-drive, link-torque-sensed arms and degrades with overfitting on the planetary-gear Kinova.
- **Robots that stop between commands:** Hwang 2020 shows history dependence even when the robot stops.
- **Teach mode on harmonic drives:** Ruderman shows residual torsion at standstill depends on the path. A gravity-compensated, hand-guided arm comes to rest wherever friction and hysteresis leave it.
- **Dual encoders:**
  - KEBA, Weigand and Ruderman: an output-side encoder measures Δq = θ_motor/N − q_link directly, and with it gear effects "have no influence on accuracy".
  - The 2023 *Actuators* paper (DOI 10.3390/act12120454, double-encoder dynamic identification) [V-meta] uses the motor/link mismatch as a deformation measurement.
  - A PoliTo thesis states backlash identification is straightforward with encoders on both sides [V-meta].
- **For us:** the camera plus per-frame EIV variables acts as a *link-side encoder at capture time only*. The EIV corrections Δθ are measurements of link-minus-motor angle and should be analysed as data, as in Tobuschat eq. 41, which projects residuals into joint space.

---

## Reusable equations (for the Ceres solver)

Notation: q = encoder (motor-side) reading, θ = link angle used in forward kinematics, u = pixel observation, X = 3D point.

1. **Joint model with EIV** (Wampler 1995; our existing option):
   `θ_{f,j} = q_{f,j} + δ_j + ε_{f,j}` with prior residual `ε_{f,j}/σ_q`.
   - For a teach-mode, bimodal flank state, a Gaussian prior is misspecified. Either:
     - (a) use a robust (Cauchy or Huber) prior on ε with scale about b/2, or
     - (b) a two-component mixture: `ε = s_{f,j}·b_j/2 + η`, with s ∈ {−1, +1} chosen per frame by alternating minimisation (discrete EM). This is only identifiable with many frames per joint.
2. **Direction-dependent offset** (Alam/Ibaraki, KEBA, Wampler), when the approach direction or torque sign is known:
   `θ_j = q_j + δ_j + s_j·b_j/2` (+ optional angle map g_j^{s}(q_j)).
   - `s_j = sign(last motor velocity before stop)`, or `s_j = sign(τ_grav,j)` when `|τ_grav,j| > τ_thr` (gravity preloads the flank).
   - Under a strictly unidirectional protocol, s_j is constant and b_j is absorbed into δ_j. It is then identifiable only from diagnostic two-direction data.
3. **Pixel covariance from joint noise** (Pradeep 2010), an alternative to per-frame variables:
   `Σ_obs = Σ_c + J_θ Σ_θ J_θᵀ`.
   - It must be applied *per frame*, as a 2N x 2N block over all corners in the frame, because joint noise is shared by every corner. Per-frame ε variables (item 1) are the exact and cheaper equivalent in Ceres. Keep item 1.
4. **Virtual Cartesian noise / distance weighting** (Tenhumberg 2022):
   `C̃_u = σ_u² I + σ_x² J_π J_πᵀ`, where `J_π = ∂π_KB4(X_c)/∂X_c` (2x3).
   - Implement as a per-observation whitening `L⁻¹` (Cholesky of C̃_u), recomputed each outer iteration.
   - σ_x is the expected model error in metres (they used 1 cm with σ_u = 0.2 px; for us start at about 1-2 mm and tune on held-out data).
   - This turns the cost into "Cartesian-error-like" units and stops near-camera corners dominating.
5. **Lumped-mass compliance virtual joint** (Tobuschat 2026; Tenhumberg 2021):
   `θ_i += k_i · (a_iᵀ τ_i)`, with `τ_i = r m_i t_i × g + Σ_{j>i} m_j (t_{i→j} + r t_j) × g`, r = ½, using vectors from the uncalibrated chain.
   - Fix one mass, or use CAD masses (Tenhumberg found calibrating masses gave no gain).
   - Joints with axis ∥ g (J1) are unidentifiable.
   - Initialise masses ≠ 0.
6. **Piecewise-linear joint correction** (Tobuschat): `θ_i += PL_i(q_i)` with nodes at spacing Δ.
   - Choose Δ so that every segment has ≥10 samples, and add a second-difference smoothness prior.
   - Discard or extrapolate-flat outside the covered range.
7. **Minimal geometric parameters** (Lu POE eq. 11; Tobuschat 4-5 DoF per joint): per joint, 2 axis-tilt angles plus 2 origin shifts ⊥ the axis, and 6 for the base. Use POE or Hayati-β instead of classic DH for the parallel J2/J3.
8. **Gravity Fourier basis** for configuration-dependent terms (Lu eq. 19), β(q2, q3) with 13 terms; SVD-reduce.
9. **Goodness of fit** (Wampler): `χ² = Σ normalised residuals²`, expected `N_obs·m`, `σ = √(2 N_obs m)`.
   - Prior-normalised parameter covariance: `(D̃ᵀD̃)⁻¹`.
   - Recalibration consistency: `(p1 − p2)ᵀ C⁻¹ (p1 − p2) ~ χ²_n`.
10. **Observability** (Nahvi): from the column-scaled Jacobian, `O1 = (Πσ)^{1/L}/√n`, `O4 = σ_min²/σ_max`; keep the condition number below about 100.
11. **Repeatability** (ISO 9283; Tobuschat): `RP = l̄ + 3 S_l` over repeated visits of the same pose.
12. **Hysteresis (only with logged history)** (Ruderman): Bouc-Wen-like `ẋ = Δ̇ − ψ|Δ̇||x|^{η−1}x − ξΔ̇|x|^η`, plus `τ = w(k1Δ + k3Δ³) + (1−w)β(x)`. It needs a motor trajectory per capture, so it is not usable with our current data.

---

## Recommendations for calico-robocal (prioritised)

Written before the experiments in [README.md](README.md). Status on the PiPER data: gravity-side backlash
(item P3.4 with the gravity-torque sign) and minimal geometry (P3.2) both improve held-out error; distance
weighting (P3.1) is neutral on that data; per-capture joint errors (P3.3, Gaussian or Cauchy) and a per-capture
gear flank over-fit, as the literature predicts for teach-mode data. P1 and P2 need a new capture.

**Diagnosis supported by the full texts:**
- Teach-mode, gravity-compensated, motor-encoder data on a non-harmonic arm leave the flank and lost-motion state unknown and *bimodal* (Wampler, KEBA, Ruderman).
- Every model that tries to absorb this shows a train/held-out gap: MUKCa Kinova, van Wyk, Lu, and our own drive model.
- The only sub-0.1 mm results in this survey (Tobuschat, MUKCa on harmonic drives) either enforce a unidirectional approach or have negligible backlash.
- So the first two steps are protocol, not modelling.

### P1. Lost-motion / repeatability diagnostic (1-2 hours per arm; decides everything else)

**Poses:** 10 per arm, spread over the workspace. Include 3 with J2/J3 under large gravity torque, 3 near zero gravity torque on J2/J3/J5, and the rest mixed.

**Capture sequence (position control, not teach mode):**
1. For each pose and each direction d ∈ {+, −}, move to `q* − d·Δ_pre` on all joints simultaneously.
   - Use `Δ_pre = 3°`, which is ≥10x the observed 0.1-0.15° and safely beyond plausible lost motion of low-cost reducers.
2. Move slowly (≤5°/s on the final segment) to `q*`.
3. Wait, then capture.
   - Wait ≥2 s after the joint-velocity reading drops below threshold. For comparison: Tenhumberg paused 1 s, Moore dwelled 5 s, Ruderman simulates slow creep.
   - Capture 3 frames 0.5 s apart. If board-corner drift between frames exceeds 0.1 px, extend the wait.
4. Repeat 5 times for each direction, interleaved: 10 x 2 x 5 = 100 captures per arm.

**Solve (geometry and hand-eye frozen at the current solution):** solve only per-frame ε, the EIV joint corrections.

**Report per joint:**
- Same-direction scatter `σ_same`, which is true repeatability (also give ISO RP in pixels and mm at the board).
- Lost motion `b_j = mean(ε|+) − mean(ε|−)`, and its dependence on `τ_grav,j`.
- Teach-mode scatter, from re-entering the same poses by hand 5 times, for comparison.

**Decision rule:**

| Outcome | Interpretation | What to do |
|---|---|---|
| `σ_same ≪ 0.1°` and `b_j ≈ 0.1-0.3°` | Error is reducible by protocol | Go to P2 |
| `σ_same ≈ 0.1°` | Floor is non-repeatable (encoder resolution, controller deadband, creep) | No static model will help; stop adding drive models |

### P2. Production capture protocol (replaces teach-mode captures)

1. **Waypoints:** use teach mode only to *record* waypoints. Replay them under position control.
2. **Pose count:** 150-300 poses per arm.
   - Main effect expected by about 50 poses (Tenhumberg Fig. 6; Stepanova: 50 → about 2 mm, 100 → about 1 mm in simulation; Conenna: 50 observability-chosen poses).
   - Use ≥200 if joint maps (P5) are planned, since each segment needs ≥10 samples (Tobuschat).
3. **Unidirectional final approach for every joint:** the last `Δ_pre = 3°` at ≤5°/s, always in the same sign per joint.
   - For J2/J3 (and J5 when loaded), choose the sign such that the final motion **lifts against gravity**, so motor torque and gravity load the same flank.
   - Where `|τ_grav,j|` falls below a threshold (set from P1, where b_j starts to scatter), either skip the pose or rely on the unidirectional approach alone. Do not mix.
4. **Settle and capture:** settle ≥2 s, take 3 frames per pose, and reject the pose if frames disagree by more than 0.1 px RMS.
5. **Log per capture:** commanded q, encoder q, motor current/effort, the approach sign per joint, the trajectory into the pose (for possible history models), timestamps and temperature.
6. **Sessions:** collect in ≥3 sessions (different days or power cycles) for session-level validation. Warm up 10-15 minutes with similar motions before recording; motor heating changes compliance and friction, so the warm-up is cheap insurance (Tobuschat used 5 h for an industrial arm; for us this is a cheap precaution, not literature-dictated).
7. **Deployment:** the arm must use the **same approach rule at deployment**, otherwise the calibration does not transfer (Hwang's distribution-match result; Tobuschat's limitation).

### P3. Solver changes, in order of expected value

1. **Distance-aware weighting (eq. 4 above).** Low cost; Tenhumberg reports more than 20% lower Cartesian error for a slightly higher pixel RMS. Evaluate in mm at the gripper or board, not in pixels.
2. **Minimal geometric parameters (eq. 7)** for both arms. Add Gaussian priors at manufacturing tolerance (start at σ = 1 mm and 0.3°; adjust with the χ² test), and drop directions whose prior-normalised singular value s_i < 1 (no information beyond the prior; Wampler). Expect modest gains. Pradeep and van Wyk suggest offsets carry most of the error on well-made arms, but PiPER tolerances are unknown.
3. **Keep the per-frame EIV (eq. 1)** with a **robust prior** (Cauchy, scale = b/2 from P1) instead of a Gaussian while any teach-mode data remain. With P2 data, switch back to a Gaussian with σ = σ_same from P1.
4. **Direction or torque-sign term (eq. 2)**, only where the sign is logged (P1 data or mixed datasets). Fit b_j jointly with the offsets. Keep it only if held-out error improves.
5. **Compliance (eq. 5)**, only with P2 data and only if residual-vs-τ plots show a trend. Start with CAD masses and fit k_2…k_5. For a real test, add a second session with a known mass (e.g. 0.5 kg) at the flange, as in Conenna and Klimchik. Our earlier c·τ result in teach mode is not evidence against compliance under position control.
6. **Joint-angle maps (eq. 6) or the Fourier basis (eq. 8)**, only if the projected joint residuals (Tobuschat eq. 41: `Δq_i = J_iᵀr/J_iᵀJ_i`) show smooth structure vs q_i, consistent across folds.
7. **GP or NN residual:** last, and probably not worth it. Gains are tens of percent and degrade off the training set (Moore offset grid; Lu train/test gap). Static inputs cannot predict flank state; history models (Hwang) need trajectory logs and matched motion distributions.

### P4. Validation protocol (adopt before any model change)

1. **Session-level k-fold** (Tobuschat temporal CV): train on all sessions but one, test on the held-out session. Report mean, 95th percentile and max in px **and** in mm at the board or TCP (via depth), plus the fold-to-fold spread.
2. **Report three numbers separately:**
   - (a) held-out same-direction (P2-style) error;
   - (b) held-out teach-mode or mixed-direction error (the ISO vAP analogue);
   - (c) the repeatability floor RP from P1.

   A model is only "better" if (a) drops by more than the fold-to-fold spread.
3. **χ² check (Wampler):** with the stated σ_u (pixels), σ_q (joints) and priors, χ² should be within N·m ± 3√(2Nm). A χ² far above that after P2 means unmodelled structure; at or near expectation means stop modelling.
4. **Parameter stability:** the spread of each parameter across folds should be small relative to its value (Tobuschat Tab. II). Unstable compliance or mass values signal unidentifiability, so reduce the parameter set.
5. **Observability report:** O1 and O4 on the column-scaled Jacobian for each arm, and the condition number (Schroer: < 100). Use greedy D-optimal (O1) selection to thin or plan poses (Conenna used DETMAX on 393 candidates to pick 50).

### Expected gains

- P1 + P2 are the only steps with direct literature support for removing an error of our type. The size of the gain is unknown until P1 measures σ_same vs b_j.
- If σ_same is close to 0.1°, the ~0.45° held-out error is mostly irreducible without a link-side measurement. The options then are vision-in-the-loop correction at deployment (FidEx-style) or hardware with output encoders.
- Geometry (P3.2) and distance weighting (P3.1) are safe improvements, each probably worth a modest fraction of the residual.
- Compliance and joint maps are only worth adding once controlled-approach data exist, and only if session-level held-out error confirms them.
