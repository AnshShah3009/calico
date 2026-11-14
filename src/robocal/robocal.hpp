// robocal: offline joint calibration of a multi-arm robot cell with wrist and static cameras.
//
// One nonlinear least-squares problem over
//   - per-joint zero offsets dq (theta = q_encoder + dq), per-joint free / fixed,
//   - each wrist camera's mount on its arm's tip link (T_tip_cam, optical frame),
//   - each arm base in the world frame (fixed for the gauge arm, free + optional prior otherwise),
//   - each static camera in the world frame,
//   - one pose per calibration board (boards are keyed by the samples' "session"),
//   - optionally camera intrinsics (last pass only),
// with pixel reprojection residuals of board corners, a Huber loss, an optional level-plane
// prior on one flat board, and priors on dq / free bases.
#ifndef ROBOCAL_ROBOCAL_HPP_
#define ROBOCAL_ROBOCAL_HPP_

#include "boards.hpp"
#include "camera_models.hpp"
#include "detect.hpp"
#include "view_cost.hpp"
#include "se3.hpp"
#include "urdf_chain.hpp"

#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace ceres {
class Problem;
}

namespace robocal {

struct ArmConfig {
    std::string name;
    std::string urdf;      // URDF path (or empty when urdf_xml is set)
    std::string urdf_xml;  // inline URDF (tests)
    std::string base_link;
    std::string tip_link;  // link the wrist camera is mounted on
    std::vector<double> drive_fixed;  // drive.shared false: [c_1..c_n, h_1..h_n] held (transferred from another session)
    bool camera_mount_fixed = false;
    bool geometry = false;            // calibrate 4 geometric terms per joint (tilts / shifts perpendicular to the axis)
    double geometry_sigma_mm = 1.0, geometry_sigma_deg = 0.3;  // priors (manufacturing tolerance)  // hold the wrist camera mount at T_tip_cam_init (e.g. transferred from another session)
    std::string ee_link;   // optional: fixed child of tip_link, for the T_EC export
    std::vector<std::string> joints;
    Iso3 T_world_base = Iso3::Identity();
    bool base_fixed = false;
    double base_prior_trans_m = 0.0;  // <= 0: no prior on a free base
    double base_prior_rot_deg = 0.0;
    std::vector<double> dq_init;       // rad, size dof (default 0)
    std::vector<bool> dq_free;         // size dof
    std::vector<double> dq_sigma_rad;  // prior sigma of free joints, size dof
    // Encoder noise (rad, 1 sigma per joint and frame). > 0: every frame gets its own joint
    // errors with this prior (errors-in-variables), so noisy encoders are weighted correctly
    // and the reported uncertainties stay honest.
    double joint_noise_rad = 0.0;               // largest per-joint sigma (> 0: EIV on)
    std::string joint_noise_loss = "gaussian";  // gaussian | cauchy (robust per joint: two-sided gear-play jumps)
    std::vector<double> joint_noise_joint_rad;  // per joint (joint_noise_deg: scalar or list)
    std::string camera;                // wrist camera name ("" = none)
    std::string samples;               // toolkit *_handeye_samples.jsonl
    Iso3 T_tip_cam_init = Iso3::Identity();  // tip link -> camera optical frame
    bool has_optical_in_body = false;
    Vec3 optical_in_body_rpy = Vec3::Zero();  // camera body -> optical (ROS: -pi/2, 0, -pi/2)
};

struct CameraConfig {
    std::string name;
    std::string camera_info;  // ROS camera_info YAML (optional: else model / size / f_init)
    std::string model = "equidistant";  // or auto: fit each of models and keep the best (hold-out error)
    std::vector<std::string> models = {"equidistant", "kb_rt"};  // double_sphere available, not default
    int width = 0;
    int height = 0;
    double f_init = 0.0;            // initial focal length (px) when there is no camera_info
    double export_distance_m = 0.25;  // kb4_pupil: the central .equidistant.yaml is fitted for points this far
    bool intrinsics_prepass = false;  // per-camera intrinsics fit (free view poses) before the joint solve
    bool intrinsics_given = false;
    CameraIntrinsics intrinsics;  // used when intrinsics_given (tests)
    bool optimize_intrinsics = false;
    bool is_static = false;
    bool handheld = false;    // camera (or one rigid board set) moved: one free pose per frame
    bool free_views = false;  // boards moved independently: one free pose per board view
    Iso3 T_world_cam_init = Iso3::Identity();  // static cameras: world -> optical
    std::string observations;                 // static cameras: jsonl or center_obs.yaml
    std::string default_board = "0";
    bool has_optical_in_body = false;
    Vec3 optical_in_body_rpy = Vec3::Zero();
};

struct LevelPrior {
    bool enabled = false;
    std::string board = "0";
    double sigma_deg = 0.3;
};

struct SolverOptions {
    double huber_px = 1.0;       // <= 0: plain least squares
    double distance_sigma_mm = 0.0;  // > 0: per-corner weight 1/sqrt(s_px^2 + (s_mm f / depth)^2) (Tenhumberg 2022)
    double distance_sigma_px = 0.3;  // the pixel part s_px of that weight
    int holdout_every = 5;       // every n-th view per camera is held out (< 2: none)
    int max_iterations = 200;
    double outlier_floor_px = 3.0;
    double outlier_mad_k = 5.0;  // point dropped above max(floor, median + k * 1.4826 * MAD)
    int min_points_per_view = 4;
    int threads = 4;
    bool verbose = false;
    int bootstrap = 10;  // intrinsics repeatability: refits on resampled frames (0: off)
    double pixel_sigma_px = 0.0;  // detection noise (1 sigma per axis); 0: estimated after pass 2
    // View-level robust weights (after pass 3): a view whose residual spread s_v exceeds the pixel
    // noise s_0 (motion blur, a sheet bending in that view) is down-weighted by s_0 / s_v, so a few
    // bad sequences cannot pull the lens. Rounds of re-weighting; 0: off.
    int view_weight_rounds = 0;
    // Data-only eigen analysis (condition, weak directions): dense in every free parameter incl.
    // per-view poses, O(n^3). -1 auto = only with arms (stage 2); 0 off; 1 on.
    int observability = -1;
    double view_distance_min_m = 0, view_distance_max_m = 0;  // view_distance_m: [min, max]; 0 = all
    // Time blocks of source frames (frames list order): hold-out and bootstrap resample whole blocks,
    // because neighbouring frames of a recording carry correlated errors. 0 = per frame.
    int block_frames = 40;
    // corner: Huber per corner (iteratively reweighted between passes); view: Huber on the whole view's
    // summed error (the pre-2026-11-13 behaviour).
    std::string robust = "corner";
    // Intrinsics are calibrated in their own stage; a robot solve refuses optimize_intrinsics unless
    // this is set (the robot run does not cover the image well enough).
    bool allow_intrinsics_in_robot_solve = false;
    // Minimiser: lm (Levenberg-Marquardt, default) | dogleg | dogleg_subspace (trust region) |
    // lbfgs | bfgs | nlcg (line search; bounds on the drive model are dropped).
    std::string method = "lm";
    // auto | dense_qr | dense_schur | sparse_schur | iterative_schur | sparse_normal_cholesky
    std::string linear_solver = "auto";
    std::string backend = "auto";  // auto | ceres | schur_lm (stage-1 problems: stage1_solver.hpp)
    // Global search around the main joint solve (pass 2): none | multistart (perturbed restarts,
    // keep the best) | basin_hopping (annealed: perturb, local solve, Metropolis accept at a
    // temperature falling to 0). Perturbations: joint zeros, camera mounts, free bases.
    std::string global = "none";
    int global_iterations = 8;
    double global_temperature = 0.05;  // relative cost increase accepted with prob 1/e at the start
    double global_step_deg = 5.0;
    double global_step_m = 0.01;
    unsigned global_seed = 1;
    // Camera mount start: auto (closed-form hand-eye from the data vs the CAD value, whichever makes
    // the static boards' implied poses agree best) | cad | data.
    std::string mount_init = "auto";
    // Robot solve sanity gate: a wrist / static camera whose training RMS exceeds this (px) fails
    // the solve (e.g. a camera mount wrong by 180 deg converges to a wrong minimum). <= 0: off.
    double max_rms_px = 1.5;
};

// Drive train between motor-side encoders and links: torsional compliance and reducer backlash
// under gravity, linear in its parameters (precomputed regressors per frame):
//   theta_j = q_j + dq_j + c_j (tau_j - tau_ref_j) + h_j (s_j - s_ref_j)
// tau_j: gravity torque from the URDF inertials; s_j in [-1, 1]: the gear flank the link rests on
// (from the capture protocol or the logged final travel); tau_ref / s_ref: the dataset's working
// load, so dq is the zero at that load. c >= 0 (rad / N m), h >= 0 (half backlash, rad).
struct DriveConfig {
    bool model = false;
    // against_gravity: every move ends against gravity (fixed + where |tau| < tau_min_nm), so
    //                  s = sign(tau); logged: frames carry the signed final travel per joint;
    // none: nothing known (s = sign(tau) only where |tau| >> friction).
    std::string approach = "against_gravity";
    bool shared = true;            // one set of drive parameters for all arms (same hardware)
    double tau_min_nm = 0.05;
    std::vector<double> friction_nm = {0.3, 0.3, 0.25, 0.1, 0.1, 0.05};
    double h_prior_deg = 0.1, h_sigma_deg = 0.15;
    std::vector<double> c_sigma = {1.5e-3, 1.5e-3, 1.5e-3, 2e-3, 2e-3, 2e-3};  // rad / N m
    Vec3 gravity = Vec3(0, 0, -9.81);  // world frame
    std::vector<bool> backlash_joints, compliance_joints;  // optional masks: false = term off (held at 0)
    bool flank_select = false;  // training frames: pick each joint's gear flank (+-1) that fits the images best (two-state EIV)
};

// Pass / fail gates on calibrated intrinsics (hand-held / free-view sessions). <= 0: not checked.
struct IntrinsicsGates {
    double max_holdout_rms_px = 0.5;   // per point, held-out frames with their pose refitted alone
    double min_theta_deg = 80.0;       // largest off-axis angle actually measured
    double min_coverage = 0.8;         // share of image cells (16 x 9, inside 95 deg) with corners
    double max_curve_sd_px = 0.3;      // bootstrap spread of the lens curve, up to the measured angle
    double max_mean_radial_px = 0.15;  // radial bias in any off-axis bin with >= 50 points
};

struct Config {
    std::string world_frame = "base";
    std::vector<ArmConfig> arms;
    std::vector<CameraConfig> cameras;
    std::vector<BoardSpec> boards;  // AprilGrids detected by calico (static over the run)
    std::string frames;             // frame manifest: images + joints (YAML list or JSON lines)
    std::string dump_rounds_dir;    // detection.dump_rounds: corners + lens after each guided round
    std::string detections_cache;   // detections of the frames: reused when present, else written
    // Reliable-tag filter for the robot solve (0 = off), through each camera's own (fixed) lens:
    // keep only AprilGrid tags whose centre lies within detection.max_theta_deg of the optical axis
    // and whose mean edge is >= detection.min_tag_px; a board view keeps >= min_tags_per_board tags.
    double max_theta_deg = 0.0;
    double min_tag_px = 0.0;
    std::string reference_board;    // world frame when no arm base is fixed (default: level / first board)
    // Frames whose still_s (time the arm had been stationary before capture) is below this are
    // dropped: joint states arrive late, so a frame taken while the arm moved pairs the image
    // with stale joints. 0: no check.
    double min_still_s = 0.0;
    DetectOptions detect;
    LevelPrior level;               // legacy single level board (samples / sessions)
    SolverOptions solver;
    IntrinsicsGates gates;
    DriveConfig drive;

    // Boards held level: name -> sigma (deg). From boards[].level and level_board.
    std::map<std::string, double> LevelBoards() const;

    static Config Load(const std::string& path);
};

// The equidistant (4-coefficient Kannala-Brandt) model closest to another model's projection over
// rays up to theta_max_deg: what tools that only know ROS "equidistant" can load. rms / max (px)
// of the approximation over those rays.
CameraIntrinsics FitEquidistant(const CameraIntrinsics& src, double theta_max_deg, double* rms, double* max);
// Any model fitted to another's projection over the same rays (e.g. to start a solve from an
// equidistant result).
CameraIntrinsics FitModel(const CameraIntrinsics& src, LensModel model, double theta_max_deg, double* rms, double* max,
                          double distance_m = 0.0);

// Largest pixel difference between two intrinsics' projections over rays up to theta_max_deg
// inside the image (repeatability of two sessions); rms over the same rays in *rms.
double CompareIntrinsics(const CameraIntrinsics& a, const CameraIntrinsics& b, double theta_max_deg, double* rms);
// Repeatability of two lens fits, three ways (index 0 raw, 1 after the best rotation of B's rays
// onto A -- what a stage-2 mount / pose absorbs --, 2 with B's principal point set to A's), as max
// and rms pixel distance over a ray fan to theta_max_deg, overall and per 5 deg ring.
struct IntrinsicsComparison {
    struct Ring {
        double max[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
        int n = 0;
    };
    double theta_max_deg = 0, max[3] = {0, 0, 0}, rms[3] = {0, 0, 0}, rotation_deg = 0;
    double d_params[4] = {0, 0, 0, 0};  // B - A: fx, fy, cx, cy
    int n = 0;
    std::vector<Ring> rings;
};
IntrinsicsComparison CompareIntrinsicsDetailed(const CameraIntrinsics& a, const CameraIntrinsics& b,
                                               double theta_max_deg, double distance_m = 0.0);

// ROS camera_info YAML (equidistant / plumb_bob / rational_polynomial) -> intrinsics.
CameraIntrinsics LoadCameraInfo(const std::string& path, const std::string& name);


struct ErrorStats {
    int n_views = 0;
    int n_points = 0;
    double rms_point = std::numeric_limits<double>::quiet_NaN();  // sqrt(mean |e|^2)
    double rms_axis = std::numeric_limits<double>::quiet_NaN();   // sqrt(mean e_x^2, e_y^2)
    double median = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

struct WeakDirection {
    double sigma_ratio = 0.0;  // singular value / largest (column-normalized Jacobian)
    std::vector<std::pair<std::string, double>> components;
};

struct ThetaBin {
    double lo_deg = 0, hi_deg = 0;
    int n = 0;
    double rms_radial = 0, rms_tangential = 0, mean_radial = 0;  // px
};

struct LensReport {
    std::vector<ThetaBin> bins;  // training corners, by off-axis angle
    double theta_max_deg = 0;  // upper edge of the highest off-axis bin with >= 20 corners
    double fold_deg = 180;     // angle where the fitted model stops being one-to-one (180: never)
    double coverage = 0;
    int cells_hit = 0, cells_total = 0;
    int bootstrap_runs = 0;
    double fx_sd = 0, fy_sd = 0, cx_sd = 0, cy_sd = 0;
    std::vector<std::pair<double, double>> curve_sd;  // (theta deg, sd px)
    double max_curve_sd = 0;
    bool gated = false;
    bool pass = true;
    std::vector<std::string> failures;
};

struct Diagnostics {
    std::map<std::string, LensReport> lens;  // per camera
    std::map<std::string, ErrorStats> train;    // per camera
    std::map<std::string, ErrorStats> holdout;  // per camera
    int n_free_params = 0;
    int n_residuals = 0;
    double condition = 0.0;
    std::vector<double> smallest_sigma_ratio;
    std::vector<WeakDirection> weak;
    std::vector<std::string> zero_columns;
    std::map<std::string, double> sigma;  // named 1-sigma (deg / mm / px), scaled by residual variance
    bool covariance_ok = false;
    bool observability_done = false;  // eigen analysis ran (solver.observability)
    int outlier_points = 0;
    int dropped_views = 0;
    double final_cost = 0.0;
    double pixel_sigma = 1.0;  // px, used to weight the image residuals
    bool solve_failed = false;  // robot solve sanity gate
    std::string fail_reason;
};

class RobotCalibrator {
public:
    explicit RobotCalibrator(Config cfg);
    ~RobotCalibrator();

    // Chains, intrinsics and (when paths are configured) samples / observations.
    void Load();
    // Views supplied directly (tests / other front ends). Call after Load().
    void AddView(const View& v);

    void Initialize();
    void Solve();

    const Diagnostics& diagnostics() const { return diag_; }
    const Config& config() const { return cfg_; }

    std::vector<double> DeltaQ(int arm) const;
    Iso3 TipCam(int arm) const;
    Iso3 WorldBase(int arm) const;
    Iso3 WorldCamera(int cam) const;  // static cameras
    Iso3 FrameCamera(int cam, int frame) const;  // hand-held cameras, per frame
    Iso3 WorldBoard(const std::string& board) const;  // static boards
    Iso3 LinkBoard(const std::string& board) const;   // mounted boards: mount link -> board
    std::vector<double> BoardShape(const std::string& board) const;
    const CameraIntrinsics& Intrinsics(int cam) const;
    int ArmIndex(const std::string& name) const;
    int CameraIndex(const std::string& name) const;

    std::string Summary() const;
    void WriteYaml(const std::string& path) const;
    // ROS camera_info of every camera with optimize_intrinsics, into dir/<name>.camera_info.yaml.
    std::vector<std::string> WriteCameraInfos(const std::string& dir) const;
    // One URDF of the calibrated cell: world link, every arm with its joint zero offsets baked
    // into the joint origins (it takes encoder angles), camera optical frames, static cameras.
    void WriteCalibratedUrdf(const std::string& path) const;
    // Every corner's reprojection residual (px), per view: image, camera, board, point id, pixel,
    // residual, hold-out flag, off-axis angle. For diagnosing blur, boards, frames.
    void WriteResiduals(const std::string& path) const;

private:
    struct ArmState {
        UrdfChain chain;
        bool has_ee = false;
        Iso3 T_tip_ee = Iso3::Identity();
        std::vector<double> dq;
        std::vector<double> drv;     // drive model: c (dof), h (dof)
        std::vector<double> geo;     // geometry terms (4 per joint), ArmConfig::geometry
        std::vector<double> tau_ref, s_ref;
        std::vector<int> drv_const;  // entries held (no spread in their regressor)
        PoseParam tip_cam;
        PoseParam world_base;
        int camera = -1;
    };
    struct CamState {
        CameraIntrinsics intr;
        PoseParam world_cam;
        bool is_static = false;
        bool handheld = false;
        bool free_views = false;
    };
    struct BoardState {
        PoseParam pose;  // static: world -> board
        bool mounted = false;
        int arm = -1;
        Iso3 T_tip_link = Iso3::Identity();
        PoseParam mount;  // mounted: link -> board
        bool mount_fixed = false;
        bool mount_init_given = false;
        double shape[5] = {0, 0, 0, 0, 0};
        bool fit_points = false;
        double point_sigma_m = 0.0003;
        std::vector<double> heights;  // per board point (m), when fit_points
        bool fit_warp = false;
        double warp_sigma_m = 0.0003;
        double warp[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // a2 a3 a4 a5 b1 b3 b4 b5 (m), when fit_warp
        bool fit_shape = false;
        bool fit_scale = false;
        double shape_sigma_m = 0.003;
        double scale_sigma = 0.01;
        double print_scale[2] = {0, 0};  // prior centre of shape[3] / shape[4]
        BoardGeometry geom;
        bool initialized = false;
        bool fixed = false;  // world reference (no fixed arm base)
    };
    struct Flags {
        bool free_dq = false;
        bool free_intrinsics = false;
        bool only_static = false;
        bool include_holdout = false;
        bool use_loss = true;
        bool no_priors = false;  // data-only problem (observability)
        bool pose7 = true;       // hand-held frame pose as one 7-parameter block (q, t): Schur-eliminable
    };

    void LoadSamples(int arm, const std::string& path);
    void LoadFrames(const std::string& path);
    void FilterReliableTags(bool erase);  // detection.max_theta_deg / min_tag_px
    bool EstimateBoardPose(const View& v, Iso3& T_cam_board, double& rms) const;
    void EstimateMissingPoses();
    void IntrinsicsPrepass(int cam);
    void DumpCorners(int cam, int round) const;  // detection.dump_rounds
    int BlockOf(const View& v) const;           // time block of the view's source frame
    int UpdateCornerWeights();
    int SelectFlanks();  // drive.flank_select                  // solver.robust: corner (IRLS Huber weights); # down-weighted
    int GuidedPass(int cam, bool rectified);  // corners found through the fitted lens; returns # added
    std::map<int, std::vector<std::pair<int, std::string>>> frame_images_;  // camera -> (frame, image)
    void LoadObservations(int cam, const std::string& path);
    void AssignHoldout();
    void BuildProblem(ceres::Problem& problem, const Flags& f);
    double RunSolver(const Flags& f, const std::string& label);
    // false: not applicable. cov != nullptr: no solve, marginal covariances of the lens / shape blocks.
    bool RunStage1Solver(const Flags& f, const std::string& label, double* final_cost,
                         std::map<const double*, Eigen::MatrixXd>* cov = nullptr);
    struct Snapshot {
        std::vector<ArmState> arms;
        std::map<std::string, BoardState> boards;
        std::vector<CamState> cams;
        std::map<std::pair<int, int>, PoseParam> frame_poses;
        std::map<std::pair<int, int>, std::vector<double>> joint_eps;
    };
    Snapshot TakeSnapshot() const { return {arms_, boards_, cams_, frame_poses_, joint_eps_}; }
    void Restore(const Snapshot& s) {
        arms_ = s.arms;
        boards_ = s.boards;
        cams_ = s.cams;
        frame_poses_ = s.frame_poses;
        joint_eps_ = s.joint_eps;
    }
    void Perturb(std::mt19937& rng, double scale);
    void InitMountsFromData();
    double MountDispersion(int arm, const Iso3& X) const;
    void GlobalSearch(const Flags& f);
    bool line_search_ = false;
    bool ViewResiduals(const View& v, std::vector<double>& r) const;
    void DropInvalidViews(const std::string& when);
    int RejectOutliers();
    void ComputeErrorStats();
    void ComputeObservability(const Flags& f);
    Iso3 CameraInWorld(const View& v) const;
    Iso3 MountLinkInWorld(const BoardState& b, const std::vector<double>& q) const;
    void SetupBoards();
    void ComputeDriveRegressors(bool verbose);
    double* DriveBlock(int arm);
    ViewCost MakeViewCost(const View& v, std::vector<double*>* blocks, std::vector<int>* sizes, bool pose7 = false) const;

    Config cfg_;
    std::vector<ArmState> arms_;
    std::vector<CamState> cams_;
    std::map<std::string, BoardState> boards_;
    std::map<std::pair<int, int>, PoseParam> frame_poses_;  // (camera, frame) of hand-held cameras
    std::map<std::pair<int, int>, std::vector<double>> joint_eps_;  // (arm, frame): encoder errors
    std::map<std::pair<int, int>, std::vector<double>> frame_amb_;  // (arm, frame): flank ambiguity per joint
    std::set<std::pair<int, int>> frames_initialized_;
    void InitHandheld();
    void RefitHeldOutFrames();
    void ComputeLensReports();
    void BootstrapIntrinsics(int cam);
    void ApplyGates();
    std::vector<View> views_;
    Diagnostics diag_;
    bool loaded_ = false;
    bool initialized_ = false;
    bool final_free_intrinsics_ = false;
    Flags final_flags_;
    bool distance_logged_ = false;
    const std::map<const View*, int>* boot_mult_ = nullptr;  // bootstrap: residual blocks per view
    double pixel_sigma_ = 1.0;  // weights image residuals against the priors (absolute units)
};

}  // namespace robocal

#endif  // ROBOCAL_ROBOCAL_HPP_
