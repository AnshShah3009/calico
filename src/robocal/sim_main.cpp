// calico-robocal-sim: end-to-end simulation of the calibration workflow.
//
//   generate --out=DIR [--seed=N]
//       A ground-truth cell (true joint zero offsets, camera mounts, right base, center camera,
//       per-camera fisheye intrinsics, a bowed board) and the nominal system calibration starts
//       from (CAD URDF, dq = 0, CAD mounts / bases, a rough focal length). Writes rendered images
//       and configs for every step:
//         DIR/stage1_<cam>/   hand-held intrinsics session per camera (off the robot)
//         DIR/stage2/         robot run: wrist route + center frames, encoder joints
//         DIR/gt.yaml         the truth (read only by `evaluate`)
//   evaluate --sim=DIR
//       Scores DIR/stage1_<cam>/out/<cam>.camera_info.yaml, DIR/stage2/out/robot_calib.yaml and
//       DIR/stage2/out/calibrated_cell.urdf against the truth, on validation poses never used in
//       calibration, next to the nominal (uncalibrated) system.
#include "focus.hpp"
#include "render.hpp"
#include "robocal.hpp"

#include <apriltag/tag36h11.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace robocal;
using namespace robocal::sim;

namespace {

constexpr double kDeg = M_PI / 180.0;
const char* kSides[2] = {"left", "right"};
const char* kCams[3] = {"left", "right", "center"};

void MkDir(const std::string& p) { mkdir(p.c_str(), 0755); }

// Piper-like 6-DoF arm (CAD). The calibration only ever sees this nominal model.
std::string ArmUrdf(const std::string& p) {
    std::ostringstream o;
    auto joint = [&](const std::string& name, const std::string& type, const std::string& parent,
                     const std::string& child, const std::string& xyz, const std::string& rpy, const std::string& axis) {
        o << "  <joint name=\"" << p << name << "\" type=\"" << type << "\">\n    <parent link=\"" << p << parent
          << "\"/>\n    <child link=\"" << p << child << "\"/>\n    <origin xyz=\"" << xyz << "\" rpy=\"" << rpy << "\"/>\n";
        if (!axis.empty()) {
            o << "    <axis xyz=\"" << axis << "\"/>\n    <limit lower=\"-3.0\" upper=\"3.0\" effort=\"10\" velocity=\"3\"/>\n";
        }
        o << "  </joint>\n";
    };
    o << "<?xml version=\"1.0\"?>\n<robot name=\"" << p << "arm\">\n";
    // Inertials: AgileX piper_description masses mapped onto this chain; link6 carries the gripper
    // (0.457 kg) and the wrist camera (0.06 kg). Must match kLinkMass (the simulator's truth).
    const std::map<std::string, std::pair<double, std::string>> inert = {
        {"link1", {0.71, "0 0 0"}},     {"link2", {1.17, "0 0 0.14"}},  {"link3", {0.50, "0.01 0 0.125"}},
        {"link4", {0.38, "0 0 0"}},     {"link5", {0.383, "0 0 0.05"}},
        {"link6", {0.517, "-0.0052224 0 0.0317679"}}};
    for (const char* l : {"base_link", "link1", "link2", "link3", "link4", "link5", "link6", "end_effector"}) {
        auto it = inert.find(l);
        if (it == inert.end()) {
            o << "  <link name=\"" << p << l << "\"/>\n";
            continue;
        }
        o << "  <link name=\"" << p << l << "\">\n    <inertial>\n      <origin xyz=\"" << it->second.second
          << "\" rpy=\"0 0 0\"/>\n      <mass value=\"" << it->second.first
          << "\"/>\n      <inertia ixx=\"1e-3\" ixy=\"0\" ixz=\"0\" iyy=\"1e-3\" iyz=\"0\" izz=\"1e-3\"/>\n"
          << "    </inertial>\n  </link>\n";
    }
    joint("joint1", "revolute", "base_link", "link1", "0 0 0.123", "0 0 0", "0 0 1");
    joint("joint2", "revolute", "link1", "link2", "0 0 0", "0 0 0", "0 1 0");
    joint("joint3", "revolute", "link2", "link3", "0 0 0.285", "0 0 0", "0 1 0");
    joint("joint4", "revolute", "link3", "link4", "0.02 0 0.25", "0 0 0", "0 0 1");
    joint("joint5", "revolute", "link4", "link5", "0 0 0", "0 0 0", "0 1 0");
    joint("joint6", "revolute", "link5", "link6", "0 0 0.091", "0 0 0", "0 0 1");
    joint("ee", "fixed", "link6", "end_effector", "0 0 0.1358", "0 0 0", "");
    o << "</robot>\n";
    return o.str();
}

std::vector<std::string> Joints(const std::string& s) {
    std::vector<std::string> j;
    for (int i = 1; i <= 6; ++i) j.push_back(s + "_joint" + std::to_string(i));
    return j;
}

CameraIntrinsics Fisheye(double f, double cx, double cy, double dk1) {
    CameraIntrinsics c;
    c.model = LensModel::Equidistant;
    c.width = 1920;  // AR0234: 1920 x 1200, 3.0 um pixels, global shutter
    c.height = 1200;
    c.params = {f, f * 1.0006, cx, cy, -0.0115 + dk1, 0.0008, -0.00018, 0.00003};
    return c;
}

struct SimOptions {
    double scale = 1.0;           // multiplies the unknown joint offsets (J2-J5), mount and base errors
    double joint_noise_deg = 0.0; // encoder noise on the recorded joints (not on the truth)
    int stage1_frames = 80;
    int route_poses = 30;         // stage-2 calibration poses per arm
    bool stage1_edges = true;     // push half the session frames off-axis (image edge coverage)
    bool backlash = true;         // gravity-loaded backlash + compliance between encoder and link
    std::string approach = "random";  // final approach per joint: random | unidirectional | against_gravity
    bool drive_model = true;          // stage-2 config: estimate the drive model (compliance + backlash)
    double model_joint_noise_deg = 0.03;  // errors-in-variables sigma written to the stage-2 config
    bool second_session = true;           // a second intrinsics session per camera (repeatability)
    std::string lens_model = "auto";      // what the intrinsics configs ask calico to fit
};

// Operating range each camera is focused for: wrist cameras work close, the center camera
// watches the whole cell.
struct Range {
    double near_m, far_m;
};
Range OperatingRange(int c) { return c == 2 ? Range{0.30, 1.50} : Range{0.13, 0.30}; }

// Prints used off the robot (focus + intrinsics session), as specified for the real cell:
//   wrist cameras (13-30 cm):  2 x A4 AprilGrid 7 x 10, 20 mm tags (254 x 176 mm), ids 100 / 170
//   center camera (30-80 cm):  2 x A2 AprilGrid 7 x 10, 35 mm tags (445 x 308 mm), ids 300 / 370
//   center far focus check:    A2 AprilGrid 3 x 4, 100 mm tags (490 x 360 mm), ids 450
std::vector<BoardSpec> SessionBoards(int c) {
    const double tag = c == 2 ? 0.035 : 0.020;
    const int id0 = c == 2 ? 300 : 100;
    std::vector<BoardSpec> b(2);
    b[0] = {"a", "tag36h11", 7, 10, tag, 0.3, id0, false, 0.3};
    b[1] = {"b", "tag36h11", 7, 10, tag, 0.3, id0 + 70, false, 0.3};
    return b;
}

// Session layout: board a flat on the table, board b next to it tilted ~35 deg (a tent), so every
// frame sees two planes.
std::vector<Iso3> SessionPlacement(const std::vector<BoardSpec>& b) {
    const double tilt = 0.6;
    const double ya = -(0.5 * b[0].Height() + 0.02);
    const double yb = 0.5 * b[1].Height() * std::cos(tilt) + 0.02;
    return {Place(b[0], Vec3(0.0, ya, 0.0), Vec3(0, 0, 0.03)),
            Place(b[1], Vec3(0.01, yb, 0.5 * b[1].Height() * std::sin(tilt) + 0.002), Vec3(-tilt, 0, -0.05))};
}
BoardSpec FarFocusBoard() { return {"far", "tag36h11", 3, 4, 0.100, 0.3, 450, false, 0.3}; }

// A fixed-focus M12 fisheye: thin lens, focused by screwing it in its thread. Moving the lens
// z metres beyond its infinity position focuses it at s = F (F + z) / z and scales the focal
// length by (1 + z / F) -- so focus must be locked before the intrinsics are calibrated.
struct LensTruth {
    double f_inf_px = 667.0;  // ~2.0 mm EFL / 3.0 um pixels (200 deg M12 fisheye on an AR0234)
    double pitch_m = 3.0e-6;
    double f_number = 2.2;    // assumed (the S191 lens is F2.4)
    double z_factory_m = 0.0;  // as delivered
    double z_m = 0.0;          // as set by the focus step
    static constexpr double kThreadPerDeg = 0.5e-3 / 360.0;  // M12 x 0.5: lens travel per degree
    double F() const { return f_inf_px * pitch_m; }
    double FocusDistance(double z) const { return z > 1e-12 ? F() * (F() + z) / z : 1e9; }
    double FocalPx(double z) const { return f_inf_px * (1.0 + z / F()); }
    double ZFor(double s) const { return F() * F() / (s - F()); }
    double DefocusPxPerDioptre(double z) const { return F() / f_number * FocalPx(z); }
};

// Everything the simulator knows and calibration must recover.
struct Truth {
    std::vector<double> dq[2];
    Iso3 base_cad[2], base[2];
    Iso3 mount_cad[2], mount[2];  // link6 -> camera optical
    Iso3 center;                  // world -> center optical
    LensTruth lens[3];
    double cx[3], cy[3], dk1[3];
    CameraIntrinsics intr[3];     // at the focus that was set (lens[c].z_m)
    double wall_bow = 0.0015;
    double session_bow = 0.002;
    bool backlash = false;        // stage-2 frames rendered with gravity-loaded backlash + compliance
    std::string approach = "random";  // final approach policy (capture and deployment)
    // The lens family of the truth: Kannala-Brandt (what calico fits by default) or Double Sphere
    // (another family: tests what a model mismatch costs).
    std::string lens_family = "kb";
    double pupil_e1_m = 0.0;  // non-central truth lens (--pupil-mm): entrance pupil z0 = e1 (theta / 90 deg)^2
    void SetFocus(int c, double z) {
        lens[c].z_m = z;
        const double f = lens[c].FocalPx(z);
        if (lens_family == "double_sphere") {
            // xi -0.17, alpha 0.58: a typical 190-200 deg M12 fisheye; fx scaled so that 95 deg lands
            // where the Kannala-Brandt lens puts it (~1075 px from the centre at f = 667).
            const double s = f / 667.0;
            intr[c].model = LensModel::DoubleSphere;
            intr[c].width = 1920;
            intr[c].height = 1200;
            intr[c].params = {527.0 * s * (1 + 0.5 * dk1[c]), 527.0 * s * 1.0006, cx[c], cy[c], -0.17 + dk1[c], 0.58};
        } else {
            intr[c] = Fisheye(f, cx[c], cy[c], dk1[c]);
            if (pupil_e1_m != 0.0) {  // the KB4 lens with its pupil shift (rendered and evaluated)
                intr[c].model = LensModel::KannalaBrandtPupil;
                intr[c].params.push_back(pupil_e1_m);
            }
        }
    }
};

// Optics of camera c as set (render model).
Lens RenderLens(const Truth& t, int c) {
    Lens l;
    l.fov_deg = 200.0;        // image circle ~ the sensor diagonal: the corners reach ~100 deg
    l.vignette = 0.45;
    l.base_blur_px = 0.55;    // lens + Bayer demosaic (luma)
    l.edge_blur_px = 0.7;
    l.focus_m = t.lens[c].FocusDistance(t.lens[c].z_m);
    l.defocus_px = t.lens[c].DefocusPxPerDioptre(t.lens[c].z_m);
    l.decenter_px = 0.15;
    l.decenter_dir = 0.6 + 2.0 * c;
    l.pupil_e1_m = t.pupil_e1_m;
    return l;
}

Truth MakeTruth(double k = 1.0) {
    Truth t;
    // J1 and J6 zeros are known mechanically to < 1 deg (not scaled); J2-J5 can be far off.
    t.dq[0] = {0.6 * kDeg, 5.5 * kDeg * k, -7.0 * kDeg * k, 9.0 * kDeg * k, -6.5 * kDeg * k, -0.8 * kDeg};
    t.dq[1] = {-0.5 * kDeg, -4.5 * kDeg * k, 8.0 * kDeg * k, -11.0 * kDeg * k, 7.5 * kDeg * k, 0.7 * kDeg};
    t.base_cad[0] = MakeIso(Vec3(0.0, 0.3, 0.0), Vec3(0, 0, 0));
    t.base_cad[1] = MakeIso(Vec3(0.0, -0.3, 0.0), Vec3(0, 0, 0));
    t.base[0] = t.base_cad[0];  // the world frame is the left base
    t.base[1] = t.base_cad[1] * MakeIso(k * Vec3(0.006, -0.004, 0.002), k * Vec3(0.2 * kDeg, -0.2 * kDeg, 1.0 * kDeg));
    // Wrist camera: behind / above link6, looking along it (out over the fingers).
    t.mount_cad[0] = t.mount_cad[1] = MakeIso(Vec3(-0.045, 0.0, 0.03), Vec3(0.0, 0.12, 0.0));
    t.mount[0] = t.mount_cad[0] * MakeIso(k * Vec3(0.008, -0.005, 0.004), k * Vec3(2.0 * kDeg, -3.0 * kDeg, 1.5 * kDeg));
    t.mount[1] = t.mount_cad[1] * MakeIso(k * Vec3(-0.006, 0.007, -0.003), k * Vec3(-2.5 * kDeg, 2.0 * kDeg, -1.0 * kDeg));
    t.center = LookAt(Vec3(0.10, 0.0, 0.72), Vec3(0.45, 0.0, 0.0), Vec3(1, 0, 0));
    // ~200 deg fisheyes on 1920 x 1200: 77 deg at the left / right edges, 47 deg top / bottom,
    // ~97 deg in the corners.
    const double f_inf[3] = {666.1, 668.4, 664.9}, cx[3] = {958.2, 963.5, 955.1}, cy[3] = {601.7, 597.9, 604.4};
    const double dk1[3] = {0.0004, -0.0003, 0.0002};
    // As delivered: left focused at infinity, right at ~1 m, center screwed in too far (~17 cm).
    const double z_factory_um[3] = {0.0, 4.0, 24.0};
    for (int c = 0; c < 3; ++c) {
        t.lens[c].f_inf_px = f_inf[c];
        t.lens[c].z_factory_m = z_factory_um[c] * 1e-6;
        t.cx[c] = cx[c];
        t.cy[c] = cy[c];
        t.dk1[c] = dk1[c];
        t.SetFocus(c, t.lens[c].z_factory_m);
    }
    return t;
}

// Realistic per-frame capture settings.
Shot HandheldShot(std::mt19937& rng) {  // stage 1: hand-held, mostly paused, sometimes moving
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Shot s;
    const bool moving = U(rng) < 0.2;
    const double w_sd = (moving ? 30.0 : 3.0) * kDeg, v_sd = moving ? 0.08 : 0.01;
    s.w = Vec3(N(rng), N(rng), N(rng)) * w_sd;
    s.v = Vec3(N(rng), N(rng), N(rng)) * v_sd;
    s.exposure_s = 0.004 + 0.008 * U(rng);
    s.readout_s = 0.0;  // AR0234: global shutter
    s.ae_target = 0.26 + 0.08 * U(rng);
    return s;
}

Shot RobotShot(std::mt19937& rng, bool moving_camera) {  // stage 2: arm settled at the pose
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Shot s;
    if (moving_camera) {
        const double w_sd = (U(rng) < 0.1 ? 2.0 : 0.3) * kDeg;  // 1 in 10: still ringing
        s.w = Vec3(N(rng), N(rng), N(rng)) * w_sd;
        s.v = Vec3(N(rng), N(rng), N(rng)) * 0.002;
    }
    s.exposure_s = 0.004 + 0.006 * U(rng);
    s.readout_s = 0.0;
    s.ae_target = 0.26 + 0.08 * U(rng);
    return s;
}

Iso3 Fk(const UrdfChain& chain, const std::vector<double>& q, const std::vector<double>& dq) {
    std::vector<double> th(q.size());
    for (size_t i = 0; i < q.size(); ++i) th[i] = q[i] + dq[i];
    return chain.ForwardIso(th);
}

// Solid links of one arm for rendering (true joint angles): pedestal, upper arm, forearm, wrist,
// gripper body and two open fingers.
std::vector<Capsule> ArmCapsules(const UrdfChain& chain, const Iso3& base, const std::vector<double>& theta,
                                 double table_z) {
    const std::vector<Iso3> F = chain.FrameChain(theta);  // joint1 .. joint6 child frames
    auto P = [&](int i) { return Vec3(base * F[i].translation()); };
    const Iso3 L6 = base * F[5];
    std::vector<Capsule> c;
    c.push_back({Vec3(base * Vec3(0, 0, table_z)), P(0), 0.06, 0.55, 2});
    c.push_back({P(1), P(2), 0.045, 0.80, 2});
    c.push_back({P(2), P(3), 0.038, 0.80, 2});
    c.push_back({P(3), P(5), 0.034, 0.75, 2});
    // Compact gripper: the wrist camera (4.5 cm off the axis, 3 cm up) looks past it.
    c.push_back({Vec3(L6 * Vec3(0, 0, 0.0)), Vec3(L6 * Vec3(0, 0, 0.045)), 0.024, 0.45, 4});
    for (double y : {-0.028, 0.028}) {
        c.push_back({Vec3(L6 * Vec3(0, y, 0.06)), Vec3(L6 * Vec3(0, y, 0.125)), 0.006, 0.30, 4});
    }
    return c;
}

// Idle arm: turned outward (away from the cell centre), forearm level, gripper down, ~9 cm
// above the table -- out of the other arm's way and out of the boards' line of sight.
const std::vector<double> kPark[2] = {{1.2, 0.3, 1.6, 0.0, 1.24, 0.0}, {-1.2, 0.3, 1.6, 0.0, 1.24, 0.0}};

double SegSegDist(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2) {
    const Vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
    double s = 0, t = 0;
    if (a <= 1e-12 && e <= 1e-12) return r.norm();
    if (a <= 1e-12) {
        t = std::clamp(f / e, 0.0, 1.0);
    } else {
        const double c = d1.dot(r);
        if (e <= 1e-12) {
            s = std::clamp(-c / a, 0.0, 1.0);
        } else {
            const double b = d1.dot(d2), den = a * e - b * b;
            s = den > 1e-12 ? std::clamp((b * f - c * e) / den, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0) {
                t = 0;
                s = std::clamp(-c / a, 0.0, 1.0);
            } else if (t > 1) {
                t = 1;
                s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return ((p1 + d1 * s) - (p2 + d2 * t)).norm();
}

// Distance from a segment to a board rectangle (sampled along the segment).
double SegBoardDist(const Vec3& a, const Vec3& b, const Iso3& T_board, double w, double h) {
    const Iso3 inv = T_board.inverse();
    double best = 1e9;
    for (int i = 0; i <= 12; ++i) {
        const Vec3 p = inv * (a + (b - a) * (i / 12.0));
        const Vec3 c(std::clamp(p.x(), 0.0, w), std::clamp(p.y(), 0.0, h), 0.0);
        best = std::min(best, (p - c).norm());
    }
    return best;
}

// The moving arm must clear the table, the static boards and the other arm.
bool ArmPoseIsFree(const std::vector<Capsule>& moving, const std::vector<Capsule>& other,
                   const std::vector<std::pair<Iso3, Vec2>>& boards, double table_z) {
    for (size_t i = 1; i < moving.size(); ++i) {  // [0] is the pedestal, on the table by design
        const Capsule& c = moving[i];
        if (std::min(c.a.z(), c.b.z()) - c.r < table_z + 0.015) return false;
        for (const auto& b : boards) {
            if (SegBoardDist(c.a, c.b, b.first, b.second.x(), b.second.y()) < c.r + 0.015) return false;
        }
        for (const Capsule& o : other) {
            if (SegSegDist(c.a, c.b, o.a, o.b) < c.r + o.r + 0.02) return false;
        }
    }
    return true;
}

std::vector<double> Theta(const std::vector<double>& q, const std::vector<double>& dq) {
    std::vector<double> th(q.size());
    for (size_t i = 0; i < q.size(); ++i) th[i] = q[i] + dq[i];
    return th;
}

// Drive train of each joint: motor-side encoder -> reducer with backlash and torsional
// compliance -> link. PiPER does not publish its reducers; these are typical small planetary
// values (3-20 arcmin backlash). Masses / COMs: the AgileX piper_description URDF, mapped onto
// this chain (gripper 0.45 kg + wrist camera 0.06 kg).
struct Drive {
    double backlash_deg;   // total play at the output
    double stiffness;      // N m / rad at the output
    double friction;       // N m: below this gravity torque the output stays where the motor left it
};
const Drive kDrive[6] = {{0.20, 1500, 0.30}, {0.25, 1500, 0.30}, {0.25, 1200, 0.25},
                         {0.35, 500, 0.10},  {0.35, 500, 0.10},  {0.35, 300, 0.05}};

struct LinkMass {
    double m;
    Vec3 com;  // in the child frame of joint i (FrameChain index)
};
const std::vector<LinkMass> kLinkMass[6] = {
    {{0.71, Vec3(0, 0, 0)}},                                          // link1
    {{1.17, Vec3(0, 0, 0.14)}},                                       // link2 (upper arm)
    {{0.50, Vec3(0.01, 0, 0.125)}},                                   // link3 (forearm)
    {{0.38, Vec3(0, 0, 0)}},                                          // link4
    {{0.383, Vec3(0, 0, 0.05)}},                                      // link5
    {{0.457, Vec3(0, 0, 0.032)}, {0.06, Vec3(-0.045, 0.0, 0.03)}},    // link6: gripper + wrist camera
};

// Gravity torque about each joint axis (N m), output-side angles theta, base in the world.
std::vector<double> GravityTorques(const UrdfChain& chain6, const Iso3& base, const std::vector<double>& theta) {
    const std::vector<Iso3> F = chain6.FrameChain(theta);  // child frames of joint1..joint6
    std::vector<Vec3> c;
    std::vector<double> m;
    for (int i = 0; i < 6; ++i) {
        for (const LinkMass& lm : kLinkMass[i]) {
            c.push_back(base * F[i] * lm.com);
            m.push_back(lm.m);
        }
    }
    const Vec3 g(0, 0, -9.81);
    std::vector<double> tau(6, 0.0);
    size_t k0 = 0;
    for (int j = 0; j < 6; ++j) {
        const Iso3 J = base * F[j];
        const Vec3 axis = J.linear() * chain6.joints()[j].axis, p = J.translation();
        Vec3 sum = Vec3::Zero();
        for (size_t k = k0; k < c.size(); ++k) sum += (c[k] - p).cross(m[k] * g);
        tau[j] = axis.dot(sum);
        k0 += kLinkMass[j].size();
    }
    return tau;
}

// Output angle minus motor-side (encoder) angle for each joint, at rest: a joint loaded by
// gravity sits on the gravity-side tooth flanks and twists by tau / k; a joint gravity barely
// loads stays where the motor pushed it last (the output trails the motor's last direction).
std::vector<double> DriveOffset(const UrdfChain& chain6, const Iso3& base, const std::vector<double>& q_motor,
                                const std::vector<double>& dq, const std::vector<double>& last_dir, bool on) {
    std::vector<double> off(6, 0.0);
    if (!on) return off;
    const std::vector<double> tau = GravityTorques(chain6, base, Theta(q_motor, dq));
    for (int j = 0; j < 6; ++j) {
        const double half = 0.5 * kDrive[j].backlash_deg * kDeg;
        const double play = std::fabs(tau[j]) > kDrive[j].friction ? (tau[j] > 0 ? half : -half)
                                                                  : (last_dir[j] >= 0 ? -half : half);
        off[j] = play + tau[j] / kDrive[j].stiffness;
    }
    return off;
}

// Direction of each joint's final motion before a capture (sign of the last motor velocity).
//   random: as the move happened; unidirectional: always +; against_gravity: lift into the pose
//   (-sign(tau)), + where gravity barely loads the joint.
std::vector<double> ApproachDirections(const std::string& policy, const UrdfChain& chain6, const Iso3& base,
                                       const std::vector<double>& q, const std::vector<double>& dq,
                                       const std::vector<double>& move_vel) {
    std::vector<double> d(6, 1.0);
    if (policy == "random") {
        for (int j = 0; j < 6; ++j) d[j] = move_vel[j] >= 0 ? 1.0 : -1.0;
    } else if (policy == "against_gravity") {
        const std::vector<double> tau = GravityTorques(chain6, base, Theta(q, dq));
        for (int j = 0; j < 6; ++j) d[j] = std::fabs(tau[j]) >= 0.05 ? (tau[j] > 0 ? -1.0 : 1.0) : 1.0;
    }
    return d;
}

std::vector<double> Add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> s(a.size());
    for (size_t i = 0; i < a.size(); ++i) s[i] = a[i] + b[i];
    return s;
}

std::vector<double> RandomPose(int arm, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> q(6);
    q[0] = (arm == 0 ? -0.55 : 0.55) + 0.5 * U(rng);
    q[1] = 0.62 + 0.35 * U(rng);
    q[2] = 1.05 + 0.40 * U(rng);
    q[3] = 0.6 * U(rng);
    q[4] = M_PI - q[1] - q[2] - 0.30 + 0.45 * U(rng);
    q[5] = 1.2 * U(rng);
    return q;
}

std::string Q(const std::vector<double>& q) {
    std::ostringstream o;
    o << "[" << std::setprecision(17);
    for (size_t j = 0; j < q.size(); ++j) o << (j ? ", " : "") << q[j];
    o << "]";
    return o.str();
}

std::string BoardLine(const BoardSpec& s, bool fit_shape, bool level, const std::string& extra = "") {
    std::ostringstream o;
    o << "  - {name: " << s.name << ", rows: " << s.rows << ", cols: " << s.cols << ", tag_size: " << s.tag_size
      << ", tag_spacing: " << s.tag_spacing << ", first_id: " << s.first_id << (level ? ", level: true" : "")
      << (fit_shape ? ", fit_shape: true" : "") << extra << "}\n";
    return o.str();
}

// ---------------------------------------------------------------------------------------- focus
// Camera on a tripod, board on a stand facing it, centred, at distance d (small tilt).
Iso3 BoardFacing(const BoardSpec& s, double d, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    // Camera at the origin looking along +x (world z up); board normal back toward the camera.
    Iso3 T = Iso3::Identity();
    T.linear() << 0, 0, -1,  // board x = world -y, board y = world z, board normal = -x
        -1, 0, 0,            //
        0, 1, 0;
    T.linear() = T.linear() * Eigen::AngleAxisd(0.12 * U(rng), Vec3::UnitX()).toRotationMatrix() *
                 Eigen::AngleAxisd(0.12 * U(rng), Vec3::UnitY()).toRotationMatrix();
    T.translation() = Vec3(d, 0.01 * U(rng), 0.01 * U(rng)) - T.linear() * Vec3(s.Width() / 2, s.Height() / 2, 0);
    return T;
}

const Iso3 kTripod = LookAt(Vec3::Zero(), Vec3(1, 0, 0), Vec3(0, 0, -1));

std::string FocusConfig(int c, const std::vector<BoardSpec>& boards) {
    const Range r = OperatingRange(c);
    std::ostringstream o;
    o << "# Focus " << kCams[c] << ": set the lens for " << r.near_m << " - " << r.far_m << " m, then lock it\n"
      << "boards:\n";
    for (const BoardSpec& b : boards) o << BoardLine(b, false, false);
    o << "cameras:\n  - {name: " << kCams[c] << ", model: equidistant, width: 1920, height: 1200, f_init: 650}\n"
      << "detection: {min_tags_per_board: 2}\n"
      << "focus: {near_m: " << r.near_m << ", far_m: " << r.far_m << ", max_off_axis_deg: 35}\n";
    return o.str();
}

// Step 0: the operator sweeps the focus ring through focus with the board at the target
// distance; calico-focus reads the blur off every frame. ring.yaml (truth) maps frames to ring angles.
void GenerateFocusSweep(const std::string& dir, int c, Truth t, apriltag_family_t* fam, unsigned seed) {
    const std::string fdir = dir + "/focus_" + kCams[c];
    MkDir(fdir);
    MkDir(fdir + "/sweep");
    std::vector<BoardSpec> specs = SessionBoards(c);
    if (c == 2) specs.push_back(FarFocusBoard());
    std::ofstream(fdir + "/focus.yaml") << FocusConfig(c, specs);
    std::vector<BoardTexture> board(1);
    board[0].spec = specs[0];
    board[0].Render(fam);
    const Range r = OperatingRange(c);
    const double target = FocusTarget(r.near_m, r.far_m);
    LensTruth& L = t.lens[c];
    const double a_star = (L.ZFor(target) - L.z_factory_m) / LensTruth::kThreadPerDeg;
    const double dir_sign = a_star >= 0 ? 1.0 : -1.0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::ofstream ring(fdir + "/sweep/ring.yaml");
    ring << "# truth: focus ring angle (deg from the delivered position) of each sweep frame\n";
    int n = 0;
    const int steps = static_cast<int>(std::ceil(std::fabs(a_star))) + 8;
    for (int k = 0; k <= steps; ++k) {
        const double a = dir_sign * k;
        t.SetFocus(c, L.z_factory_m + a * LensTruth::kThreadPerDeg);
        Renderer rend{t.intr[c]};
        rend.lens = RenderLens(t, c);
        rend.Init();
        for (int rep = 0; rep < 2; ++rep) {
            const double d = target * (1.0 + 0.03 * U(rng));
            const std::vector<Iso3> pose = {BoardFacing(board[0].spec, d, rng)};
            Shot s;
            s.exposure_s = 0.010;
            char name[64];
            std::snprintf(name, sizeof(name), "%s_%03d.png", kCams[c], n++);
            cv::imwrite(fdir + "/sweep/" + name, rend.RenderReal(kTripod, board, pose, SceneExtras(), Lighting(), s, rng));
            ring << name << ": " << std::setprecision(6) << a << "\n";
        }
    }
    std::cout << "focus " << kCams[c] << ": " << n << " sweep frames, target " << std::setprecision(3) << target
              << " m (ring " << a_star << " deg from delivered)\n";
}

// After calico-focus picked the sharpest sweep frame: lock the ring there.
double LockedRing(const std::string& dir, int c) {
    const std::string fdir = dir + "/focus_" + kCams[c];
    const YAML::Node rep = YAML::LoadFile(fdir + "/out/focus_report.yaml");
    const std::string best = rep["best_frame"].as<std::string>();
    if (best.empty()) throw std::runtime_error("focus " + std::string(kCams[c]) + ": no sharpest frame in the report");
    return YAML::LoadFile(fdir + "/sweep/ring.yaml")[best].as<double>();
}

// Near / far check with the lens locked: both ends of the range should be equally blurred.
void GenerateFocusCheck(const std::string& dir, int c, const Truth& t, apriltag_family_t* fam, unsigned seed) {
    const std::string fdir = dir + "/focus_" + kCams[c];
    MkDir(fdir + "/check");
    const Range r = OperatingRange(c);
    std::vector<BoardTexture> near_b(1), far_b(1);
    near_b[0].spec = SessionBoards(c)[0];
    far_b[0].spec = c == 2 ? FarFocusBoard() : SessionBoards(c)[0];
    near_b[0].Render(fam);
    far_b[0].Render(fam);
    Renderer rend{t.intr[c]};
    rend.lens = RenderLens(t, c);
    rend.Init();
    std::mt19937 rng(seed);
    int n = 0;
    for (int end = 0; end < 2; ++end) {
        const std::vector<BoardTexture>& b = end == 0 ? near_b : far_b;
        for (int rep = 0; rep < 4; ++rep) {
            const double d = (end == 0 ? r.near_m * 1.02 : r.far_m * 0.98) * (1.0 + 0.02 * (rep - 1.5) / 1.5);
            Shot s;
            s.exposure_s = 0.010;
            char name[64];
            std::snprintf(name, sizeof(name), "%s_%s_%d.png", kCams[c], end == 0 ? "near" : "far", rep);
            cv::imwrite(fdir + "/check/" + name,
                        rend.RenderReal(kTripod, b, {BoardFacing(b[0].spec, d, rng)}, SceneExtras(), Lighting(), s, rng));
            ++n;
        }
    }
}

// --------------------------------------------------------------------------------------- stage 1
void GenerateStage1(const std::string& dir, int ci, const Truth& t, apriltag_family_t* fam, unsigned seed,
                    const SimOptions& opt, const std::string& suffix = "") {
    const std::string cam = kCams[ci];
    const std::string sdir = dir + "/stage1" + suffix + "_" + cam;
    MkDir(sdir);
    MkDir(sdir + "/images");
    // Two 6x8 AprilGrids on a table, disjoint ids; board b is laminated (glare) and bowed.
    const std::vector<BoardSpec> specs = SessionBoards(ci);
    const double k = specs[0].tag_size / 0.024;  // the center camera's session is scaled up
    std::vector<BoardTexture> boards(2);
    boards[0].spec = specs[0];
    const std::vector<Iso3> place = SessionPlacement(specs);
    boards[0].T_world_board = place[0];
    boards[1].spec = specs[1];
    boards[1].T_world_board = place[1];
    boards[1].bow_x = t.session_bow;
    boards[1].gloss = 0.5;
    for (BoardTexture& b : boards) b.Render(fam);
    Renderer rend{t.intr[ci]};
    rend.lens = RenderLens(t, ci);
    rend.Init();
    SceneExtras table;
    table.table = true;
    table.table_z = -0.0006;  // the prints lie on the table
    table.table_min = k * Vec2(-0.5, -0.6);
    table.table_max = k * Vec2(0.5, 0.6);
    Lighting light;
    light.lamp = k * Vec3(0.25, -0.35, 0.9);
    light.range_m = 0.9 * k;
    std::vector<Iso3> pose_list;
    for (const BoardTexture& b : boards) pose_list.push_back(b.T_world_board);
    std::ofstream pose_log(sdir + "/camera_poses.txt");  // truth, for the video only
    pose_log << std::setprecision(10);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::ofstream frames(sdir + "/frames.yaml");
    frames << "frames:\n";
    int kept = 0;
    const Range range = OperatingRange(ci);
    const double h_lo = ci == 2 ? 0.30 : 0.13, h_hi = ci == 2 ? 0.80 : 0.30;
    for (int tries = 0; tries < 6000 && kept < opt.stage1_frames; ++tries) {
        // Hand-held over the boards at working distance, rolled freely; every other frame
        // tilted far off-axis so the boards reach the edge of the 190 deg image circle.
        const Vec3 target(0.1 * k * U(rng), 0.2 * k * U(rng), 0.0);
        const double h = h_lo + (h_hi - h_lo) * (0.5 + 0.5 * U(rng));
        const Vec3 eye = target + Vec3(0.12 * k * U(rng), 0.12 * k * U(rng), h);
        Iso3 T_wc = LookAt(eye, target, Vec3(1, 0, 0));
        T_wc.linear() = T_wc.linear() * Eigen::AngleAxisd(M_PI * U(rng), Vec3::UnitZ()).toRotationMatrix();
        if (opt.stage1_edges && kept % 2) {
            T_wc.linear() = T_wc.linear() *
                            Eigen::AngleAxisd(1.25 * U(rng), Vec3(U(rng), U(rng), 0).normalized()).toRotationMatrix();
        }
        int visible = 0;
        for (const BoardTexture& b : boards) visible += BoardInView(T_wc, b, t.intr[ci], opt.stage1_edges ? 92.0 : 45.0);
        if (visible < 1) continue;
        char name[64];
        std::snprintf(name, sizeof(name), "images/%s_%05d.png", cam.c_str(), kept++);
        const Shot shot = HandheldShot(rng);
        cv::imwrite(sdir + "/" + name, rend.RenderReal(T_wc, boards, pose_list, table, light, shot, rng));
        frames << "  - {camera: " << cam << ", image: " << name << "}\n";
        pose_log << name;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) pose_log << " " << T_wc.matrix()(r, c);
        }
        pose_log << " " << shot.w.norm() / kDeg << "\n";
    }
    (void)range;
    std::ofstream cfg(sdir + "/stage1.yaml");
    cfg << "# Stage 1: " << cam << " intrinsics, hand-held session over two 6x8 AprilGrids (off the robot,\n"
        << "# lens focus already locked by the focus step)\n"
        << "frames: frames.yaml\nreference_board: a\n"
        << "boards:\n" << BoardLine(boards[0].spec, true, false) << BoardLine(boards[1].spec, true, false)
        << "cameras:\n  - {name: " << cam << ", mode: handheld, model: " << opt.lens_model << ", width: 1920, height: 1200, "
        << "f_init: 650, intrinsics_prepass: true, optimize_intrinsics: true}\n"
        << "detection: {cache: detections.jsonl, min_tags_per_board: 2}\n"
        << "intrinsics_gates: {max_holdout_rms_px: 0.5, min_theta_deg: 85, min_coverage: 0.8, "
        << "max_curve_sd_px: 0.3, max_mean_radial_px: 0.15}\n"
        << "solver: {huber_px: 1.0, holdout_every: 5, bootstrap: 10, threads: 16}\n";
    std::cout << "stage 1 " << cam << ": " << kept << " frames\n";
}

// --------------------------------------------------------------------------------------- stage 2
// Static cell boards. 190 deg lenses spread few pixels over each degree (f ~ 330 px), so the
// center camera (0.7-0.9 m away) needs ~55 mm tags to resolve them.
std::vector<BoardTexture> CellBoards(const Truth& t, apriltag_family_t* fam) {
    std::vector<BoardTexture> b(4);
    b[0].spec = {"table", "tag36h11", 5, 7, 0.055, 0.3, 0, true, 0.3};
    b[0].T_world_board = Place(b[0].spec, Vec3(0.42, 0.0, -0.03), Vec3(0, 0, 0.05));
    b[1].spec = {"wedgeL", "tag36h11", 3, 4, 0.055, 0.3, 48, false, 0.3};
    b[1].T_world_board = Place(b[1].spec, Vec3(0.40, 0.36, 0.03), Vec3(0.5, 0.0, -0.3));
    b[1].gloss = 0.5;  // laminated
    b[2].spec = {"wedgeR", "tag36h11", 3, 4, 0.055, 0.3, 60, false, 0.3};
    b[2].T_world_board = Place(b[2].spec, Vec3(0.40, -0.36, 0.03), Vec3(-0.5, 0.0, 0.3));
    b[3].spec = {"wall", "tag36h11", 4, 6, 0.060, 0.3, 72, false, 0.3};
    b[3].T_world_board = Place(b[3].spec, Vec3(0.78, 0.0, 0.17), Vec3(0.0, -1.2, 0.0));
    b[3].bow_x = t.wall_bow;
    for (BoardTexture& x : b) x.Render(fam);
    return b;
}

// What happens around one capture: the arm arrives at a pose and its structure rings
// (decaying oscillation); joint states reach the recorder kLag late. The recorder waits until
// the *reported* joints have been stationary for kDwell, then grabs the image and the joints.
struct Arrival {
    std::vector<double> target, approach_vel, amp, phase;
    double tau = 0.12, freq = 6.0;
    std::vector<double> Q(double t) const {  // true joints, t from arrival
        std::vector<double> q(target.size());
        for (size_t j = 0; j < q.size(); ++j) {
            q[j] = target[j] + (t >= 0 ? amp[j] * std::exp(-t / tau) * std::cos(2 * M_PI * freq * t + phase[j])
                                       : amp[j] * std::cos(phase[j]) + approach_vel[j] * t);
        }
        return q;
    }
};
constexpr double kLag = 0.08;    // joint-state latency (s)
constexpr double kDwell = 0.40;  // recorder: stationary time required before a capture
constexpr double kStillTol = 0.01 * kDeg;

Arrival MakeArrival(const std::vector<double>& from, const std::vector<double>& to, std::mt19937& rng) {
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 2 * M_PI);
    Arrival a;
    a.target = to;
    double norm = 0;
    for (size_t j = 0; j < to.size(); ++j) norm += (to[j] - from[j]) * (to[j] - from[j]);
    norm = std::max(1e-9, std::sqrt(norm));
    for (size_t j = 0; j < to.size(); ++j) {
        a.approach_vel.push_back(0.35 * (to[j] - from[j]) / norm);  // ~20 deg/s at arrival
        a.amp.push_back(0.25 * kDeg * N(rng));
        a.phase.push_back(U(rng));
    }
    return a;
}

// Capture time per the recorder's rule (or a glitched early trigger), and the still time it logs.
void CaptureTime(const Arrival& arr, bool glitch, double& t_c, double& still_s) {
    if (glitch) {
        t_c = 0.03;
        still_s = 0.0;
        return;
    }
    double still_since = -1;
    for (double t = 0.0; t < 5.0; t += 0.01) {
        const std::vector<double> a = arr.Q(t - kLag), b = arr.Q(t - kLag - 0.1);
        double d = 0;
        for (size_t j = 0; j < a.size(); ++j) d = std::max(d, std::fabs(a[j] - b[j]));
        if (d < kStillTol) {
            if (still_since < 0) still_since = t;
            if (t - still_since >= kDwell) {
                t_c = t;
                still_s = t - still_since;
                return;
            }
        } else {
            still_since = -1;
        }
    }
    t_c = 5.0;
    still_s = 5.0;
}

// Camera body velocity (camera frame) at time t from the joint trajectory.
void CameraTwist(const std::function<Iso3(double)>& T_of_t, double t, Vec3& v, Vec3& w) {
    const double h = 1e-3;
    const Iso3 A = T_of_t(t - h), B = T_of_t(t + h);
    const Iso3 D = A.inverse() * B;
    const Eigen::AngleAxisd aa(D.linear());
    w = aa.axis() * aa.angle() / (2 * h);
    v = D.translation() / (2 * h);
}

void GenerateStage2(const std::string& dir, const Truth& t, apriltag_family_t* fam, unsigned seed,
                    const SimOptions& opt) {
    const std::string sdir = dir + "/stage2";
    MkDir(sdir);
    MkDir(sdir + "/images");
    MkDir(dir + "/robot");
    std::vector<BoardTexture> boards = CellBoards(t, fam);
    std::vector<Iso3> poses;
    for (const BoardTexture& b : boards) poses.push_back(b.T_world_board);
    std::vector<UrdfChain> chains, ee_chains;
    for (int a = 0; a < 2; ++a) {
        const std::string s = kSides[a];
        std::ofstream(dir + "/robot/" + s + ".urdf") << ArmUrdf(s + "_");
        chains.push_back(UrdfChain::FromString(ArmUrdf(s + "_"), s + "_base_link", s + "_link6", Joints(s)));
        ee_chains.push_back(UrdfChain::FromString(ArmUrdf(s + "_"), s + "_base_link", s + "_end_effector", Joints(s)));
    }
    auto scene = [&](const std::vector<double>& q_left, const std::vector<double>& q_right) {
        SceneExtras x;
        x.table = true;
        x.table_z = -0.031;
        const std::vector<double>* q[2] = {&q_left, &q_right};
        for (int a = 0; a < 2; ++a) {
            const std::vector<Capsule> c = ArmCapsules(ee_chains[a], t.base[a], Theta(*q[a], t.dq[a]), x.table_z);
            x.capsules.insert(x.capsules.end(), c.begin(), c.end());
        }
        return x;
    };
    std::vector<std::pair<Iso3, Vec2>> bd;
    for (const BoardTexture& b : boards) bd.push_back({b.T_world_board, Vec2(b.spec.Width(), b.spec.Height())});
    std::mt19937 rng(seed);
    std::normal_distribution<double> enc(0.0, opt.joint_noise_deg * kDeg);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    auto recorded = [&](std::vector<double> q) {  // what the encoders report
        if (opt.joint_noise_deg > 0) {
            for (double& v : q) v += enc(rng);
        }
        return q;
    };
    const Lighting light;
    Renderer center{t.intr[2]};
    center.lens = RenderLens(t, 2);
    center.Init();
    std::ofstream frames(sdir + "/frames.yaml");
    frames << "frames:\n";
    std::ofstream capture_log(sdir + "/capture_log.txt");  // truth: capture timing per pose
    capture_log << "# image t_capture_s still_s glitch max_joint_error_deg drive_deg(link - motor, J1..J6)\n";
    int n_frames = 0, n_glitch = 0;
    for (int a = 0; a < 2; ++a) {
        Renderer wrist{t.intr[a]};
        wrist.lens = RenderLens(t, a);
        wrist.Init();
        const std::string s = kSides[a];
        int kept = 0;
        int collisions = 0;
        std::vector<double> prev = kPark[a];
        for (int tries = 0; tries < 80 * opt.route_poses && kept < opt.route_poses; ++tries) {
            const std::vector<double> q = RandomPose(a, rng);
            {
                const std::vector<Capsule> mv = ArmCapsules(ee_chains[a], t.base[a], Theta(q, t.dq[a]), -0.031);
                const std::vector<Capsule> ot =
                    ArmCapsules(ee_chains[1 - a], t.base[1 - a], Theta(kPark[1 - a], t.dq[1 - a]), -0.031);
                if (!ArmPoseIsFree(mv, ot, bd, -0.031)) {
                    ++collisions;
                    continue;
                }
            }
            const Iso3 T_target = t.base[a] * Fk(chains[a], q, t.dq[a]) * t.mount[a];
            int visible = 0;
            for (const BoardTexture& b : boards) visible += BoardInView(T_target, b, t.intr[a], 88.0);
            if (visible == 0) continue;
            // Arrive, ring, wait for the recorder's stationarity rule (2 in 30 captures glitch).
            Arrival arr = MakeArrival(prev, q, rng);
            const std::vector<double> d_dir = ApproachDirections(opt.approach, chains[a], t.base[a], q, t.dq[a], arr.approach_vel);
            for (size_t j = 0; j < 6; ++j) arr.approach_vel[j] = d_dir[j] * std::max(std::fabs(arr.approach_vel[j]), 1e-3);
            prev = q;
            const bool glitch = kept % 15 == 7;
            double t_c, still_s;
            CaptureTime(arr, glitch, t_c, still_s);
            n_glitch += glitch;
            // Motor side (encoders) vs links: backlash + compliance under gravity.
            const std::vector<double> q_true = arr.Q(t_c), q_rep = arr.Q(t_c - kLag);
            const std::vector<double> off = DriveOffset(chains[a], t.base[a], q_true, t.dq[a], arr.approach_vel, opt.backlash);
            const std::vector<double> park_off =
                DriveOffset(chains[1 - a], t.base[1 - a], kPark[1 - a], t.dq[1 - a], std::vector<double>(6, 1.0), opt.backlash);
            double max_err = 0;
            for (size_t j = 0; j < q.size(); ++j) max_err = std::max(max_err, std::fabs(q_true[j] - q_rep[j]) / kDeg);
            auto cam_at = [&](double tt) { return t.base[a] * Fk(chains[a], Add(arr.Q(tt), off), t.dq[a]) * t.mount[a]; };
            Shot shot = RobotShot(rng, false);
            CameraTwist(cam_at, t_c, shot.v, shot.w);
            const SceneExtras cell = scene(a == 0 ? Add(q_true, off) : Add(kPark[0], park_off),
                                           a == 1 ? Add(q_true, off) : Add(kPark[1], park_off));
            char name[64];
            std::snprintf(name, sizeof(name), "images/%s_%03d.png", s.c_str(), kept);
            cv::imwrite(sdir + "/" + name, wrist.RenderReal(cam_at(t_c), boards, poses, cell, light, shot, rng));
            const std::vector<double> q_rec = recorded(q_rep);
            std::vector<double> travel(6);  // what the recorder logs: signed final travel (deg)
            for (int j = 0; j < 6; ++j) travel[j] = (d_dir[j] >= 0 ? 2.0 : -2.0);
            const std::string joints = "joints: {" + s + ": " + Q(q_rec) + ", " + kSides[1 - a] + ": " +
                                       Q(recorded(kPark[1 - a])) + "}, travel_deg: {" + s + ": " + Q(travel) + "}";
            std::ostringstream still;
            still << std::setprecision(3) << still_s;
            frames << "  - {camera: " << s << ", image: " << name << ", still_s: " << still.str() << ", " << joints << "}\n";
            capture_log << name << " " << t_c << " " << still_s << " " << glitch << " " << max_err << " drive_deg";
            for (double o : off) capture_log << " " << o / kDeg;
            capture_log << "\n";
            ++n_frames;
            if (kept % 3 == 0) {  // the center camera: static, sees the boards (arms occlude some)
                std::snprintf(name, sizeof(name), "images/center_%s_%03d.png", s.c_str(), kept);
                cv::imwrite(sdir + "/" + name,
                            center.RenderReal(t.center, boards, poses, cell, light, RobotShot(rng, false), rng));
                frames << "  - {camera: center, image: " << name << ", still_s: " << still.str() << ", " << joints
                       << "}\n";
                ++n_frames;
            }
            ++kept;
        }
        std::cout << "stage 2 " << s << ": " << kept << " poses (" << collisions << " colliding samples rejected)\n";
    }
    // Stage 2 config: the nominal system (what calibration starts from) + stage 1 intrinsics.
    std::ofstream cfg(sdir + "/stage2.yaml");
    cfg << "# Stage 2: robot calibration, intrinsics from stage 1 held constant. Nominal (CAD) start.\n"
        << "world_frame: base\nframes: frames.yaml\n"
        << "# joint states lag the images: only frames taken after >= 0.3 s stationary are used\n"
        << "capture: {min_still_s: 0.3}\narms:\n";
    for (int a = 0; a < 2; ++a) {
        const std::string s = kSides[a];
        cfg << "  - name: " << s << "\n    urdf: ../robot/" << s << ".urdf\n    base_link: " << s << "_base_link\n"
            << "    tip_link: " << s << "_link6\n    ee_link: " << s << "_end_effector\n    joints: [";
        const std::vector<std::string> j = Joints(s);
        for (size_t k = 0; k < j.size(); ++k) cfg << (k ? ", " : "") << j[k];
        cfg << "]\n    T_world_base: " << PoseYaml(t.base_cad[a]) << "\n    base_fixed: " << (a == 0 ? "true" : "false")
            << "\n    # No marker on the robot: J1 (base yaw) and J6 (camera roll) zeros are not observable from\n"
            << "    # the images. They are known mechanically to < 1 deg: prior 0 +- 0.5 deg. J2-J5: free.\n"
            << "    delta_q_free: [true, true, true, true, true, true]\n"
            << "    delta_q_sigma_deg: [0.5, 20, 20, 20, 20, 0.5]\n"
            << "    camera: " << s << "\n    T_tip_cam_init: " << PoseYaml(t.mount_cad[a]) << "\n";
        if (opt.joint_noise_deg > 0 || opt.model_joint_noise_deg > 0) {
            cfg << "    joint_noise_deg: " << std::max(opt.joint_noise_deg, opt.model_joint_noise_deg)
                << "   # encoder noise + unmodelled drive play (errors-in-variables)\n";
        }
    }
    if (opt.drive_model) {
        cfg << "# drive train between encoders and links (compliance + backlash under gravity)\n"
            << "drive: {model: true, approach: " << (opt.approach == "against_gravity" ? "against_gravity" : "logged")
            << ", shared: true}\n";
    }
    cfg << "cameras:\n";
    for (int c = 0; c < 3; ++c) {
        cfg << "  - {name: " << kCams[c] << ", camera_info: ../stage1_" << kCams[c] << "/out/" << kCams[c]
            << ".camera_info.yaml" << (c == 2 ? ", static: true" : "") << "}\n";
    }
    cfg << "boards:\n";
    for (int i = 0; i < 4; ++i) cfg << BoardLine(boards[i].spec, i == 3, i == 0);
    cfg << "detection: {cache: detections.jsonl, min_tags_per_board: 2}\n"
        << "solver: {huber_px: 1.0, holdout_every: 5, threads: 16}\n";
    std::cout << "stage 2: " << n_frames << " frames (" << n_glitch << " captures triggered before the arm settled)\n";
}

void WriteTruth(const std::string& path, const Truth& t) {
    std::ofstream o(path);
    o << std::setprecision(17);
    for (int a = 0; a < 2; ++a) {
        o << kSides[a] << ":\n  dq: " << Q(t.dq[a]) << "\n  base: " << PoseYaml(t.base[a]) << "\n  mount: "
          << PoseYaml(t.mount[a]) << "\n  base_cad: " << PoseYaml(t.base_cad[a]) << "\n  mount_cad: "
          << PoseYaml(t.mount_cad[a]) << "\n";
    }
    o << "center: " << PoseYaml(t.center) << "\n";
    o << "backlash: " << (t.backlash ? "true" : "false") << "\napproach: " << t.approach << "\n";
    o << "wall_bow: " << t.wall_bow << "\nsession_bow: " << t.session_bow << "\nintrinsics:\n";
    for (int c = 0; c < 3; ++c) o << "  " << kCams[c] << ": " << Q(t.intr[c].params) << "\n";
    o << "lens_family: " << t.lens_family << "\n";
    o << "pupil_e1_m: " << t.pupil_e1_m << "\n";
    o << "lens_z_m: {";  // focus ring as locked (lens travel beyond infinity focus)
    for (int c = 0; c < 3; ++c) o << (c ? ", " : "") << kCams[c] << ": " << t.lens[c].z_m;
    o << "}\nfocus_m: {";
    for (int c = 0; c < 3; ++c) o << (c ? ", " : "") << kCams[c] << ": " << t.lens[c].FocusDistance(t.lens[c].z_m);
    o << "}\n";
}

// -------------------------------------------------------------------------------------- evaluate
Iso3 ReadXyzRpy(const YAML::Node& n) {
    if (n["matrix"]) {
        Iso3 T = Iso3::Identity();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) T.linear()(r, c) = n["matrix"][r][c].as<double>();
            T.translation()[r] = n["matrix"][r][3].as<double>();
        }
        return T;
    }
    return MakeIso(Vec3(n["xyz"][0].as<double>(), n["xyz"][1].as<double>(), n["xyz"][2].as<double>()),
                   Vec3(n["rpy"][0].as<double>(), n["rpy"][1].as<double>(), n["rpy"][2].as<double>()));
}

CameraIntrinsics ReadCameraInfo(const std::string& path) { return LoadCameraInfo(path, "cam"); }

struct Stat {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double max() const { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }
    double median() const {
        if (v.empty()) return 0;
        std::vector<double> s = v;
        std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
        return s[s.size() / 2];
    }
    double p95() const {
        if (v.empty()) return 0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s[std::min(s.size() - 1, static_cast<size_t>(0.95 * s.size()))];
    }
    double rms() const {
        double q = 0;
        for (double x : v) q += x * x;
        return v.empty() ? 0 : std::sqrt(q / v.size());
    }
};

Truth ReadTruth(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    Truth t = MakeTruth();
    for (int a = 0; a < 2; ++a) {
        const YAML::Node s = n[kSides[a]];
        t.dq[a].clear();
        for (const auto& v : s["dq"]) t.dq[a].push_back(v.as<double>());
        t.base[a] = ReadXyzRpy(s["base"]);
        t.mount[a] = ReadXyzRpy(s["mount"]);
        t.base_cad[a] = ReadXyzRpy(s["base_cad"]);
        t.mount_cad[a] = ReadXyzRpy(s["mount_cad"]);
    }
    t.center = ReadXyzRpy(n["center"]);
    t.wall_bow = n["wall_bow"].as<double>();
    t.backlash = n["backlash"] && n["backlash"].as<bool>();
    if (n["lens_family"]) t.lens_family = n["lens_family"].as<std::string>();
    if (n["pupil_e1_m"]) t.pupil_e1_m = n["pupil_e1_m"].as<double>();
    if (n["approach"]) t.approach = n["approach"].as<std::string>();
    for (int c = 0; c < 3; ++c) {
        t.lens[c].z_m = n["lens_z_m"][kCams[c]].as<double>();
        t.intr[c].params.clear();
        for (const auto& v : n["intrinsics"][kCams[c]]) t.intr[c].params.push_back(v.as<double>());
        if (n["lens_family"] && n["lens_family"].as<std::string>() == "double_sphere") {
            t.intr[c].model = LensModel::DoubleSphere;
        }
        if (t.pupil_e1_m != 0.0 && t.intr[c].params.size() == 9) t.intr[c].model = LensModel::KannalaBrandtPupil;
    }
    return t;
}

// The calibrated drive model (robot_calib.yaml arms.<arm>.drive), and the link angles it predicts.
struct DriveCal {
    bool on = false;
    std::string approach;
    std::vector<double> c, h, tau_ref, s_ref, unloaded;
};

DriveCal ReadDriveCal(const YAML::Node& arm) {
    DriveCal d;
    if (const YAML::Node n = arm["drive"]) {
        d.on = true;
        d.approach = n["approach"].as<std::string>();
        for (const auto& v : n["compliance_rad_per_nm"]) d.c.push_back(v.as<double>());
        for (const auto& v : n["half_backlash_rad"]) d.h.push_back(v.as<double>());
        for (const auto& v : n["tau_ref_nm"]) d.tau_ref.push_back(v.as<double>());
        for (const auto& v : n["s_ref"]) d.s_ref.push_back(v.as<double>());
        for (const auto& v : n["delta_q_unloaded_rad"]) d.unloaded.push_back(v.as<double>());
    }
    return d;
}

// Encoders q (+ the final-travel direction d per joint, as logged) -> the model's link angles.
std::vector<double> ModelTheta(const UrdfChain& chain6, const Iso3& base_cal, const std::vector<double>& q,
                               const std::vector<double>& dq_cal, const DriveCal& drv, const std::vector<double>& d) {
    std::vector<double> th = Theta(q, dq_cal);
    if (!drv.on) return th;
    const std::vector<double> tau = chain6.GravityTorques(th, base_cal.linear());
    for (int j = 0; j < 6; ++j) {
        const double sg = tau[j] >= 0 ? 1.0 : -1.0;
        double sj;
        if (drv.approach == "against_gravity") {
            sj = std::fabs(tau[j]) >= 0.05 ? sg : -1.0;
        } else {  // logged direction d, calico's default friction table
            const double f = kDrive[j].friction;
            const double w = 0.5 * (1.0 + std::tanh((std::fabs(tau[j]) - f) / (0.3 * f)));
            sj = w * sg - (1.0 - w) * d[j];
        }
        th[j] += drv.c[j] * (tau[j] - drv.tau_ref[j]) + drv.h[j] * (sj - drv.s_ref[j]);
    }
    return th;
}

int Evaluate(const std::string& dir) {
    const Truth t = ReadTruth(dir + "/gt.yaml");
    const Truth factory = MakeTruth();  // the lenses as delivered (before the focus step)
    int failures = 0;
    auto verdict = [&](bool ok) {
        if (!ok) ++failures;
        return ok ? "ok" : "FAIL";
    };
    std::cout << std::fixed;
    Renderer truth_cam[3];  // the real lenses: projection model + unmodelled decentering
    for (int c = 0; c < 3; ++c) {
        truth_cam[c].cam = t.intr[c];
        truth_cam[c].lens = RenderLens(t, c);
        truth_cam[c].InitCircle();
    }

    std::cout << "== focus: lens set for the operating range (harmonic mean) ==\n";
    for (int c = 0; c < 3; ++c) {
        const Range r = OperatingRange(c);
        const double target = FocusTarget(r.near_m, r.far_m);
        const LensTruth& L = t.lens[c];
        const double s = L.FocusDistance(L.z_m), s0 = factory.lens[c].FocusDistance(factory.lens[c].z_factory_m);
        auto worst_dpt = [&](double sf) { return std::max(std::fabs(1 / r.near_m - 1 / sf), std::fabs(1 / r.far_m - 1 / sf)); };
        const double best = 0.5 * (1 / r.near_m - 1 / r.far_m);
        const double k = L.DefocusPxPerDioptre(L.z_m) / 4.0;  // blur sigma (px) per dioptre
        const YAML::Node chk = YAML::LoadFile(dir + "/focus_" + kCams[c] + "/check_out/focus_report.yaml");
        std::cout << "  " << std::setw(6) << kCams[c] << ": range " << std::setprecision(2) << r.near_m << "-" << r.far_m
                  << " m, target " << std::setprecision(3) << target << " m | delivered focus "
                  << (s0 > 100 ? std::string("inf") : std::to_string(s0).substr(0, 5)) << " m: worst defocus blur "
                  << worst_dpt(s0) * k << " px | locked " << s << " m: " << worst_dpt(s) * k << " px (best possible "
                  << best * k << ")  check: near " << chk["near_blur_px"].as<double>() << " / far "
                  << chk["far_blur_px"].as<double>() << " px, " << (chk["balanced"].as<bool>() ? "balanced" : "NOT balanced")
                  << "  " << verdict(worst_dpt(s) <= 1.3 * best + 0.3 && chk["balanced"].as<bool>()) << "\n";
        std::cout << "          focusing moved f by " << std::setprecision(2) << std::showpos
                  << (L.FocalPx(L.z_m) / factory.lens[c].FocalPx(factory.lens[c].z_factory_m) - 1) * 100 << std::noshowpos
                  << " %: intrinsics from before the focus step would be wrong by that much\n";
    }

    std::cout << "== stage 1: intrinsics vs the real lens (rays to 90 deg inside the sensor) ==\n";
    CameraIntrinsics cal_intr[3];
    for (int c = 0; c < 3; ++c) {
        const std::string path = dir + "/stage1_" + kCams[c] + "/out/" + kCams[c] + ".camera_info.yaml";
        cal_intr[c] = ReadCameraInfo(path);
        std::string header, q_line;
        {
            std::ifstream f(path);
            std::getline(f, header);
            std::getline(f, q_line);
        }
        // Pixel error of the calibrated lens over a fan of rays (no pose involved). Includes
        // the decentering the Kannala-Brandt model cannot represent.
        Stat px;
        for (double th = 2; th <= 90; th += 2) {
            for (double ph = 0; ph < 360; ph += 10) {
                const Vec3 X(std::sin(th * kDeg) * std::cos(ph * kDeg), std::sin(th * kDeg) * std::sin(ph * kDeg),
                             std::cos(th * kDeg));
                double a[2], b[2];
                truth_cam[c].TrueProject(X, a);
                if (a[0] < 0 || a[1] < 0 || a[0] >= t.intr[c].width || a[1] >= t.intr[c].height) continue;
                ProjectPoint(cal_intr[c].model, cal_intr[c].params.data(), Eigen::Matrix<double, 3, 1>(X), b);
                px.add(std::hypot(a[0] - b[0], a[1] - b[1]));
            }
        }
        const bool same = cal_intr[c].model == t.intr[c].model;
        const double dfx = cal_intr[c].params[0] - t.intr[c].params[0];
        std::cout << "  " << std::setw(6) << kCams[c] << " (" << LensModelName(cal_intr[c].model) << " fit, truth "
                  << LensModelName(t.intr[c].model) << "): ";
        if (same) std::cout << "fx err " << std::setprecision(3) << std::showpos << dfx << std::noshowpos << " px ("
                            << dfx / t.intr[c].params[0] * 100 << " %), ";
        std::cout << std::setprecision(3) << "cx/cy err "
                  << cal_intr[c].params[2] - t.intr[c].params[2] << "/" << cal_intr[c].params[3] - t.intr[c].params[3]
                  << " px, lens error over rays: rms " << px.rms() << " p95 " << px.p95() << " max " << px.max() << " px  "
                  << verdict(px.p95() < 0.25 && px.max() < 0.5) << "\n         " << q_line << "\n";
    }

    // Stage 2 may refine (or solve from scratch) the intrinsics with the robot data: then those
    // are what the robot model uses.
    bool stage2_intr = false;
    for (int c = 0; c < 3; ++c) {
        const std::string p2 = dir + "/stage2/out/" + kCams[c] + ".camera_info.yaml";
        if (!std::ifstream(p2).good()) continue;
        if (!stage2_intr) std::cout << "== stage 2 intrinsics (solved with the robot data) vs the real lens ==\n";
        stage2_intr = true;
        cal_intr[c] = ReadCameraInfo(p2);
        Stat px, by[3];
        for (double th = 2; th <= 90; th += 2) {
            for (double ph = 0; ph < 360; ph += 10) {
                const Vec3 X(std::sin(th * kDeg) * std::cos(ph * kDeg), std::sin(th * kDeg) * std::sin(ph * kDeg),
                             std::cos(th * kDeg));
                double a[2], b[2];
                truth_cam[c].TrueProject(X, a);
                if (a[0] < 0 || a[1] < 0 || a[0] >= t.intr[c].width || a[1] >= t.intr[c].height) continue;
                ProjectPoint(cal_intr[c].model, cal_intr[c].params.data(), Eigen::Matrix<double, 3, 1>(X), b);
                const double e = std::hypot(a[0] - b[0], a[1] - b[1]);
                px.add(e);
                by[th <= 45 ? 0 : (th <= 75 ? 1 : 2)].add(e);
            }
        }
        const double dfx = cal_intr[c].params[0] - t.intr[c].params[0];
        std::cout << "  " << std::setw(6) << kCams[c] << ": fx err " << std::setprecision(3) << std::showpos << dfx
                  << std::noshowpos << " px, cx/cy err " << cal_intr[c].params[2] - t.intr[c].params[2] << "/"
                  << cal_intr[c].params[3] - t.intr[c].params[3] << " px, lens error over rays p95: 0-45 "
                  << by[0].p95() << ", 45-75 " << by[1].p95() << ", 75-90 " << by[2].p95() << " px, max " << px.max()
                  << " px  " << verdict(px.p95() < 0.25 && px.max() < 0.5) << "\n";
    }
    const YAML::Node res = YAML::LoadFile(dir + "/stage2/out/robot_calib.yaml");
    std::vector<double> dq_cal[2];
    DriveCal drv[2];
    for (int a = 0; a < 2; ++a) {
        const YAML::Node arm = res["arms"][kSides[a]];
        for (const auto& v : arm["delta_q_rad"]) dq_cal[a].push_back(v.as<double>());
        drv[a] = ReadDriveCal(arm);
    }
    // Gauge. Without a marker on the robot, J1's zero is the same as a yaw of that arm's base and
    // J6's zero the same as a roll of the wrist camera about J6: the priors (0 +- 0.5 deg) pick
    // them. The left base is the world, so a left-J1 error rotates the solved world about the
    // left J1 axis by d1 = dq1_cal - dq1_true. Everything is compared in that rotated world.
    const double d1[2] = {dq_cal[0][0] - t.dq[0][0], dq_cal[1][0] - t.dq[1][0]};
    const double d6[2] = {dq_cal[0][5] - t.dq[0][5], dq_cal[1][5] - t.dq[1][5]};
    auto Rz = [](double a) {
        Iso3 T = Iso3::Identity();
        T.linear() = Eigen::AngleAxisd(a, Vec3::UnitZ()).toRotationMatrix();
        return T;
    };
    const Iso3 G = t.base[0] * Rz(d1[0]) * t.base[0].inverse();
    auto pose_err = [](const Iso3& A, const Iso3& B, double& mm, double& deg) {
        const Iso3 E = A.inverse() * B;
        mm = E.translation().norm() * 1e3;
        deg = RotationAngleDeg(E.linear(), Mat3::Identity());
    };
    std::vector<UrdfChain> cad;
    for (int a = 0; a < 2; ++a) {
        cad.push_back(UrdfChain::FromFile(dir + "/robot/" + kSides[a] + ".urdf", std::string(kSides[a]) + "_base_link",
                                          std::string(kSides[a]) + "_link6", Joints(kSides[a])));
    }
    // Drive train over the workspace (random approach directions): the mean is what a static
    // joint-zero calibration can absorb; the spread is the floor no zero offset can remove.
    std::vector<double> mean_off[2] = {std::vector<double>(6, 0.0), std::vector<double>(6, 0.0)};
    auto random_dir = [](std::mt19937& g) {
        std::vector<double> d(6);
        for (double& x : d) x = (g() & 1) ? 1.0 : -1.0;
        return d;
    };
    // Deployment: the robot ends its moves per the same policy as the capture.
    auto eval_dir = [&](int a, const std::vector<double>& q, std::mt19937& g) {
        std::vector<double> rnd = random_dir(g);
        return ApproachDirections(t.approach, cad[a], t.base[a], q, t.dq[a], rnd);
    };
    if (t.backlash) {
        std::cout << "== drive train: link - encoder angle under gravity (backlash + compliance), 2000 poses, "
                  << t.approach << " approach ==\n";
        std::mt19937 g(77);
        for (int a = 0; a < 2; ++a) {
            Stat st[6];
            int loaded[6] = {0, 0, 0, 0, 0, 0};
            for (int k = 0; k < 2000; ++k) {
                const std::vector<double> q = RandomPose(a, g);
                const std::vector<double> off = DriveOffset(cad[a], t.base[a], q, t.dq[a], eval_dir(a, q, g), true);
                const std::vector<double> tau = GravityTorques(cad[a], t.base[a], Theta(q, t.dq[a]));
                for (int j = 0; j < 6; ++j) {
                    st[j].add(off[j] / kDeg);
                    loaded[j] += std::fabs(tau[j]) > kDrive[j].friction;
                }
            }
            for (int j = 0; j < 6; ++j) {
                double m = 0, sd = 0;
                for (double v : st[j].v) m += v / st[j].v.size();
                for (double v : st[j].v) sd += (v - m) * (v - m) / st[j].v.size();
                mean_off[a][j] = m * kDeg;
                std::cout << "  " << kSides[a] << " J" << j + 1 << ": backlash " << std::setprecision(2)
                          << kDrive[j].backlash_deg << " deg, gravity-loaded in " << std::setprecision(0)
                          << 100.0 * loaded[j] / 2000 << " % of poses, offset mean " << std::setprecision(3) << std::showpos
                          << m << std::noshowpos << " sd " << std::sqrt(sd) << " deg\n";
            }
        }
    }
    std::cout << "== stage 2: robot calibration vs truth ==\n";
    for (int a = 0; a < 2; ++a) {
        const std::string s = kSides[a];
        const YAML::Node arm = res["arms"][s];
        std::cout << "  " << s << " dq (deg)  truth:";
        for (int j = 0; j < 6; ++j) std::cout << " " << std::setprecision(2) << std::showpos << t.dq[a][j] / kDeg;
        if (drv[a].on) {
            std::cout << std::noshowpos << "\n  " << s << " drive model (calibrated / true): stiffness N m/rad, backlash arcmin:";
            for (int j = 0; j < 6; ++j) {
                std::cout << "\n     J" << j + 1 << ": " << std::setprecision(0) << std::noshowpos
                          << (drv[a].c[j] > 1e-9 ? 1.0 / drv[a].c[j] : 0.0) << " / " << kDrive[j].stiffness << ",  "
                          << std::setprecision(1) << 2 * drv[a].h[j] / kDeg * 60 << " / " << kDrive[j].backlash_deg * 60;
            }
        }
        std::cout << std::showpos;
        std::cout << "\n  " << s
                  << (drv[a].on ? " dq err, unloaded zero (deg)       :"
                                : (t.backlash ? " dq err vs effective zero (zero + mean drive offset):" : " dq err (deg)       :"));
        for (int j = 0; j < 6; ++j) {
            const double e = drv[a].on ? (drv[a].unloaded[j] - t.dq[a][j]) / kDeg
                                       : (dq_cal[a][j] - t.dq[a][j] - mean_off[a][j]) / kDeg;
            std::cout << " " << std::setprecision(3) << e << std::noshowpos;
            if ((j > 0 && j < 5 && std::fabs(e) > (t.backlash ? 0.1 : 0.05))) ++failures, std::cout << "(FAIL)";
            std::cout << std::showpos;
        }
        std::cout << std::noshowpos << "\n    J1, J6 unobservable without a robot marker: solved "
                  << std::setprecision(3) << dq_cal[a][0] / kDeg << " / " << dq_cal[a][5] / kDeg
                  << " deg = the prior; the true " << t.dq[a][0] / kDeg << " / " << t.dq[a][5] / kDeg
                  << " deg goes into base yaw / camera roll  "
                  << verdict(std::fabs(dq_cal[a][0]) < 0.05 * kDeg && std::fabs(dq_cal[a][5]) < 0.05 * kDeg) << "\n";
        double mm, deg;
        pose_err(t.mount[a], Rz(d6[a]) * ReadXyzRpy(arm["T_tip_cam_optical"]), mm, deg);
        std::cout << "  " << s << " camera mount err (J6 gauge removed) " << std::setprecision(3) << mm << " mm / " << deg
                  << " deg  " << verdict(mm < (t.backlash ? 1.5 : 1.0) && deg < (t.backlash ? 0.2 : 0.1)) << "\n";
    }
    {
        double mm, deg;
        pose_err(G * t.base[1], ReadXyzRpy(res["arms"]["right"]["base_to_base"]) * Rz(d1[1]), mm, deg);
        std::cout << "  right base err (J1 gauges removed) " << mm << " mm / " << deg << " deg  "
                  << verdict(mm < 1.0 && deg < 0.1) << "\n";
        pose_err(G * t.center, ReadXyzRpy(res["sensors"]["center"]["T_base_cam"]), mm, deg);
        std::cout << "  center camera err " << mm << " mm / " << deg << " deg  " << verdict(mm < 2.0 && deg < 0.1) << "\n";
    }
    const double bow = res["boards"]["wall"]["shape"]["bow_x_mm"].as<double>();
    std::cout << "  wall board bow " << bow << " mm (true " << t.wall_bow * 1e3 << ")  "
              << verdict(std::fabs(bow - t.wall_bow * 1e3) < 0.3) << "\n";

    // Validation: poses never used in calibration. The calibrated model is read back from the
    // exported URDF (encoder joints in, offsets baked in) -- this also checks the export.
    std::cout << "== validation: 300 unseen poses per arm, calibrated URDF vs nominal (CAD) ==\n";
    const std::string urdf = dir + "/stage2/out/calibrated_cell.urdf";
    std::mt19937 rng(424242);
    std::vector<Vec3> table_pts;
    {
        apriltag_family_t* fam = tag36h11_create();
        const std::vector<BoardTexture> cell = CellBoards(t, fam);
        tag36h11_destroy(fam);
        const BoardSpec& s = cell[0].spec;
        for (int id = s.first_id; id < s.first_id + s.NumTags(); ++id) {
            for (const Vec3& p : s.TagCorners(id)) table_pts.push_back(cell[0].T_world_board * p);
        }
    }
    for (int a = 0; a < 2; ++a) {
        const std::string s = kSides[a];
        const UrdfChain cal_cam = UrdfChain::FromFile(urdf, "base", s + "_optical_frame", Joints(s));
        const UrdfChain cal_ee = UrdfChain::FromFile(urdf, "base", s + "_end_effector", Joints(s));
        const Iso3 base_cal = ReadXyzRpy(res["arms"][s]["base_to_base"]);
        const Iso3 mount_cal = ReadXyzRpy(res["arms"][s]["T_tip_cam_optical"]);
        // The calibrated model's link angles: encoders + zeros (+ the drive model when solved).
        auto model_theta = [&](const std::vector<double>& q, const std::vector<double>& d) {
            return ModelTheta(cad[a], base_cal, q, dq_cal[a], drv[a], d);
        };
        const UrdfChain cad_ee = UrdfChain::FromFile(dir + "/robot/" + s + ".urdf", s + "_base_link",
                                                     s + "_end_effector", Joints(s));
        Stat cam_mm, cam_deg, ee_mm, ee_deg, ee_nom_mm, px_cal, px_nom, or_cam_mm, or_cam_deg, or_ee_mm, or_px;
        const std::vector<double> zero(6, 0.0);
        // Oracle: true geometry, zeros = truth + mean drive offset: the best any static model
        // (joint zeros, no backlash term) can do on these poses.
        const std::vector<double> dq_or = Add(t.dq[a], mean_off[a]);
        for (int k = 0; k < 300; ++k) {
            const std::vector<double> q = RandomPose(a, rng);
            // The links: encoder angles + drive offset (gravity, backlash, deployment approach side).
            const std::vector<double> dir_q = eval_dir(a, q, rng);
            const std::vector<double> qo = Add(q, DriveOffset(cad[a], t.base[a], q, t.dq[a], dir_q, t.backlash));
            const std::vector<double> th_m = model_theta(q, dir_q);
            const Iso3 gt_cam = t.base[a] * Fk(cad[a], qo, t.dq[a]) * t.mount[a];
            {
                double mm, deg;
                pose_err(gt_cam, t.base[a] * Fk(cad[a], q, dq_or) * t.mount[a], mm, deg);
                or_cam_mm.add(mm);
                or_cam_deg.add(deg);
                pose_err(t.base[a] * cad_ee.ForwardIso(Theta(qo, t.dq[a])), t.base[a] * cad_ee.ForwardIso(Theta(q, dq_or)), mm, deg);
                or_ee_mm.add(mm);
            }
            const Iso3 cal = base_cal * cad[a].ForwardIso(th_m) * mount_cal;
            if (!drv[a].on && k < 5) {  // the exported URDF must agree with the solved parameters
                const Iso3 E = cal.inverse() * cal_cam.ForwardIso(q);
                if (E.translation().norm() > 1e-6) std::cout << "  WARNING: calibrated URDF differs from the solution\n";
            }
            const Iso3 nom_cam = t.base_cad[a] * Fk(cad[a], q, zero) * t.mount_cad[a];
            double mm, deg;
            pose_err(G * gt_cam, cal, mm, deg);
            cam_mm.add(mm);
            cam_deg.add(deg);
            const Iso3 gt_ee = t.base[a] * cad_ee.ForwardIso(Theta(qo, t.dq[a]));
            pose_err(G * gt_ee, base_cal * cad_ee.ForwardIso(th_m), mm, deg);
            ee_mm.add(mm);
            ee_deg.add(deg);
            ee_nom_mm.add(((t.base_cad[a] * cad_ee.ForwardIso(q)).translation() - gt_ee.translation()).norm() * 1e3);
            // Table board corners seen by the wrist camera: the calibrated model (and the nominal
            // geometry with the calibrated lens) against where the real camera sees them.
            for (size_t i = 0; i < table_pts.size(); i += 5) {
                const Vec3 Xg = gt_cam.inverse() * table_pts[i];
                if (std::atan2(std::hypot(Xg.x(), Xg.y()), Xg.z()) > 88 * kDeg) continue;
                double g[2], c[2], n[2];
                truth_cam[a].TrueProject(Xg, g);
                if (g[0] < 0 || g[1] < 0 || g[0] >= t.intr[a].width || g[1] >= t.intr[a].height) continue;
                ProjectPoint(cal_intr[a].model, cal_intr[a].params.data(),
                             Eigen::Matrix<double, 3, 1>(cal.inverse() * (G * table_pts[i])), c);
                ProjectPoint(cal_intr[a].model, cal_intr[a].params.data(),
                             Eigen::Matrix<double, 3, 1>(nom_cam.inverse() * table_pts[i]), n);
                px_cal.add(std::hypot(c[0] - g[0], c[1] - g[1]));
                px_nom.add(std::hypot(n[0] - g[0], n[1] - g[1]));
                const Iso3 T_or = t.base[a] * Fk(cad[a], q, dq_or) * t.mount[a];
                double o[2];
                truth_cam[a].TrueProject(T_or.inverse() * table_pts[i], o);
                or_px.add(std::hypot(o[0] - g[0], o[1] - g[1]));
            }
        }
        // Without backlash the oracle is exact: the limits are absolute. With it, the calibrated
        // model may be at most a little worse than the oracle.
        const double cam_mm_lim = t.backlash ? 1.25 * or_cam_mm.p95() + 0.3 : 1.0;
        const double cam_deg_lim = t.backlash ? 1.25 * or_cam_deg.p95() + 0.03 : 0.1;
        const double ee_lim = t.backlash ? 1.25 * or_ee_mm.p95() + 0.3 : 1.0;
        const double px_lim = t.backlash ? 1.25 * or_px.p95() + 0.3 : 0.6;
        std::cout << "  " << s << " wrist camera pose err: p95 " << std::setprecision(3) << cam_mm.p95() << " mm / "
                  << cam_deg.p95() << " deg";
        if (t.backlash) std::cout << "  (oracle " << or_cam_mm.p95() << " mm / " << or_cam_deg.p95() << " deg)";
        std::cout << "  " << verdict(cam_mm.p95() < cam_mm_lim && cam_deg.p95() < cam_deg_lim) << "\n";
        std::cout << "  " << s << " gripper (end effector) position err: calibrated p95 " << ee_mm.p95() << " mm, max "
                  << ee_mm.max() << " mm";
        if (t.backlash) std::cout << " (oracle p95 " << or_ee_mm.p95() << " mm)";
        std::cout << "  |  nominal p95 " << std::setprecision(1) << ee_nom_mm.p95() << " mm  " << verdict(ee_mm.p95() < ee_lim)
                  << "\n";
        std::cout << "  " << s << " gripper orientation err p95 " << std::setprecision(2) << ee_deg.p95()
                  << " deg (= the J6 zero the prior could not see: " << std::fabs(d6[a]) / kDeg << " deg roll)\n";
        std::cout << "  " << s << " table corners in the wrist image: calibrated rms " << std::setprecision(3)
                  << px_cal.rms() << " p95 " << px_cal.p95() << " px";
        if (t.backlash) std::cout << " (oracle p95 " << or_px.p95() << ")";
        std::cout << "  |  nominal rms " << std::setprecision(1) << px_nom.rms() << " px  " << verdict(px_cal.p95() < px_lim)
                  << "\n";
    }
    std::cout << (failures ? "SIMULATION FAILED: " + std::to_string(failures) + " check(s)\n" : "SIMULATION PASSED\n");
    return failures ? 1 : 0;
}


// ------------------------------------------------------------------------------- board study
// Same hand-held session (camera poses, blur, noise, light), one A4-sized print of each board
// type at a time; every detector scored against the true corner positions, then a full
// intrinsics calibration per type (run by tools/robocal_board_study.sh).
struct StudyCase {
    std::string name;
    BoardSpec spec;
    std::string apriltag = "kaess";
    std::string refine = "xfit";
};

std::vector<StudyCase> StudyCases() {
    std::vector<StudyCase> v;
    StudyCase a;
    a.spec = {"board", "tag36h11", 6, 8, 0.024, 0.3, 0, false, 0.3};  // the A4 6x8 in use (243 x 180 mm)
    a.name = "aprilgrid_subpix";  // the previous detector: cornerSubPix only
    a.refine = "subpix";
    v.push_back(a);
    a.name = "aprilgrid";
    a.refine = "xfit";
    v.push_back(a);
    a.name = "aprilgrid_at3";
    a.apriltag = "apriltag3";
    v.push_back(a);
    StudyCase c;
    c.name = "charuco";
    c.spec.name = "board";
    c.spec.type = BoardType::Charuco;
    c.spec.family = "DICT_5X5_1000";
    c.spec.rows = 9;
    c.spec.cols = 12;
    c.spec.square_size = 0.020;  // 240 x 180 mm, 88 corners, marker / square 0.75
    c.spec.marker_size = 0.015;
    v.push_back(c);
    c.name = "charuco_r06";      // marker / square 0.6: more room around each saddle
    c.spec.rows = 7;
    c.spec.cols = 10;
    c.spec.square_size = 0.024;  // 240 x 168 mm, 54 corners
    c.spec.marker_size = 0.0144;
    v.push_back(c);
    StudyCase k;
    k.name = "checkerboard";
    k.spec.name = "board";
    k.spec.type = BoardType::Checkerboard;
    k.spec.rows = 9;
    k.spec.cols = 12;
    k.spec.square_size = 0.020;
    v.push_back(k);
    return v;
}

std::string StudyBoardLine(const StudyCase& c) {
    std::ostringstream o;
    const BoardSpec& s = c.spec;
    o << "  - {name: board, type: " << BoardTypeName(s.type) << ", rows: " << s.rows << ", cols: " << s.cols;
    if (s.type == BoardType::AprilGrid) {
        o << ", tag_size: " << s.tag_size << ", tag_spacing: " << s.tag_spacing << ", first_id: " << s.first_id;
    } else {
        o << ", square_size: " << s.square_size;
        if (s.type == BoardType::Charuco) o << ", marker_size: " << s.marker_size << ", dictionary: " << s.family;
    }
    o << "}\n";
    return o.str();
}

int BoardStudy(const std::string& dir, int n_frames, unsigned seed) {
    MkDir(dir);
    Truth t = MakeTruth();
    // Lens focused as the focus step would (harmonic mean of the wrist range).
    const Range r = OperatingRange(0);
    t.SetFocus(0, t.lens[0].ZFor(FocusTarget(r.near_m, r.far_m)));
    WriteTruth(dir + "/gt.yaml", t);
    Renderer rend{t.intr[0]};
    rend.lens = RenderLens(t, 0);
    rend.Init();
    apriltag_family_t* fam = tag36h11_create();
    std::vector<StudyCase> cases = StudyCases();
    std::vector<std::vector<BoardTexture>> tex(cases.size(), std::vector<BoardTexture>(1));
    for (size_t i = 0; i < cases.size(); ++i) {
        tex[i][0].spec = cases[i].spec;
        tex[i][0].T_world_board = Place(cases[i].spec, Vec3(0, 0, 0), Vec3(0, 0, 0.05));
        tex[i][0].Render(fam);
        MkDir(dir + "/" + cases[i].name);
        MkDir(dir + "/" + cases[i].name + "/images");
    }
    SceneExtras table;
    table.table = true;
    table.table_z = -0.0006;
    table.table_min = Vec2(-0.5, -0.6);
    table.table_max = Vec2(0.5, 0.6);
    Lighting light;
    light.lamp = Vec3(0.25, -0.35, 0.9);
    light.range_m = 0.9;
    const BoardSpec& ref = cases[0].spec;
    BoardTexture ref_tex;
    ref_tex.spec = ref;
    ref_tex.T_world_board = tex[0][0].T_world_board;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    // Per case and off-axis bin: visible points, detected, errors.
    const std::vector<double> bins = {0, 30, 60, 75, 85, 100};
    const int NB = static_cast<int>(bins.size()) - 1;
    struct Acc {
        std::vector<int> vis, det, out;
        std::vector<Stat> err;
        std::vector<double> rad_sum;
        int frames_det = 0, pts = 0;
        double max_theta = 0;
        double ms = 0;
    };
    std::vector<Acc> acc(cases.size());
    for (Acc& a : acc) {
        a.vis.assign(NB, 0);
        a.det.assign(NB, 0);
        a.out.assign(NB, 0);
        a.err.assign(NB, Stat());
        a.rad_sum.assign(NB, 0.0);
    }
    std::vector<std::ofstream> frames(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        frames[i].open(dir + "/" + cases[i].name + "/frames.yaml");
        frames[i] << "frames:\n";
    }
    int kept = 0;
    for (int tries = 0; tries < 8000 && kept < n_frames; ++tries) {
        const Vec3 target(0.06 * U(rng), 0.05 * U(rng), 0.0);
        const double h = 0.12 + 0.20 * (0.5 + 0.5 * U(rng));
        const Vec3 eye = target + Vec3(0.10 * U(rng), 0.10 * U(rng), h);
        Iso3 T_wc = LookAt(eye, target, Vec3(1, 0, 0));
        T_wc.linear() = T_wc.linear() * Eigen::AngleAxisd(M_PI * U(rng), Vec3::UnitZ()).toRotationMatrix();
        if (kept % 2) {
            T_wc.linear() = T_wc.linear() *
                            Eigen::AngleAxisd(1.25 * U(rng), Vec3(U(rng), U(rng), 0).normalized()).toRotationMatrix();
        }
        if (!BoardInView(T_wc, ref_tex, t.intr[0], 92.0)) continue;
        const Shot shot = HandheldShot(rng);
        const unsigned noise_seed = rng();
        char name[64];
        std::snprintf(name, sizeof(name), "images/f_%04d.png", kept);
        for (size_t i = 0; i < cases.size(); ++i) {
            std::mt19937 nr(noise_seed);  // identical noise / exposure draws for every board type
            const std::vector<Iso3> pose = {tex[i][0].T_world_board};
            const cv::Mat img = rend.RenderReal(T_wc, tex[i], pose, table, light, shot, nr);
            cv::imwrite(dir + "/" + cases[i].name + "/" + name, img);
            frames[i] << "  - {camera: cam, image: " << name << "}\n";
            DetectOptions o;
            o.apriltag = cases[i].apriltag;
            o.refine = cases[i].refine;
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<BoardDetection> dets = DetectBoards(img, {cases[i].spec}, o);
            acc[i].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            // Truth: every board point in front of the camera, inside the sensor.
            const BoardSpec& s = cases[i].spec;
            std::map<int, Vec2> found;
            for (const BoardDetection& d : dets) {
                for (size_t k = 0; k < d.pids.size(); ++k) found[d.pids[k]] = d.pts[k];
            }
            acc[i].frames_det += !found.empty();
            for (int pid = 0; pid < s.NumPoints(); ++pid) {
                const Vec3 Xc = T_wc.inverse() * (pose[0] * s.Point(pid));
                const double th = std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) / kDeg;
                if (th >= bins.back()) continue;
                double uv[2];
                rend.TrueProject(Xc, uv);
                if (uv[0] < 2 || uv[1] < 2 || uv[0] > t.intr[0].width - 3 || uv[1] > t.intr[0].height - 3) continue;
                int b = 0;
                while (b + 1 < NB && th >= bins[b + 1]) ++b;
                ++acc[i].vis[b];
                auto it = found.find(pid);
                if (it == found.end()) continue;
                ++acc[i].det[b];
                ++acc[i].pts;
                const Vec2 e = it->second - Vec2(uv[0], uv[1]);
                acc[i].max_theta = std::max(acc[i].max_theta, th);
                if (e.norm() > 2.0) {
                    ++acc[i].out[b];
                    continue;
                }
                acc[i].err[b].add(e.norm());
                const Vec2 radial = (Vec2(uv[0], uv[1]) - Vec2(t.intr[0].params[2], t.intr[0].params[3])).normalized();
                acc[i].rad_sum[b] += e.dot(radial);
            }
        }
        ++kept;
    }
    tag36h11_destroy(fam);
    std::ofstream rep(dir + "/detection_report.txt");
    std::ostringstream o;
    o << std::fixed << "Board study: " << kept << " hand-held frames, ~200 deg fisheye (1920x1200), 12-32 cm, "
      << "half tilted toward the image edge\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const Acc& a = acc[i];
        o << "\n" << cases[i].name << ": board points " << cases[i].spec.NumPoints() << ", frames with a detection "
          << a.frames_det << "/" << kept << ", points " << a.pts << ", largest angle detected " << std::setprecision(1)
          << a.max_theta << " deg, " << std::setprecision(0) << a.ms / kept << " ms/frame\n";
        o << "   off-axis    detected/visible   err median   p95    radial bias   outliers>2px\n";
        for (int b = 0; b < NB; ++b) {
            o << "   " << std::setw(3) << std::setprecision(0) << bins[b] << "-" << std::setw(3) << bins[b + 1] << " deg   "
              << std::setw(6) << a.det[b] << "/" << std::setw(6) << a.vis[b] << " (" << std::setw(3)
              << (a.vis[b] ? 100.0 * a.det[b] / a.vis[b] : 0.0) << "%)   " << std::setprecision(3) << std::setw(6)
              << a.err[b].median() << "  " << std::setw(6) << a.err[b].p95() << "   " << std::showpos << std::setw(7)
              << (a.err[b].v.empty() ? 0.0 : a.rad_sum[b] / a.err[b].v.size()) << std::noshowpos << "      "
              << a.out[b] << "\n";
        }
        std::ofstream cfg(dir + "/" + cases[i].name + "/stage1.yaml");
        cfg << "# Board study: intrinsics from the " << cases[i].name << " session\n"
            << "frames: frames.yaml\nreference_board: board\nboards:\n" << StudyBoardLine(cases[i])
            << "cameras:\n  - {name: cam, mode: handheld, model: equidistant, width: 1920, height: 1200, "
            << "f_init: 650, intrinsics_prepass: true, optimize_intrinsics: true}\n"
            << "detection: {cache: detections.jsonl, min_tags_per_board: 2, apriltag: " << cases[i].apriltag
            << ", refine: " << cases[i].refine << "}\n"
            << "intrinsics_gates: {max_holdout_rms_px: 0.5, min_theta_deg: 85, min_coverage: 0.8, "
            << "max_curve_sd_px: 0.3, max_mean_radial_px: 0.15}\n"
            << "solver: {huber_px: 1.0, holdout_every: 5, bootstrap: 10, threads: 16}\n";
    }
    rep << o.str();
    std::cout << o.str();
    return 0;
}

// Lens error of each study case's calibration over a fan of rays (truth incl. decentering).
int BoardStudyReport(const std::string& dir) {
    const Truth t = ReadTruth(dir + "/gt.yaml");
    Renderer truth;
    truth.cam = t.intr[0];
    truth.lens = RenderLens(t, 0);
    truth.InitCircle();
    std::cout << std::fixed << "intrinsics per board type (lens error over rays inside the sensor):\n";
    for (const StudyCase& c : StudyCases()) {
        const std::string path = dir + "/" + c.name + "/out/cam.camera_info.yaml";
        const std::string failed = dir + "/" + c.name + "/out/cam.camera_info.FAILED.yaml";
        const bool ok = std::ifstream(path).good();
        if (!ok && !std::ifstream(failed).good()) {
            std::cout << "  " << std::setw(16) << c.name << ": no result\n";
            continue;
        }
        const CameraIntrinsics cal = ReadCameraInfo(ok ? path : failed);
        Stat by[4];
        const double lim[4] = {45, 75, 85, 95};
        for (double th = 1; th <= 95; th += 2) {
            for (double ph = 0; ph < 360; ph += 10) {
                const Vec3 X(std::sin(th * kDeg) * std::cos(ph * kDeg), std::sin(th * kDeg) * std::sin(ph * kDeg),
                             std::cos(th * kDeg));
                double a[2], b[2];
                truth.TrueProject(X, a);
                if (a[0] < 0 || a[1] < 0 || a[0] >= t.intr[0].width || a[1] >= t.intr[0].height) continue;
                ProjectPoint(cal.model, cal.params.data(), Eigen::Matrix<double, 3, 1>(X), b);
                int k = 0;
                while (k < 3 && th > lim[k]) ++k;
                by[k].add(std::hypot(a[0] - b[0], a[1] - b[1]));
            }
        }
        std::string hold;
        {
            std::ifstream f(ok ? path : failed);
            std::getline(f, hold);
            std::getline(f, hold);
        }
        std::cout << "  " << std::setw(16) << c.name << (ok ? " PASS" : " FAIL") << "  fx err " << std::setprecision(3)
                  << std::showpos << cal.params[0] - t.intr[0].params[0] << std::noshowpos << " px | lens err p95: 0-45 "
                  << by[0].p95() << ", 45-75 " << by[1].p95() << ", 75-85 " << by[2].p95() << ", 85-95 " << by[3].p95()
                  << " px\n      " << hold << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------------- drawing
// Third-person views of each step, rendered from an observer pinhole camera.
struct Observer {
    CameraIntrinsics cam;
    Iso3 T_wc;
    Renderer rend;
    explicit Observer(const Vec3& eye, const Vec3& target, double f = 1150.0, int w = 1600, int h = 1000, int ss = 2)
        : rend{CameraIntrinsics()} {
        cam.model = LensModel::PlumbBob;
        cam.width = w;
        cam.height = h;
        cam.params = {f, f, 0.5 * w, 0.5 * h, 0, 0, 0, 0, 0};
        rend.ss = ss;
        T_wc = LookAt(eye, target, Vec3(0, 0, -1));  // image y points down in the world
        rend.cam = cam;
        rend.Init();
    }
    bool Project(const Vec3& X, cv::Point2f& uv) const {
        const Vec3 c = T_wc.inverse() * X;
        if (c.z() < 0.02) return false;
        uv = cv::Point2f(static_cast<float>(cam.params[0] * c.x() / c.z() + cam.params[2]),
                         static_cast<float>(cam.params[1] * c.y() / c.z() + cam.params[3]));
        return true;
    }
};

cv::Mat Colorize(const cv::Mat& gray, const cv::Mat& mat) {
    cv::Mat out(gray.size(), CV_8UC3);
    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            const float v = gray.at<unsigned char>(r, c);
            cv::Vec3f tint(1.0f, 0.97f, 0.93f);                          // background (BGR)
            switch (mat.at<unsigned char>(r, c)) {
                case 1: tint = cv::Vec3f(1.0f, 1.0f, 1.0f); break;       // board
                case 2: tint = cv::Vec3f(1.0f, 0.90f, 0.78f); break;     // arm
                case 4: tint = cv::Vec3f(0.85f, 0.85f, 0.95f); break;    // gripper
                case 9: tint = cv::Vec3f(0.74f, 0.86f, 1.0f); break;     // table
                default: break;
            }
            out.at<cv::Vec3b>(r, c) = cv::Vec3b(cv::saturate_cast<uchar>(v * tint[0]),
                                                cv::saturate_cast<uchar>(v * tint[1]),
                                                cv::saturate_cast<uchar>(v * tint[2]));
        }
    }
    return out;
}

void Text(cv::Mat& img, const std::string& s, cv::Point p, double scale, const cv::Scalar& color, int th = 2) {
    int base = 0;
    const cv::Size sz = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale, th, &base);
    cv::rectangle(img, p + cv::Point(-6, -sz.height - 8), p + cv::Point(sz.width + 6, base + 4), cv::Scalar(250, 250, 250),
                  cv::FILLED);
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, color, th, cv::LINE_AA);
}

void Skeleton(cv::Mat& img, const Observer& ob, const std::vector<Vec3>& pts, const cv::Scalar& col, int th) {
    std::vector<cv::Point2f> uv;
    for (const Vec3& p : pts) {
        cv::Point2f q;
        if (ob.Project(p, q)) uv.push_back(q);
    }
    for (size_t i = 1; i < uv.size(); ++i) cv::line(img, uv[i - 1], uv[i], col, th, cv::LINE_AA);
    for (const cv::Point2f& q : uv) cv::circle(img, q, th + 3, col, cv::FILLED, cv::LINE_AA);
}

// Camera drawn as a small pyramid along its optical axis (+z of the optical frame).
void Frustum(cv::Mat& img, const Observer& ob, const Iso3& T_cam, const cv::Scalar& col, double d = 0.07,
             const std::string& label = "") {
    const Vec3 o = T_cam.translation();
    std::vector<cv::Point2f> q(4);
    cv::Point2f c0;
    if (!ob.Project(o, c0)) return;
    const double h = 0.75 * d;
    const Vec3 cs[4] = {Vec3(-h, -h * 0.6, d), Vec3(h, -h * 0.6, d), Vec3(h, h * 0.6, d), Vec3(-h, h * 0.6, d)};
    for (int i = 0; i < 4; ++i) {
        if (!ob.Project(T_cam * cs[i], q[i])) return;
        cv::line(img, c0, q[i], col, 2, cv::LINE_AA);
    }
    for (int i = 0; i < 4; ++i) cv::line(img, q[i], q[(i + 1) % 4], col, 2, cv::LINE_AA);
    if (!label.empty()) Text(img, label, cv::Point(static_cast<int>(c0.x) + 10, static_cast<int>(c0.y) - 10), 0.6, col, 2);
}

// White strip above the render for the captions (they never cover the scene).
cv::Mat Header(const cv::Mat& img, int h = 140) {
    cv::Mat out(img.rows + h, img.cols, img.type(), cv::Scalar(250, 250, 250));
    img.copyTo(out(cv::Rect(0, h, img.cols, img.rows)));
    return out;
}

void Arrow(cv::Mat& img, const Observer& ob, const Vec3& from, const Vec3& to, const cv::Scalar& col,
           const std::string& label) {
    cv::Point2f a, b;
    if (!ob.Project(from, a) || !ob.Project(to, b)) return;
    cv::arrowedLine(img, a, b, col, 3, cv::LINE_AA, 0, 0.25);
    Text(img, label, cv::Point(static_cast<int>(b.x) + 12, static_cast<int>(b.y) + 6), 0.65, col, 2);
}

std::vector<Vec3> SkeletonPoints(const UrdfChain& ee_chain, const Iso3& base, const std::vector<double>& theta) {
    std::vector<Vec3> p = {base.translation()};
    for (const Iso3& F : ee_chain.FrameChain(theta)) {
        const Vec3 x = base * F.translation();
        if ((x - p.back()).norm() > 1e-6) p.push_back(x);
    }
    return p;
}

std::vector<std::vector<double>> FrameJoints(const std::string& path, const std::string& camera, const std::string& arm) {
    std::vector<std::vector<double>> out;
    for (const auto& f : YAML::LoadFile(path)["frames"]) {
        if (f["camera"].as<std::string>() != camera || !f["joints"] || !f["joints"][arm]) continue;
        std::vector<double> q;
        for (const auto& v : f["joints"][arm]) q.push_back(v.as<double>());
        out.push_back(q);
    }
    return out;
}

// ----------------------------------------------------------------------------------------- video
Iso3 Interp(const Iso3& A, const Iso3& B, double t) {
    Iso3 T = Iso3::Identity();
    T.linear() = Eigen::Quaterniond(A.linear()).slerp(t, Eigen::Quaterniond(B.linear())).toRotationMatrix();
    T.translation() = (1 - t) * A.translation() + t * B.translation();
    return T;
}

double Smooth(double t) { return t * t * (3 - 2 * t); }

std::vector<double> Lerp(const std::vector<double>& a, const std::vector<double>& b, double t) {
    std::vector<double> q(a.size());
    for (size_t i = 0; i < a.size(); ++i) q[i] = (1 - t) * a[i] + t * b[i];
    return q;
}

CameraIntrinsics Scaled(const CameraIntrinsics& c, int w) {
    CameraIntrinsics s = c;
    const double k = static_cast<double>(w) / c.width;
    s.width = w;
    s.height = static_cast<int>(std::lround(c.height * k));
    for (int i = 0; i < 4; ++i) s.params[i] *= k;
    return s;
}

struct VideoWriterPng {
    std::string dir;
    int n = 0;
    void Add(const cv::Mat& f) {
        char name[64];
        std::snprintf(name, sizeof(name), "/f%05d.png", n++);
        cv::imwrite(dir + name, f, {cv::IMWRITE_PNG_COMPRESSION, 1});
    }
};

// Canvas 1600 x 900: caption bar, big observer view on the left, two camera panels + legend right.
cv::Mat Compose(const std::string& title, const std::string& sub, const cv::Mat& big, const cv::Mat& v1,
                const std::string& l1, const cv::Mat& v2, const std::string& l2,
                const std::vector<std::pair<std::string, cv::Scalar>>& legend) {
    cv::Mat c(900, 1600, CV_8UC3, cv::Scalar(246, 246, 246));
    cv::putText(c, title, {20, 36}, cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
    cv::putText(c, sub, {20, 66}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    big.copyTo(c(cv::Rect(0, 80, big.cols, big.rows)));
    auto panel = [&](const cv::Mat& v, const std::string& l, int y) {
        if (v.empty()) return;
        v.copyTo(c(cv::Rect(1600 - v.cols, y, v.cols, v.rows)));
        cv::rectangle(c, cv::Rect(1600 - v.cols, y, v.cols, 26), cv::Scalar(246, 246, 246), cv::FILLED);
        cv::putText(c, l, {1600 - v.cols + 8, y + 19}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    };
    panel(v1, l1, 80);
    panel(v2, l2, 80 + 300 + 6);
    int y = 80 + 2 * 306 + 30;
    for (const auto& kv : legend) {
        cv::putText(c, kv.first, {1600 - 533 + 10, y}, cv::FONT_HERSHEY_SIMPLEX, 0.58, kv.second, 2, cv::LINE_AA);
        y += 30;
    }
    return c;
}

// Camera body velocity (camera frame) between two poses dt apart.
Shot MotionShot(const Iso3& A, const Iso3& B, double dt) {
    Shot s;
    const Iso3 D = A.inverse() * B;
    const Eigen::AngleAxisd aa(D.linear());
    s.w = aa.axis() * aa.angle() / dt;
    s.v = D.translation() / dt;
    s.exposure_s = 0.008;
    return s;
}

// Blur vs focus-ring plot for the focus segment.
cv::Mat FocusPlot(const std::vector<double>& ang, const std::vector<double>& blur, size_t upto, int best, int W, int H) {
    cv::Mat p(H, W, CV_8UC3, cv::Scalar(252, 252, 252));
    double a0 = 1e9, a1 = -1e9, b1 = 0;
    for (size_t i = 0; i < ang.size(); ++i) {
        a0 = std::min(a0, ang[i]);
        a1 = std::max(a1, ang[i]);
        if (blur[i] > 0) b1 = std::max(b1, blur[i]);
    }
    b1 = std::max(1.0, std::ceil(b1 * 2) / 2);
    const int L = 52, R = 14, T = 34, B = 40;
    auto X = [&](double a) { return L + (a - a0) / std::max(1e-9, a1 - a0) * (W - L - R); };
    auto Y = [&](double b) { return H - B - b / b1 * (H - T - B); };
    for (double b = 0; b <= b1 + 1e-9; b += 0.5) {
        cv::line(p, cv::Point2d(L, Y(b)), cv::Point2d(W - R, Y(b)), cv::Scalar(225, 225, 225), 1);
        std::ostringstream o;
        o << std::fixed << std::setprecision(1) << b;
        cv::putText(p, o.str(), cv::Point2d(12, Y(b) + 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);
    }
    cv::putText(p, "edge blur sigma (px) vs focus ring (deg)", {L, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(40, 40, 40), 1,
                cv::LINE_AA);
    for (double a = std::ceil(a0 / 5) * 5; a <= a1; a += 5) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(0) << a;
        cv::putText(p, o.str(), cv::Point2d(X(a) - 8, H - B + 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(110, 110, 110),
                    1, cv::LINE_AA);
    }
    for (size_t i = 0; i < upto && i < ang.size(); ++i) {
        if (blur[i] <= 0) continue;
        cv::circle(p, cv::Point2d(X(ang[i]), Y(blur[i])), i + 1 == upto ? 6 : 3,
                   i + 1 == upto ? cv::Scalar(214, 120, 42) : cv::Scalar(120, 120, 120), cv::FILLED, cv::LINE_AA);
    }
    if (best >= 0 && static_cast<size_t>(best) < upto) {
        cv::circle(p, cv::Point2d(X(ang[best]), Y(blur[best])), 10, cv::Scalar(60, 150, 30), 2, cv::LINE_AA);
        cv::putText(p, "sharpest", cv::Point2d(X(ang[best]) - 30, Y(blur[best]) - 14), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(60, 150, 30), 1, cv::LINE_AA);
    }
    return p;
}

// Full-frame text card (title, subtitle, coloured lines, optional bar chart).
struct CardBar {
    std::string label;
    double value;
    std::string text;
    cv::Scalar color;
};

cv::Mat Card(const std::string& title, const std::string& sub,
             const std::vector<std::pair<std::string, cv::Scalar>>& lines, const std::vector<CardBar>& bars = {},
             const std::string& bars_title = "", double bars_max = 0) {
    cv::Mat c(900, 1600, CV_8UC3, cv::Scalar(246, 246, 246));
    cv::putText(c, title, {60, 110}, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(30, 30, 30), 3, cv::LINE_AA);
    cv::putText(c, sub, {62, 160}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(90, 90, 90), 2, cv::LINE_AA);
    cv::line(c, {60, 190}, {1540, 190}, cv::Scalar(210, 210, 210), 2);
    int y = 250;
    for (const auto& l : lines) {
        if (l.first.empty()) {
            y += 18;
            continue;
        }
        cv::putText(c, l.first, {70, y}, cv::FONT_HERSHEY_SIMPLEX, 0.78, l.second, 2, cv::LINE_AA);
        y += 44;
    }
    if (!bars.empty()) {
        y += 20;
        cv::putText(c, bars_title, {70, y}, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(60, 60, 60), 2, cv::LINE_AA);
        y += 25;
        const int x0 = 560, x1 = 1450;
        double mx = bars_max;
        for (const CardBar& b : bars) mx = std::max(mx, b.value);
        for (const CardBar& b : bars) {
            cv::putText(c, b.label, {70, y + 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
            const int w = static_cast<int>((x1 - x0) * b.value / std::max(1e-9, mx));
            cv::rectangle(c, cv::Rect(x0, y + 6, std::max(3, w), 32), b.color, cv::FILLED);
            cv::putText(c, b.text, {x0 + std::max(3, w) + 12, y + 31}, cv::FONT_HERSHEY_SIMPLEX, 0.7, b.color, 2,
                        cv::LINE_AA);
            y += 50;
        }
    }
    return c;
}

int Video(const std::string& dir, const std::string& out, const std::string& notes_path) {
    MkDir(out);
    const std::string fdir = out + "/frames";
    MkDir(fdir);
    VideoWriterPng vw{fdir};
    const Truth t = ReadTruth(dir + "/gt.yaml");
    const YAML::Node res = YAML::LoadFile(dir + "/stage2/out/robot_calib.yaml");
    apriltag_family_t* fam = tag36h11_create();
    std::vector<BoardTexture> cell = CellBoards(t, fam);
    std::vector<UrdfChain> cad_ee, cad6;
    std::vector<double> dq_cal[2];
    DriveCal drv_cal[2];
    Iso3 base_cal[2], mount_cal[2];
    for (int a = 0; a < 2; ++a) {
        const std::string s = kSides[a];
        cad_ee.push_back(UrdfChain::FromString(ArmUrdf(s + "_"), s + "_base_link", s + "_end_effector", Joints(s)));
        cad6.push_back(UrdfChain::FromString(ArmUrdf(s + "_"), s + "_base_link", s + "_link6", Joints(s)));
        for (const auto& v : res["arms"][s]["delta_q_rad"]) dq_cal[a].push_back(v.as<double>());
        base_cal[a] = ReadXyzRpy(res["arms"][s]["base_to_base"]);
        mount_cal[a] = ReadXyzRpy(res["arms"][s]["T_tip_cam_optical"]);
        drv_cal[a] = ReadDriveCal(res["arms"][s]);
    }
    // The real arm: encoders + zero offsets + its drive train (backlash, sag); every move in the
    // video ends with + travel. The calibrated model sees the same encoders and travel.
    const std::vector<double> dir_plus(6, 1.0);
    auto true_theta = [&](int a, const std::vector<double>& q) {
        return Theta(Add(q, DriveOffset(cad6[a], t.base[a], q, t.dq[a], dir_plus, t.backlash)), t.dq[a]);
    };
    auto model_theta = [&](int a, const std::vector<double>& q, double w) {  // w: 0 nominal -> 1 calibrated
        const std::vector<double> cal = ModelTheta(cad6[a], base_cal[a], q, dq_cal[a], drv_cal[a], dir_plus);
        std::vector<double> th(6);
        for (int j = 0; j < 6; ++j) th[j] = q[j] + w * (cal[j] - q[j]);
        return th;
    };
    // The solved world is the left base as its J1 prior defines it: rotate it onto the true one
    // for drawing (the unobservable left-J1 zero; everything relative is unaffected).
    Iso3 Gi = Iso3::Identity();
    {
        Iso3 R = Iso3::Identity();
        R.linear() = Eigen::AngleAxisd(dq_cal[0][0] - t.dq[0][0], Vec3::UnitZ()).toRotationMatrix();
        Gi = (t.base[0] * R * t.base[0].inverse()).inverse();
    }
    CameraIntrinsics cal_intr[3];
    for (int c = 0; c < 3; ++c) {
        cal_intr[c] = ReadCameraInfo(dir + "/stage1_" + std::string(kCams[c]) + "/out/" + kCams[c] + ".camera_info.yaml");
    }
    const std::vector<std::vector<double>> route[2] = {FrameJoints(dir + "/stage2/frames.yaml", "left", "left"),
                                                       FrameJoints(dir + "/stage2/frames.yaml", "right", "right")};
    const cv::Scalar kNom(32, 104, 235), kCal(214, 120, 42), kWrist(160, 90, 30), kCen(60, 150, 30), kInk(30, 30, 30),
        kGrey(120, 120, 120);
    std::vector<std::pair<Iso3, Vec2>> bd;
    for (const BoardTexture& b : cell) bd.push_back({b.T_world_board, Vec2(b.spec.Width(), b.spec.Height())});
    auto scene = [&](const std::vector<double>& ql, const std::vector<double>& qr, std::vector<Iso3>& poses) {
        SceneExtras x;
        x.table = true;
        x.table_z = -0.031;
        const std::vector<double>* q[2] = {&ql, &qr};
        for (int a = 0; a < 2; ++a) {
            const std::vector<Capsule> c = ArmCapsules(cad_ee[a], t.base[a], true_theta(a, *q[a]), x.table_z);
            x.capsules.insert(x.capsules.end(), c.begin(), c.end());
        }
        poses.clear();
        for (const BoardTexture& b : cell) poses.push_back(b.T_world_board);
        return x;
    };
    auto free_pair = [&](const std::vector<double>& ql, const std::vector<double>& qr) {
        const auto cl = ArmCapsules(cad_ee[0], t.base[0], true_theta(0, ql), -0.031);
        const auto cr = ArmCapsules(cad_ee[1], t.base[1], true_theta(1, qr), -0.031);
        return ArmPoseIsFree(cl, cr, bd, -0.031) && ArmPoseIsFree(cr, cl, bd, -0.031);
    };
    // A move between two poses, with a lift waypoint if the straight joint path collides.
    auto plan = [&](int a, const std::vector<double>& from, const std::vector<double>& to, const std::vector<double>& other,
                    int steps) {
        auto ok_path = [&](const std::vector<std::vector<double>>& wps) {
            for (size_t w = 1; w < wps.size(); ++w) {
                for (int k = 0; k <= 10; ++k) {
                    const auto q = Lerp(wps[w - 1], wps[w], k / 10.0);
                    if (!(a == 0 ? free_pair(q, other) : free_pair(other, q))) return false;
                }
            }
            return true;
        };
        std::vector<std::vector<double>> wps = {from, to};
        if (!ok_path(wps)) {
            std::vector<double> lift = Lerp(from, to, 0.5);
            lift[1] -= 0.45;  // shoulder back
            lift[2] -= 0.35;  // elbow up
            wps = {from, lift, to};
            if (!ok_path(wps)) return std::vector<std::vector<double>>();
        }
        std::vector<std::vector<double>> out;
        for (size_t w = 1; w < wps.size(); ++w) {
            for (int k = 1; k <= steps; ++k) out.push_back(Lerp(wps[w - 1], wps[w], Smooth(k / static_cast<double>(steps))));
        }
        return out;
    };
    std::mt19937 rng(3);
    // Live views: observer (left), two real-camera views (right, 1920x1200 scaled to 480x300).
    const Observer ob(Vec3(-0.70, -1.15, 1.05), Vec3(0.30, 0.0, 0.14), 760.0, 1067, 820, 2);
    const int PW = 480;
    const double kv = PW / 1920.0;
    const CameraIntrinsics vin[3] = {Scaled(t.intr[0], PW), Scaled(t.intr[1], PW), Scaled(t.intr[2], PW)};
    Renderer vr[3] = {Renderer{vin[0]}, Renderer{vin[1]}, Renderer{vin[2]}};
    for (int c = 0; c < 3; ++c) {
        vr[c].ss = 2;
        vr[c].lens = RenderLens(t, c).Scaled(kv);
        vr[c].Init();
    }
    const Lighting cell_light;
    // What the camera records (real optics + sensor), with the tags it actually sees outlined.
    auto cam_view = [&](int ci, const Iso3& T_wc, const std::vector<BoardTexture>& bs, const std::vector<Iso3>& poses,
                        const SceneExtras& x, const Lighting& light, const Shot& shot, int* n_tags,
                        std::vector<Vec2>* corners, cv::Mat* mat_out = nullptr) {
        cv::Mat mat;
        const cv::Mat gray = vr[ci].RenderReal(T_wc, bs, poses, x, light, shot, rng, &mat);
        cv::Mat img;
        cv::cvtColor(gray, img, cv::COLOR_GRAY2BGR);
        int n = 0;
        for (size_t b = 0; b < bs.size(); ++b) {
            const BoardSpec& sp = bs[b].spec;
            for (int id = sp.first_id; id < sp.first_id + sp.NumTags(); ++id) {
                std::vector<cv::Point> quad;
                for (const Vec3& pb : sp.TagCorners(id)) {
                    const Vec3 Xc = T_wc.inverse() * (poses[b] * pb);
                    if (std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) > 99 * kDeg) break;
                    double uv[2];
                    vr[ci].TrueProject(Xc, uv);
                    const int u = static_cast<int>(std::lround(uv[0])), v = static_cast<int>(std::lround(uv[1]));
                    if (u < 1 || v < 1 || u >= img.cols - 1 || v >= img.rows - 1 || mat.at<unsigned char>(v, u) != 1) break;
                    quad.push_back({u, v});
                }
                if (quad.size() != 4) continue;
                ++n;
                cv::polylines(img, quad, true, cv::Scalar(40, 200, 60), 1, cv::LINE_AA);
                if (corners) {
                    for (const cv::Point& p : quad) corners->push_back(Vec2(p.x, p.y));
                }
            }
        }
        if (n_tags) *n_tags = n;
        if (mat_out) *mat_out = mat;
        return img;
    };
    auto fit = [&](const cv::Mat& img, int w) {
        cv::Mat o;
        cv::resize(img, o, cv::Size(w, static_cast<int>(std::lround(img.rows * double(w) / img.cols))), 0, 0, cv::INTER_AREA);
        if (o.channels() == 1) cv::cvtColor(o, o, cv::COLOR_GRAY2BGR);
        return o;
    };

    YAML::Node notes;
    if (!notes_path.empty()) notes = YAML::LoadFile(notes_path);
    auto hold = [&](const cv::Mat& m, int n) {
        for (int k = 0; k < n; ++k) vw.Add(m);
    };
    // ---- Title and the pipeline at a glance
    {
        hold(Card("calico: fisheye + robot calibration, end to end",
                  "Simulated bimanual PiPER cell: AR0234 1920x1200 + 200 deg M12 fisheyes, no marker on the robot",
                  {{"Everything you see is rendered from a known truth and calibrated by the real tools:", kInk},
                   {"calico-focus, calico-robocal (intrinsics), calico-robocal (robot), then scored on unseen poses.", kInk},
                   {"", kInk},
                   {"Rendered like the real cameras: lamp + arm shadows, glossy prints, vignetting, motion blur,", kGrey},
                   {"depth- and field-dependent focus blur, auto exposure, noise, a lens distortion no model has,", kGrey},
                   {"joint states arriving late, gear backlash and sag under gravity.", kGrey}}),
             45);
        hold(Card("The pipeline", "what the operator does, and what calico solves",
                  {{"0  Focus (calico-focus): board at the harmonic mean of the working range, turn the lens to the", kInk},
                   {"   sharpest edge blur, lock it, check near = far. Focus moves f by ~1 %: it comes first.", kGrey},
                   {"1  Intrinsics, off the robot: camera moved by hand over AprilGrids. X-junction corner fit,", kInk},
                   {"   guided second pass for the image edge, quality gates -> camera_info (then held fixed).", kGrey},
                   {"2  Robot run: arm visits poses; image + joints captured only after 0.4 s still (joint states lag);", kInk},
                   {"   each frame logs still time and each joint's final travel direction.", kGrey},
                   {"3  Joint solve: joint zeros (J1 / J6 by prior: no robot marker), camera mounts, second base,", kInk},
                   {"   static camera, boards + their bow, drive model (backlash, sag), per-frame joint noise.", kGrey},
                   {"4  Outputs: robot_calib.yaml (+ drive terms), calibrated URDF, per-camera camera_info.", kInk}}),
             105);
    }

    // ---- F. Focus: the left lens arrives focused at infinity; the ring is swept through focus.
    {
        const std::string fd = dir + "/focus_left";
        const YAML::Node ring = YAML::LoadFile(fd + "/sweep/ring.yaml");
        const YAML::Node rep = YAML::LoadFile(fd + "/out/focus_report.yaml");
        const YAML::Node chk = YAML::LoadFile(fd + "/check_out/focus_report.yaml");
        std::vector<std::string> names;
        std::vector<double> ang, blur, dist;
        int best = -1;
        for (const auto& f : rep["frames"]) {
            names.push_back(f["image"].as<std::string>());
            ang.push_back(ring[names.back()].as<double>());
            blur.push_back(f["ok"].as<bool>() ? f["blur_px"].as<double>() : -1.0);
            dist.push_back(f["distance_m"].as<double>());
            if (names.back() == rep["best_frame"].as<std::string>()) best = static_cast<int>(names.size()) - 1;
        }
        const Range r = OperatingRange(0);
        auto show = [&](const std::string& path, const std::string& title, const std::string& sub, const cv::Mat& plot,
                        const std::vector<std::pair<std::string, cv::Scalar>>& legend, int repeat) {
            const cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            const cv::Rect crop(static_cast<int>(t.intr[0].params[2]) - 267, static_cast<int>(t.intr[0].params[3]) - 205, 534,
                                410);
            cv::Mat big;
            cv::resize(img(crop), big, cv::Size(1068, 820), 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(big, big, cv::COLOR_GRAY2BGR);
            cv::putText(big, "image centre, 2x zoom", {16, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(30, 30, 220), 2,
                        cv::LINE_AA);
            for (int k = 0; k < repeat; ++k) {
                vw.Add(Compose(title, sub, big.colRange(0, 1067), fit(img, PW), "left camera, full 1920x1200 frame", plot,
                               "calico-focus", legend));
            }
        };
        std::ostringstream sub;
        sub << std::fixed << std::setprecision(1) << "Wrist range " << r.near_m * 100 << "-" << r.far_m * 100
            << " cm -> focus at the harmonic mean " << FocusTarget(r.near_m, r.far_m) * 100
            << " cm. Board on a stand there; the ring is turned through focus.";
        for (size_t i = 0; i < names.size(); ++i) {
            std::ostringstream l1, l2, l3;
            l1 << std::fixed << std::setprecision(1) << "focus ring: " << ang[i] << " deg from delivered (inf)";
            l2 << std::fixed << std::setprecision(3) << "board distance (PnP): " << dist[i] << " m";
            l3 << std::fixed << std::setprecision(2) << "edge blur: " << blur[i] << " px";
            show(fd + "/sweep/" + names[i], "Step 0 - focus, before anything is calibrated", sub.str(),
                 FocusPlot(ang, blur, i + 1, -1, PW, 300), {{l1.str(), kInk}, {l2.str(), kInk}, {l3.str(), kCal}},
                 i + 1 == names.size() ? 1 : (i % 2 ? 2 : 1));
        }
        if (best >= 0) {
            std::ostringstream l1, l2;
            l1 << std::fixed << std::setprecision(1) << "locked at ring " << ang[best] << " deg: blur " << std::setprecision(2)
               << blur[best] << " px";
            l2 << std::fixed << std::setprecision(2) << "check: near " << chk["near_blur_px"].as<double>() << " px, far "
               << chk["far_blur_px"].as<double>() << " px";
            show(fd + "/sweep/" + names[best], "Step 0 - focus: back to the sharpest setting, lock the lens", sub.str(),
                 FocusPlot(ang, blur, names.size(), best, PW, 300),
                 {{l1.str(), kCen},
                  {l2.str(), kInk},
                  {chk["balanced"].as<bool>() ? "near / far balanced: ok" : "near / far NOT balanced", kInk},
                  {"focus moves f: intrinsics only after this", kNom}},
                 30);
        }
    }

    // ---- A. Stage 1: hand-held intrinsics session (left camera)
    {
        const std::vector<BoardSpec> specs = SessionBoards(0);
        std::vector<BoardTexture> sb(2);
        sb[0].spec = specs[0];
        const std::vector<Iso3> place = SessionPlacement(specs);
        sb[0].T_world_board = place[0];
        sb[1].spec = specs[1];
        sb[1].T_world_board = place[1];
        sb[1].bow_x = t.session_bow;
        sb[1].gloss = 0.5;
        for (BoardTexture& b : sb) b.Render(fam);
        std::vector<Iso3> sp = {sb[0].T_world_board, sb[1].T_world_board};
        SceneExtras sc;
        sc.table = true;
        sc.table_z = -0.0006;
        sc.table_min = Vec2(-0.5, -0.6);
        sc.table_max = Vec2(0.5, 0.6);
        Lighting light;
        light.lamp = Vec3(0.25, -0.35, 0.9);
        light.range_m = 0.9;
        const Observer o1(Vec3(-0.80, -0.70, 0.80), Vec3(0.0, 0.0, 0.10), 820.0, 1067, 820, 2);
        cv::Mat obs_mat;
        const cv::Mat obs_bg = Colorize(o1.rend.Render(o1.T_wc, sb, sp, sc, rng, &obs_mat, false), obs_mat);
        std::vector<Iso3> cams;
        {
            std::ifstream in(dir + "/stage1_left/camera_poses.txt");
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                std::string name;
                ls >> name;
                Iso3 T = Iso3::Identity();
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) ls >> (c < 3 ? T.linear()(r, c) : T.translation()[r]);
                }
                cams.push_back(T);
            }
        }
        // Coverage from what the calibration actually used: the detection cache of the session.
        std::map<std::string, std::vector<Vec2>> det_pts;  // image file name -> detected corners
        {
            std::ifstream in(dir + "/stage1_left/detections.jsonl");
            std::string line;
            while (std::getline(in, line)) {
                if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
                const YAML::Node n = YAML::Load(line);
                const std::string img = n["image"].as<std::string>();
                const std::string key = img.substr(img.find_last_of('/') + 1);
                for (const auto& p : n["pts"]) det_pts[key].push_back(Vec2(p[0].as<double>(), p[1].as<double>()));
            }
        }
        std::vector<std::string> cam_names;
        {
            std::ifstream in(dir + "/stage1_left/camera_poses.txt");
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                std::string name;
                ls >> name;
                cam_names.push_back(name.substr(name.find_last_of('/') + 1));
            }
        }
        const CameraIntrinsics& ci = cal_intr[0];
        const double kc = static_cast<double>(PW) / 1920.0;
        auto theta_of = [&](const Vec2& p) {
            Vec3 b;
            if (!UnprojectPixel(ci.model, ci.params.data(), p, b)) return 0.0;
            return std::atan2(std::hypot(b.x(), b.y()), b.z()) / kDeg;
        };
        auto ring_px = [&](double th) {
            double uv[2];
            ProjectPoint(ci.model, ci.params.data(),
                         Eigen::Matrix<double, 3, 1>(std::sin(th * kDeg), 0.0, std::cos(th * kDeg)), uv);
            return (uv[0] - ci.params[2]) * kc;
        };
        // The coverage gate's grid: 16 x 9 cells over the image.
        const int GX = 16, GY = 9;
        std::vector<int> cell(GX * GY, 0);
        cv::Mat dots(vin[0].height, vin[0].width, CV_8UC3, cv::Scalar(34, 34, 34));
        double max_theta = 0;
        int n_pts = 0;
        auto add_frame = [&](const std::string& name) {
            auto it = det_pts.find(name);
            if (it == det_pts.end()) return;
            for (const Vec2& p : it->second) {
                const int gx = std::clamp(static_cast<int>(p.x() / 1920.0 * GX), 0, GX - 1);
                const int gy = std::clamp(static_cast<int>(p.y() / 1200.0 * GY), 0, GY - 1);
                cell[gy * GX + gx] = 1;
                cv::circle(dots, cv::Point2d(p.x() * kc, p.y() * kc), 1, cv::Scalar(60, 210, 80), cv::FILLED);
                max_theta = std::max(max_theta, theta_of(p));
                ++n_pts;
            }
        };
        auto coverage_view = [&]() {
            cv::Mat v = dots.clone();
            const double cw = static_cast<double>(v.cols) / GX, chh = static_cast<double>(v.rows) / GY;
            cv::Mat fill = v.clone();
            for (int gy = 0; gy < GY; ++gy) {
                for (int gx = 0; gx < GX; ++gx) {
                    const cv::Rect r(static_cast<int>(gx * cw), static_cast<int>(gy * chh), static_cast<int>(cw) + 1,
                                     static_cast<int>(chh) + 1);
                    if (cell[gy * GX + gx]) cv::rectangle(fill, r, cv::Scalar(70, 120, 60), cv::FILLED);
                    cv::rectangle(v, r, cv::Scalar(80, 80, 80), 1);
                }
            }
            cv::addWeighted(fill, 0.35, v, 0.65, 0, v);
            for (int k = 0; k < GY; ++k) {
                for (int g = 0; g < GX; ++g) cv::rectangle(v, cv::Rect(static_cast<int>(g * cw), static_cast<int>(k * chh),
                                                                       static_cast<int>(cw) + 1, static_cast<int>(chh) + 1),
                                                           cv::Scalar(85, 85, 85), 1);
            }
            const cv::Point2d c(ci.params[2] * kc, ci.params[3] * kc);
            for (double th : {30.0, 60.0, 90.0}) {
                cv::circle(v, c, static_cast<int>(ring_px(th)), cv::Scalar(200, 170, 110), 1, cv::LINE_AA);
                cv::putText(v, std::to_string(static_cast<int>(th)) + " deg", c + cv::Point2d(ring_px(th) * 0.71 + 3, -ring_px(th) * 0.71),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 170, 110), 1, cv::LINE_AA);
            }
            return v;
        };
        auto cov_text = [&]() {
            int filled = 0;
            for (int x : cell) filled += x;
            std::ostringstream o;
            o << std::fixed << std::setprecision(0) << "coverage " << 100.0 * filled / (GX * GY) << " % of cells, "
              << n_pts << " corners, to " << max_theta << " deg";
            return o.str();
        };
        const int n_show = static_cast<int>(cams.size());
        for (int k = 0; k < n_show; ++k) {
            const Iso3 from = k == 0 ? cams[0] : cams[k - 1];
            Iso3 prev = from;
            // The first captures in real time (motion blur, pause, capture); the rest faster.
            const int moving = k < 6 ? 9 : 3, still = k < 6 ? 3 : 1;
            const double dt = 0.6 / moving;  // a hand move between captures: ~0.6 s
            for (int f = 0; f < moving + still; ++f) {
                const double u = f < moving ? Smooth((f + 1) / static_cast<double>(moving)) : 1.0;
                const Iso3 T = Interp(from, cams[k], u);
                Shot shot = f < moving ? MotionShot(prev, T, dt) : Shot();  // moving: blurred; paused: sharp
                prev = T;
                cv::Mat big = obs_bg.clone();
                for (int j = 0; j < k; ++j) Frustum(big, o1, cams[j], cv::Scalar(190, 160, 120), 0.03);
                Frustum(big, o1, T, kWrist, 0.06, "camera (hand-held)");
                int n = 0;
                cv::Mat view = cam_view(0, T, sb, sp, sc, light, shot, &n, nullptr);
                if (f == moving && k < static_cast<int>(cam_names.size())) add_frame(cam_names[k]);  // captured
                if (f >= moving) cv::rectangle(view, cv::Rect(0, 0, view.cols, view.rows), cv::Scalar(255, 255, 255), 6);
                std::ostringstream lg;
                lg << "capture " << (k + 1) << " of " << cams.size();
                vw.Add(Compose("Stage 1 - intrinsics, off the robot (lens focus locked)",
                               "Camera moved by hand over two A4 6x8 AprilGrids (b laminated + bowed 2 mm). Every capture "
                               "adds its detected corners to the coverage map.",
                               big, view, "live camera (~200 deg fisheye): " + std::to_string(n) + " tags", coverage_view(),
                               "detected corners, all captures so far (16 x 9 gate grid)",
                               {{lg.str(), kInk},
                                {f < moving ? "moving: motion blur" : "paused: capture", f < moving ? kGrey : kCen},
                                {cov_text(), kCal},
                                {"tilting the boards into the corners reaches", kInk},
                                {"the 85-95 deg zone that the robot views use", kInk}}));
            }
        }
        // Hold the final coverage.
        {
            cv::Mat big = obs_bg.clone();
            for (const Iso3& c : cams) Frustum(big, o1, c, cv::Scalar(190, 160, 120), 0.03);
            cv::Mat last;
            cv::resize(coverage_view(), last, cv::Size(), 1.0, 1.0);
            hold(Compose("Stage 1 - intrinsics: the whole image is measured",
                         "All captures: detected corners over the 16 x 9 grid of the coverage gate, rings at 30 / 60 / 90 deg "
                         "off axis.",
                         big, last, "detected corners, all captures", cv::Mat(), "",
                         {{cov_text(), kCal},
                          {"+ guided pass corners at the edge (not shown)", kInk},
                          {"gates: PASS (hold-out 0.019 px, curve spread", kInk},
                          {"0.06 px) -> left.camera_info.yaml, then fixed", kCen}}),
                 45);
        }
    }

    // ---- A2. Detection close-up: decoded corners vs the guided pass at the image edge
    {
        const std::vector<BoardSpec> specs = SessionBoards(0);
        std::vector<BoardSpec> boards = specs;
        std::vector<Iso3> T_wb = SessionPlacement(specs);
        const CameraIntrinsics ci = cal_intr[0];
        // Every session frame: decoded corners + the guided pass (as calico-robocal runs it, through
        // the fitted lens); show the frame where the guided pass recovers the most.
        struct Shot1 {
            std::string name;
            cv::Mat gray;
            std::vector<Vec2> decoded, guided;
            std::vector<double> th_dec, th_gui;
        };
        auto analyse = [&](const std::string& name, const Iso3& T_wc, Shot1& r) {
            r.name = name;
            r.gray = cv::imread(dir + "/stage1_left/" + name, cv::IMREAD_GRAYSCALE);
            if (r.gray.empty()) return;
            DetectOptions dopt;
            const std::vector<BoardDetection> dets = DetectBoards(r.gray, boards, dopt);
            const cv::Mat lin = LinearizeGray(r.gray, dopt.gamma);
            for (const BoardDetection& d : dets) {
                std::set<int> have(d.pids.begin(), d.pids.end());
                const BoardSpec& b = boards[d.board];
                const Iso3 T = T_wc.inverse() * T_wb[d.board];
                for (size_t k = 0; k < d.pts.size(); ++k) {
                    r.decoded.push_back(d.pts[k]);
                    const Vec3 Xc = T * b.Point(d.pids[k]);
                    r.th_dec.push_back(std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) / kDeg);
                }
                for (int pid = 0; pid < b.NumPoints(); ++pid) {
                    if (have.count(pid)) continue;
                    auto proj = [&](const Vec3& Xb, Vec2& uv) {
                        const Vec3 Xc = T * Xb;
                        if (std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) > 100 * kDeg) return false;
                        double u[2];
                        ProjectPoint(ci.model, ci.params.data(), Eigen::Matrix<double, 3, 1>(Xc), u);
                        uv = Vec2(u[0], u[1]);
                        return u[0] > 4 && u[1] > 4 && u[0] < 1915 && u[1] < 1195;
                    };
                    const Vec3 X = b.Point(pid);
                    if (T.linear().col(2).dot(T * X) >= 0) continue;
                    Vec2 uv, ux, uy;
                    if (!proj(X, uv) || !proj(X + Vec3(1e-4, 0, 0), ux) || !proj(X + Vec3(0, 1e-4, 0), uy)) continue;
                    Eigen::Matrix2d A;
                    A.col(0) = (ux - uv) / 1e-4;
                    A.col(1) = (uy - uv) / 1e-4;
                    if (!RefinePredictedSaddle(r.gray, lin, b, pid, uv, A, dopt, 0.8)) continue;
                    r.guided.push_back(uv);
                    const Vec3 Xc = T * X;
                    r.th_gui.push_back(std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) / kDeg);
                }
            }
        };
        Shot1 best;
        {
            std::ifstream in(dir + "/stage1_left/camera_poses.txt");
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                std::string name;
                ls >> name;
                Iso3 T = Iso3::Identity();
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) ls >> (c < 3 ? T.linear()(r, c) : T.translation()[r]);
                }
                Shot1 r;
                analyse(name, T, r);
                if (r.guided.size() > best.guided.size()) best = r;
            }
        }
        const cv::Mat& gray = best.gray;
        if (!gray.empty()) {
            const std::vector<Vec2>& decoded = best.decoded;
            const std::vector<Vec2>& guided = best.guided;
            const std::vector<double>& th_dec = best.th_dec;
            const std::vector<double>& th_gui = best.th_gui;
            cv::Mat col;
            cv::cvtColor(gray, col, cv::COLOR_GRAY2BGR);
            for (const Vec2& p : decoded) cv::circle(col, cv::Point2d(p.x(), p.y()), 4, cv::Scalar(40, 200, 60), cv::FILLED, cv::LINE_AA);
            for (const Vec2& p : guided) cv::circle(col, cv::Point2d(p.x(), p.y()), 6, kNom, 2, cv::LINE_AA);
            cv::Mat big;
            cv::resize(col, big, cv::Size(1067, 667), 0, 0, cv::INTER_AREA);
            cv::Mat bigc(820, 1067, CV_8UC3, cv::Scalar(246, 246, 246));
            big.copyTo(bigc(cv::Rect(0, 40, 1067, 667)));
            // Zoom on the guided corners (or the outermost decoded ones).
            Vec2 c0(960, 600);
            {
                const std::vector<Vec2>& src = guided.empty() ? decoded : guided;
                double far = -1;
                for (const Vec2& p : src) {
                    const double r = (p - Vec2(960, 600)).norm();
                    if (r > far) far = r, c0 = p;
                }
            }
            auto zoom = [&](const Vec2& c, int half_w, int half_h) {
                const int x = std::clamp(static_cast<int>(c.x()) - half_w, 0, 1920 - 2 * half_w);
                const int y = std::clamp(static_cast<int>(c.y()) - half_h, 0, 1200 - 2 * half_h);
                cv::Mat z;
                cv::resize(col(cv::Rect(x, y, 2 * half_w, 2 * half_h)), z, cv::Size(PW, 300), 0, 0, cv::INTER_NEAREST);
                return z;
            };
            const Vec2 c1 = decoded.empty() ? Vec2(960, 600) : decoded[decoded.size() / 2];
            auto pct = [](const std::vector<double>& v, double lo) {
                int n = 0;
                for (double x : v) n += x >= lo;
                return n;
            };
            std::ostringstream l1, l2, l3;
            l1 << decoded.size() << " corners from decoded tags (" << pct(th_dec, 80) << " past 80 deg)";
            l2 << "+" << guided.size() << " from the guided pass (" << pct(th_gui, 80) << " past 80 deg)";
            double mx = 0;
            for (double x : th_dec) mx = std::max(mx, x);
            for (double x : th_gui) mx = std::max(mx, x);
            l3 << std::fixed << std::setprecision(1) << "furthest corner: " << mx << " deg off axis";
            hold(Compose("Detection - where the calibration is decided: the image edge",
                         "Green: corners of decoded tags (X-junction fit, ~0.01 px). Orange rings: tags too squeezed to decode, "
                         "corners still found by the guided pass.",
                         bigc, zoom(c0, 120, 75), "2x zoom at the edge (guided pass)", zoom(c1, 120, 75),
                         "2x zoom near the centre (decoded tags)",
                         {{l1.str(), cv::Scalar(40, 160, 60)}, {l2.str(), kNom}, {l3.str(), kInk},
                          {"board study: AprilGrid beat ChArUco + checkerboard", kInk}}),
                 75);
        }
    }

    // ---- B. Stage 2: robot run (left arm route, then right arm route)
    {
        std::vector<double> cur[2] = {kPark[0], kPark[1]};
        const double dt = 0.1;
        for (int a = 0; a < 2; ++a) {
            const int n_poses = a == 0 ? 10 : 6;
            std::vector<Iso3> captured;
            for (int k = 0; k < n_poses; ++k) {
                const std::vector<double>& target = route[a][k];
                auto path = plan(a, cur[a], target, cur[1 - a], 10);
                if (path.empty()) continue;  // no collision-free move found: skip this pose in the video
                for (int h = 0; h < 5; ++h) path.push_back(target);
                Iso3 prev = t.base[a] * cad6[a].ForwardIso(true_theta(a, cur[a])) * t.mount[a];
                for (size_t f = 0; f < path.size(); ++f) {
                    cur[a] = path[f];
                    std::vector<Iso3> pz;
                    const SceneExtras sc = scene(cur[0], cur[1], pz);
                    cv::Mat mat;
                    cv::Mat big = Colorize(ob.rend.Render(ob.T_wc, cell, pz, sc, rng, &mat, false), mat);
                    for (const Iso3& c : captured) Frustum(big, ob, c, cv::Scalar(190, 160, 120), 0.035);
                    const Iso3 T_w = t.base[a] * cad6[a].ForwardIso(true_theta(a, cur[a])) * t.mount[a];
                    Frustum(big, ob, T_w, kWrist, 0.06, std::string(kSides[a]) + " wrist cam");
                    Frustum(big, ob, t.center, kCen, 0.10, "center cam");
                    const size_t still_frames = f + 5 >= path.size() ? f + 6 - path.size() : 0;  // 1..5 when arrived
                    const bool settled = still_frames >= 4;  // joints stationary long enough: capture
                    int nw = 0, nc = 0;
                    cv::Mat wv = cam_view(a, T_w, cell, pz, sc, cell_light, MotionShot(prev, T_w, dt), &nw, nullptr);
                    prev = T_w;
                    cv::Mat cv2v = cam_view(2, t.center, cell, pz, sc, cell_light, Shot(), &nc, nullptr);
                    if (settled) {
                        cv::rectangle(wv, cv::Rect(0, 0, wv.cols, wv.rows), cv::Scalar(255, 255, 255), 6);
                        cv::rectangle(cv2v, cv::Rect(0, 0, cv2v.cols, cv2v.rows), cv::Scalar(255, 255, 255), 6);
                    }
                    if (f + 1 == path.size()) captured.push_back(T_w);
                    std::ostringstream lg, st;
                    lg << std::string(kSides[a]) << " arm: pose " << (k + 1) << " of 30 (showing " << n_poses << ")";
                    if (still_frames == 0) st << "moving";
                    else if (!settled) st << "arrived, waiting: still " << std::fixed << std::setprecision(1) << 0.1 * still_frames << " s";
                    else st << "still >= 0.4 s: capture image + joints";
                    vw.Add(Compose("Stage 2 - robot run (intrinsics fixed from stage 1, no marker on the robot)",
                                   std::string("The ") + kSides[a] + " arm visits poses, the other is parked. Joint states lag: "
                                   "capture only after the arm has been still for 0.4 s.",
                                   big, wv, std::string(kSides[a]) + " wrist camera: " + std::to_string(nw) + " tags", cv2v,
                                   "center camera: " + std::to_string(nc) + " tags",
                                   {{lg.str(), kInk},
                                    {st.str(), settled ? kCen : kGrey},
                                    {"unknowns: J2-J5 zeros (J1, J6: prior), mounts,", kInk},
                                    {"right base, center camera, boards", kInk}}));
                }
            }
            // back to park before the other arm runs
            auto home = plan(a, cur[a], kPark[a], cur[1 - a], 10);
            for (const auto& q : home) {
                cur[a] = q;
                std::vector<Iso3> pz;
                const SceneExtras sc = scene(cur[0], cur[1], pz);
                cv::Mat mat;
                cv::Mat big = Colorize(ob.rend.Render(ob.T_wc, cell, pz, sc, rng, &mat, false), mat);
                Frustum(big, ob, t.center, kCen, 0.10, "center cam");
                vw.Add(Compose("Stage 2 - robot run (intrinsics fixed from stage 1, no marker on the robot)",
                               "Parking the arm.", big, cv::Mat(), "", cv::Mat(), "", {}));
            }
            if (home.empty()) cur[a] = kPark[a];
        }
    }

    // ---- C + D: the solve, then unseen motions (model drawn in the true world: Gi)
    auto model_base = [&](int a, double w) { return Interp(t.base_cad[a], Gi * base_cal[a], w); };
    auto skeleton_model = [&](int a, const std::vector<double>& q, double w) {  // w: 0 nominal -> 1 calibrated
        return SkeletonPoints(cad_ee[a], model_base(a, w), model_theta(a, q, w));
    };
    auto corner_overlay = [&](cv::Mat& img, int a, const std::vector<double>& q, double w, const std::vector<Iso3>& poses,
                              Stat* sn, Stat* sc, const cv::Mat& vis) {
        const Iso3 T_true = t.base[a] * cad6[a].ForwardIso(true_theta(a, q)) * t.mount[a];
        auto model_cam = [&](double ww) {
            return model_base(a, ww) * cad6[a].ForwardIso(model_theta(a, q, ww)) * Interp(t.mount_cad[a], mount_cal[a], ww);
        };
        const Iso3 T_m = model_cam(w), T_n = model_cam(0.0), T_c = model_cam(1.0);
        const CameraIntrinsics ci = Scaled(cal_intr[a], PW);
        for (size_t b = 0; b < cell.size(); ++b) {
            const BoardSpec& sp = cell[b].spec;
            for (int id = sp.first_id; id < sp.first_id + sp.NumTags(); ++id) {
                for (const Vec3& pb : sp.TagCorners(id)) {
                    const Vec3 X = poses[b] * pb;
                    const Vec3 Xt = T_true.inverse() * X;
                    if (std::atan2(std::hypot(Xt.x(), Xt.y()), Xt.z()) > 95 * kDeg) continue;
                    double ut[2], um[2];
                    vr[a].TrueProject(Xt, ut);
                    if (ut[0] < 2 || ut[1] < 2 || ut[0] > img.cols - 2 || ut[1] > img.rows - 2) continue;
                    if (!vis.empty() && vis.at<unsigned char>(static_cast<int>(ut[1]), static_cast<int>(ut[0])) != 1) continue;
                    ProjectPoint(ci.model, ci.params.data(), Eigen::Matrix<double, 3, 1>(T_m.inverse() * X), um);
                    const cv::Scalar col = w >= 0.999 ? kCal : (w <= 0.001 ? kNom : kNom * (1 - w) + kCal * w);
                    cv::circle(img, cv::Point2d(um[0], um[1]), 3, col, 1, cv::LINE_AA);
                    if (sn && sc) {  // full-resolution pixel errors
                        const CameraIntrinsics& cf = cal_intr[a];
                        double un[2], uc[2], ug[2];
                        ProjectPoint(cf.model, cf.params.data(), Eigen::Matrix<double, 3, 1>(T_n.inverse() * X), un);
                        ProjectPoint(cf.model, cf.params.data(), Eigen::Matrix<double, 3, 1>(T_c.inverse() * X), uc);
                        Renderer full;
                        full.cam = t.intr[a];
                        full.lens = RenderLens(t, a);
                        full.InitCircle();
                        full.TrueProject(Xt, ug);
                        sn->add(std::hypot(un[0] - ug[0], un[1] - ug[1]));
                        sc->add(std::hypot(uc[0] - ug[0], uc[1] - ug[1]));
                    }
                }
            }
        }
    };
    {
        const std::vector<double>& ql = route[0][4];
        std::vector<Iso3> pz;
        const SceneExtras sc = scene(ql, kPark[1], pz);
        cv::Mat mat;
        const cv::Mat bg = Colorize(ob.rend.Render(ob.T_wc, cell, pz, sc, rng, &mat, false), mat);
        const Iso3 T_w = t.base[0] * cad6[0].ForwardIso(true_theta(0, ql)) * t.mount[0];
        cv::Mat vis0;
        const cv::Mat wv0 = cam_view(0, T_w, cell, pz, sc, cell_light, Shot(), nullptr, nullptr, &vis0);
        const std::vector<Vec3> tru = SkeletonPoints(cad_ee[0], t.base[0], true_theta(0, ql));
        std::ostringstream truth;
        truth << std::fixed << std::setprecision(2) << "truth:";
        for (int j = 0; j < 6; ++j) truth << " " << t.dq[0][j] / kDeg;
        for (int f = 0; f < 75; ++f) {
            const double w = f < 15 ? 0.0 : (f < 60 ? Smooth((f - 15) / 45.0) : 1.0);
            cv::Mat big = bg.clone();
            const std::vector<Vec3> m = skeleton_model(0, ql, w);
            const cv::Scalar col = kNom * (1 - w) + kCal * w;
            Skeleton(big, ob, m, col, 3);
            std::ostringstream e;
            e << std::fixed << std::setprecision(w > 0.98 ? 2 : 1) << "model gripper error " << (m.back() - tru.back()).norm() * 1e3 << " mm";
            Arrow(big, ob, tru.back(), m.back(), col, e.str());
            cv::Mat wv = wv0.clone();
            corner_overlay(wv, 0, ql, w, pz, nullptr, nullptr, vis0);
            std::ostringstream dq;
            dq << std::fixed << std::setprecision(2) << "left zeros (deg):";
            for (int j = 0; j < 6; ++j) {
                dq << " " << w * (drv_cal[0].on ? drv_cal[0].unloaded[j] : dq_cal[0][j]) / kDeg;
            }
            vw.Add(Compose("Solve - one joint optimisation over all frames",
                           "The model is pulled onto the real arm: J2-J5 zeros, camera mounts, right base, center camera, boards.",
                           big, wv, "left wrist: green = seen tags, rings = model prediction", cv::Mat(), "",
                           {{w < 0.01 ? "nominal CAD model (orange)" : (w > 0.99 ? "calibrated model (blue)" : "optimising ..."), col},
                            {dq.str(), kInk},
                            {truth.str(), kGrey},
                            {"J1, J6 = prior 0 (no robot marker): their", kGrey},
                            {"true 0.6 / -0.8 deg -> world yaw / camera roll", kGrey}}));
        }
    }
    {
        std::mt19937 vr2(99);
        std::vector<double> cl = kPark[0], cr = kPark[1];
        for (int seg = 0; seg < 7; ++seg) {
            std::vector<double> nl, nr;
            for (int tries = 0; tries < 4000; ++tries) {
                nl = RandomPose(0, vr2);
                nr = RandomPose(1, vr2);
                if (!free_pair(nl, nr)) continue;
                bool ok = true;
                for (int k = 0; k <= 12 && ok; ++k) ok = free_pair(Lerp(cl, nl, k / 12.0), Lerp(cr, nr, k / 12.0));
                if (ok) break;
            }
            for (int f = 1; f <= 14; ++f) {
                const double u = Smooth(std::min(1.0, f / 11.0));
                const auto ql = Lerp(cl, nl, u), qr = Lerp(cr, nr, u);
                std::vector<Iso3> pz;
                const SceneExtras sc = scene(ql, qr, pz);
                cv::Mat mat;
                cv::Mat big = Colorize(ob.rend.Render(ob.T_wc, cell, pz, sc, rng, &mat, false), mat);
                double e_nom[2], e_cal[2];
                for (int a = 0; a < 2; ++a) {
                    const auto& q = a == 0 ? ql : qr;
                    const auto tru = SkeletonPoints(cad_ee[a], t.base[a], true_theta(a, q));
                    const auto nom = skeleton_model(a, q, 0.0), cal = skeleton_model(a, q, 1.0);
                    Skeleton(big, ob, nom, kNom, 2);
                    Skeleton(big, ob, cal, kCal, 3);
                    e_nom[a] = (nom.back() - tru.back()).norm() * 1e3;
                    e_cal[a] = (cal.back() - tru.back()).norm() * 1e3;
                }
                Stat fn, fc;
                cv::Mat vl, vrr;
                cv::Mat wl = cam_view(0, t.base[0] * cad6[0].ForwardIso(true_theta(0, ql)) * t.mount[0], cell, pz, sc, cell_light, Shot(),
                                      nullptr, nullptr, &vl);
                corner_overlay(wl, 0, ql, 0.0, pz, &fn, &fc, vl);
                corner_overlay(wl, 0, ql, 1.0, pz, nullptr, nullptr, vl);
                cv::Mat wr = cam_view(1, t.base[1] * cad6[1].ForwardIso(true_theta(1, qr)) * t.mount[1], cell, pz, sc, cell_light, Shot(),
                                      nullptr, nullptr, &vrr);
                corner_overlay(wr, 1, qr, 0.0, pz, nullptr, nullptr, vrr);
                corner_overlay(wr, 1, qr, 1.0, pz, nullptr, nullptr, vrr);
                std::ostringstream l1, l2, l3;
                l1 << std::fixed << std::setprecision(1) << "gripper error, nominal: " << e_nom[0] << " / " << e_nom[1] << " mm";
                l2 << std::fixed << std::setprecision(2) << "gripper error, calibrated: " << e_cal[0] << " / " << e_cal[1] << " mm";
                l3 << std::fixed << std::setprecision(1) << "left wrist corners: nominal " << fn.rms() << " px, calibrated "
                   << std::setprecision(2) << fc.rms() << " px";
                vw.Add(Compose("Validation - motions never used for calibration",
                               "Orange = nominal CAD model, blue = calibrated model (zeros + drive model). Both get the same encoder angles.",
                               big, wl, "left wrist: rings = model prediction, green = seen", wr,
                               "right wrist: rings = model prediction, green = seen",
                               {{l1.str(), kNom}, {l2.str(), kCal}, {l3.str(), kInk}}));
            }
            cl = nl;
            cr = nr;
        }
    }
    // ---- E. Drive model: what it found, and what it is worth
    if (drv_cal[0].on) {
        std::vector<std::pair<std::string, cv::Scalar>> lines = {
            {"Encoders sit on the motors: gear backlash and sag under gravity move the links. calico models both:", kInk},
            {"theta = q + dq + compliance * gravity torque + half backlash * gear flank (flank from the logged travel)", kGrey},
            {"", kInk}};
        const int nj = 6;
        std::ostringstream bl, st;
        bl << "backlash found / true (arcmin):";
        st << std::fixed << std::setprecision(0) << "stiffness found / true (N m/rad):";
        for (int j = 0; j < nj; ++j) {
            bl << "  J" << j + 1 << " " << std::fixed << std::setprecision(0) << 2 * drv_cal[0].h[j] / kDeg * 60 << "/"
               << kDrive[j].backlash_deg * 60;
            if (drv_cal[0].c[j] > 1e-9) st << "  J" << j + 1 << " " << 1.0 / drv_cal[0].c[j] << "/" << kDrive[j].stiffness;
        }
        lines.push_back({bl.str(), kCal});
        lines.push_back({st.str(), kCal});
        lines.push_back({"(J2 / J3 backlash and J1, J5, J6 sag are not excited by these poses: held at the prior)", kGrey});
        std::vector<CardBar> bars;
        if (notes && notes["drive_comparison"]) {
            const cv::Scalar cols[4] = {kNom, cv::Scalar(60, 140, 220), cv::Scalar(120, 160, 60), kCal};
            int k = 0;
            for (const auto& e : notes["drive_comparison"]) {
                std::ostringstream v;
                v << std::fixed << std::setprecision(1) << e[1].as<double>() << " mm";
                bars.push_back({e[0].as<std::string>(), e[1].as<double>(), v.str(), cols[std::min(k++, 3)]});
            }
        }
        hold(Card("Backlash and gravity", "same robot run, same images: only the model changes", lines, bars,
                  "gripper position error, p95 over 300 unseen poses:"),
             120);
    }
    // ---- F. Results
    if (notes && notes["results"]) {
        std::vector<std::pair<std::string, cv::Scalar>> lines;
        for (const auto& e : notes["results"]) {
            const std::string l = e.as<std::string>();
            lines.push_back({l, l.rfind("  ", 0) == 0 ? kGrey : kInk});
        }
        hold(Card("Results (simulation, scored against the truth)", "run with every effect on", lines), 150);
    }
    tag36h11_destroy(fam);
    std::cout << vw.n << " frames in " << fdir << "\n";
    return 0;
}

std::string Arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    const std::string p = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind(p, 0) == 0) return a.substr(p.size());
    }
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: calico-robocal-sim focus --out=DIR [--seed=N]\n"
                     "       calico-robocal-sim generate --out=DIR [--seed=N] [--scale=S] [--joint-noise-deg=D]\n"
                     "                                   [--stage1-frames=N] [--stage1-edges=0|1] [--route-poses=N]\n"
                     "                                   [--backlash=0|1] [--approach=random|unidirectional|against_gravity]\n"
                     "                                   [--eiv-deg=D] [--drive-model=0|1] [--only=stage2]\n"
                     "                                   [--lens=kb|double_sphere] [--lens-model=auto|equidistant|...]\n"
                     "                                   [--second-session=0|1] [--print=toner|clean]\n"
                     "       calico-robocal-sim evaluate --sim=DIR\n"
                     "       calico-robocal-sim video --sim=DIR [--out=DIR] [--notes=FILE]\n"
                     "       calico-robocal-sim boards --out=DIR [--frames=N]   (board type study)\n"
                     "       calico-robocal-sim boards-report --sim=DIR\n"
                     "focus writes the focus-ring sweeps (run calico-focus on them); generate locks each lens at the\n"
                     "sweep frame calico-focus picked, then renders the focus check, stage 1 and stage 2.\n";
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        {   // Prints: toner defects (default) or clean.
            BoardTexture::Toner& tn = BoardTexture::DefaultToner();
            tn.on = Arg(argc, argv, "print", "toner") != "clean";
            tn.seed = static_cast<unsigned>(std::stoul(Arg(argc, argv, "seed", "1")));
            if (tn.on) BoardTexture::DefaultRes() = 0.0001;  // resolve dot gain and ragged edges
        }
        if (cmd == "focus") {
            const std::string dir = Arg(argc, argv, "out");
            if (dir.empty()) throw std::runtime_error("focus needs --out=DIR");
            const unsigned seed = static_cast<unsigned>(std::stoul(Arg(argc, argv, "seed", "1")));
            MkDir(dir);
            Truth t = MakeTruth(std::stod(Arg(argc, argv, "scale", "1")));
            t.lens_family = Arg(argc, argv, "lens", "kb");
            t.pupil_e1_m = std::stod(Arg(argc, argv, "pupil-mm", "0")) * 1e-3;
            for (int c = 0; c < 3; ++c) t.SetFocus(c, t.lens[c].z_factory_m);
            apriltag_family_t* fam = tag36h11_create();
            for (int c = 0; c < 3; ++c) GenerateFocusSweep(dir, c, t, fam, seed * 100 + 50 + c);
            tag36h11_destroy(fam);
            return 0;
        }
        if (cmd == "generate") {
            const std::string dir = Arg(argc, argv, "out");
            if (dir.empty()) throw std::runtime_error("generate needs --out=DIR");
            const unsigned seed = static_cast<unsigned>(std::stoul(Arg(argc, argv, "seed", "1")));
            SimOptions opt;
            opt.scale = std::stod(Arg(argc, argv, "scale", "1"));
            opt.joint_noise_deg = std::stod(Arg(argc, argv, "joint-noise-deg", "0"));
            opt.stage1_frames = std::stoi(Arg(argc, argv, "stage1-frames", "80"));
            opt.stage1_edges = Arg(argc, argv, "stage1-edges", "1") != "0";
            opt.route_poses = std::stoi(Arg(argc, argv, "route-poses", "30"));
            opt.backlash = Arg(argc, argv, "backlash", "1") != "0";
            opt.approach = Arg(argc, argv, "approach", "random");
            if (opt.approach != "random" && opt.approach != "unidirectional" && opt.approach != "against_gravity") {
                throw std::runtime_error("--approach: random, unidirectional or against_gravity");
            }
            opt.drive_model = Arg(argc, argv, "drive-model", "1") != "0";
            opt.model_joint_noise_deg = std::stod(Arg(argc, argv, "eiv-deg", "0.03"));
            const bool stage2_only = Arg(argc, argv, "only", "") == "stage2";
            const bool stage1_only = Arg(argc, argv, "only", "") == "stage1";  // intrinsics sessions only
            const std::string only_cam = Arg(argc, argv, "cam", "");          // with stage1: one camera
            MkDir(dir);
            Truth t = MakeTruth(opt.scale);
            t.lens_family = Arg(argc, argv, "lens", "kb");
            if (t.lens_family != "kb" && t.lens_family != "double_sphere") throw std::runtime_error("--lens: kb or double_sphere");
            t.pupil_e1_m = std::stod(Arg(argc, argv, "pupil-mm", "0")) * 1e-3;
            opt.second_session = Arg(argc, argv, "second-session", "1") != "0";
            opt.lens_model = Arg(argc, argv, "lens-model", "auto");
            t.backlash = opt.backlash;
            t.approach = opt.approach;
            apriltag_family_t* fam = tag36h11_create();
            for (int c = 0; c < 3; ++c) {
                const double ring = LockedRing(dir, c);
                t.SetFocus(c, t.lens[c].z_factory_m + ring * LensTruth::kThreadPerDeg);
                std::cout << "focus " << kCams[c] << ": ring locked at " << ring << " deg -> focused at "
                          << t.lens[c].FocusDistance(t.lens[c].z_m) << " m\n";
                if (!stage2_only && !stage1_only) GenerateFocusCheck(dir, c, t, fam, seed * 100 + 60 + c);
            }
            for (int c = 0; c < 3 && !stage2_only; ++c) {
                if (!only_cam.empty() && only_cam != kCams[c]) continue;
                GenerateStage1(dir, c, t, fam, seed * 100 + c, opt);
                // A second, independent session of the same camera: the repeatability check.
                if (opt.second_session) GenerateStage1(dir, c, t, fam, seed * 100 + 40 + c, opt, "b");
            }
            if (!stage1_only) GenerateStage2(dir, t, fam, seed * 100 + 7, opt);
            tag36h11_destroy(fam);
            WriteTruth(dir + "/gt.yaml", t);
            std::cout << "simulation written to " << dir << "\n";
            return 0;
        }
        if (cmd == "video") {
            const std::string dir = Arg(argc, argv, "sim");
            if (dir.empty()) throw std::runtime_error("video needs --sim=DIR");
            return Video(dir, Arg(argc, argv, "out", dir + "/video"), Arg(argc, argv, "notes", ""));
        }
        if (cmd == "lens-error") {  // a camera_info against the truth lens (incl. decentering)
            const std::string dir = Arg(argc, argv, "sim");
            const std::string cam = Arg(argc, argv, "cam", "left");
            const std::string file = Arg(argc, argv, "file");
            const Truth t = ReadTruth(dir + "/gt.yaml");
            int c = 0;
            while (c < 3 && cam != kCams[c]) ++c;
            if (c == 3 || file.empty()) throw std::runtime_error("lens-error --sim=DIR --cam=left|right|center --file=X");
            Renderer truth;
            truth.cam = t.intr[c];
            truth.lens = RenderLens(t, c);
            truth.InitCircle();
            const CameraIntrinsics cal = LoadCameraInfo(file, cam);
            Stat by[3];
            for (double th = 2; th <= 90; th += 2) {
                for (double ph = 0; ph < 360; ph += 10) {
                    const Vec3 X(std::sin(th * kDeg) * std::cos(ph * kDeg), std::sin(th * kDeg) * std::sin(ph * kDeg),
                                 std::cos(th * kDeg));
                    double a[2], b[2];
                    truth.TrueProject(X, a);
                    if (a[0] < 0 || a[1] < 0 || a[0] >= t.intr[c].width || a[1] >= t.intr[c].height) continue;
                    ProjectPoint(cal.model, cal.params.data(), Eigen::Matrix<double, 3, 1>(X), b);
                    by[th <= 45 ? 0 : (th <= 75 ? 1 : 2)].add(std::hypot(a[0] - b[0], a[1] - b[1]));
                }
            }
            std::cout << std::fixed << std::setprecision(3) << cam << " " << LensModelName(cal.model)
                      << ": lens error p95 0-45 " << by[0].p95() << ", 45-75 " << by[1].p95() << ", 75-90 " << by[2].p95()
                      << " px\n";
            return 0;
        }
        if (cmd == "boards") {
            const std::string dir = Arg(argc, argv, "out");
            if (dir.empty()) throw std::runtime_error("boards needs --out=DIR");
            return BoardStudy(dir, std::stoi(Arg(argc, argv, "frames", "80")),
                              static_cast<unsigned>(std::stoul(Arg(argc, argv, "seed", "5"))));
        }
        if (cmd == "boards-report") {
            const std::string dir = Arg(argc, argv, "sim");
            if (dir.empty()) throw std::runtime_error("boards-report needs --sim=DIR");
            return BoardStudyReport(dir);
        }
        if (cmd == "evaluate") {
            const std::string dir = Arg(argc, argv, "sim");
            if (dir.empty()) throw std::runtime_error("evaluate needs --sim=DIR");
            return Evaluate(dir);
        }
        throw std::runtime_error("unknown command " + cmd);
    } catch (const std::exception& e) {
        std::cerr << "calico-robocal-sim: " << e.what() << "\n";
        return 1;
    }
}
