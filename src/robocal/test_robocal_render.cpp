// End-to-end test on rendered images: two arms with fisheye wrist cameras, a static center
// camera, four static AprilGrids (different layouts / tag sizes, disjoint id blocks, one flat).
// calico detects the tags itself, estimates board poses, fits the wrist intrinsics from a 2 %
// focal error, and solves joint offsets, mounts, base, camera and boards in one problem.
#include "render.hpp"
#include "robocal.hpp"

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <unistd.h>

using namespace robocal;
using namespace robocal::sim;

namespace {

constexpr double kDeg = M_PI / 180.0;
int g_failed = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "    CHECK FAILED: " << #cond << " -- " << msg << "\n"; \
            ++g_failed;                                                          \
        }                                                                        \
    } while (0)

std::string ArmUrdf(const std::string& p) {
    std::ostringstream o;
    auto joint = [&](const std::string& name, const std::string& type, const std::string& parent,
                     const std::string& child, const std::string& xyz, const std::string& axis) {
        o << "<joint name=\"" << p << name << "\" type=\"" << type << "\"><parent link=\"" << p << parent
          << "\"/><child link=\"" << p << child << "\"/><origin xyz=\"" << xyz << "\" rpy=\"0 0 0\"/>";
        if (!axis.empty()) o << "<axis xyz=\"" << axis << "\"/>";
        o << "</joint>\n";
    };
    o << "<robot name=\"" << p << "arm\">\n";
    for (const char* l : {"base_link", "link1", "link2", "link3", "link4", "link5", "link6"}) {
        o << "<link name=\"" << p << l << "\"/>\n";
    }
    joint("joint1", "revolute", "base_link", "link1", "0 0 0.12", "0 0 1");
    joint("joint2", "revolute", "link1", "link2", "0 0 0", "0 1 0");
    joint("joint3", "revolute", "link2", "link3", "0 0 0.28", "0 1 0");
    joint("joint4", "revolute", "link3", "link4", "0 0 0.24", "0 0 1");
    joint("joint5", "revolute", "link4", "link5", "0 0 0", "0 1 0");
    joint("joint6", "revolute", "link5", "link6", "0 0 0.08", "0 0 1");
    o << "</robot>\n";
    return o.str();
}

// Four static boards (one flat, one with a 2 mm bow), rendered textures.
std::vector<BoardTexture> StaticBoards(apriltag_family_t* fam) {
    std::vector<BoardTexture> boards(4);
    boards[0].spec = {"flat", "tag36h11", 6, 8, 0.024, 0.3, 0, true, 0.3};
    boards[0].T_world_board = Place(boards[0].spec, Vec3(0.40, 0.0, -0.03), Vec3(0, 0, 0.1));
    boards[1].spec = {"wedgeL", "tag36h11", 3, 4, 0.040, 0.3, 48, false, 0.3};
    boards[1].T_world_board = Place(boards[1].spec, Vec3(0.44, 0.24, 0.02), Vec3(0.5, 0.0, -0.4));
    boards[2].spec = {"wedgeR", "tag36h11", 5, 5, 0.030, 0.3, 60, false, 0.3};
    boards[2].T_world_board = Place(boards[2].spec, Vec3(0.44, -0.24, 0.02), Vec3(-0.5, 0.0, 0.4));
    boards[3].spec = {"wall", "tag36h11", 4, 6, 0.035, 0.3, 85, false, 0.3};
    boards[3].T_world_board = Place(boards[3].spec, Vec3(0.66, 0.0, 0.12), Vec3(0.0, -1.2, 0.0));
    boards[3].bow_x = 0.002;  // a bowed print
    for (BoardTexture& b : boards) b.Render(fam);
    return boards;
}

std::string BoardsYaml(const std::vector<BoardTexture>& boards) {
    std::ostringstream o;
    o << "boards:\n";
    for (const BoardTexture& b : boards) {
        const BoardSpec& s = b.spec;
        o << "  - {name: " << s.name << ", rows: " << s.rows << ", cols: " << s.cols << ", tag_size: " << s.tag_size
          << ", tag_spacing: " << s.tag_spacing << ", first_id: " << s.first_id << (s.level ? ", level: true" : "")
          << (b.bow_x != 0 ? ", fit_shape: true" : "");
        if (!s.mount_arm.empty()) {
            o << ", mount: {arm: " << s.mount_arm << ", link: " << s.mount_link << ", fixed: true, T_link_board_init: "
              << PoseYaml(s.T_link_board_init) << "}";
        }
        o << "}\n";
    }
    return o.str();
}

// Scenario A: two arms, wrist cameras, a static center camera, four static boards (one bowed)
// and a marker board on the left gripper with a known mount (makes the left J6 zero solvable).
void ArmsScenario(apriltag_family_t* fam, const CameraIntrinsics& kb, const Renderer& rend,
                  const std::string& calibrated_info) {
    std::cout << "[ RUN ] arms (intrinsics fixed from the hand-held session): wrist + center cameras, "
                 "4 static boards (1 bowed), gripper marker\n";
    char tmpl[] = "/tmp/robocal_render_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<BoardTexture> boards = StaticBoards(fam);
    BoardTexture marker;
    marker.spec = {"grip", "tag36h11", 3, 3, 0.030, 0.3, 120, false, 0.3};
    marker.spec.mount_arm = "left";
    marker.spec.mount_link = "left_link6";
    marker.spec.mount_fixed = true;
    marker.spec.has_mount_init = true;
    // Back of the wrist, facing away from the wrist camera (up when the camera looks down).
    marker.spec.T_link_board_init = Place(marker.spec, Vec3(-0.06, 0.0, -0.02), Vec3(M_PI, 0.0, 0.0));
    marker.Render(fam);
    boards.push_back(marker);

    const std::vector<double> dq_true[2] = {{0.0, 2.85 * kDeg, -3.6 * kDeg, 4.0 * kDeg, 9.1 * kDeg, 1.5 * kDeg},
                                            {0.0, 1.64 * kDeg, -3.9 * kDeg, -2.3 * kDeg, 15.5 * kDeg, 0.0}};
    const Iso3 tip_true[2] = {MakeIso(Vec3(0.045, 0.006, 0.03), Vec3(0.04, -0.12, 0.03)),
                              MakeIso(Vec3(0.043, -0.012, 0.032), Vec3(-0.03, -0.10, -0.05))};
    const Iso3 base_true[2] = {MakeIso(Vec3(0.0, 0.3, 0.0), Vec3(0, 0, 0)),
                               MakeIso(Vec3(0.006, -0.304, 0.002), Vec3(0.004, -0.003, 0.018))};
    const Iso3 center_true = LookAt(Vec3(0.10, 0.0, 0.70), Vec3(0.45, 0.0, 0.0), Vec3(1, 0, 0));
    const char* sides[2] = {"left", "right"};
    std::vector<UrdfChain> chains;
    for (int a = 0; a < 2; ++a) {
        const std::string s = sides[a];
        std::ofstream(dir + "/" + s + ".urdf") << ArmUrdf(s + "_");
        chains.push_back(UrdfChain::FromString(ArmUrdf(s + "_"), s + "_base_link", s + "_link6",
                                               {s + "_joint1", s + "_joint2", s + "_joint3", s + "_joint4",
                                                s + "_joint5", s + "_joint6"}));
    }
    auto fk = [&](int a, const std::vector<double>& q) {
        std::vector<double> th(6);
        for (int j = 0; j < 6; ++j) th[j] = q[j] + dq_true[a][j];
        return base_true[a] * chains[a].ForwardIso(th);
    };
    auto poses_with_marker = [&](const std::vector<double>& q_left) {
        std::vector<Iso3> poses;
        for (size_t i = 0; i + 1 < boards.size(); ++i) poses.push_back(boards[i].T_world_board);
        poses.push_back(fk(0, q_left) * marker.spec.T_link_board_init);
        return poses;
    };
    auto joints_yaml = [](const std::string& arm, const std::vector<double>& q) {
        std::ostringstream o;
        o << arm << ": [" << std::setprecision(17);
        for (int j = 0; j < 6; ++j) o << (j ? ", " : "") << q[j];
        o << "]";
        return o.str();
    };
    const std::vector<double> left_park = {-0.4, 0.2, 0.3, 0.0, 1.0, 0.0};

    std::ofstream manifest(dir + "/frames.yaml");
    manifest << "frames:\n";
    int n_wrist = 0, n_center = 0;
    for (int a = 0; a < 2; ++a) {
        const std::string s = sides[a];
        int kept = 0;
        for (int tries = 0; tries < 400 && kept < 26; ++tries) {
            std::vector<double> q(6);
            q[0] = (a == 0 ? -0.55 : 0.55) + 0.5 * U(rng);
            q[1] = 0.70 + 0.35 * U(rng);
            q[2] = 1.00 + 0.40 * U(rng);
            q[3] = 0.6 * U(rng);
            q[4] = M_PI - q[1] - q[2] - 0.35 + 0.45 * U(rng);
            q[5] = 1.2 * U(rng);
            const Iso3 T_wc = fk(a, q) * tip_true[a];
            int visible = 0;
            for (int i = 0; i < 4; ++i) visible += BoardInView(T_wc, boards[i], kb);
            if (visible == 0) continue;
            const std::vector<double>& q_left = a == 0 ? q : left_park;
            const std::vector<Iso3> poses = poses_with_marker(q_left);
            const std::string name = s + "_" + std::to_string(kept) + ".png";
            cv::imwrite(dir + "/" + name, rend.Render(T_wc, boards, poses, rng));
            manifest << "  - {camera: " << s << ", image: " << name << ", joints: {" << joints_yaml(s, q);
            if (a == 1) manifest << ", " << joints_yaml("left", left_park);
            manifest << "}}\n";
            ++kept;
            ++n_wrist;
            if (a == 0) {  // the center camera sees the gripper marker at this left-arm pose
                const std::string cname = "center_" + std::to_string(n_center++) + ".png";
                cv::imwrite(dir + "/" + cname, rend.Render(center_true, boards, poses, rng));
                manifest << "  - {camera: center, image: " << cname << ", joints: {" << joints_yaml("left", q) << "}}\n";
            }
        }
    }
    manifest.close();
    std::cout << "rendered " << n_wrist << " wrist + " << n_center << " center frames in " << dir << "\n";

    std::ofstream(dir + "/center.camera_info.yaml") << CameraInfoYaml(kb);
    const Iso3 tip_init[2] = {tip_true[0] * MakeIso(Vec3(0.008, -0.006, 0.005), Vec3(0.03, -0.04, 0.05)),
                              tip_true[1] * MakeIso(Vec3(-0.007, 0.006, -0.004), Vec3(-0.04, 0.03, -0.03))};
    std::ofstream cfg(dir + "/robocal.yaml");
    cfg << "frames: frames.yaml\ndetection: {cache: detections.jsonl}\narms:\n";
    for (int a = 0; a < 2; ++a) {
        const std::string s = sides[a];
        cfg << "  - name: " << s << "\n    urdf: " << s << ".urdf\n    base_link: " << s << "_base_link\n    tip_link: " << s
            << "_link6\n    joints: [" << s << "_joint1, " << s << "_joint2, " << s << "_joint3, " << s << "_joint4, " << s
            << "_joint5, " << s << "_joint6]\n    T_world_base: "
            << PoseYaml(a == 0 ? base_true[0] : MakeIso(Vec3(0, -0.3, 0), Vec3(0, 0, 0))) << "\n    base_fixed: "
            << (a == 0 ? "true" : "false") << "\n    delta_q_free: [false, true, true, true, true, "
            << (a == 0 ? "true" : "false") << "]\n    camera: " << s << "\n    T_tip_cam_init: " << PoseYaml(tip_init[a]) << "\n";
    }
    cfg << "cameras:\n";
    for (int a = 0; a < 2; ++a) cfg << "  - {name: " << sides[a] << ", camera_info: " << calibrated_info << "}\n";
    cfg << "  - {name: center, camera_info: " << calibrated_info << ", static: true}\n";
    cfg << BoardsYaml(boards);
    cfg << "solver: {huber_px: 1.0, holdout_every: 5}\n";
    cfg.close();

    Config config = Config::Load(dir + "/robocal.yaml");
    RobotCalibrator cal(config);
    cal.Load();
    cal.Initialize();
    cal.Solve();
    cal.WriteYaml(dir + "/robot_calib.yaml");
    std::cout << cal.Summary();

    for (int a = 0; a < 2; ++a) {
        const std::vector<double> dq = cal.DeltaQ(a);
        for (int j = 1; j < (a == 0 ? 6 : 5); ++j) {
            const double err = std::fabs(dq[j] - dq_true[a][j]) / kDeg;
            CHECK(err < (j == 5 ? 0.1 : 0.05), sides[a] << " dq[" << j + 1 << "] error " << err << " deg");
        }
        const Iso3 E = tip_true[a].inverse() * cal.TipCam(a);
        CHECK(E.translation().norm() < 1e-3, sides[a] << " camera mount error " << E.translation().norm() * 1e3 << " mm");
        CHECK(RotationAngleDeg(E.linear(), Mat3::Identity()) < 0.1, sides[a] << " camera mount rotation");
    }
    const Iso3 Eb = base_true[1].inverse() * cal.WorldBase(1);
    CHECK(Eb.translation().norm() < 1e-3, "right base error " << Eb.translation().norm() * 1e3 << " mm");
    const Iso3 Ec = center_true.inverse() * cal.WorldCamera(cal.CameraIndex("center"));
    CHECK(Ec.translation().norm() < 2e-3, "center camera error " << Ec.translation().norm() * 1e3 << " mm");
    CHECK(RotationAngleDeg(Ec.linear(), Mat3::Identity()) < 0.1, "center camera rotation");
    for (int i = 0; i < 4; ++i) {
        const Iso3 E = boards[i].T_world_board.inverse() * cal.WorldBoard(boards[i].spec.name);
        CHECK(E.translation().norm() < 1.5e-3, "board " << boards[i].spec.name << " error " << E.translation().norm() * 1e3 << " mm");
    }
    const double bow = cal.BoardShape("wall")[0];
    CHECK(std::fabs(bow - 0.002) < 0.3e-3, "wall bow " << bow * 1e3 << " mm (true 2.0)");
    for (const char* c : {"left", "right", "center"}) {
        CHECK(cal.diagnostics().train.at(c).rms_point < 0.5, c << " train rms " << cal.diagnostics().train.at(c).rms_point);
    }
    CHECK(cal.diagnostics().weak.empty(), "no weak directions expected (J6 is tied by the fixed marker mount)");

    // Second run reads the cached detections and must give the same answer.
    RobotCalibrator again(Config::Load(dir + "/robocal.yaml"));
    again.Load();
    again.Initialize();
    again.Solve();
    CHECK(std::fabs(again.DeltaQ(0)[4] - cal.DeltaQ(0)[4]) < 1e-9, "cached detections reproduce the solve");
}

// Scenario B: an intrinsics session -- one camera moved by hand over the static boards, no robot.
std::string HandheldScenario(apriltag_family_t* fam, const CameraIntrinsics& kb, const Renderer& rend) {
    std::cout << "[ RUN ] hand-held intrinsics session over 4 static boards (1 bowed)\n";
    char tmpl[] = "/tmp/robocal_handheld_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<BoardTexture> boards = StaticBoards(fam);
    std::ofstream manifest(dir + "/frames.yaml");
    manifest << "frames:\n";
    int kept = 0;
    for (int tries = 0; tries < 1500 && kept < 60; ++tries) {
        const Vec3 target(0.45 + 0.15 * U(rng), 0.25 * U(rng), 0.03);
        const Vec3 eye = target + Vec3(-0.15 + 0.1 * U(rng), 0.2 * U(rng), 0.0) +
                         Vec3(0, 0, 0.18 + 0.15 * (0.5 + 0.5 * U(rng)));
        Iso3 T_wc = LookAt(eye, target, Vec3(1, 0, 0));
        T_wc.linear() = T_wc.linear() * Eigen::AngleAxisd(1.5 * U(rng), Vec3::UnitZ()).toRotationMatrix();
        // Push half the frames off-axis so boards land near the image edge (coverage to ~85 deg).
        if (kept % 2) {
            T_wc.linear() = T_wc.linear() * Eigen::AngleAxisd(0.9 * U(rng), Vec3(U(rng), U(rng), 0).normalized()).toRotationMatrix();
        }
        int visible = 0;
        for (const BoardTexture& b : boards) visible += BoardInView(T_wc, b, kb, 84.0);
        if (visible < 2) continue;
        const std::string name = "hand_" + std::to_string(kept++) + ".png";
        cv::imwrite(dir + "/" + name, rend.Render(T_wc, boards, rng));
        manifest << "  - {camera: cam, image: " << name << "}\n";
    }
    manifest.close();
    std::ofstream cfg(dir + "/robocal.yaml");
    cfg << "frames: frames.yaml\nreference_board: flat\ncameras:\n"
        << "  - {name: cam, mode: handheld, model: equidistant, width: 1280, height: 720, f_init: 440.0, "
        << "intrinsics_prepass: true, optimize_intrinsics: true}\n"
        << BoardsYaml(boards) << "solver: {huber_px: 1.0, holdout_every: 5, bootstrap: 8}\n"
        << "intrinsics_gates: {max_holdout_rms_px: 0.5, min_theta_deg: 75, min_coverage: 0.6, max_curve_sd_px: 0.5}\n";
    cfg.close();
    RobotCalibrator cal(Config::Load(dir + "/robocal.yaml"));
    cal.Load();
    cal.Initialize();
    cal.Solve();
    cal.WriteYaml(dir + "/robot_calib.yaml");
    std::cout << cal.Summary();
    const std::vector<std::string> infos = cal.WriteCameraInfos(dir);
    CHECK(infos.size() == 1, "one camera_info exported");
    const LensReport& L = cal.diagnostics().lens.at("cam");
    CHECK(L.gated && L.pass, "intrinsics gates pass");
    CHECK(L.bootstrap_runs == 8 && L.max_curve_sd < 0.5, "bootstrap lens curve spread " << L.max_curve_sd << " px");
    const CameraIntrinsics& ci = cal.Intrinsics(0);
    CHECK(std::fabs(ci.params[0] / kb.params[0] - 1) < 1e-3, "hand-held fx " << ci.params[0] << " (true 430)");
    CHECK(std::fabs(ci.params[2] - kb.params[2]) < 0.5, "hand-held cx " << ci.params[2]);
    CHECK(std::fabs(ci.params[4] - kb.params[4]) < 2e-3, "hand-held k1 " << ci.params[4]);
    const Iso3 ref = boards[0].T_world_board;
    for (int i = 1; i < 4; ++i) {
        const Iso3 truth = ref.inverse() * boards[i].T_world_board;
        const Iso3 E = truth.inverse() * cal.WorldBoard(boards[i].spec.name);
        CHECK(E.translation().norm() < 1e-3, "hand-held board " << boards[i].spec.name << " relative pose error "
                                                                << E.translation().norm() * 1e3 << " mm");
    }
    const double bow = cal.BoardShape("wall")[0];
    CHECK(std::fabs(bow - 0.002) < 0.3e-3, "hand-held wall bow " << bow * 1e3 << " mm (true 2.0)");
    const ErrorStats& h = cal.diagnostics().holdout.at("cam");
    CHECK(h.n_views > 0 && h.rms_point < 0.5, "hand-held hold-out rms " << h.rms_point << " over " << h.n_views << " views");
    return infos.empty() ? std::string() : infos.front();
}

// Scenario C: static camera, boards moved independently in front of it (free_views).
void FreeViewsScenario(apriltag_family_t* fam, const CameraIntrinsics& kb, const Renderer& rend) {
    std::cout << "[ RUN ] free views: static camera, boards moved independently\n";
    char tmpl[] = "/tmp/robocal_freeviews_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<BoardTexture> boards = StaticBoards(fam);
    const Iso3 T_wc = Iso3::Identity();  // camera at the origin looking along +z
    std::ofstream manifest(dir + "/frames.yaml");
    manifest << "frames:\n";
    int kept = 0;
    for (int tries = 0; tries < 2000 && kept < 45; ++tries) {
        std::vector<Iso3> poses;
        int visible = 0;
        for (BoardTexture& b : boards) {
            // Somewhere in front of the camera, tilted toward it, up to ~80 deg off-axis.
            const double th = 80 * kDeg * std::sqrt(0.5 + 0.5 * U(rng)), ph = M_PI * U(rng);
            const double dist = 0.22 + 0.18 * (0.5 + 0.5 * U(rng));
            const Vec3 c = dist * Vec3(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
            const Iso3 face = LookAt(c, c - Vec3(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)),
                                     Vec3(0, 0, 1));
            Iso3 T = face;
            T.linear() = face.linear() * Eigen::AngleAxisd(0.6 * U(rng), Vec3(U(rng), U(rng), 0).normalized()).toRotationMatrix();
            // LookAt toward the camera: the print's z points at the camera (its front faces it).
            T.translation() = c - T.linear() * Vec3(b.spec.Width() / 2, b.spec.Height() / 2, 0);
            b.T_world_board = T;
            poses.push_back(T);
            visible += BoardInView(T_wc, b, kb, 84.0);
        }
        if (visible < 1) continue;
        const std::string name = "fv_" + std::to_string(kept++) + ".png";
        cv::imwrite(dir + "/" + name, rend.Render(T_wc, boards, poses, rng));
        manifest << "  - {camera: cam, image: " << name << "}\n";
    }
    manifest.close();
    std::ofstream cfg(dir + "/robocal.yaml");
    cfg << "frames: frames.yaml\ncameras:\n"
        << "  - {name: cam, mode: free_views, model: equidistant, width: 1280, height: 720, f_init: 440.0, "
        << "intrinsics_prepass: true, optimize_intrinsics: true}\n"
        << BoardsYaml(boards) << "solver: {huber_px: 1.0, holdout_every: 5, bootstrap: 6}\n"
        << "intrinsics_gates: {max_holdout_rms_px: 0.5, min_theta_deg: 70, min_coverage: 0.5, max_curve_sd_px: 0.5}\n";
    cfg.close();
    RobotCalibrator cal(Config::Load(dir + "/robocal.yaml"));
    cal.Load();
    cal.Initialize();
    cal.Solve();
    std::cout << cal.Summary();
    const CameraIntrinsics& ci = cal.Intrinsics(0);
    CHECK(std::fabs(ci.params[0] / kb.params[0] - 1) < 1e-3, "free-view fx " << ci.params[0] << " (true 430)");
    CHECK(std::fabs(ci.params[2] - kb.params[2]) < 0.5, "free-view cx " << ci.params[2]);
    const double bow = cal.BoardShape("wall")[0];
    CHECK(std::fabs(bow - 0.002) < 0.3e-3, "free-view wall bow " << bow * 1e3 << " mm (true 2.0)");
    CHECK(cal.diagnostics().lens.at("cam").pass, "free-view gates pass");
}

}  // namespace

int main() {
    try {
        CameraIntrinsics kb;
        kb.model = LensModel::Equidistant;
        kb.width = 1280;
        kb.height = 720;
        kb.params = {430.0, 430.4, 638.0, 362.0, -0.0115, 0.0008, -0.0002, 0.00003};
        Renderer rend{kb};
        rend.Init();
        apriltag_family_t* fam = tag36h11_create();
        const std::string info = HandheldScenario(fam, kb, rend);  // stage 1: intrinsics, exported
        FreeViewsScenario(fam, kb, rend);
        if (!info.empty()) ArmsScenario(fam, kb, rend, info);         // stage 2: intrinsics constant
        tag36h11_destroy(fam);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failed) {
        std::cerr << g_failed << " check(s) failed\n";
        return 1;
    }
    std::cout << "rendered robocal tests passed\n";
    return 0;
}
