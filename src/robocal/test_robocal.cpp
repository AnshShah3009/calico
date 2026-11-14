// Synthetic end-to-end tests for robocal: a two-arm cell with wrist fisheye cameras, a static
// center camera and three boards (one flat), known joint zero offsets, written to disk in the
// calibration_toolkit formats (samples jsonl, camera_info yaml, URDF) and solved through Config.
#include <algorithm>
#include "robocal.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace robocal;

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
        o << "  <joint name=\"" << p << name << "\" type=\"" << type << "\">\n"
          << "    <parent link=\"" << p << parent << "\"/><child link=\"" << p << child << "\"/>\n"
          << "    <origin xyz=\"" << xyz << "\" rpy=\"0 0 0\"/>\n";
        if (!axis.empty()) o << "    <axis xyz=\"" << axis << "\"/><limit lower=\"-3\" upper=\"3\" effort=\"1\" velocity=\"1\"/>\n";
        o << "  </joint>\n";
    };
    o << "<robot name=\"" << p << "arm\">\n";
    for (const char* l : {"base_link", "link1", "link2", "link3", "link4", "link5", "link6", "end_effector"}) {
        o << "  <link name=\"" << p << l << "\"/>\n";
    }
    joint("joint1", "revolute", "base_link", "link1", "0 0 0.12", "0 0 1");
    joint("joint2", "revolute", "link1", "link2", "0 0 0", "0 1 0");
    joint("joint3", "revolute", "link2", "link3", "0 0 0.28", "0 1 0");
    joint("joint4", "revolute", "link3", "link4", "0 0 0.24", "0 0 1");
    joint("joint5", "revolute", "link4", "link5", "0 0 0", "0 1 0");
    joint("joint6", "revolute", "link5", "link6", "0 0 0.08", "0 0 1");
    joint("ee", "fixed", "link6", "end_effector", "0 0 0.10", "");
    o << "</robot>\n";
    return o.str();
}

std::vector<Vec3> GridPoints() {  // Kalibr 6x8 AprilGrid, 24 mm tags, spacing 0.3: tag corners
    std::vector<Vec3> pts;
    const double s = 0.024, g = 0.3 * s;
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 8; ++c) {
            const double x0 = c * (s + g), y0 = r * (s + g);
            pts.emplace_back(x0, y0, 0);
            pts.emplace_back(x0 + s, y0, 0);
            pts.emplace_back(x0 + s, y0 + s, 0);
            pts.emplace_back(x0, y0 + s, 0);
        }
    }
    return pts;
}

Iso3 LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 z = (target - eye).normalized();
    Vec3 x = up.cross(z);
    if (x.norm() < 1e-6) x = Vec3::UnitX().cross(z);
    x.normalize();
    const Vec3 y = z.cross(x);
    Iso3 T = Iso3::Identity();
    T.linear().col(0) = x;
    T.linear().col(1) = y;
    T.linear().col(2) = z;
    T.translation() = eye;
    return T;
}

struct Scene {
    std::string dir;
    std::vector<double> dq_true[2];
    Iso3 tip_cam_true[2];
    Iso3 base_true[2];
    Iso3 center_true;
    std::map<std::string, Iso3> boards;
    CameraIntrinsics kb;
};

void WriteCameraInfo(const std::string& path, const CameraIntrinsics& c) {
    std::ofstream f(path);
    f << "width: " << c.width << "\nheight: " << c.height << "\ndistortion_model: equidistant\n";
    f << std::setprecision(17) << "d: [" << c.params[4] << ", " << c.params[5] << ", " << c.params[6] << ", "
      << c.params[7] << "]\n";
    f << "k: [" << c.params[0] << ", 0, " << c.params[2] << ", 0, " << c.params[1] << ", " << c.params[3]
      << ", 0, 0, 1]\n";
}

void EmitMatrix(std::ostream& o, const Iso3& T) {
    o << "[";
    for (int r = 0; r < 4; ++r) {
        o << (r ? "," : "") << "[";
        for (int c = 0; c < 4; ++c) o << (c ? "," : "") << std::setprecision(17) << T.matrix()(r, c);
        o << "]";
    }
    o << "]";
}

// Writes samples / observations; returns the number of views written per source.
Scene BuildScene(const std::string& dir, unsigned seed, int n_poses) {
    Scene sc;
    sc.dir = dir;
    std::mt19937 rng(seed);
    std::normal_distribution<double> px_noise(0.0, 0.3);
    std::normal_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    sc.kb.model = LensModel::Equidistant;
    sc.kb.width = 1920;
    sc.kb.height = 1080;
    sc.kb.params = {641.0, 641.5, 958.0, 542.0, -0.0115, 0.0008, -0.0002, 0.00003};

    sc.dq_true[0] = {0.0, 2.85 * kDeg, -3.6 * kDeg, 4.0 * kDeg, 9.1 * kDeg, 0.0};
    sc.dq_true[1] = {0.0, 1.64 * kDeg, -3.9 * kDeg, -2.3 * kDeg, 15.5 * kDeg, 0.0};
    sc.tip_cam_true[0] = MakeIso(Vec3(0.045, 0.006, 0.03), Vec3(0.04, -0.12, 0.03));
    sc.tip_cam_true[1] = MakeIso(Vec3(0.043, -0.012, 0.032), Vec3(-0.03, -0.10, -0.05));
    sc.base_true[0] = MakeIso(Vec3(0.0, 0.3, 0.0), Vec3(0, 0, 0));
    sc.base_true[1] = MakeIso(Vec3(0.006, -0.304, 0.002), Vec3(0.004, -0.003, 0.018));
    sc.center_true = LookAt(Vec3(0.12, 0.0, 0.72), Vec3(0.42, 0.0, 0.0), Vec3(1, 0, 0));
    const Vec3 c0(-0.096, -0.072, 0);  // board centre offset (6x8 grid ~0.192 x 0.144 m)
    auto board = [&](const Vec3& centre, const Vec3& rpy) {
        Iso3 T = MakeIso(centre, rpy);
        T.translation() = centre + T.linear() * c0;
        return T;
    };
    sc.boards["flat"] = board(Vec3(0.40, 0.0, -0.03), Vec3(0, 0, 0.1));
    sc.boards["wedgeL"] = board(Vec3(0.46, 0.22, 0.02), Vec3(0.52, 0.0, -0.4));
    sc.boards["wedgeR"] = board(Vec3(0.46, -0.22, 0.02), Vec3(-0.52, 0.0, 0.4));

    const std::vector<Vec3> grid = GridPoints();
    const std::string joints[2] = {"left_", "right_"};

    auto project_view = [&](const Iso3& T_world_cam, const std::string& bname, std::ostream& out,
                            const std::vector<double>* q, const std::string& side, int idx) -> bool {
        const Iso3 T_wo = sc.boards.at(bname);
        const Iso3 T_cw = T_world_cam.inverse();
        const Vec3 n_w = T_wo.linear().col(2);
        std::vector<Vec3> obj;
        std::vector<Vec2> img;
        for (const Vec3& p : grid) {
            const Vec3 Xw = T_wo * p;
            if ((T_world_cam.translation() - Xw).dot(n_w) <= 0.02) continue;  // board faces the camera
            const Vec3 Xc = T_cw * Xw;
            if (std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) > 85.0 * kDeg) continue;
            double uv[2];
            ProjectPoint(sc.kb.model, sc.kb.params.data(), Eigen::Matrix<double, 3, 1>(Xc), uv);
            uv[0] += px_noise(rng);
            uv[1] += px_noise(rng);
            if (uv[0] < 2 || uv[1] < 2 || uv[0] > 1917 || uv[1] > 1077) continue;
            obj.push_back(p);
            img.emplace_back(uv[0], uv[1]);
        }
        if (obj.size() < 24) return false;
        // detector PnP: truth + ~1 mm / 0.2 deg
        Iso3 T_cb = T_cw * T_wo;
        T_cb.translation() += 0.001 * Vec3(unit(rng), unit(rng), unit(rng));
        T_cb.linear() = T_cb.linear() * Eigen::AngleAxisd(0.2 * kDeg, Vec3(unit(rng), unit(rng), unit(rng)).normalized()).toRotationMatrix();
        out << std::setprecision(17) << "{\"frame_idx\": " << idx << ", \"side\": \"" << side
            << "\", \"session\": \"" << bname << "\"";
        if (q) {
            out << ", \"q\": {";
            for (int j = 0; j < 6; ++j) out << (j ? ", " : "") << "\"" << side << "_joint" << j + 1 << "\": " << (*q)[j];
            out << "}, \"q6\": [";
            for (int j = 0; j < 6; ++j) out << (j ? ", " : "") << (*q)[j];
            out << "]";
        }
        out << ", \"corners\": [";
        for (size_t i = 0; i < img.size(); ++i) out << (i ? ", " : "") << "[" << img[i].x() << ", " << img[i].y() << "]";
        out << "], \"object_points\": [";
        for (size_t i = 0; i < obj.size(); ++i) out << (i ? ", " : "") << "[" << obj[i].x() << ", " << obj[i].y() << ", 0.0]";
        out << "], \"T_CW\": ";
        EmitMatrix(out, T_cb);
        out << ", \"n\": " << obj.size() / 4 << "}\n";
        return true;
    };

    for (int a = 0; a < 2; ++a) {
        const std::string side = a == 0 ? "left" : "right";
        std::ofstream urdf(dir + "/" + side + ".urdf");
        urdf << ArmUrdf(side + "_");
        urdf.close();
        const UrdfChain chain = UrdfChain::FromString(
            ArmUrdf(side + "_"), side + "_base_link", side + "_link6",
            {side + "_joint1", side + "_joint2", side + "_joint3", side + "_joint4", side + "_joint5", side + "_joint6"});
        std::ofstream out(dir + "/" + side + "_handeye_samples.jsonl");
        const double yaw0 = a == 0 ? -0.55 : 0.55;  // turn toward the cell centre
        int idx = 0;
        for (int k = 0; k < n_poses; ++k) {
            std::vector<double> q(6);
            q[0] = yaw0 + 0.55 * U(rng);
            q[1] = 0.70 + 0.35 * U(rng);
            q[2] = 1.00 + 0.40 * U(rng);
            q[3] = 0.6 * U(rng);
            q[4] = M_PI - q[1] - q[2] - 0.35 + 0.45 * U(rng);
            q[5] = 1.2 * U(rng);
            std::vector<double> theta(6);
            for (int j = 0; j < 6; ++j) theta[j] = q[j] + sc.dq_true[a][j];
            const Iso3 T_wc = sc.base_true[a] * chain.ForwardIso(theta) * sc.tip_cam_true[a];
            for (const auto& kv : sc.boards) {
                if (project_view(T_wc, kv.first, out, &q, side, idx)) ++idx;
            }
        }
    }
    std::ofstream cen(dir + "/center_samples.jsonl");
    for (int k = 0; k < 6; ++k) {
        for (const auto& kv : sc.boards) project_view(sc.center_true, kv.first, cen, nullptr, "center", k);
    }
    cen.close();
    for (const char* c : {"left", "right", "center"}) WriteCameraInfo(dir + "/" + c + ".camera_info.yaml", sc.kb);
    return sc;
}

void WriteConfig(const Scene& sc, const std::string& path, bool free_j1_left) {
    auto pose = [](std::ostream& o, const Iso3& T) {
        const Vec3 x = T.translation(), r = RotationToRpy(T.linear());
        o << std::setprecision(17) << "{xyz: [" << x.x() << ", " << x.y() << ", " << x.z() << "], rpy: [" << r.x()
          << ", " << r.y() << ", " << r.z() << "]}";
    };
    // Initial guesses: CAD-like, off from the truth.
    const Iso3 tip_init[2] = {sc.tip_cam_true[0] * MakeIso(Vec3(0.008, -0.006, 0.005), Vec3(0.03, -0.04, 0.05)),
                              sc.tip_cam_true[1] * MakeIso(Vec3(-0.007, 0.006, -0.004), Vec3(-0.04, 0.03, -0.03))};
    const Iso3 center_init = sc.center_true * MakeIso(Vec3(0.03, -0.02, 0.02), Vec3(0.04, 0.03, -0.05));
    std::ofstream o(path);
    o << "world_frame: base\narms:\n";
    for (int a = 0; a < 2; ++a) {
        const std::string s = a == 0 ? "left" : "right";
        o << "  - name: " << s << "\n    urdf: " << s << ".urdf\n    base_link: " << s << "_base_link\n"
          << "    tip_link: " << s << "_link6\n    ee_link: " << s << "_end_effector\n"
          << "    joints: [" << s << "_joint1, " << s << "_joint2, " << s << "_joint3, " << s << "_joint4, " << s
          << "_joint5, " << s << "_joint6]\n";
        o << "    T_world_base: ";
        pose(o, a == 0 ? sc.base_true[0] : MakeIso(Vec3(0, -0.3, 0), Vec3(0, 0, 0)));
        o << "\n    base_fixed: " << (a == 0 ? "true" : "false") << "\n";
        o << "    delta_q_free: [" << (a == 0 && free_j1_left ? "true" : "false") << ", true, true, true, true, false]\n";
        o << "    delta_q_sigma_deg: 20\n    camera: " << s << "\n    samples: " << s << "_handeye_samples.jsonl\n";
        o << "    optical_in_body_rpy: [-1.5707963267948966, 0, -1.5707963267948966]\n";
        o << "    T_tip_cam_init: ";
        pose(o, tip_init[a]);
        o << "\n";
    }
    o << "cameras:\n";
    for (const char* c : {"left", "right"}) o << "  - name: " << c << "\n    camera_info: " << c << ".camera_info.yaml\n";
    o << "  - name: center\n    camera_info: center.camera_info.yaml\n    static: true\n"
      << "    observations: center_samples.jsonl\n    T_world_cam_init: ";
    pose(o, center_init);
    o << "\nlevel_board:\n  board: flat\n  sigma_deg: 0.3\nsolver:\n  huber_px: 1.0\n  holdout_every: 5\n";
}

std::string MakeTempDir() {
    char tmpl[] = "/tmp/robocal_test_XXXXXX";
    const char* d = mkdtemp(tmpl);
    if (!d) throw std::runtime_error("mkdtemp failed");
    return d;
}

void TestProjectionPast90() {
    std::cout << "[ RUN ] equidistant projection past 90 deg\n";
    const double p[8] = {641, 641, 960, 540, 0, 0, 0, 0};
    double uv[2];
    // 92.9 deg off-axis, on +x: must land right of the centre, ~ f * theta from it.
    const Eigen::Matrix<double, 3, 1> X(1.0, 0.0, -0.05);
    CHECK(ProjectPoint(LensModel::Equidistant, p, X, uv), "projection valid");
    const double theta = std::atan2(1.0, -0.05);
    CHECK(std::fabs(uv[0] - (960 + 641 * theta)) < 1e-9, "u = cx + f*theta, got " << uv[0]);
    const double pp[9] = {641, 641, 960, 540, 0, 0, 0, 0, 0};
    CHECK(!ProjectPoint(LensModel::PlumbBob, pp, X, uv), "pinhole must reject z < 0");
}

// Every wide-angle model: project -> unproject round trip out to 100 deg, and the equidistant
// approximation / repeatability helpers.
// kb4_pupil (non-central): a 3D point projects through the moving pupil; its pixel unprojects to
// the ray direction, and the point lies on the ray from (0, 0, z0(theta)) along it. With e1 = 0
// the model is the equidistant one; at infinity it is the equidistant one for any e1.
void TestPupilModel() {
    std::cout << "[ RUN ] kb4_pupil: 3D round trip at 0.1-1 m to 100 deg, e1 = 0 / infinity reduce to equidistant\n";
    const std::vector<double> eq = {667, 667.4, 958, 601, -0.0111, 0.0008, -0.00018, 0.00003};
    std::vector<double> pu = eq;
    pu.push_back(0.0012);  // e1 = 1.2 mm (real AR0234 + 200 deg M12)
    std::vector<double> pu0 = eq;
    pu0.push_back(0.0);
    double worst_ray = 0, worst_e0 = 0, worst_inf = 0, shift_near = 0;
    for (double dist : {0.1, 0.25, 1.0}) {
        for (double th = 0.5; th <= 100; th += 1.5) {
            for (double ph = 0; ph < 360; ph += 30) {
                const Eigen::Vector3d u(std::sin(th * M_PI / 180) * std::cos(ph * M_PI / 180),
                                        std::sin(th * M_PI / 180) * std::sin(ph * M_PI / 180), std::cos(th * M_PI / 180));
                const Eigen::Vector3d X = dist * u;
                double uv[2], uv0[2], uve[2];
                CHECK(ProjectPoint(LensModel::KannalaBrandtPupil, pu.data(), Eigen::Matrix<double, 3, 1>(X), uv),
                      "kb4_pupil projects " << th << " deg at " << dist << " m");
                Eigen::Vector3d b;
                CHECK(UnprojectPixel(LensModel::KannalaBrandtPupil, pu.data(), Eigen::Vector2d(uv[0], uv[1]), b),
                      "kb4_pupil unprojects " << th << " deg at " << dist << " m");
                b.normalize();
                const double thb = std::atan2(std::hypot(b.x(), b.y()), b.z());
                const Eigen::Vector3d o(0, 0, PupilOffset(LensModel::KannalaBrandtPupil, pu.data(), thb));
                const Eigen::Vector3d d = (X - o).normalized();
                worst_ray = std::max(worst_ray, std::atan2(b.cross(d).norm(), b.dot(d)) * 180 / M_PI);
                ProjectPoint(LensModel::KannalaBrandtPupil, pu0.data(), Eigen::Matrix<double, 3, 1>(X), uv0);
                ProjectPoint(LensModel::Equidistant, eq.data(), Eigen::Matrix<double, 3, 1>(X), uve);
                worst_e0 = std::max(worst_e0, std::hypot(uv0[0] - uve[0], uv0[1] - uve[1]));
                if (dist == 0.1 && th > 79 && th < 81) shift_near = std::max(shift_near, std::hypot(uv[0] - uve[0], uv[1] - uve[1]));
                double ud[2];
                ProjectDirection(LensModel::KannalaBrandtPupil, pu.data(), Eigen::Matrix<double, 3, 1>(u), ud);
                ProjectPoint(LensModel::Equidistant, eq.data(), Eigen::Matrix<double, 3, 1>(u), uve);
                worst_inf = std::max(worst_inf, std::hypot(ud[0] - uve[0], ud[1] - uve[1]));
            }
        }
    }
    std::cout << "      ray error " << worst_ray << " deg, e1 = 0 vs equidistant " << worst_e0 << " px, direction vs equidistant "
              << worst_inf << " px, pupil shift at 0.1 m / 80 deg " << shift_near << " px\n";
    CHECK(worst_ray < 1e-6, "kb4_pupil: point off its unprojected ray by " << worst_ray << " deg");
    CHECK(worst_e0 < 1e-9, "kb4_pupil with e1 = 0 differs from equidistant by " << worst_e0 << " px");
    CHECK(worst_inf < 1e-9, "kb4_pupil direction projection differs from equidistant by " << worst_inf << " px");
    CHECK(shift_near > 1.0, "kb4_pupil: e1 = 1.2 mm should move a point at 0.1 m / 80 deg by > 1 px, got " << shift_near);
}

void TestWideModels() {
    std::cout << "[ RUN ] wide-angle models: round trip to 100 deg, equidistant approximation\n";
    struct M {
        LensModel m;
        std::vector<double> p;
    };
    const std::vector<M> models = {
        {LensModel::Equidistant, {667, 667.4, 958, 601, -0.0111, 0.0008, -0.00018, 0.00003}},
        {LensModel::KannalaBrandtRT, {667, 667.4, 958, 601, -0.0111, 0.0008, -0.00018, 0.00003, 2e-4, -1.5e-4}},
        {LensModel::DoubleSphere, {527, 527.3, 958, 601, -0.17, 0.58}},
    };
    for (const M& mm : models) {
        double worst = 0;
        for (double th = 0.5; th <= 100; th += 1.5) {
            for (double ph = 0; ph < 360; ph += 20) {
                const Eigen::Vector3d X(std::sin(th * M_PI / 180) * std::cos(ph * M_PI / 180),
                                        std::sin(th * M_PI / 180) * std::sin(ph * M_PI / 180), std::cos(th * M_PI / 180));
                double uv[2];
                CHECK(ProjectPoint(mm.m, mm.p.data(), Eigen::Matrix<double, 3, 1>(X), uv),
                      LensModelName(mm.m) << " projects " << th << " deg");
                Eigen::Vector3d b;
                CHECK(UnprojectPixel(mm.m, mm.p.data(), Eigen::Vector2d(uv[0], uv[1]), b),
                      LensModelName(mm.m) << " unprojects " << th << " deg");
                const Eigen::Vector3d bn = b.normalized();
                worst = std::max(worst, std::atan2(bn.cross(X).norm(), bn.dot(X)) * 180 / M_PI);
            }
        }
        CHECK(worst < 1e-6, LensModelName(mm.m) << " round trip error " << worst << " deg");
        CHECK(KannalaBrandtFoldDeg(mm.m, mm.p.data()) >= 120.0, LensModelName(mm.m) << " must not fold within the lens");
        CameraIntrinsics c;
        c.model = mm.m;
        c.width = 1920;
        c.height = 1200;
        c.params = mm.p;
        double rms = 0, mx = 0;
        const CameraIntrinsics eq = FitEquidistant(c, 90, &rms, &mx);
        std::cout << "      " << LensModelName(mm.m) << ": equidistant approximation rms " << rms << " px, max " << mx
                  << " px\n";
        if (mm.m == LensModel::Equidistant) CHECK(mx < 1e-3, "equidistant approximates itself, max " << mx);
        double r2 = 0;
        CHECK(CompareIntrinsics(c, c, 90, &r2) < 1e-9, "identical intrinsics compare equal");
        const IntrinsicsComparison same = CompareIntrinsicsDetailed(c, c, 85);
        CHECK(same.max[0] < 1e-9 && same.max[1] < 1e-6 && same.rotation_deg < 1e-6, "identical: zero difference");
        // Principal point moved: the pp-matched difference vanishes, the rotation-aligned one
        // shrinks in rms (near the axis a pp shift is a rotation; not far off axis), the raw one is the shift.
        CameraIntrinsics s = c;
        s.params[2] += 0.2;
        s.params[3] -= 0.2;
        const IntrinsicsComparison d = CompareIntrinsicsDetailed(c, s, 85);
        CHECK(std::fabs(d.rms[0] - 0.2 * std::sqrt(2.0)) < 1e-3, "pp shift: raw rms " << d.rms[0]);
        CHECK(d.max[2] < 1e-9, "pp shift: pp-matched max " << d.max[2]);
        CHECK(d.rms[1] < 0.5 * d.rms[0] && d.max[1] < d.max[0] && d.rotation_deg > 0.01,
              "pp shift: rotation-aligned rms " << d.rms[1] << " max " << d.max[1] << ", raw max " << d.max[0]);
    }
}

void TestFoldDetection() {
    std::cout << "[ RUN ] Kannala-Brandt fold detection\n";
    // k1 strongly negative: theta_d turns back near 60 deg; unprojection past it must refuse.
    const double p[8] = {640, 640, 960, 540, -0.35, 0, 0, 0};
    const double fold = KannalaBrandtFoldDeg(LensModel::Equidistant, p);
    CHECK(fold > 50 && fold < 70, "fold expected near 60 deg, got " << fold);
    double uv[2];
    const double th = (fold + 10) * M_PI / 180;
    ProjectPoint(LensModel::Equidistant, p, Eigen::Matrix<double, 3, 1>(std::sin(th), 0, std::cos(th)), uv);
    Eigen::Vector3d b;
    const bool ok = UnprojectPixel(LensModel::Equidistant, p, Eigen::Vector2d(uv[0], uv[1]), b);
    CHECK(!ok || std::atan2(std::hypot(b.x(), b.y()), b.z()) * 180 / M_PI < fold,
          "unprojection must not return an angle past the fold");
}

void TestJointSolve(bool free_j1_left) {
    std::cout << "[ RUN ] synthetic bimanual joint solve" << (free_j1_left ? " (J1 left freed: expect a weak direction)" : "")
              << "\n";
    const std::string dir = MakeTempDir();
    const Scene sc = BuildScene(dir, 7u, 40);
    WriteConfig(sc, dir + "/robocal.yaml", free_j1_left);
    Config cfg = Config::Load(dir + "/robocal.yaml");
    RobotCalibrator cal(cfg);
    cal.Load();
    cal.Initialize();
    cal.Solve();
    cal.WriteYaml(dir + "/robot_calib.yaml");
    std::cout << cal.Summary();
    const Diagnostics& d = cal.diagnostics();

    if (free_j1_left) {
        bool found = false;
        for (const WeakDirection& w : d.weak) {
            for (const auto& c : w.components) found = found || (c.first == "left.dq[1]" && std::fabs(c.second) > 0.3);
        }
        CHECK(found, "freeing left J1 (gauge with the board / base yaw) must show as a weak direction");
        return;
    }
    CHECK(d.weak.empty(), "no weak direction expected with J1 / J6 fixed");
    for (int a = 0; a < 2; ++a) {
        const std::vector<double> dq = cal.DeltaQ(a);
        for (int j = 0; j < 6; ++j) {
            const double err = std::fabs(dq[j] - sc.dq_true[a][j]) / kDeg;
            CHECK(err < 0.05, "arm " << a << " dq[" << j + 1 << "] error " << err << " deg");
        }
        const Iso3 E = sc.tip_cam_true[a].inverse() * cal.TipCam(a);
        CHECK(E.translation().norm() < 0.5e-3, "arm " << a << " tip_cam translation error " << E.translation().norm() * 1e3 << " mm");
        CHECK(RotationAngleDeg(E.linear(), Mat3::Identity()) < 0.05, "arm " << a << " tip_cam rotation error");
    }
    const Iso3 Eb = sc.base_true[1].inverse() * cal.WorldBase(1);
    CHECK(Eb.translation().norm() < 0.5e-3, "right base error " << Eb.translation().norm() * 1e3 << " mm");
    CHECK(RotationAngleDeg(Eb.linear(), Mat3::Identity()) < 0.05, "right base rotation error");
    const Iso3 Ec = sc.center_true.inverse() * cal.WorldCamera(cal.CameraIndex("center"));
    CHECK(Ec.translation().norm() < 1.0e-3, "center camera error " << Ec.translation().norm() * 1e3 << " mm");
    CHECK(RotationAngleDeg(Ec.linear(), Mat3::Identity()) < 0.05, "center camera rotation error");
    for (const char* c : {"left", "right", "center"}) {
        const ErrorStats& t = d.train.at(c);
        const ErrorStats& h = d.holdout.at(c);
        CHECK(t.rms_point > 0.3 && t.rms_point < 0.6, c << " train rms " << t.rms_point << " (0.3 px noise -> ~0.42)");
        CHECK(h.n_points == 0 || h.rms_point < 0.7, c << " holdout rms " << h.rms_point);
    }
    CHECK(d.covariance_ok, "covariance computed");
    // The written file is valid YAML with the expected top-level keys.
    YAML::Node out = YAML::LoadFile(dir + "/robot_calib.yaml");
    CHECK(out["schema"].as<std::string>() == "workcell_calib_v1", "schema key");
    CHECK(out["arms"]["left"]["delta_q_rad"].size() == 6, "left delta_q_rad");
    CHECK(out["sensors"]["center"]["T_base_cam"]["xyz"].size() == 3, "center pose");
}

}  // namespace

int main() {
    try {
        TestProjectionPast90();
        TestWideModels();
        TestPupilModel();
        TestFoldDetection();
        TestJointSolve(false);
        TestJointSolve(true);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failed) {
        std::cerr << g_failed << " check(s) failed\n";
        return 1;
    }
    std::cout << "all robocal tests passed\n";
    return 0;
}
