#include "robocal.hpp"
#include "log_time.hpp"
#include "stage1_solver.hpp"

#include <ceres/ceres.h>
#include <sys/stat.h>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Eigenvalues>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace robocal {
namespace {

// cv::setNumThreads(1) for a scope (restores the previous setting).
struct CvSerial {
    int prev = cv::getNumThreads();
    CvSerial() { cv::setNumThreads(1); }
    ~CvSerial() { cv::setNumThreads(prev); }
};

// Frame / view pose q[4] then t[3] (PoseParam layout) as one parameter block.
using Pose7Manifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;

constexpr double kDeg = M_PI / 180.0;

template <typename T>
using M3 = Eigen::Matrix<T, 3, 3>;
template <typename T>
using V3 = Eigen::Matrix<T, 3, 1>;

// ----------------------------------------------------------------------------- cost functions

template <typename T>
bool ProjectViewResiduals(const View& v, LensModel model, const T* intr, const M3<T>& Rwc,
                          const V3<T>& twc, const M3<T>& Rwo, const V3<T>& two, T* r) {
    const M3<T> Rcw = Rwc.transpose();
    for (size_t i = 0; i < v.obj.size(); ++i) {
        const V3<T> Xw = Rwo * v.obj[i].cast<T>() + two;
        const V3<T> Xc = Rcw * (Xw - twc);
        T uv[2];
        if (!ProjectPoint(model, intr, Xc, uv)) return false;
        r[2 * i] = uv[0] - T(v.img[i].x());
        r[2 * i + 1] = uv[1] - T(v.img[i].y());
    }
    return true;
}

// Static camera: blocks cam_q, cam_t, board_q, board_t, intrinsics.
struct StaticViewCost {
    const View* view;
    LensModel model;
    bool pose7 = false;  // blocks: board pose in the camera (q, t) as one 7-block, intrinsics

    template <typename T>
    bool operator()(T const* const* p, T* r) const {
        if (pose7) {
            const Eigen::Map<const Eigen::Quaternion<T>> Qco(p[0]);
            const Eigen::Map<const V3<T>> tco(p[0] + 4);
            return ProjectViewResiduals(*view, model, p[1], M3<T>(M3<T>::Identity()), V3<T>(V3<T>::Zero()),
                                        M3<T>(Qco.normalized().toRotationMatrix()), V3<T>(tco), r);
        }
        const Eigen::Map<const Eigen::Quaternion<T>> Qwc(p[0]);
        const Eigen::Map<const V3<T>> twc(p[1]);
        const Eigen::Map<const Eigen::Quaternion<T>> Qwo(p[2]);
        const Eigen::Map<const V3<T>> two(p[3]);
        return ProjectViewResiduals(*view, model, p[4], M3<T>(Qwc.normalized().toRotationMatrix()),
                                    V3<T>(twc), M3<T>(Qwo.normalized().toRotationMatrix()),
                                    V3<T>(two), r);
    }
};

struct DeltaQPrior {
    std::vector<double> dq0;
    std::vector<double> inv_sigma;

    template <typename T>
    bool operator()(T const* const* p, T* r) const {
        for (size_t i = 0; i < dq0.size(); ++i) r[i] = (p[0][i] - T(dq0[i])) * T(inv_sigma[i]);
        return true;
    }
};

struct PosePrior {
    Eigen::Quaterniond q0;
    Vec3 t0;
    double inv_sigma_rot;  // 1 / rad
    double inv_sigma_t;    // 1 / m

    template <typename T>
    bool operator()(const T* q, const T* t, T* r) const {
        const Eigen::Map<const Eigen::Quaternion<T>> Q(q);
        const Eigen::Quaternion<T> d = q0.cast<T>().conjugate() * Q.normalized();
        const T s = d.w() < T(0) ? T(-2) : T(2);  // small-angle rotation vector
        for (int i = 0; i < 3; ++i) {
            r[i] = s * d.vec()[i] * T(inv_sigma_rot);
            r[3 + i] = (t[i] - T(t0[i])) * T(inv_sigma_t);
        }
        return true;
    }
};

// Flat board: its normal parallel to the world z axis (sign-free).
struct LevelPriorCost {
    double weight;

    template <typename T>
    bool operator()(const T* q, T* r) const {
        const Eigen::Map<const Eigen::Quaternion<T>> Q(q);
        const M3<T> R = Q.normalized().toRotationMatrix();
        r[0] = T(weight) * R(0, 2);
        r[1] = T(weight) * R(1, 2);
        return true;
    }
};

// Board shape prior: bow / twist (m) around zero, print scale (relative) around the known print scale.
struct ShapePrior {
    double inv_sigma_shape;
    double inv_sigma_scale;
    double scale0[2] = {0, 0};

    template <typename T>
    bool operator()(const T* s, T* r) const {
        for (int i = 0; i < 3; ++i) r[i] = s[i] * T(inv_sigma_shape);
        for (int i = 3; i < 5; ++i) r[i] = (s[i] - T(scale0[i - 3])) * T(inv_sigma_scale);
        return true;
    }
};

// ----------------------------------------------------------------------------- YAML helpers

std::string JoinPath(const std::string& dir, const std::string& p) {
    if (p.empty() || p[0] == '/' || dir.empty()) return p;
    if (p.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        return std::string(home ? home : "") + p.substr(1);
    }
    return dir + "/" + p;
}

std::string DirName(const std::string& path) {
    const size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

Vec3 ReadVec3(const YAML::Node& n) {
    if (!n || !n.IsSequence() || n.size() != 3) throw std::runtime_error("expected a 3-vector");
    return Vec3(n[0].as<double>(), n[1].as<double>(), n[2].as<double>());
}

Iso3 ReadMatrix4(const YAML::Node& m) {
    Iso3 T = Iso3::Identity();
    if (!m.IsSequence() || m.size() < 3) throw std::runtime_error("expected a 4x4 matrix");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.linear()(r, c) = m[r][c].as<double>();
        T.translation()[r] = m[r][3].as<double>();
    }
    return T;
}

// {xyz, rpy} | {matrix} | {translation_m, quaternion_xyzw} | bare 4x4.
Iso3 ReadPose(const YAML::Node& n) {
    if (!n) throw std::runtime_error("missing pose");
    if (n.IsSequence()) return ReadMatrix4(n);
    if (n["matrix"]) return ReadMatrix4(n["matrix"]);
    if (n["xyz"] || n["rpy"]) {
        return MakeIso(n["xyz"] ? ReadVec3(n["xyz"]) : Vec3::Zero(),
                       n["rpy"] ? ReadVec3(n["rpy"]) : Vec3::Zero());
    }
    if (n["translation_m"] && n["quaternion_xyzw"]) {
        const YAML::Node q = n["quaternion_xyzw"];
        Eigen::Quaterniond Q(q[3].as<double>(), q[0].as<double>(), q[1].as<double>(),
                             q[2].as<double>());
        Iso3 T = Iso3::Identity();
        T.linear() = Q.normalized().toRotationMatrix();
        T.translation() = ReadVec3(n["translation_m"]);
        return T;
    }
    throw std::runtime_error("pose needs xyz/rpy, matrix or translation_m/quaternion_xyzw");
}

template <typename T>
std::vector<T> ReadList(const YAML::Node& n) {
    std::vector<T> out;
    if (!n) return out;
    for (const auto& e : n) out.push_back(e.as<T>());
    return out;
}

}  // namespace

CameraIntrinsics LoadCameraInfo(const std::string& path, const std::string& name) {
    YAML::Node n = YAML::LoadFile(path);
    CameraIntrinsics c;
    c.name = name;
    c.width = n["width"] ? n["width"].as<int>() : (n["image_width"] ? n["image_width"].as<int>() : 0);
    c.height =
        n["height"] ? n["height"].as<int>() : (n["image_height"] ? n["image_height"].as<int>() : 0);
    std::vector<double> K, D;
    if (n["k"]) K = ReadList<double>(n["k"]);
    else if (n["K"]) K = ReadList<double>(n["K"]);
    else if (n["camera_matrix"]) K = ReadList<double>(n["camera_matrix"]["data"]);
    if (n["d"]) D = ReadList<double>(n["d"]);
    else if (n["D"]) D = ReadList<double>(n["D"]);
    else if (n["distortion_coefficients"]) D = ReadList<double>(n["distortion_coefficients"]["data"]);
    const std::string model = n["distortion_model"] ? n["distortion_model"].as<std::string>() : "";
    if (K.size() != 9) throw std::runtime_error(path + ": camera matrix must have 9 entries");
    c.model = LensModelFromName(model, D.size());
    const size_t nd = static_cast<size_t>(DistortionCount(c.model));
    if (D.size() > nd) throw std::runtime_error(path + ": too many distortion coefficients for " + model);
    D.resize(nd, 0.0);
    c.params = {K[0], K[4], K[2], K[5]};
    c.params.insert(c.params.end(), D.begin(), D.end());
    return c;
}

namespace {
struct RayFit {
    LensModel model;
    Eigen::Vector3d X;
    double u, v;
    template <typename T>
    bool operator()(T const* const* p, T* r) const {
        T uv[2];
        if (!ProjectDirection(model, p[0], Eigen::Matrix<T, 3, 1>(X.cast<T>()), uv)) return false;
        r[0] = uv[0] - T(u);
        r[1] = uv[1] - T(v);
        return true;
    }
};

std::vector<Eigen::Vector3d> RayFan(double theta_max_deg) {
    std::vector<Eigen::Vector3d> rays;
    for (double th = 1.0; th <= theta_max_deg + 1e-9; th += 1.0) {
        for (double ph = 0; ph < 360; ph += 7.5) {
            const double t = th * M_PI / 180.0, f = ph * M_PI / 180.0;
            rays.emplace_back(std::sin(t) * std::cos(f), std::sin(t) * std::sin(f), std::cos(t));
        }
    }
    return rays;
}

bool InImage(const CameraIntrinsics& c, const double* uv) {
    return std::isfinite(uv[0]) && uv[0] >= 0 && uv[1] >= 0 && uv[0] < c.width && uv[1] < c.height;
}
}  // namespace

CameraIntrinsics FitModel(const CameraIntrinsics& src, LensModel model, double theta_max_deg, double* rms, double* max,
                          double distance_m) {
    CameraIntrinsics out = src;
    out.model = model;
    out.params = {src.params[0], src.params[1], src.params[2], src.params[3]};
    for (double v : DefaultDistortion(model)) out.params.push_back(v);
    if (model == LensModel::DoubleSphere) {  // same focal scale near the axis: fx / (1 + xi)
        out.params[0] *= 0.8;
        out.params[1] *= 0.8;
        out.params[4] = -0.2;
        out.params[5] = 0.55;
    }
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector2d>> obs;
    for (const Eigen::Vector3d& X : RayFan(theta_max_deg)) {
        double uv[2];
        // distance_m > 0: the source projects points at that distance (non-central kb4_pupil: the central
        // approximation for a working range); else directions.
        const bool ok = distance_m > 0 ? ProjectPoint(src.model, src.params.data(), Eigen::Vector3d(X * distance_m), uv)
                                       : ProjectDirection(src.model, src.params.data(), X, uv);
        if (!ok || !InImage(src, uv)) continue;
        obs.push_back({X, Eigen::Vector2d(uv[0], uv[1])});
    }
    // Coarse to fine: the near-axis rays first (fixes the focal scale), then all of them.
    for (double lim : {std::min(45.0, theta_max_deg), theta_max_deg}) {
        ceres::Problem problem;
        for (const auto& ob : obs) {
            if (std::acos(std::clamp(ob.first.z(), -1.0, 1.0)) * 180.0 / M_PI > lim + 1e-9) continue;
            auto* cost = new ceres::DynamicAutoDiffCostFunction<RayFit, 4>(new RayFit{model, ob.first, ob.second.x(), ob.second.y()});
            cost->AddParameterBlock(static_cast<int>(out.params.size()));
            cost->SetNumResiduals(2);
            problem.AddResidualBlock(cost, nullptr, out.params.data());
        }
        ceres::Solver::Options o;
        o.linear_solver_type = ceres::DENSE_QR;
        o.max_num_iterations = 300;
        ceres::Solver::Summary sum;
        if (problem.NumResidualBlocks() > 0) ceres::Solve(o, &problem, &sum);
    }
    double sq = 0, mx = 0;
    for (const auto& ob : obs) {
        double uv[2];
        ProjectDirection(model, out.params.data(), ob.first, uv);
        const double e = std::hypot(uv[0] - ob.second.x(), uv[1] - ob.second.y());
        sq += e * e;
        mx = std::max(mx, e);
    }
    if (rms) *rms = obs.empty() ? 0.0 : std::sqrt(sq / obs.size());
    if (max) *max = mx;
    return out;
}

CameraIntrinsics FitEquidistant(const CameraIntrinsics& src, double theta_max_deg, double* rms, double* max) {
    return FitModel(src, LensModel::Equidistant, theta_max_deg, rms, max, 0.0);
}

double CompareIntrinsics(const CameraIntrinsics& a, const CameraIntrinsics& b, double theta_max_deg, double* rms) {
    double sq = 0, mx = 0;
    int n = 0;
    for (const Eigen::Vector3d& X : RayFan(theta_max_deg)) {
        double ua[2], ub[2];
        if (!ProjectDirection(a.model, a.params.data(), X, ua) || !ProjectDirection(b.model, b.params.data(), X, ub)) continue;
        if (!InImage(a, ua) || !InImage(b, ub)) continue;
        const double e = std::hypot(ua[0] - ub[0], ua[1] - ub[1]);
        sq += e * e;
        mx = std::max(mx, e);
        ++n;
    }
    if (rms) *rms = n ? std::sqrt(sq / n) : 0.0;
    return mx;
}

IntrinsicsComparison CompareIntrinsicsDetailed(const CameraIntrinsics& a, const CameraIntrinsics& b,
                                               double theta_max_deg, double distance_m) {
    // distance_m > 0: compare where points at that distance land (non-central models differ with
    // distance); else directions (points at infinity).
    auto proj = [distance_m](const CameraIntrinsics& m, const Eigen::Vector3d& X, double* uv) {
        return distance_m > 0 ? ProjectPoint(m.model, m.params.data(), Eigen::Vector3d(X * distance_m), uv)
                              : ProjectDirection(m.model, m.params.data(), X, uv);
    };
    IntrinsicsComparison c;
    c.theta_max_deg = theta_max_deg;
    for (int i = 0; i < 4; ++i) c.d_params[i] = b.params[i] - a.params[i];
    const std::vector<Eigen::Vector3d> fan = RayFan(theta_max_deg);
    // Rays both models project inside their images, with A's pixel.
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector2d>> rays;
    for (const Eigen::Vector3d& X : fan) {
        double ua[2], ub[2];
        if (!proj(a, X, ua) || !proj(b, X, ub)) continue;
        if (!InImage(a, ua) || !InImage(b, ub)) continue;
        rays.push_back({X, Eigen::Vector2d(ua[0], ua[1])});
    }
    // B with A's principal point (diagnostic: curve shape only).
    CameraIntrinsics bc = b;
    bc.params[2] = a.params[2];
    bc.params[3] = a.params[3];
    auto diff = [&](const CameraIntrinsics& m, const Eigen::Matrix3d& R, const Eigen::Vector3d& X,
                    const Eigen::Vector2d& ua, Eigen::Vector2d* e) {
        double u[2];
        const Eigen::Vector3d RX = R * X;
        if (!proj(m, RX, u)) return false;
        *e = Eigen::Vector2d(u[0], u[1]) - ua;
        return true;
    };
    // Best rotation of B's rays onto A (Gauss-Newton, numeric Jacobian): a camera rotation is
    // absorbed by the stage-2 mount / pose, so it is not a lens difference.
    Eigen::Vector3d w = Eigen::Vector3d::Zero();
    auto rot = [](const Eigen::Vector3d& v) {
        const double n = v.norm();
        return n < 1e-15 ? Eigen::Matrix3d::Identity() : Eigen::AngleAxisd(n, v / n).toRotationMatrix();
    };
    for (int it = 0; it < 8 && !rays.empty(); ++it) {
        Eigen::Matrix3d JtJ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Jtr = Eigen::Vector3d::Zero();
        const Eigen::Matrix3d R0 = rot(w);
        for (const auto& r : rays) {
            Eigen::Vector2d e0;
            if (!diff(b, R0, r.first, r.second, &e0)) continue;
            Eigen::Matrix<double, 2, 3> J;
            bool ok = true;
            for (int k = 0; k < 3 && ok; ++k) {
                Eigen::Vector3d wk = w;
                wk[k] += 1e-7;
                Eigen::Vector2d ek;
                ok = diff(b, rot(wk), r.first, r.second, &ek);
                J.col(k) = (ek - e0) / 1e-7;
            }
            if (!ok) continue;
            JtJ += J.transpose() * J;
            Jtr += J.transpose() * e0;
        }
        const Eigen::Vector3d step = -JtJ.ldlt().solve(Jtr);
        w += step;
        if (step.norm() < 1e-12) break;
    }
    c.rotation_deg = w.norm() * 180.0 / M_PI;
    const Eigen::Matrix3d Rw = rot(w);
    const int nring = static_cast<int>(std::ceil(theta_max_deg / 5.0));
    c.rings.assign(nring, {});
    double sq[3] = {0, 0, 0};
    int n = 0;
    for (const auto& r : rays) {
        Eigen::Vector2d er, ew, ec;
        if (!diff(b, Eigen::Matrix3d::Identity(), r.first, r.second, &er) || !diff(b, Rw, r.first, r.second, &ew) ||
            !diff(bc, Eigen::Matrix3d::Identity(), r.first, r.second, &ec)) {
            continue;
        }
        const double th = std::acos(std::clamp(r.first.z() / r.first.norm(), -1.0, 1.0)) * 180.0 / M_PI;
        IntrinsicsComparison::Ring& g = c.rings[std::min(nring - 1, static_cast<int>(th / 5.0 - 1e-9))];
        const double e[3] = {er.norm(), ew.norm(), ec.norm()};
        for (int k = 0; k < 3; ++k) {
            g.max[k] = std::max(g.max[k], e[k]);
            g.sq[k] += e[k] * e[k];
            c.max[k] = std::max(c.max[k], e[k]);
            sq[k] += e[k] * e[k];
        }
        ++g.n;
        ++n;
    }
    for (int k = 0; k < 3; ++k) c.rms[k] = n ? std::sqrt(sq[k] / n) : 0.0;
    c.n = n;
    return c;
}

namespace {

// One toolkit sample (jsonl line, or a frame of center_obs.yaml) -> view.
bool ParseSample(const YAML::Node& s, const std::vector<std::string>& joints, const std::string& default_board,
                 View& v) {
    v.frame_idx = s["frame_idx"] ? s["frame_idx"].as<int>() : 0;
    if (s["session"]) v.board = s["session"].as<std::string>();
    else if (s["board"]) v.board = s["board"].as<std::string>();
    else v.board = default_board;
    const YAML::Node img = s["corners"] ? s["corners"] : s["image_points"];
    const YAML::Node obj = s["object_points"];
    if (!img || !obj || img.size() != obj.size() || img.size() == 0) return false;
    for (size_t i = 0; i < img.size(); ++i) {
        v.img.emplace_back(img[i][0].as<double>(), img[i][1].as<double>());
        v.obj.emplace_back(obj[i][0].as<double>(), obj[i][1].as<double>(), obj[i][2].as<double>());
    }
    if (!joints.empty()) {
        v.q.clear();
        const YAML::Node qmap = s["q"];
        bool by_name = qmap && qmap.IsMap();
        if (by_name) {
            for (const std::string& j : joints) by_name = by_name && qmap[j];
        }
        if (by_name) {
            for (const std::string& j : joints) v.q.push_back(qmap[j].as<double>());
        } else if (s["q6"]) {
            v.q = ReadList<double>(s["q6"]);
        }
        if (v.q.size() != joints.size()) return false;
    }
    const YAML::Node T = s["T_CW"] ? s["T_CW"] : s["T_cam_board"];
    if (T && T.IsSequence() && T.size() >= 3) {
        v.has_T_cam_board = true;
        v.T_cam_board = ReadMatrix4(T);
    }
    return true;
}

void EmitPose(YAML::Emitter& out, const Iso3& T) {
    const Vec3 xyz = T.translation();
    const Vec3 rpy = RotationToRpy(T.linear());
    out << YAML::BeginMap;
    out << YAML::Key << "xyz" << YAML::Value << YAML::Flow << YAML::BeginSeq << xyz.x() << xyz.y()
        << xyz.z() << YAML::EndSeq;
    out << YAML::Key << "rpy" << YAML::Value << YAML::Flow << YAML::BeginSeq << rpy.x() << rpy.y()
        << rpy.z() << YAML::EndSeq;
    out << YAML::Key << "matrix" << YAML::Value << YAML::BeginSeq;
    const Eigen::Matrix4d M = T.matrix();
    for (int r = 0; r < 4; ++r) {
        out << YAML::Flow << YAML::BeginSeq;
        for (int c = 0; c < 4; ++c) out << M(r, c);
        out << YAML::EndSeq;
    }
    out << YAML::EndSeq << YAML::EndMap;
}

void EmitStats(YAML::Emitter& out, const ErrorStats& s) {
    out << YAML::BeginMap;
    out << YAML::Key << "n_views" << YAML::Value << s.n_views;
    out << YAML::Key << "n_points" << YAML::Value << s.n_points;
    out << YAML::Key << "rms_point_px" << YAML::Value << s.rms_point;
    out << YAML::Key << "rms_axis_px" << YAML::Value << s.rms_axis;
    out << YAML::Key << "median_px" << YAML::Value << s.median;
    out << YAML::Key << "p95_px" << YAML::Value << s.p95;
    out << YAML::Key << "max_px" << YAML::Value << s.max;
    out << YAML::EndMap;
}

ErrorStats StatsOf(std::vector<double> norms, int n_views) {
    ErrorStats s;
    s.n_views = n_views;
    s.n_points = static_cast<int>(norms.size());
    if (norms.empty()) return s;
    double sq = 0.0;
    for (double e : norms) sq += e * e;
    s.rms_point = std::sqrt(sq / norms.size());
    s.rms_axis = s.rms_point / std::sqrt(2.0);
    std::sort(norms.begin(), norms.end());
    s.median = norms[norms.size() / 2];
    s.p95 = norms[std::min(norms.size() - 1, static_cast<size_t>(0.95 * norms.size()))];
    s.max = norms.back();
    return s;
}

}  // namespace

// ----------------------------------------------------------------------------- Config

Config Config::Load(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    // Absolute: relative paths in the config (and the detection cache keys) must not depend on
    // the working directory the tool is started from.
    std::string dir = DirName(path);
    if (char* real = realpath(dir.c_str(), nullptr)) {
        dir = real;
        std::free(real);
    }
    Config cfg;
    if (root["world_frame"]) cfg.world_frame = root["world_frame"].as<std::string>();
    for (const auto& a : root["arms"]) {
        ArmConfig arm;
        arm.name = a["name"].as<std::string>();
        arm.urdf = JoinPath(dir, a["urdf"].as<std::string>());
        arm.base_link = a["base_link"].as<std::string>();
        arm.tip_link = a["tip_link"].as<std::string>();
        if (a["ee_link"]) arm.ee_link = a["ee_link"].as<std::string>();
        arm.joints = ReadList<std::string>(a["joints"]);
        const size_t n = arm.joints.size();
        if (a["T_world_base"]) arm.T_world_base = ReadPose(a["T_world_base"]);
        if (a["base_fixed"]) arm.base_fixed = a["base_fixed"].as<bool>();
        if (a["base_prior"]) {
            arm.base_prior_trans_m = a["base_prior"]["trans_m"].as<double>(0.0);
            arm.base_prior_rot_deg = a["base_prior"]["rot_deg"].as<double>(0.0);
        }
        arm.dq_init = a["delta_q_init_rad"] ? ReadList<double>(a["delta_q_init_rad"])
                                            : std::vector<double>(n, 0.0);
        if (a["delta_q_free"]) {
            for (const auto& e : a["delta_q_free"]) arm.dq_free.push_back(e.as<bool>());
        } else {
            arm.dq_free.assign(n, true);
            if (n > 0) arm.dq_free.front() = false;
            if (n > 1) arm.dq_free.back() = false;
        }
        const double sigma_deg = a["delta_q_sigma_deg"] && a["delta_q_sigma_deg"].IsScalar()
                                     ? a["delta_q_sigma_deg"].as<double>()
                                     : 20.0;
        if (a["delta_q_sigma_deg"] && a["delta_q_sigma_deg"].IsSequence()) {
            for (double s : ReadList<double>(a["delta_q_sigma_deg"])) arm.dq_sigma_rad.push_back(s * kDeg);
        } else {
            arm.dq_sigma_rad.assign(n, sigma_deg * kDeg);
        }
        if (arm.dq_init.size() != n || arm.dq_free.size() != n || arm.dq_sigma_rad.size() != n) {
            throw std::runtime_error("arm " + arm.name + ": delta_q lists must have one entry per joint");
        }
        if (const YAML::Node jn = a["joint_noise_deg"]) {
            if (jn.IsSequence()) {
                for (double v : ReadList<double>(jn)) arm.joint_noise_joint_rad.push_back(v * kDeg);
                if (arm.joint_noise_joint_rad.size() != n) {
                    throw std::runtime_error("arm " + arm.name + ": joint_noise_deg needs one entry per joint");
                }
            } else {
                arm.joint_noise_joint_rad.assign(n, jn.as<double>() * kDeg);
            }
            for (double v : arm.joint_noise_joint_rad) arm.joint_noise_rad = std::max(arm.joint_noise_rad, v);
        }
        if (a["camera"]) arm.camera = a["camera"].as<std::string>();
        if (a["samples"]) arm.samples = JoinPath(dir, a["samples"].as<std::string>());
        if (a["T_tip_cam_init"]) arm.T_tip_cam_init = ReadPose(a["T_tip_cam_init"]);
        if (a["optical_in_body_rpy"]) {
            arm.has_optical_in_body = true;
            arm.optical_in_body_rpy = ReadVec3(a["optical_in_body_rpy"]);
        }
        // Seed from the camera body mount (CAD / xacro), composed with body -> optical.
        if (a["T_tip_cam_body_init"]) {
            if (!arm.has_optical_in_body) {
                throw std::runtime_error("arm " + arm.name + ": T_tip_cam_body_init needs optical_in_body_rpy");
            }
            Iso3 T_body_opt = Iso3::Identity();
            T_body_opt.linear() = RpyToRotation(arm.optical_in_body_rpy);
            arm.T_tip_cam_init = ReadPose(a["T_tip_cam_body_init"]) * T_body_opt;
        }
        cfg.arms.push_back(arm);
    }
    for (const auto& c : root["cameras"]) {
        CameraConfig cam;
        cam.name = c["name"].as<std::string>();
        if (c["camera_info"]) cam.camera_info = JoinPath(dir, c["camera_info"].as<std::string>());
        if (c["model"]) cam.model = c["model"].as<std::string>();
        if (c["models"]) cam.models = ReadList<std::string>(c["models"]);
        if (c["width"]) cam.width = c["width"].as<int>();
        if (c["height"]) cam.height = c["height"].as<int>();
        if (c["f_init"]) cam.f_init = c["f_init"].as<double>();
        if (c["export_distance_m"]) cam.export_distance_m = c["export_distance_m"].as<double>();
        if (c["intrinsics_prepass"]) cam.intrinsics_prepass = c["intrinsics_prepass"].as<bool>();
        if (c["optimize_intrinsics"]) cam.optimize_intrinsics = c["optimize_intrinsics"].as<bool>();
        if (c["static"]) cam.is_static = c["static"].as<bool>();
        if (c["handheld"]) cam.handheld = c["handheld"].as<bool>();
        if (c["mode"]) {
            const std::string m = c["mode"].as<std::string>();
            cam.is_static = m == "static";
            cam.handheld = m == "handheld" || m == "free_views";
            cam.free_views = m == "free_views";
            if (m != "static" && m != "handheld" && m != "free_views" && m != "wrist") {
                throw std::runtime_error("camera " + cam.name + ": mode must be wrist, static, handheld or free_views");
            }
        }
        if (c["T_world_cam_init"]) cam.T_world_cam_init = ReadPose(c["T_world_cam_init"]);
        if (c["observations"]) cam.observations = JoinPath(dir, c["observations"].as<std::string>());
        if (c["board"]) cam.default_board = c["board"].as<std::string>();
        if (c["optical_in_body_rpy"]) {
            cam.has_optical_in_body = true;
            cam.optical_in_body_rpy = ReadVec3(c["optical_in_body_rpy"]);
        }
        cfg.cameras.push_back(cam);
    }
    for (const auto& b : root["boards"]) {
        BoardSpec spec;
        spec.name = b["name"].as<std::string>();
        spec.type = BoardTypeFromName(b["type"] ? b["type"].as<std::string>() : "aprilgrid");
        if (spec.type == BoardType::Charuco) spec.family = "DICT_5X5_1000";
        if (b["family"]) spec.family = b["family"].as<std::string>();
        if (b["dictionary"]) spec.family = b["dictionary"].as<std::string>();
        spec.rows = b["rows"].as<int>();
        spec.cols = b["cols"].as<int>();
        if (b["tag_size"]) spec.tag_size = b["tag_size"].as<double>();
        if (b["square_size"]) spec.square_size = b["square_size"].as<double>();
        if (b["marker_size"]) spec.marker_size = b["marker_size"].as<double>();
        if (b["legacy"]) spec.legacy = b["legacy"].as<bool>();
        if (b["tag_spacing"]) spec.tag_spacing = b["tag_spacing"].as<double>();
        if (b["first_id"]) spec.first_id = b["first_id"].as<int>();
        if (b["level"]) spec.level = b["level"].as<bool>();
        if (b["level_sigma_deg"]) spec.level_sigma_deg = b["level_sigma_deg"].as<double>();
        if (const YAML::Node m = b["mount"]) {
            spec.mount_arm = m["arm"].as<std::string>();
            if (m["link"]) spec.mount_link = m["link"].as<std::string>();
            if (m["T_link_board_init"]) {
                spec.has_mount_init = true;
                spec.T_link_board_init = ReadPose(m["T_link_board_init"]);
            }
            if (m["fixed"]) spec.mount_fixed = m["fixed"].as<bool>();
        }
        if (b["fit_shape"]) spec.fit_shape = b["fit_shape"].as<bool>();
        if (b["fit_scale"]) spec.fit_scale = b["fit_scale"].as<bool>();
        if (spec.fit_scale && !spec.fit_shape) {
            // The x / y scale lives in the board shape block, which is only free with fit_shape:
            // without this, fit_scale alone would silently hold the scale at print_scale_pct.
            std::cout << "[robocal] board " << spec.name << ": fit_scale needs the shape block, fit_shape enabled "
                      << "(bow / twist with their prior shape_sigma_mm)" << std::endl;
            spec.fit_shape = true;
        }
        if (b["fit_points"]) spec.fit_points = b["fit_points"].as<bool>();
        if (b["point_sigma_mm"]) spec.point_sigma_m = b["point_sigma_mm"].as<double>() * 1e-3;
        if (b["fit_warp"]) spec.fit_warp = b["fit_warp"].as<bool>();
        if (b["warp_sigma_mm"]) spec.warp_sigma_m = b["warp_sigma_mm"].as<double>() * 1e-3;
        if (b["shape_sigma_mm"]) spec.shape_sigma_m = b["shape_sigma_mm"].as<double>() * 1e-3;
        if (b["scale_sigma_pct"]) spec.scale_sigma = b["scale_sigma_pct"].as<double>() * 1e-2;
        if (const YAML::Node ps = b["print_scale_pct"]) {
            if (!ps.IsSequence() || ps.size() != 2)
                throw std::runtime_error("board " + spec.name + ": print_scale_pct needs [x, y]");
            spec.print_scale[0] = ps[0].as<double>() * 1e-2;
            spec.print_scale[1] = ps[1].as<double>() * 1e-2;
        }
        cfg.boards.push_back(spec);
    }
    ValidateBoards(cfg.boards);
    if (root["frames"]) cfg.frames = JoinPath(dir, root["frames"].as<std::string>());
    if (root["capture"] && root["capture"]["min_still_s"]) cfg.min_still_s = root["capture"]["min_still_s"].as<double>();
    if (const YAML::Node d = root["drive"]) {
        DriveConfig& dc = cfg.drive;
        dc.model = d["model"] ? d["model"].as<bool>() : true;
        if (d["approach"]) dc.approach = d["approach"].as<std::string>();
        if (dc.approach != "against_gravity" && dc.approach != "logged" && dc.approach != "none") {
            throw std::runtime_error("drive.approach: against_gravity, logged or none");
        }
        if (d["shared"]) dc.shared = d["shared"].as<bool>();
        if (d["tau_min_nm"]) dc.tau_min_nm = d["tau_min_nm"].as<double>();
        if (d["friction_nm"]) dc.friction_nm = ReadList<double>(d["friction_nm"]);
        if (d["backlash_prior_deg"]) dc.h_prior_deg = 0.5 * d["backlash_prior_deg"].as<double>();
        if (d["backlash_sigma_deg"]) dc.h_sigma_deg = 0.5 * d["backlash_sigma_deg"].as<double>();
        if (d["compliance_sigma"]) dc.c_sigma = ReadList<double>(d["compliance_sigma"]);
        if (d["gravity"]) {
            const std::vector<double> g = ReadList<double>(d["gravity"]);
            if (g.size() != 3) throw std::runtime_error("drive.gravity: [x, y, z] in the world frame");
            dc.gravity = Vec3(g[0], g[1], g[2]);
        }
    }
    if (root["reference_board"]) cfg.reference_board = root["reference_board"].as<std::string>();
    if (const YAML::Node g = root["intrinsics_gates"]) {
        IntrinsicsGates& G = cfg.gates;
        if (g["max_holdout_rms_px"]) G.max_holdout_rms_px = g["max_holdout_rms_px"].as<double>();
        if (g["min_theta_deg"]) G.min_theta_deg = g["min_theta_deg"].as<double>();
        if (g["min_coverage"]) G.min_coverage = g["min_coverage"].as<double>();
        if (g["max_curve_sd_px"]) G.max_curve_sd_px = g["max_curve_sd_px"].as<double>();
        if (g["max_mean_radial_px"]) G.max_mean_radial_px = g["max_mean_radial_px"].as<double>();
    }
    if (const YAML::Node d = root["detection"]) {
        if (d["subpixel"]) cfg.detect.subpixel = d["subpixel"].as<bool>();
        if (d["min_tags_per_board"]) cfg.detect.min_tags_per_board = d["min_tags_per_board"].as<int>();
        if (d["min_points_per_board"]) cfg.detect.min_points_per_board = d["min_points_per_board"].as<int>();
        if (d["apriltag"]) cfg.detect.apriltag = d["apriltag"].as<std::string>();
        if (d["refine"]) cfg.detect.refine = d["refine"].as<std::string>();
        if (d["gamma"]) cfg.detect.gamma = d["gamma"].as<double>();
        if (d["check_spacers"]) cfg.detect.check_spacers = d["check_spacers"].as<bool>();
        if (d["guided"]) cfg.detect.guided = d["guided"].as<bool>();
        if (d["rectified"]) cfg.detect.rectified = d["rectified"].as<bool>();
        if (d["rectified_cache"]) cfg.detect.rectified_cache = JoinPath(dir, d["rectified_cache"].as<std::string>());
        if (d["guided_rounds"]) cfg.detect.guided_rounds = d["guided_rounds"].as<int>();
        if (d["dump_rounds"]) cfg.dump_rounds_dir = JoinPath(dir, d["dump_rounds"].as<std::string>());
        if (d["cache"]) cfg.detections_cache = JoinPath(dir, d["cache"].as<std::string>());
        if (d["max_theta_deg"]) cfg.max_theta_deg = d["max_theta_deg"].as<double>();
        if (d["min_tag_px"]) cfg.min_tag_px = d["min_tag_px"].as<double>();
    }
    if (const YAML::Node l = root["level_board"]) {
        cfg.level.enabled = true;
        if (l["board"]) cfg.level.board = l["board"].as<std::string>();
        if (l["sigma_deg"]) cfg.level.sigma_deg = l["sigma_deg"].as<double>();
    }
    if (const YAML::Node s = root["solver"]) {
        SolverOptions& o = cfg.solver;
        if (s["huber_px"]) o.huber_px = s["huber_px"].as<double>();
        if (s["holdout_every"]) o.holdout_every = s["holdout_every"].as<int>();
        if (s["max_iterations"]) o.max_iterations = s["max_iterations"].as<int>();
        if (s["outlier_floor_px"]) o.outlier_floor_px = s["outlier_floor_px"].as<double>();
        if (s["outlier_mad_k"]) o.outlier_mad_k = s["outlier_mad_k"].as<double>();
        if (s["min_points_per_view"]) o.min_points_per_view = s["min_points_per_view"].as<int>();
        if (s["threads"]) o.threads = s["threads"].as<int>();
        if (s["verbose"]) o.verbose = s["verbose"].as<bool>();
        if (s["bootstrap"]) o.bootstrap = s["bootstrap"].as<int>();
        if (s["pixel_sigma_px"]) o.pixel_sigma_px = s["pixel_sigma_px"].as<double>();
        if (const YAML::Node w = s["view_weights"]) {  // true (= 2 rounds) | false | rounds
            const std::string t = w.as<std::string>();
            o.view_weight_rounds = t == "true" ? 2 : t == "false" ? 0 : w.as<int>();
        }
        if (const YAML::Node vd = s["view_distance_m"]) {  // [min, max] board-centre distance
            if (!vd.IsSequence() || vd.size() != 2) throw std::runtime_error("solver.view_distance_m needs [min, max]");
            o.view_distance_min_m = vd[0].as<double>();
            o.view_distance_max_m = vd[1].as<double>();
        }
        if (s["block_frames"]) o.block_frames = s["block_frames"].as<int>();
        if (s["robust"]) {
            o.robust = s["robust"].as<std::string>();
            if (o.robust != "corner" && o.robust != "view") throw std::runtime_error("solver.robust: corner | view");
        }
        if (const YAML::Node ob = s["observability"]) {  // auto | true | false
            const std::string t = ob.as<std::string>();
            o.observability = t == "auto" ? -1 : (t == "true" || t == "1") ? 1 : 0;
        }
        if (s["allow_intrinsics_in_robot_solve"]) {
            o.allow_intrinsics_in_robot_solve = s["allow_intrinsics_in_robot_solve"].as<bool>();
        }
        if (s["method"]) o.method = s["method"].as<std::string>();
        if (s["linear_solver"]) o.linear_solver = s["linear_solver"].as<std::string>();
        if (s["backend"]) {
            o.backend = s["backend"].as<std::string>();
            if (o.backend != "auto" && o.backend != "ceres" && o.backend != "schur_lm") {
                throw std::runtime_error("solver.backend: auto | ceres | schur_lm");
            }
        }
        if (s["global"]) o.global = s["global"].as<std::string>();
        if (s["global_iterations"]) o.global_iterations = s["global_iterations"].as<int>();
        if (s["global_temperature"]) o.global_temperature = s["global_temperature"].as<double>();
        if (s["global_step_deg"]) o.global_step_deg = s["global_step_deg"].as<double>();
        if (s["global_step_m"]) o.global_step_m = s["global_step_m"].as<double>();
        if (s["global_seed"]) o.global_seed = s["global_seed"].as<unsigned>();
        if (s["mount_init"]) o.mount_init = s["mount_init"].as<std::string>();
        if (s["max_rms_px"]) o.max_rms_px = s["max_rms_px"].as<double>();
        if (o.mount_init != "auto" && o.mount_init != "cad" && o.mount_init != "data") {
            throw std::runtime_error("solver.mount_init: auto, cad or data");
        }
        const std::set<std::string> methods = {"lm", "dogleg", "dogleg_subspace", "lbfgs", "bfgs", "nlcg"};
        if (!methods.count(o.method)) throw std::runtime_error("solver.method: lm, dogleg, dogleg_subspace, lbfgs, bfgs, nlcg");
        if (o.global != "none" && o.global != "multistart" && o.global != "basin_hopping") {
            throw std::runtime_error("solver.global: none, multistart, basin_hopping");
        }
    }
    return cfg;
}

std::map<std::string, double> Config::LevelBoards() const {
    std::map<std::string, double> out;
    for (const BoardSpec& b : boards) {
        if (b.level) out[b.name] = b.level_sigma_deg;
    }
    if (level.enabled) out[level.board] = level.sigma_deg;
    return out;
}

// ----------------------------------------------------------------------------- calibrator

RobotCalibrator::RobotCalibrator(Config cfg) : cfg_(std::move(cfg)) {}
RobotCalibrator::~RobotCalibrator() = default;

int RobotCalibrator::ArmIndex(const std::string& name) const {
    for (size_t i = 0; i < cfg_.arms.size(); ++i) {
        if (cfg_.arms[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

int RobotCalibrator::CameraIndex(const std::string& name) const {
    for (size_t i = 0; i < cfg_.cameras.size(); ++i) {
        if (cfg_.cameras[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

void RobotCalibrator::Load() {
    cams_.clear();
    for (const CameraConfig& c : cfg_.cameras) {
        CamState s;
        if (c.intrinsics_given) {
            s.intr = c.intrinsics;
        } else if (!c.camera_info.empty()) {
            s.intr = LoadCameraInfo(c.camera_info, c.name);
        } else {
            if (c.width <= 0 || c.height <= 0 || c.f_init <= 0) {
                throw std::runtime_error("camera " + c.name + ": give camera_info, or width / height / f_init");
            }
            s.intr.model = LensModelFromName(c.model, c.model == "rational_polynomial" ? 8 : 5);
            s.intr.width = c.width;
            s.intr.height = c.height;
            s.intr.params = {c.f_init, c.f_init, 0.5 * (c.width - 1), 0.5 * (c.height - 1)};
            for (double v : DefaultDistortion(s.intr.model)) s.intr.params.push_back(v);
        }
        s.intr.name = c.name;
        if (static_cast<int>(s.intr.params.size()) != s.intr.NumParams()) {
            throw std::runtime_error("camera " + c.name + ": intrinsics have the wrong size");
        }
        s.is_static = c.is_static;
        s.handheld = c.handheld;
        s.free_views = c.free_views;
        s.world_cam = PoseParam::FromIso(c.T_world_cam_init);
        cams_.push_back(s);
    }
    arms_.clear();
    for (const ArmConfig& a : cfg_.arms) {
        ArmState s;
        s.chain = a.urdf_xml.empty()
                      ? UrdfChain::FromFile(a.urdf, a.base_link, a.tip_link, a.joints)
                      : UrdfChain::FromString(a.urdf_xml, a.base_link, a.tip_link, a.joints);
        if (!a.ee_link.empty()) {
            const UrdfChain ee = a.urdf_xml.empty()
                                     ? UrdfChain::FromFile(a.urdf, a.tip_link, a.ee_link, {})
                                     : UrdfChain::FromString(a.urdf_xml, a.tip_link, a.ee_link, {});
            s.has_ee = true;
            s.T_tip_ee = ee.ForwardIso({});
        }
        s.camera = a.camera.empty() ? -1 : CameraIndex(a.camera);
        if (!a.camera.empty() && s.camera < 0) {
            throw std::runtime_error("arm " + a.name + ": unknown camera " + a.camera);
        }
        if (s.camera >= 0 && (cams_[s.camera].is_static || cams_[s.camera].handheld)) {
            throw std::runtime_error("arm " + a.name + ": camera " + a.camera + " is static / hand-held, not a wrist camera");
        }
        s.dq = a.dq_init;
        {
            const int n = s.chain.dof();
            s.drv.assign(2 * n, 0.0);
            for (int j = 0; j < n; ++j) s.drv[n + j] = cfg_.drive.h_prior_deg * kDeg;
            s.tau_ref.assign(n, 0.0);
            s.s_ref.assign(n, 0.0);
            if (cfg_.drive.model && s.chain.TotalMass() <= 0) {
                throw std::runtime_error("arm " + a.name + ": drive model needs <inertial> masses in the URDF");
            }
        }
        s.tip_cam = PoseParam::FromIso(a.T_tip_cam_init);
        s.world_base = PoseParam::FromIso(a.T_world_base);
        arms_.push_back(s);
    }
    if (cfg_.drive.model && cfg_.drive.shared) {
        // One drive block (arm 0's) serves every arm: the arms must have the same joints.
        for (const ArmState& a : arms_) {
            if (a.chain.dof() != arms_[0].chain.dof()) {
                throw std::runtime_error("drive.shared: true needs arms with the same number of joints (" +
                                         std::to_string(arms_[0].chain.dof()) + " vs " + std::to_string(a.chain.dof()) +
                                         "); set drive.shared: false");
            }
        }
    }
    views_.clear();
    for (size_t i = 0; i < cfg_.arms.size(); ++i) {
        if (!cfg_.arms[i].samples.empty()) LoadSamples(static_cast<int>(i), cfg_.arms[i].samples);
    }
    for (size_t i = 0; i < cfg_.cameras.size(); ++i) {
        if (cfg_.cameras[i].is_static && !cfg_.cameras[i].observations.empty()) {
            LoadObservations(static_cast<int>(i), cfg_.cameras[i].observations);
        }
    }
    if (!cfg_.frames.empty()) LoadFrames(cfg_.frames);
    FilterReliableTags(true);
    loaded_ = true;
}

// detection.max_theta_deg / min_tag_px: big tags near the optical axis only. Off-axis fisheye corners
// and small tags carry the largest lens-model and corner-localisation errors; in the robot solve the
// intrinsics are fixed, so those errors would go straight into the joint offsets and mounts.
// Angle: bearing of the tag centre (mean of its 4 corners) through the camera's lens; size: mean
// edge in pixels. Tags missing a corner are dropped; non-AprilGrid boards get the angle test per
// point. A board view with fewer than min_tags_per_board tags left is removed (erase) or dropped.
void RobotCalibrator::FilterReliableTags(bool erase) {
    if (cfg_.max_theta_deg <= 0 && cfg_.min_tag_px <= 0) return;
    std::map<std::string, BoardType> type;
    for (const BoardSpec& b : cfg_.boards) type[b.name] = b.type;
    const double max_th = cfg_.max_theta_deg > 0 ? cfg_.max_theta_deg * M_PI / 180.0 : M_PI;
    auto off_axis = [&](const CamState& cam, const Vec2& uv, double& th) {
        Vec3 b;
        if (!UnprojectPixel(cam.intr.model, cam.intr.params.data(), uv, b)) return false;
        th = std::acos(std::max(-1.0, std::min(1.0, b.z() / b.norm())));
        return true;
    };
    std::vector<long> before(cams_.size(), 0), after(cams_.size(), 0);
    std::vector<int> views_before(cams_.size(), 0), views_after(cams_.size(), 0);
    for (View& v : views_) {
        if (v.camera < 0 || v.dropped) continue;
        const CamState& cam = cams_[v.camera];
        before[v.camera] += static_cast<long>(v.img.size());
        ++views_before[v.camera];
        std::vector<char> keep(v.img.size(), 0);
        int tags = 0;
        const auto ty = type.find(v.board);
        if (ty != type.end() && ty->second == BoardType::AprilGrid && v.pids.size() == v.img.size()) {
            std::map<int, std::array<int, 4>> corners;  // tag -> index of each corner (-1: missing)
            for (size_t i = 0; i < v.pids.size(); ++i) {
                auto it = corners.find(v.pids[i] / 4);
                if (it == corners.end()) it = corners.emplace(v.pids[i] / 4, std::array<int, 4>{-1, -1, -1, -1}).first;
                it->second[v.pids[i] % 4] = static_cast<int>(i);
            }
            for (const auto& kv : corners) {
                const std::array<int, 4>& k = kv.second;
                if (k[0] < 0 || k[1] < 0 || k[2] < 0 || k[3] < 0) continue;
                Vec2 m = Vec2::Zero();
                double edge = 0;
                for (int e = 0; e < 4; ++e) {
                    m += 0.25 * v.img[k[e]];
                    edge += 0.25 * (v.img[k[e]] - v.img[k[(e + 1) % 4]]).norm();
                }
                double th = 0;
                if (edge < cfg_.min_tag_px || !off_axis(cam, m, th) || th > max_th) continue;
                for (int e = 0; e < 4; ++e) keep[k[e]] = 1;
                ++tags;
            }
        } else {
            int n_keep = 0;
            for (size_t i = 0; i < v.img.size(); ++i) {
                double th = 0;
                if (off_axis(cam, v.img[i], th) && th <= max_th) {
                    keep[i] = 1;
                    ++n_keep;
                }
            }
            tags = n_keep >= cfg_.detect.min_points_per_board ? std::max(1, cfg_.detect.min_tags_per_board) : 0;
        }
        // Guided corners sit at the end of the view (n_guided of them); the order is kept, so the
        // survivors among them stay at the end.
        const size_t first_guided = v.img.size() - std::min(v.img.size(), static_cast<size_t>(std::max(0, v.n_guided)));
        int guided_kept = 0;
        size_t n = 0;
        for (size_t i = 0; i < v.img.size(); ++i) {
            if (!keep[i]) continue;
            if (i >= first_guided) ++guided_kept;
            v.img[n] = v.img[i];
            v.obj[n] = v.obj[i];
            if (i < v.pids.size()) v.pids[n] = v.pids[i];
            ++n;
        }
        v.img.resize(n);
        v.obj.resize(n);
        if (v.pids.size() > n) v.pids.resize(n);
        v.n_guided = guided_kept;
        v.pw.clear();
        if (tags < std::max(1, cfg_.detect.min_tags_per_board)) {
            v.dropped = true;
            continue;
        }
        after[v.camera] += static_cast<long>(n);
        ++views_after[v.camera];
    }
    if (erase) {
        views_.erase(std::remove_if(views_.begin(), views_.end(), [](const View& v) { return v.dropped; }), views_.end());
    }
    std::cout << "[robocal] reliable tags (<= " << cfg_.max_theta_deg << " deg off-axis, >= " << cfg_.min_tag_px << " px edge):";
    for (size_t c = 0; c < cams_.size(); ++c) {
        if (views_before[c]) {
            std::cout << " " << cfg_.cameras[c].name << " " << after[c] << "/" << before[c] << " corners, " << views_after[c] << "/"
                      << views_before[c] << " views;";
        }
    }
    std::cout << robocal::Took() << std::endl;
}

void RobotCalibrator::LoadSamples(int arm, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("samples not found: " + path);
    std::string line;
    int n = 0, bad = 0;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        View v;
        v.arm = arm;
        v.camera = arms_[arm].camera;
        if (v.camera < 0) throw std::runtime_error("arm " + cfg_.arms[arm].name + " has samples but no camera");
        if (ParseSample(YAML::Load(line), cfg_.arms[arm].joints, "0", v)) {
            views_.push_back(v);
            ++n;
        } else {
            ++bad;
        }
    }
    std::cout << "[robocal] " << cfg_.arms[arm].name << ": " << n << " views from " << path
              << (bad ? " (" + std::to_string(bad) + " unusable)" : "") << std::endl;
}

void RobotCalibrator::LoadObservations(int cam, const std::string& path) {
    const CameraConfig& c = cfg_.cameras[cam];
    int n = 0;
    auto add = [&](const YAML::Node& s, const std::string& board) {
        View v;
        v.camera = cam;
        if (ParseSample(s, {}, board, v)) {
            views_.push_back(v);
            ++n;
        }
    };
    if (path.size() > 6 && path.substr(path.size() - 6) == ".jsonl") {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("observations not found: " + path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r") != std::string::npos) add(YAML::Load(line), c.default_board);
        }
    } else {
        YAML::Node root = YAML::LoadFile(path);
        const std::string board = root["board"] ? root["board"].as<std::string>() : c.default_board;
        const YAML::Node frames = root["frames"] ? root["frames"] : root;
        for (const auto& f : frames) add(f, board);
    }
    std::cout << "[robocal] " << c.name << ": " << n << " static views from " << path << std::endl;
}

// Frame manifest: one entry per image.
//   {camera: left, image: frames/left_000.png, joints: {left: [q1..qn]}}   (or q: {joint: value})
// Wrist cameras need their arm's joints; static cameras need none. Relative image paths
// resolve against the manifest's directory.
void RobotCalibrator::LoadFrames(const std::string& path) {
    if (cfg_.boards.empty()) throw std::runtime_error("frames given but no boards configured");
    const std::string dir = DirName(path);
    std::vector<YAML::Node> entries;
    if (path.size() > 6 && path.substr(path.size() - 6) == ".jsonl") {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("frames not found: " + path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r") != std::string::npos) entries.push_back(YAML::Load(line));
        }
    } else {
        YAML::Node root = YAML::LoadFile(path);
        const YAML::Node list = root["frames"] ? root["frames"] : root;
        for (const auto& e : list) entries.push_back(e);
    }
    struct Job {
        int camera = -1, arm = -1, idx = 0;
        std::map<int, std::vector<double>> arm_q;
        std::map<int, std::vector<double>> travel;
        std::string image;
    };
    std::vector<Job> jobs;
    int not_still = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const YAML::Node& e = entries[i];
        // Joint states reach the recorder with a delay: a frame is only trusted when the arm had
        // been stationary for a while before the image and the joints were taken.
        if (cfg_.min_still_s > 0 && e["still_s"] && e["still_s"].as<double>() < cfg_.min_still_s) {
            ++not_still;
            continue;
        }
        Job j;
        j.idx = static_cast<int>(i);
        j.camera = CameraIndex(e["camera"].as<std::string>());
        if (j.camera < 0) throw std::runtime_error("frames: unknown camera " + e["camera"].as<std::string>());
        j.image = JoinPath(dir, e["image"].as<std::string>());
        for (size_t a = 0; a < arms_.size(); ++a) {
            if (arms_[a].camera == j.camera) j.arm = static_cast<int>(a);
            // Joints of every arm in the frame: by arm name, or by joint name under q.
            const ArmConfig& ac = cfg_.arms[a];
            std::vector<double> q;
            if (e["joints"] && e["joints"][ac.name]) {
                q = ReadList<double>(e["joints"][ac.name]);
            } else if (e["q"] && e["q"].IsMap()) {
                bool all = true;
                for (const std::string& n : ac.joints) all = all && e["q"][n];
                if (all) {
                    for (const std::string& n : ac.joints) q.push_back(e["q"][n].as<double>());
                }
            }
            if (!q.empty()) {
                if (q.size() != ac.joints.size()) {
                    throw std::runtime_error("frame " + j.image + ": arm " + ac.name + " needs " +
                                             std::to_string(ac.joints.size()) + " joints");
                }
                j.arm_q[static_cast<int>(a)] = q;
                if (e["travel_deg"] && e["travel_deg"][ac.name]) {
                    std::vector<double> tr = ReadList<double>(e["travel_deg"][ac.name]);
                    if (tr.size() != q.size()) throw std::runtime_error("frame " + j.image + ": travel_deg size");
                    for (double& x : tr) x *= kDeg;
                    j.travel[static_cast<int>(a)] = tr;
                }
            }
        }
        if (j.arm >= 0 && !j.arm_q.count(j.arm)) {
            throw std::runtime_error("frame " + j.image + ": wrist camera of arm " + cfg_.arms[j.arm].name +
                                     " needs that arm's joints");
        }
        jobs.push_back(j);
    }

    if (not_still > 0) {
        std::cout << "[robocal] " << not_still << " frame(s) dropped: arm stationary < " << cfg_.min_still_s
                  << " s before capture (joint states may not match the image)" << std::endl;
    }

    // Detections: from the cache when it exists (same images), else detected here and cached.
    struct Det {
        std::string board;
        std::vector<int> pids;
        std::vector<Vec2> pts;
    };
    std::vector<std::vector<Det>> dets(jobs.size());
    const std::string& cache = cfg_.detections_cache;
    // Cache signature: the frame images in order, the boards and the detector settings. A cache
    // written for anything else (frames added / removed, boards or detection options changed) is
    // refused instead of silently leaving frames without detections.
    const std::string signature = [&] {
        std::ostringstream o;
        o << std::setprecision(10) << "frames " << jobs.size() << "\n";
        for (const Job& j : jobs) o << j.image << "\n";
        for (const BoardSpec& b : cfg_.boards) {
            o << "board " << b.name << " " << static_cast<int>(b.type) << " " << b.family << " " << b.rows << " " << b.cols
              << " " << b.tag_size << " " << b.tag_spacing << " " << b.first_id << " " << b.square_size << " " << b.marker_size
              << " " << b.legacy << "\n";
        }
        const DetectOptions& d = cfg_.detect;
        o << "detect " << d.subpixel << " " << d.min_tags_per_board << " " << d.min_points_per_board << " "
          << d.max_subpixel_move << " " << d.apriltag << " " << d.refine << " " << d.gamma << " " << d.check_spacers << "\n";
        uint64_t h = 1469598103934665603ULL;  // FNV-1a
        for (unsigned char ch : o.str()) h = (h ^ ch) * 1099511628211ULL;
        std::ostringstream hex;
        hex << std::hex << std::setw(16) << std::setfill('0') << h;
        return hex.str();
    }();
    bool from_cache = false;
    if (!cache.empty()) {
        std::ifstream in(cache);
        if (in) {
            from_cache = true;
            bool signed_cache = false;
            std::string line;
            while (std::getline(in, line)) {
                if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
                const YAML::Node n = YAML::Load(line);
                if (n["cache_meta"]) {
                    const std::string sig = n["cache_meta"]["signature"].as<std::string>();
                    if (sig != signature) {
                        throw std::runtime_error("detections cache " + cache +
                                                 " was written for other frames, boards or detection settings (delete it)");
                    }
                    signed_cache = true;
                    continue;
                }
                const int k = n["frame"].as<int>();
                if (k < 0 || k >= static_cast<int>(jobs.size()) || n["image"].as<std::string>() != jobs[k].image) {
                    throw std::runtime_error("detections cache " + cache + " does not match the frames (delete it)");
                }
                Det d;
                d.board = n["board"].as<std::string>();
                if (n["pids"]) {
                    d.pids = ReadList<int>(n["pids"]);
                    const YAML::Node c = n["pts"];
                    if (c.size() != d.pids.size()) throw std::runtime_error("detections cache: bad point count");
                    for (size_t t = 0; t < c.size(); ++t) d.pts.push_back(Vec2(c[t][0].as<double>(), c[t][1].as<double>()));
                } else {  // older caches: AprilGrid tag ids + 4 corners each
                    const std::vector<int> ids = ReadList<int>(n["ids"]);
                    const YAML::Node c = n["corners"];
                    if (c.size() != 4 * ids.size()) throw std::runtime_error("detections cache: bad corner count");
                    int first = 0;
                    for (const BoardSpec& b : cfg_.boards) {
                        if (b.name == d.board) first = b.first_id;
                    }
                    for (size_t t = 0; t < ids.size(); ++t) {
                        for (int m = 0; m < 4; ++m) {
                            d.pids.push_back(4 * (ids[t] - first) + m);
                            d.pts.push_back(Vec2(c[4 * t + m][0].as<double>(), c[4 * t + m][1].as<double>()));
                        }
                    }
                }
                dets[k].push_back(d);
            }
            if (!signed_cache) {
                std::cout << "[robocal] warning: detections cache " << cache << " has no signature (older calico, or a filtered copy): "
                          << "frames, boards and detection settings are not checked against it" << std::endl;
            }
        }
    }
    if (!from_cache) {
        int missing = 0;
    const CvSerial cv_serial;  // frames run in parallel; OpenCV calls inside stay serial
#pragma omp parallel for schedule(dynamic) reduction(+ : missing) num_threads(cfg_.solver.threads)
        for (int k = 0; k < static_cast<int>(jobs.size()); ++k) {
            const cv::Mat img = cv::imread(jobs[k].image, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                ++missing;
                continue;
            }
            for (const BoardDetection& d : DetectBoards(img, cfg_.boards, cfg_.detect)) {
                dets[k].push_back({cfg_.boards[d.board].name, d.pids, d.pts});
            }
        }
        if (missing) throw std::runtime_error(std::to_string(missing) + " frame image(s) could not be read");
        if (!cache.empty()) {
            std::ofstream out(cache);
            out << std::setprecision(10);
            out << "{\"cache_meta\": {\"frames\": " << jobs.size() << ", \"signature\": \"" << signature << "\"}}\n";
            for (size_t k = 0; k < jobs.size(); ++k) {
                for (const Det& d : dets[k]) {
                    out << "{\"frame\": " << k << ", \"image\": \"" << jobs[k].image << "\", \"board\": \"" << d.board
                        << "\", \"pids\": [";
                    for (size_t t = 0; t < d.pids.size(); ++t) out << (t ? ", " : "") << d.pids[t];
                    out << "], \"pts\": [";
                    for (size_t t = 0; t < d.pts.size(); ++t) {
                        out << (t ? ", " : "") << "[" << d.pts[t].x() << ", " << d.pts[t].y() << "]";
                    }
                    out << "]}\n";
                }
            }
            std::cout << "[robocal] detections cached in " << cache << robocal::Took() << std::endl;
        }
    } else {
        std::cout << "[robocal] detections read from " << cache << robocal::Took() << std::endl;
    }

    const std::map<std::string, int> board_index = [&] {
        std::map<std::string, int> m;
        for (size_t b = 0; b < cfg_.boards.size(); ++b) m[cfg_.boards[b].name] = static_cast<int>(b);
        return m;
    }();
    std::map<std::string, int> per_board;
    int n_views = 0;
    for (size_t k = 0; k < jobs.size(); ++k) {
        const Job& j = jobs[k];
        frame_images_[j.camera].push_back({j.idx, j.image});
        for (const Det& d : dets[k]) {
            const auto bi = board_index.find(d.board);
            if (bi == board_index.end()) throw std::runtime_error("detections: unknown board " + d.board);
            const BoardSpec& spec = cfg_.boards[bi->second];
            View v;
            v.arm = j.arm;
            v.camera = j.camera;
            v.arm_q = j.arm_q;
            v.travel = j.travel;
            if (j.arm >= 0) v.q = j.arm_q.at(j.arm);
            v.frame_idx = j.idx;
            v.src_frame = j.idx;
            v.image = j.image;
            v.board = d.board;
            for (size_t t = 0; t < d.pids.size(); ++t) {
                v.obj.push_back(spec.Point(d.pids[t]));
                v.img.push_back(d.pts[t]);
                v.pids.push_back(d.pids[t]);
            }
            per_board[v.board]++;
            views_.push_back(std::move(v));
            ++n_views;
        }
    }
    std::cout << "[robocal] frames: " << jobs.size() << " images -> " << n_views << " board views (";
    for (const auto& kv : per_board) std::cout << " " << kv.first << ":" << kv.second;
    std::cout << " )" << robocal::Took() << std::endl;
}

// Board pose in the camera frame from one view: bearings rotated so the board sits on the
// axis of a virtual pinhole (valid for any field angle), IPPE (both solutions), then a
// pixel refinement through the real lens model; the lower-cost solution wins.
bool RobotCalibrator::EstimateBoardPose(const View& v, Iso3& T_cam_board, double& rms) const {
    const CamState& cam = cams_[v.camera];
    {
        // Points (nearly) on one line fix no pose: e.g. one row of chessboard corners.
        Eigen::Vector2d m = Eigen::Vector2d::Zero();
        for (const Vec3& o : v.obj) m += o.head<2>() / static_cast<double>(v.obj.size());
        Eigen::Matrix2d C = Eigen::Matrix2d::Zero();
        for (const Vec3& o : v.obj) C += (o.head<2>() - m) * (o.head<2>() - m).transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(C);
        if (v.obj.size() < 4 || es.eigenvalues()(0) < 0.02 * es.eigenvalues()(1)) return false;
    }
    std::vector<Vec3> bearings;
    std::vector<Vec3> objs;
    for (size_t i = 0; i < v.img.size(); ++i) {
        Vec3 b;
        if (UnprojectPixel(cam.intr.model, cam.intr.params.data(), v.img[i], b)) {
            bearings.push_back(b);
            objs.push_back(v.obj[i]);
        }
    }
    const std::vector<Iso3> candidates = BoardPoseCandidates(bearings, objs);
    if (candidates.empty()) return false;
    double best = std::numeric_limits<double>::infinity();
    for (const Iso3& cand : candidates) {
        PoseParam board = PoseParam::FromIso(cand);
        PoseParam ident;
        std::vector<double> intr = cam.intr.params;
        ceres::Problem problem;
        auto* cost = new ceres::DynamicAutoDiffCostFunction<StaticViewCost, 4>(new StaticViewCost{&v, cam.intr.model});
        for (int bs : {4, 3, 4, 3}) cost->AddParameterBlock(bs);
        cost->AddParameterBlock(static_cast<int>(intr.size()));
        cost->SetNumResiduals(static_cast<int>(2 * v.obj.size()));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(2.0), {ident.q, ident.t, board.q, board.t, intr.data()});
        problem.SetParameterBlockConstant(ident.q);
        problem.SetParameterBlockConstant(ident.t);
        problem.SetParameterBlockConstant(intr.data());
        problem.SetManifold(board.q, new ceres::EigenQuaternionManifold);
        ceres::Solver::Options o;
        o.linear_solver_type = ceres::DENSE_QR;
        o.max_num_iterations = 50;
        ceres::Solver::Summary sum;
        ceres::Solve(o, &problem, &sum);
        std::vector<double> r(2 * v.obj.size());
        const double* p[5] = {ident.q, ident.t, board.q, board.t, intr.data()};
        if (!StaticViewCost{&v, cam.intr.model}(p, r.data())) continue;
        double sq = 0.0;
        for (double e : r) sq += e * e;
        const double this_rms = std::sqrt(sq / v.obj.size());
        if (this_rms < best) {
            best = this_rms;
            T_cam_board = board.ToIso();
        }
    }
    rms = best;
    return std::isfinite(best) && T_cam_board.matrix().allFinite();
}

void RobotCalibrator::EstimateMissingPoses() {
    int n = 0, failed = 0, out_of_range = 0;
    for (View& v : views_) {
        if (v.dropped || v.has_T_cam_board) continue;
        Iso3 T;
        double rms = 0;
        if (EstimateBoardPose(v, T, rms)) {
            v.T_cam_board = T;
            v.has_T_cam_board = true;
            ++n;
            // solver.view_distance_m: keep only views whose board centre (mean of its points) lies
            // in [min, max] from the camera (fit the lens for a working range).
            if (cfg_.solver.view_distance_max_m > 0 && !v.obj.empty()) {
                Vec3 c = Vec3::Zero();
                for (const Vec3& o : v.obj) c += o;
                const double d = (T * (c / static_cast<double>(v.obj.size()))).norm();
                if (d < cfg_.solver.view_distance_min_m || d > cfg_.solver.view_distance_max_m) {
                    v.dropped = true;
                    ++out_of_range;
                }
            }
        } else {
            v.dropped = true;
            ++failed;
        }
    }
    if (n >= 20 && !distance_logged_) {  // board distance spread (solver.view_distance_m, near/far split)
        std::vector<double> ds;
        for (const View& v : views_) {
            if (v.dropped || !v.has_T_cam_board || v.obj.empty()) continue;
            Vec3 c = Vec3::Zero();
            for (const Vec3& o : v.obj) c += o;
            ds.push_back((v.T_cam_board * (c / static_cast<double>(v.obj.size()))).norm());
        }
        if (ds.size() >= 20) {
            std::sort(ds.begin(), ds.end());
            auto q = [&](double f) { return ds[static_cast<size_t>(f * (ds.size() - 1))]; };
            std::cout << "[robocal] board distance median " << std::setprecision(3) << q(0.5) << " m (p10 " << q(0.1)
                      << ", p90 " << q(0.9) << ", " << ds.size() << " views)" << std::endl;
            distance_logged_ = true;
        }
    }
    if (n || failed) {
        std::cout << "[robocal] board poses from PnP: " << n << " view(s)"
                  << (failed ? ", " + std::to_string(failed) + " failed (dropped)" : "")
                  << (out_of_range ? ", " + std::to_string(out_of_range) + " outside view_distance_m (dropped)" : "")
                  << robocal::Took() << std::endl;
    }
    diag_.dropped_views += failed + out_of_range;
}

// Classic per-camera calibration: intrinsics + one free board pose per view (training views).
void RobotCalibrator::IntrinsicsPrepass(int ci) {
    CamState& cam = cams_[ci];
    std::vector<View*> vs;
    for (View& v : views_) {
        if (v.camera == ci && !v.dropped && !v.holdout && v.has_T_cam_board) vs.push_back(&v);
    }
    if (vs.size() < 3) {
        std::cout << "[robocal] " << cfg_.cameras[ci].name << ": too few views for the intrinsics prepass" << std::endl;
        return;
    }
    std::vector<PoseParam> poses(vs.size());
    PoseParam ident;
    ceres::Problem problem;
    for (size_t i = 0; i < vs.size(); ++i) {
        poses[i] = PoseParam::FromIso(vs[i]->T_cam_board);
        // One 7-parameter block per view pose: Schur eliminates whole poses (separate q / t blocks
        // leave every translation in the dense reduced system: O(views^3) per iteration).
        auto* cost = new ceres::DynamicAutoDiffCostFunction<StaticViewCost, 4>(new StaticViewCost{vs[i], cam.intr.model, true});
        cost->AddParameterBlock(7);
        cost->AddParameterBlock(cam.intr.NumParams());
        cost->SetNumResiduals(static_cast<int>(2 * vs[i]->obj.size()));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(cfg_.solver.huber_px > 0 ? cfg_.solver.huber_px : 1.0),
                                 {poses[i].q, cam.intr.params.data()});
        problem.SetManifold(poses[i].q, new Pose7Manifold);
    }
    (void)ident;
    ceres::Solver::Options o;
    o.linear_solver_type = ceres::DENSE_SCHUR;
    {
        auto* ord = new ceres::ParameterBlockOrdering;
        for (auto& p : poses) ord->AddElementToGroup(p.q, 0);
        ord->AddElementToGroup(cam.intr.params.data(), 1);
        o.linear_solver_ordering.reset(ord);
    }
    o.max_num_iterations = cfg_.solver.max_iterations;
    o.num_threads = cfg_.solver.threads;
    ceres::Solver::Summary sum;
    ceres::Solve(o, &problem, &sum);
    for (size_t i = 0; i < vs.size(); ++i) vs[i]->T_cam_board = poses[i].ToIso();
    std::cout << "[robocal] " << cfg_.cameras[ci].name << " intrinsics prepass (" << vs.size() << " views): cost "
              << std::scientific << std::setprecision(4) << sum.initial_cost << " -> " << sum.final_cost
              << std::fixed << std::setprecision(3) << ", fx " << cam.intr.params[0] << " fy " << cam.intr.params[1]
              << " cx " << cam.intr.params[2] << " cy " << cam.intr.params[3] << robocal::Took() << std::endl;
    // Views of this camera: refresh their PnP poses with the new lens for the initialization.
    for (View& v : views_) {
        if (v.camera != ci || v.dropped) continue;
        Iso3 T;
        double rms;
        if (EstimateBoardPose(v, T, rms)) v.T_cam_board = T;
    }
}

double* RobotCalibrator::DriveBlock(int arm) { return arms_[cfg_.drive.shared ? 0 : arm].drv.data(); }

// Gravity torque and gear flank per wrist view, at the current dq and base: the drive model's
// regressors (centred on the dataset's working load), the per-frame flank ambiguity, and which
// drive entries the data can move at all.
void RobotCalibrator::ComputeDriveRegressors(bool verbose) {
    if (!cfg_.drive.model) return;
    const DriveConfig& dc = cfg_.drive;
    struct Acc {
        std::vector<double> tau_sum, s_sum;
        std::vector<std::vector<double>> taus, ss;
    };
    std::map<int, Acc> acc;
    frame_amb_.clear();
    auto fric = [&](int j) { return j < static_cast<int>(dc.friction_nm.size()) ? dc.friction_nm[j] : 0.1; };
    for (View& v : views_) {
        if (v.arm < 0) continue;
        ArmState& a = arms_[v.arm];
        const int n = a.chain.dof();
        std::vector<double> th(n);
        for (int j = 0; j < n; ++j) th[j] = v.q[j] + a.dq[j];
        const std::vector<double> tau = a.chain.GravityTorques(th, a.world_base.ToIso().linear(), dc.gravity);
        std::vector<double> s(n, 0.0), amb(n, 0.0);
        const auto tr = v.travel.find(v.arm);
        for (int j = 0; j < n; ++j) {
            const double sg = tau[j] >= 0 ? 1.0 : -1.0;
            if (dc.approach == "against_gravity") {
                s[j] = std::fabs(tau[j]) >= dc.tau_min_nm ? sg : -1.0;  // fixed + approach: link trails, s = -1
            } else if (dc.approach == "logged" && tr != v.travel.end()) {
                const double d = tr->second[j] >= 0 ? 1.0 : -1.0;
                const double f = fric(j);
                const double w = 0.5 * (1.0 + std::tanh((std::fabs(tau[j]) - f) / (0.3 * f)));
                s[j] = w * sg - (1.0 - w) * d;
                amb[j] = sg != -d ? 4.0 * w * (1.0 - w) : 0.0;
                if (std::fabs(tr->second[j]) < 1.0 * kDeg) amb[j] = 1.0;  // gap may not be taken up
            } else {
                const bool loaded = std::fabs(tau[j]) > 3.0 * fric(j);
                s[j] = loaded ? sg : 0.0;
                amb[j] = loaded ? 0.0 : 1.0;
            }
        }
        v.dtau = tau;
        v.ds = s;
        v.amb = amb;
        frame_amb_[{v.arm, v.frame_idx}] = amb;
        Acc& ac = acc[v.arm];
        if (ac.tau_sum.empty()) ac.tau_sum.assign(n, 0.0), ac.s_sum.assign(n, 0.0);
        if (!v.holdout && !v.dropped) {
            for (int j = 0; j < n; ++j) ac.tau_sum[j] += tau[j], ac.s_sum[j] += s[j];
            ac.taus.push_back(tau);
            ac.ss.push_back(s);
        }
    }
    // Reference load / flank per arm; spread decides which entries can be estimated.
    std::map<int, std::vector<int>> movable;  // drive entry -> #arms with spread
    for (auto& kv : acc) {
        ArmState& a = arms_[kv.first];
        const int n = a.chain.dof();
        const double N = std::max<size_t>(1, kv.second.taus.size());
        std::vector<double> sd_tau(n, 0.0), sd_s(n, 0.0);
        std::vector<int> minority(n, 0);
        for (int j = 0; j < n; ++j) {
            a.tau_ref[j] = kv.second.tau_sum[j] / N;
            a.s_ref[j] = kv.second.s_sum[j] >= 0 ? 1.0 : -1.0;
            double vt = 0, vs = 0;
            int pos = 0, neg = 0;
            for (size_t k = 0; k < kv.second.taus.size(); ++k) {
                vt += std::pow(kv.second.taus[k][j] - a.tau_ref[j], 2);
                vs += std::pow(kv.second.ss[k][j] - kv.second.s_sum[j] / N, 2);
                pos += kv.second.ss[k][j] > 0.5;
                neg += kv.second.ss[k][j] < -0.5;
            }
            sd_tau[j] = std::sqrt(vt / N);
            sd_s[j] = std::sqrt(vs / N);
            minority[j] = std::min(pos, neg);
        }
        std::vector<int>& m = movable[kv.first];
        m.assign(2 * n, 0);
        for (int j = 0; j < n; ++j) {
            m[j] = sd_tau[j] >= 0.175;                        // c_j: >= 0.01 deg of sag spread at 1e-3 rad/N m
            m[n + j] = sd_s[j] >= 0.2 && minority[j] >= 4;    // h_j: both flanks seen
        }
        if (verbose) {
            std::cout << "[robocal] drive regressors " << cfg_.arms[kv.first].name << " (" << kv.second.taus.size()
                      << " frames): tau ref / sd (N m):";
            for (int j = 0; j < n; ++j) std::cout << " " << std::setprecision(2) << a.tau_ref[j] << "/" << sd_tau[j];
            std::cout << "; minority flank frames:";
            for (int j = 0; j < n; ++j) std::cout << " " << minority[j];
            std::cout << std::endl;
        }
    }
    for (View& v : views_) {
        if (v.arm < 0 || v.dtau.empty()) continue;
        const ArmState& a = arms_[v.arm];
        for (size_t j = 0; j < v.dtau.size(); ++j) {
            v.dtau[j] -= a.tau_ref[j];
            v.ds[j] -= a.s_ref[j];
        }
    }
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        const int n = arms_[ai].chain.dof();
        std::vector<int> held;
        for (int e = 0; e < 2 * n; ++e) {
            bool any = false;
            if (dc.shared) {
                for (const auto& kv : movable) any = any || kv.second[e];
            } else if (movable.count(static_cast<int>(ai))) {
                any = movable[static_cast<int>(ai)][e];
            }
            if (!any) held.push_back(e);
        }
        arms_[ai].drv_const = held;
    }
}

// Iterative refinement around a fitted lens (TartanCalib-style, refined on the original pixels):
//   1. re-detect tags in virtual pinhole views of every frame (rectified) -- tags too squeezed to
//      decode at the fisheye edge decode there; new corners join their board's view, boards seen
//      only this way become new views;
//   2. (boards fixed to each other) predict boards that decoded nothing from the board layout
//      seen in other frames;
//   3. predict every remaining corner of every posed view and refine it (X-junction fit).
// Returns the number of corners added.
// detection.dump_rounds: <dir>/<camera>_round<k>.csv with every corner in use after detection
// (k = 0) and after each guided round, plus the lens at that point (tools/robocal_rounds.py).
void RobotCalibrator::DumpCorners(int ci, int round) const {
    if (cfg_.dump_rounds_dir.empty()) return;
    mkdir(cfg_.dump_rounds_dir.c_str(), 0755);
    const std::string base = cfg_.dump_rounds_dir + "/" + cfg_.cameras[ci].name + "_round" + std::to_string(round);
    std::ofstream o(base + ".csv");
    o << "image,board,view,pid,u,v,holdout,guided\n" << std::setprecision(9);
    for (size_t vi = 0; vi < views_.size(); ++vi) {
        const View& v = views_[vi];
        if (v.camera != ci || v.dropped) continue;
        for (size_t i = 0; i < v.img.size(); ++i) {
            o << v.image << ',' << v.board << ',' << vi << ',' << (i < v.pids.size() ? v.pids[i] : -1) << ','
              << v.img[i].x() << ',' << v.img[i].y() << ',' << (v.holdout ? 1 : 0) << ','
              << (static_cast<int>(i) >= static_cast<int>(v.img.size()) - v.n_guided ? 1 : 0) << '\n';
        }
    }
    std::ofstream l(base + "_lens.txt");
    l << std::setprecision(12) << LensModelName(cams_[ci].intr.model);
    for (double x : cams_[ci].intr.params) l << ' ' << x;
    l << '\n';
}

int RobotCalibrator::GuidedPass(int ci, bool rectified) {
    const CamState& cam = cams_[ci];
    std::map<std::string, int> board_index;
    for (size_t b = 0; b < cfg_.boards.size(); ++b) board_index[cfg_.boards[b].name] = static_cast<int>(b);
    std::map<std::string, std::map<std::string, View*>> by_image;  // image -> board -> view
    int max_frame = 0;
    for (View& v : views_) {
        if (v.camera == ci) max_frame = std::max(max_frame, v.frame_idx);
        if (v.camera != ci || v.dropped || v.image.empty() || !board_index.count(v.board) || v.pids.empty()) continue;
        by_image[v.image][v.board] = &v;
    }
    // Board-to-board layout (boards fixed to each other): T_a^-1 T_b averaged over shared frames.
    std::map<std::pair<std::string, std::string>, Iso3> rel;
    if (cams_[ci].handheld && !cams_[ci].free_views) {  // free views: boards move independently
        std::map<std::pair<std::string, std::string>, std::vector<Iso3>> acc;
        for (const auto& im : by_image) {
            for (const auto& a : im.second) {
                for (const auto& b : im.second) {
                    if (a.first == b.first || !a.second->has_T_cam_board || !b.second->has_T_cam_board) continue;
                    acc[{a.first, b.first}].push_back(a.second->T_cam_board.inverse() * b.second->T_cam_board);
                }
            }
        }
        for (const auto& kv : acc) {
            if (kv.second.size() >= 3) rel[kv.first] = AveragePoses(kv.second);
        }
    }
    std::vector<std::pair<int, std::string>> images = frame_images_[ci];
    std::unique_ptr<RectifiedDetector> rd;
    if (rectified) rd = std::make_unique<RectifiedDetector>(cam.intr);
    // detection.rectified_cache: virtual-view tags per image, from an earlier solve of the same images.
    std::map<std::string, TagsByFamily> rect_cached;
    std::vector<TagsByFamily> rect_found(images.size());
    const bool rect_read = rd && !cfg_.detect.rectified_cache.empty() && std::ifstream(cfg_.detect.rectified_cache).good();
    if (rect_read) {
        std::ifstream in(cfg_.detect.rectified_cache);
        std::string line;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const YAML::Node n = YAML::Load(line);
            TagsByFamily& t = rect_cached[n["image"].as<std::string>()];
            std::vector<TagCorners>& list = t[n["family"].as<std::string>()];
            const YAML::Node ids = n["ids"], c = n["corners"];
            for (size_t q = 0; q < ids.size(); ++q) {
                TagCorners tc;
                tc.id = ids[q].as<int>();
                for (int m = 0; m < 4; ++m) tc.p[m] = Vec2(c[4 * q + m][0].as<double>(), c[4 * q + m][1].as<double>());
                list.push_back(tc);
            }
        }
    }
    struct NewView {
        std::string board;
        std::vector<int> pids;
        std::vector<Vec3> obj;
        std::vector<Vec2> img;
        bool has_pose = false;
        Iso3 T = Iso3::Identity();
    };
    std::vector<std::vector<NewView>> fresh(images.size());
    int added = 0, tried = 0, from_rect = 0, from_layout = 0;
    auto predict = [&](const cv::Mat& gray, const cv::Mat& lin, const BoardSpec& b, const Iso3& T, const std::set<int>& have,
                       std::vector<int>& pids, std::vector<Vec3>& obj, std::vector<Vec2>& img, int& n_tried) {
        auto project = [&](const Vec3& Xb, Vec2& uv) {
            const Vec3 Xc = T * Xb;
            if (std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) > 100.0 * kDeg) return false;
            double u[2];
            ProjectPoint(cam.intr.model, cam.intr.params.data(), Eigen::Matrix<double, 3, 1>(Xc), u);
            uv = Vec2(u[0], u[1]);
            return std::isfinite(u[0]) && u[0] > 4 && u[1] > 4 && u[0] < cam.intr.width - 5 && u[1] < cam.intr.height - 5;
        };
        int n = 0;
        for (int pid = 0; pid < b.NumPoints(); ++pid) {
            if (have.count(pid)) continue;
            const Vec3 X = b.Point(pid);
            if ((T.linear().col(2)).dot(T * X) >= 0) continue;  // not facing the camera
            Vec2 uv, ux, uy;
            const double h = 1e-4;
            if (!project(X, uv) || !project(X + Vec3(h, 0, 0), ux) || !project(X + Vec3(0, h, 0), uy)) continue;
            Eigen::Matrix2d A;
            A.col(0) = (ux - uv) / h;
            A.col(1) = (uy - uv) / h;
            ++n_tried;
            if (!RefinePredictedSaddle(gray, lin, b, pid, uv, A, cfg_.detect, 0.8)) continue;
            pids.push_back(pid);
            obj.push_back(X);
            img.push_back(uv);
            ++n;
        }
        return n;
    };
    const CvSerial cv_serial;  // frames run in parallel; OpenCV calls inside stay serial
#pragma omp parallel for schedule(dynamic) reduction(+ : added, tried, from_rect, from_layout) num_threads(cfg_.solver.threads)
    for (int k = 0; k < static_cast<int>(images.size()); ++k) {
        const std::string& image = images[k].second;
        const cv::Mat gray = cv::imread(image, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;
        const cv::Mat lin = LinearizeGray(gray, cfg_.detect.gamma);
        std::map<std::string, View*> have_views;
        {
            auto it = by_image.find(image);
            if (it != by_image.end()) have_views = it->second;
        }
        // 1. rectified re-detection
        if (rd) {
            const auto hit = rect_cached.find(image);
            if (hit != rect_cached.end()) rect_found[k] = hit->second;
            else rect_found[k] = rd->FindTags(gray, cfg_.boards, cfg_.detect);
            for (const BoardDetection& d : RectifiedDetector::DetectGiven(gray, cfg_.boards, cfg_.detect, rect_found[k])) {
                const std::string& name = cfg_.boards[d.board].name;
                const BoardSpec& b = cfg_.boards[d.board];
                auto it = have_views.find(name);
                if (it != have_views.end()) {
                    View* v = it->second;
                    std::set<int> have(v->pids.begin(), v->pids.end());
                    for (size_t q = 0; q < d.pids.size(); ++q) {
                        if (have.count(d.pids[q])) continue;
                        v->pids.push_back(d.pids[q]);
                        v->obj.push_back(b.Point(d.pids[q]));
                        v->img.push_back(d.pts[q]);
                        ++v->n_guided;
                        ++from_rect;
                    }
                } else {
                    NewView nv;
                    nv.board = name;
                    nv.pids = d.pids;
                    for (size_t q = 0; q < d.pids.size(); ++q) {
                        nv.obj.push_back(b.Point(d.pids[q]));
                        nv.img.push_back(d.pts[q]);
                    }
                    from_rect += static_cast<int>(d.pids.size());
                    fresh[k].push_back(nv);
                }
            }
        }
        // 2. boards with nothing decoded, from the board layout
        if (!rel.empty()) {
            std::set<std::string> present;
            for (const auto& kv : have_views) present.insert(kv.first);
            for (const NewView& nv : fresh[k]) present.insert(nv.board);
            for (const BoardSpec& b : cfg_.boards) {
                if (present.count(b.name) || b.type != BoardType::AprilGrid) continue;
                const View* src = nullptr;
                Iso3 T = Iso3::Identity();
                for (const auto& kv : have_views) {
                    auto r = rel.find({kv.first, b.name});
                    if (r == rel.end() || !kv.second->has_T_cam_board) continue;
                    src = kv.second;
                    T = kv.second->T_cam_board * r->second;
                    break;
                }
                if (!src) continue;
                NewView nv;
                nv.board = b.name;
                const int n = predict(gray, lin, b, T, {}, nv.pids, nv.obj, nv.img, tried);
                if (n >= std::max(8, cfg_.detect.min_points_per_board)) {
                    nv.has_pose = true;
                    nv.T = T;
                    from_layout += n;
                    fresh[k].push_back(nv);
                }
            }
        }
        // 3. predicted corners of every posed view
        for (auto& kv : have_views) {
            View* v = kv.second;
            if (!v->has_T_cam_board) continue;
            const BoardSpec& b = cfg_.boards[board_index.at(v->board)];
            std::set<int> have(v->pids.begin(), v->pids.end());
            std::vector<int> pids;
            std::vector<Vec3> obj;
            std::vector<Vec2> img;
            const int n = predict(gray, lin, b, v->T_cam_board, have, pids, obj, img, tried);
            for (int q = 0; q < n; ++q) {
                v->pids.push_back(pids[q]);
                v->obj.push_back(obj[q]);
                v->img.push_back(img[q]);
            }
            v->n_guided += n;
            added += n;
        }
    }
    if (rd && !cfg_.detect.rectified_cache.empty() && !rect_read) {
        std::ofstream o(cfg_.detect.rectified_cache);
        o << std::setprecision(10);
        for (size_t k = 0; k < images.size(); ++k) {
            for (const auto& fam : rect_found[k]) {
                o << "{\"image\": \"" << images[k].second << "\", \"family\": \"" << fam.first << "\", \"ids\": [";
                for (size_t q = 0; q < fam.second.size(); ++q) o << (q ? ", " : "") << fam.second[q].id;
                o << "], \"corners\": [";
                for (size_t q = 0; q < fam.second.size(); ++q) {
                    for (int m = 0; m < 4; ++m) {
                        o << (q || m ? ", " : "") << "[" << fam.second[q].p[m].x() << ", " << fam.second[q].p[m].y() << "]";
                    }
                }
                o << "]}\n";
            }
        }
    }
    // New views (outside the parallel loop: views_ may reallocate).
    int new_views = 0;
    for (size_t k = 0; k < images.size(); ++k) {
        for (const NewView& nv : fresh[k]) {
            if (static_cast<int>(nv.pids.size()) < cfg_.detect.min_points_per_board) continue;
            View v;
            v.camera = ci;
            v.board = nv.board;
            v.frame_idx = cams_[ci].free_views ? ++max_frame : images[k].first;
            v.src_frame = images[k].first;
            v.image = images[k].second;
            v.pids = nv.pids;
            v.obj = nv.obj;
            v.img = nv.img;
            v.n_guided = static_cast<int>(nv.img.size());  // the whole view comes from the guided pass
            v.has_T_cam_board = nv.has_pose;
            v.T_cam_board = nv.T;
            views_.push_back(v);
            ++new_views;
        }
    }
    std::cout << "[robocal] " << cfg_.cameras[ci].name << " guided pass" << (rectified ? " (+ rectified re-detection)" : "")
              << ": " << added << " of " << tried << " predicted corners added, " << from_rect
              << " from rectified views, " << from_layout << " from the board layout, " << new_views << " new board views"
              << robocal::Took() << std::endl;
    return added + from_rect + from_layout;
}

void RobotCalibrator::WriteResiduals(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << "image,camera,board,pid,u,v,du,dv,holdout,theta_deg\n";
    f << std::setprecision(6);
    std::vector<double> r;
    for (const View& v : views_) {
        if (v.dropped || !ViewResiduals(v, r)) continue;
        const CamState& cam = cams_[v.camera];
        for (size_t i = 0; i < v.obj.size(); ++i) {
            Vec3 b;
            double th = 0;
            if (UnprojectPixel(cam.intr.model, cam.intr.params.data(), v.img[i], b)) {
                th = std::atan2(std::hypot(b.x(), b.y()), b.z()) / kDeg;
            }
            f << v.image << "," << cfg_.cameras[v.camera].name << "," << v.board << ","
              << (i < v.pids.size() ? v.pids[i] : -1) << "," << v.img[i].x() << "," << v.img[i].y() << ","
              << r[2 * i] << "," << r[2 * i + 1] << "," << (v.holdout ? 1 : 0) << "," << th << "\n";
        }
    }
}

void RobotCalibrator::AddView(const View& v) {
    if (!loaded_) throw std::runtime_error("AddView before Load");
    views_.push_back(v);
}

// Whole frames are held out (every k-th frame per camera), never single board views: a
// per-view stride aliases with the fixed board order in each frame and can hold out every
// view of one board (e.g. a gripper marker) -- which then constrains nothing.
int RobotCalibrator::BlockOf(const View& v) const {
    const int f = v.src_frame >= 0 ? v.src_frame : v.frame_idx;
    return cfg_.solver.block_frames > 0 ? f / cfg_.solver.block_frames : f;
}

void RobotCalibrator::AssignHoldout() {
    const int k = cfg_.solver.holdout_every;
    std::map<int, int> count;  // frames (or time blocks) seen per camera
    std::map<std::pair<int, int>, bool> frame_hold;
    // Hold out whole time blocks of source frames (a held-out frame's neighbours 0.2 s away would
    // otherwise train the same errors). Needs >= 2k blocks per camera, else per frame (warned).
    std::map<int, std::set<int>> blocks_per_cam;
    for (const View& v : views_) blocks_per_cam[v.camera].insert(BlockOf(v));
    for (View& v : views_) {
        v.holdout = false;
        if (k < 2) continue;
        const bool by_block = cfg_.solver.block_frames > 1 && v.src_frame >= 0 && cams_[v.camera].handheld &&
                              static_cast<int>(blocks_per_cam[v.camera].size()) >= 2 * k;
        const auto key = by_block ? std::make_pair(v.camera, -1 - BlockOf(v)) : std::make_pair(v.camera, v.frame_idx);
        auto it = frame_hold.find(key);
        if (it == frame_hold.end()) {
            const int i = count[v.camera]++;
            it = frame_hold.emplace(key, (i % k) == (k - 1)).first;
        }
        v.holdout = it->second;
    }
    // Every board must keep training views in every camera that sees it.
    std::map<std::pair<int, std::string>, std::pair<int, int>> seen;  // (camera, board) -> (train, held)
    for (const View& v : views_) {
        auto& c = seen[{v.camera, v.board}];
        (v.holdout ? c.second : c.first)++;
    }
    for (const auto& kv : seen) {
        if (kv.second.first == 0 && kv.second.second > 0) {
            for (View& v : views_) {
                if (v.camera == kv.first.first && v.board == kv.first.second) v.holdout = false;
            }
            std::cout << "[robocal] " << cfg_.cameras[kv.first.first].name << " / " << kv.first.second
                      << ": all views were in the hold-out set; kept them for the fit" << std::endl;
        }
    }
}

// Hand-held cameras: chain board and frame poses from the reference board outwards
// (frame from boards it sees, boards from frames that see them).
void RobotCalibrator::InitHandheld() {
    bool changed = true;
    while (changed) {
        changed = false;
        std::map<std::pair<int, int>, std::vector<Iso3>> frame_est;
        for (const View& v : views_) {
            if (!cams_[v.camera].handheld || v.dropped || !v.has_T_cam_board) continue;
            const auto key = std::make_pair(v.camera, v.frame_idx);
            const BoardState& b = boards_.at(v.board);
            if (frames_initialized_.count(key) || b.mounted || !b.initialized) continue;
            frame_est[key].push_back(b.pose.ToIso() * v.T_cam_board.inverse());
        }
        for (auto& kv : frame_est) {
            frame_poses_[kv.first] = PoseParam::FromIso(AveragePoses(kv.second));
            frames_initialized_.insert(kv.first);
            changed = true;
        }
        std::map<std::string, std::vector<Iso3>> board_est;
        for (const View& v : views_) {
            if (!cams_[v.camera].handheld || v.dropped || !v.has_T_cam_board) continue;
            const auto key = std::make_pair(v.camera, v.frame_idx);
            const BoardState& b = boards_.at(v.board);
            if (!frames_initialized_.count(key) || b.mounted || b.initialized) continue;
            board_est[v.board].push_back(frame_poses_.at(key).ToIso() * v.T_cam_board);
        }
        for (auto& kv : board_est) {
            BoardState& b = boards_.at(kv.first);
            b.pose = PoseParam::FromIso(AveragePoses(kv.second));
            b.initialized = true;
            changed = true;
        }
    }
    int unreached = 0;
    for (View& v : views_) {
        if (!cams_[v.camera].handheld || v.dropped) continue;
        if (!frames_initialized_.count(std::make_pair(v.camera, v.frame_idx))) {
            v.dropped = true;
            ++unreached;
        }
    }
    if (unreached) {
        std::cout << "[robocal] " << unreached << " hand-held view(s) not connected to the reference board: dropped"
                  << std::endl;
    }
}

// Held-out frames of hand-held cameras: their pose is refitted alone (boards and intrinsics
// fixed), so the hold-out error measures the lens and board model, not a free pose.
void RobotCalibrator::RefitHeldOutFrames() {
    std::map<std::pair<int, int>, std::vector<View*>> frames;
    for (View& v : views_) {
        if (cams_[v.camera].handheld && v.holdout && !v.dropped) frames[{v.camera, v.frame_idx}].push_back(&v);
    }
    for (auto& kv : frames) {
        if (!frames_initialized_.count(kv.first)) continue;
        ceres::Problem problem;
        for (View* v : kv.second) {
            std::vector<double*> blocks;
            std::vector<int> sizes;
            const ViewCost c = MakeViewCost(*v, &blocks, &sizes);
            auto* cost = new ceres::DynamicAutoDiffCostFunction<ViewCost, 4>(new ViewCost(c));
            for (int s : sizes) cost->AddParameterBlock(s);
            cost->SetNumResiduals(static_cast<int>(2 * v->obj.size()));
            problem.AddResidualBlock(cost, new ceres::HuberLoss((cfg_.solver.huber_px > 0 ? cfg_.solver.huber_px : 1.0) / pixel_sigma_), blocks);
            for (size_t i = 2; i < blocks.size(); ++i) problem.SetParameterBlockConstant(blocks[i]);
        }
        PoseParam& fp = frame_poses_.at(kv.first);
        problem.SetManifold(fp.q, new ceres::EigenQuaternionManifold);
        ceres::Solver::Options o;
        o.linear_solver_type = ceres::DENSE_QR;
        o.max_num_iterations = 50;
        ceres::Solver::Summary sum;
        ceres::Solve(o, &problem, &sum);
    }
}

// Camera pose in the world for one view: arm chain (wrist), the frame's own pose (hand-held /
// free_views; identity until InitHandheld has reached the frame) or the camera's pose (static).
Iso3 RobotCalibrator::CameraInWorld(const View& v) const {
    if (v.arm < 0) {
        if (cams_[v.camera].handheld) {
            const auto it = frame_poses_.find(std::make_pair(v.camera, v.frame_idx));
            return it != frame_poses_.end() ? it->second.ToIso() : Iso3::Identity();
        }
        return cams_[v.camera].world_cam.ToIso();
    }
    const ArmState& a = arms_[v.arm];
    std::vector<double> theta(v.q.size());
    for (size_t i = 0; i < theta.size(); ++i) theta[i] = v.q[i] + a.dq[i];
    return a.world_base.ToIso() * a.chain.ForwardIso(theta) * a.tip_cam.ToIso();
}

Iso3 RobotCalibrator::MountLinkInWorld(const BoardState& b, const std::vector<double>& q) const {
    const ArmState& a = arms_[b.arm];
    std::vector<double> theta(q.size());
    for (size_t i = 0; i < theta.size(); ++i) theta[i] = q[i] + a.dq[i];
    return a.world_base.ToIso() * a.chain.ForwardIso(theta) * b.T_tip_link;
}

// One BoardState per configured board and per sample session; geometry for the shape model,
// mounts resolved against the URDF.
void RobotCalibrator::SetupBoards() {
    boards_.clear();
    for (const BoardSpec& s : cfg_.boards) {
        BoardState b;
        b.geom = {0.5 * s.Width(), 0.5 * s.Height(), 0.5 * s.Width(), 0.5 * s.Height()};
        b.fit_shape = s.fit_shape;
        b.fit_scale = s.fit_scale;
        b.fit_points = s.fit_points;
        b.point_sigma_m = s.point_sigma_m;
        if (s.fit_points) b.heights.assign(static_cast<size_t>(s.NumPoints()), 0.0);
        b.fit_warp = s.fit_warp;
        b.warp_sigma_m = s.warp_sigma_m;
        b.shape_sigma_m = s.shape_sigma_m;
        b.scale_sigma = s.scale_sigma;
        b.print_scale[0] = b.shape[3] = s.print_scale[0];
        b.print_scale[1] = b.shape[4] = s.print_scale[1];
        if (!s.mount_arm.empty()) {
            b.mounted = true;
            b.arm = ArmIndex(s.mount_arm);
            if (b.arm < 0) throw std::runtime_error("board " + s.name + ": unknown mount arm " + s.mount_arm);
            const ArmConfig& ac = cfg_.arms[b.arm];
            const std::string link = s.mount_link.empty() ? ac.tip_link : s.mount_link;
            if (link != ac.tip_link) {
                const UrdfChain c = ac.urdf_xml.empty() ? UrdfChain::FromFile(ac.urdf, ac.tip_link, link, {})
                                                        : UrdfChain::FromString(ac.urdf_xml, ac.tip_link, link, {});
                b.T_tip_link = c.ForwardIso({});
            }
            b.mount_fixed = s.mount_fixed;
            b.mount_init_given = s.has_mount_init;
            b.mount = PoseParam::FromIso(s.T_link_board_init);
            if (b.mount_fixed && !s.has_mount_init) {
                throw std::runtime_error("board " + s.name + ": mount_fixed needs T_link_board_init");
            }
        }
        boards_[s.name] = b;
    }
    // Boards that only exist as sample sessions (imported corners): static, extents from points.
    for (const View& v : views_) {
        if (boards_.count(v.board)) continue;
        BoardState b;
        double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
        for (const View& w : views_) {
            if (w.board != v.board) continue;
            for (const Vec3& o : w.obj) {
                x0 = std::min(x0, o.x());
                x1 = std::max(x1, o.x());
                y0 = std::min(y0, o.y());
                y1 = std::max(y1, o.y());
            }
        }
        b.geom = {0.5 * (x0 + x1), 0.5 * (y0 + y1), std::max(1e-3, 0.5 * (x1 - x0)), std::max(1e-3, 0.5 * (y1 - y0))};
        boards_[v.board] = b;
    }
    // A wrist camera cannot be scored against a board on its own arm (the chain cancels).
    int skipped = 0;
    for (View& v : views_) {
        const BoardState& b = boards_.at(v.board);
        if (b.mounted && v.arm >= 0 && v.arm == b.arm && !v.dropped) {
            v.dropped = true;
            ++skipped;
        }
        if (b.mounted && !v.dropped && !v.arm_q.count(b.arm)) {
            throw std::runtime_error("frame " + v.image + ": board " + v.board + " is mounted on arm " +
                                     cfg_.arms[b.arm].name + " but the frame has no joints for it");
        }
    }
    if (skipped) {
        std::cout << "[robocal] " << skipped << " view(s) of a board mounted on the camera's own arm skipped"
                  << std::endl;
    }
}

void RobotCalibrator::Initialize() {
    if (!loaded_) throw std::runtime_error("Initialize before Load");
    // Intrinsics belong to their own stage: the robot run sees the boards only where the arms put
    // them, so a lens freed here fits that part of the image and extrapolates the rest (in
    // simulation: a static camera's 200 deg lens off by > 100 px beyond 45 deg, wrist lenses 2x
    // worse). Refuse unless explicitly allowed.
    if (!cfg_.arms.empty() && !cfg_.solver.allow_intrinsics_in_robot_solve) {
        for (const CameraConfig& c : cfg_.cameras) {
            if (c.optimize_intrinsics) {
                throw std::runtime_error(
                    "camera " + c.name + ": optimize_intrinsics in a robot solve. Calibrate intrinsics in their own "
                    "session (full image coverage, gates) and load the camera_info here; set "
                    "solver.allow_intrinsics_in_robot_solve: true to override");
            }
        }
    }
    // Gauge: at least one fixed arm base.
    bool any_fixed = false;
    for (const ArmConfig& a : cfg_.arms) any_fixed = any_fixed || a.base_fixed;
    if (!any_fixed && !cfg_.arms.empty()) {
        std::cout << "[robocal] no arm base is fixed: fixing " << cfg_.arms[0].name
                  << " (world frame = its configured base pose)" << std::endl;
        cfg_.arms[0].base_fixed = true;
    }
    SetupBoards();
    // free_views cameras: every board view gets its own pose; boards they alone see stay at identity.
    {
        int next = 1000000;
        std::map<std::string, bool> only_free;
        for (View& v : views_) {
            const bool fv = cams_[v.camera].free_views;
            if (fv) v.frame_idx = next++;
            auto it = only_free.find(v.board);
            only_free[v.board] = (it == only_free.end() ? true : it->second) && fv;
        }
        for (const auto& kv : only_free) {
            if (!kv.second) continue;
            BoardState& b = boards_.at(kv.first);
            b.pose = PoseParam();
            b.initialized = true;
            b.fixed = true;
        }
    }
    bool all_boards_fixed = !boards_.empty();
    for (const auto& kv : boards_) all_boards_fixed = all_boards_fixed && kv.second.fixed;
    if (cfg_.arms.empty() && !all_boards_fixed) {
        // No robot: the world frame is a board.
        std::string ref = cfg_.reference_board;
        if (ref.empty() && !cfg_.LevelBoards().empty()) ref = cfg_.LevelBoards().begin()->first;
        if (ref.empty()) {
            for (const View& v : views_) {
                if (!boards_.at(v.board).mounted) {
                    ref = v.board;
                    break;
                }
            }
        }
        if (ref.empty() || !boards_.count(ref)) throw std::runtime_error("no reference board for the world frame");
        BoardState& b = boards_.at(ref);
        b.pose = PoseParam();
        b.initialized = true;
        b.fixed = true;
        std::cout << "[robocal] world frame = board " << ref << std::endl;
    }
    AssignHoldout();
    EstimateMissingPoses();
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        if (!cfg_.cameras[ci].intrinsics_prepass) continue;
        IntrinsicsPrepass(static_cast<int>(ci));
        DumpCorners(static_cast<int>(ci), 0);
        // Corners whose tags did not decode (far off-axis, blurred, small) are still clean saddles:
        // find them through the fitted lens, refit, repeat until little is added.
        for (int round = 0; cfg_.detect.guided && round < std::max(1, cfg_.detect.guided_rounds); ++round) {
            int have = 0;
            for (const View& v : views_) have += v.camera == static_cast<int>(ci) && !v.dropped ? static_cast<int>(v.obj.size()) : 0;
            const int added = GuidedPass(static_cast<int>(ci), cfg_.detect.rectified && round == 0);
            if (added == 0) break;
            FilterReliableTags(false);
            EstimateMissingPoses();
            AssignHoldout();
            IntrinsicsPrepass(static_cast<int>(ci));
            DumpCorners(static_cast<int>(ci), round + 1);
            if (added < 0.01 * have) break;
        }
    }

    InitMountsFromData();
    auto is_static_board = [&](const std::string& n) { return !boards_.at(n).mounted; };
    std::map<std::string, std::vector<Iso3>> board_obs;
    auto collect = [&](bool fixed_bases) {
        for (const View& v : views_) {
            if (v.arm < 0 || !v.has_T_cam_board || v.holdout || v.dropped || !is_static_board(v.board)) continue;
            if (cfg_.arms[v.arm].base_fixed != fixed_bases || boards_.at(v.board).initialized) continue;
            board_obs[v.board].push_back(CameraInWorld(v) * v.T_cam_board);
        }
        for (auto& kv : board_obs) {
            BoardState& b = boards_.at(kv.first);
            if (!b.initialized && !kv.second.empty()) {
                b.pose = PoseParam::FromIso(AveragePoses(kv.second));
                b.initialized = true;
            }
        }
        board_obs.clear();
    };
    collect(true);
    // Free bases from the static boards the fixed arms already placed.
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        if (cfg_.arms[ai].base_fixed) continue;
        std::vector<Iso3> est;
        for (const View& v : views_) {
            if (v.arm != static_cast<int>(ai) || !v.has_T_cam_board || v.holdout || v.dropped) continue;
            const BoardState& b = boards_.at(v.board);
            if (b.mounted || !b.initialized) continue;
            std::vector<double> theta(v.q.size());
            for (size_t i = 0; i < theta.size(); ++i) theta[i] = v.q[i] + arms_[ai].dq[i];
            const Iso3 T_base_board = arms_[ai].chain.ForwardIso(theta) * arms_[ai].tip_cam.ToIso() * v.T_cam_board;
            est.push_back(b.pose.ToIso() * T_base_board.inverse());
        }
        if (!est.empty()) {
            const Iso3 T = AveragePoses(est);
            const Iso3 T0 = arms_[ai].world_base.ToIso();
            std::cout << "[robocal] " << cfg_.arms[ai].name << " base initialized from shared boards: "
                      << std::fixed << std::setprecision(1)
                      << (T.translation() - T0.translation()).norm() * 1e3 << " mm / "
                      << RotationAngleDeg(T.linear(), T0.linear()) << " deg from the configured pose"
                      << std::endl;
            arms_[ai].world_base = PoseParam::FromIso(T);
        }
    }
    collect(false);
    // Static cameras from the static boards they see.
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        if (!cams_[ci].is_static) continue;
        std::vector<Iso3> est;
        for (const View& v : views_) {
            if (v.arm >= 0 || v.camera != static_cast<int>(ci) || !v.has_T_cam_board || v.dropped) continue;
            const BoardState& b = boards_.at(v.board);
            if (b.mounted || !b.initialized) continue;
            est.push_back(b.pose.ToIso() * v.T_cam_board.inverse());
        }
        if (!est.empty()) cams_[ci].world_cam = PoseParam::FromIso(AveragePoses(est));
    }
    // Static boards seen only by static cameras.
    for (const View& v : views_) {
        BoardState& b = boards_.at(v.board);
        if (b.initialized || b.mounted || v.arm >= 0 || !cams_[v.camera].is_static || !v.has_T_cam_board || v.dropped) {
            continue;
        }
        b.pose = PoseParam::FromIso(cams_[v.camera].world_cam.ToIso() * v.T_cam_board);
        b.initialized = true;
    }
    InitHandheld();
    // Mounted boards: the mount link -> board transform from every camera that sees them.
    for (auto& kv : boards_) {
        BoardState& b = kv.second;
        if (!b.mounted) continue;
        if (!b.mount_fixed) {
            std::vector<Iso3> est;
            for (const View& v : views_) {
                if (v.board != kv.first || !v.has_T_cam_board || v.holdout || v.dropped) continue;
                // hand-held / free_views frames: only those InitHandheld placed via a static board
                if (v.arm < 0 && cams_[v.camera].handheld && !frames_initialized_.count(std::make_pair(v.camera, v.frame_idx))) continue;
                est.push_back(MountLinkInWorld(b, v.arm_q.at(b.arm)).inverse() * CameraInWorld(v) * v.T_cam_board);
            }
            if (!est.empty()) b.mount = PoseParam::FromIso(AveragePoses(est));
            else if (!b.mount_init_given) throw std::runtime_error("mounted board " + kv.first + " is never seen");
        }
        b.initialized = true;
    }
    for (const View& v : views_) {
        if (!v.dropped && !boards_.at(v.board).initialized) {
            throw std::runtime_error("board '" + v.board + "' has no initial pose (no view with a PnP pose)");
        }
    }
    for (const auto& kv : cfg_.LevelBoards()) {
        if (!boards_.count(kv.first) || !boards_.at(kv.first).initialized || boards_.at(kv.first).mounted) {
            throw std::runtime_error("level board '" + kv.first + "' is not an observed static board");
        }
    }
    bool has_static = false;
    for (const CamState& c : cams_) has_static = has_static || c.is_static;
    if (has_static) {
        Flags f;
        f.only_static = true;
        DropInvalidViews("static camera init");
        RunSolver(f, "static cameras (init)");
    }
    DropInvalidViews("init");
    initialized_ = true;
}

ViewCost RobotCalibrator::MakeViewCost(const View& v, std::vector<double*>* blocks, std::vector<int>* sizes, bool pose7) const {
    auto* self = const_cast<RobotCalibrator*>(this);
    const CamState& cam = cams_[v.camera];
    const BoardState& b = boards_.at(v.board);
    ViewCost c;
    c.view = &v;
    c.model = cam.intr.model;
    c.geom = b.geom;
    c.weight = 1.0 / (pixel_sigma_ * v.sigma_scale);
    auto add = [&](const double* p, int n) {
        if (blocks) blocks->push_back(const_cast<double*>(p));
        if (sizes) sizes->push_back(n);
    };
    auto eps_block = [&](int arm) -> const double* {
        auto& e = self->joint_eps_[std::make_pair(arm, v.frame_idx)];
        if (e.empty()) e.assign(arms_[arm].chain.dof(), 0.0);
        return e.data();
    };
    if (v.arm >= 0) {
        ArmState& a = self->arms_[v.arm];
        c.cam_chain = &a.chain;
        add(a.dq.data(), a.chain.dof());
        if (cfg_.arms[v.arm].joint_noise_rad > 0) {
            c.cam_eiv = true;
            add(eps_block(v.arm), a.chain.dof());
        }
        if (cfg_.drive.model && static_cast<int>(v.dtau.size()) == a.chain.dof()) {
            c.cam_drive = true;
            add(self->DriveBlock(v.arm), 2 * a.chain.dof());
        }
        add(a.tip_cam.q, 4);
        add(a.tip_cam.t, 3);
        add(a.world_base.q, 4);
        add(a.world_base.t, 3);
    } else if (cam.handheld) {
        const PoseParam& fp = frame_poses_.at(std::make_pair(v.camera, v.frame_idx));
        if (pose7) {
            c.cam_pose7 = true;
            add(fp.q, 7);
        } else {
            add(fp.q, 4);
            add(fp.t, 3);
        }
    } else {
        add(cam.world_cam.q, 4);
        add(cam.world_cam.t, 3);
    }
    if (b.mounted) {
        ArmState& a = self->arms_[b.arm];
        c.board_chain = &a.chain;
        c.board_q = v.arm_q.at(b.arm);
        c.T_tip_link = b.T_tip_link;
        add(a.dq.data(), a.chain.dof());
        if (cfg_.arms[b.arm].joint_noise_rad > 0) {
            c.board_eiv = true;
            add(eps_block(b.arm), a.chain.dof());
        }
        add(b.mount.q, 4);
        add(b.mount.t, 3);
        add(a.world_base.q, 4);
        add(a.world_base.t, 3);
    } else {
        add(b.pose.q, 4);
        add(b.pose.t, 3);
    }
    add(b.shape, 5);
    if (b.fit_warp) {
        c.board_warp = true;
        add(b.warp, 8);
    }
    if (b.fit_points && !b.heights.empty() && v.pids.size() == v.obj.size()) {
        c.board_heights = true;
        add(b.heights.data(), static_cast<int>(b.heights.size()));
    }
    add(cam.intr.params.data(), cam.intr.NumParams());
    return c;
}

bool RobotCalibrator::ViewResiduals(const View& v, std::vector<double>& r) const {
    r.assign(2 * v.obj.size(), 0.0);
    std::vector<double*> blocks;
    ViewCost c = MakeViewCost(v, &blocks, nullptr);
    c.weight = 1.0;            // pixels
    c.corner_weights = false;  // the image error itself: the IRLS weights belong to the solve only
    std::vector<const double*> cb(blocks.begin(), blocks.end());
    return c(cb.data(), r.data());
}

void RobotCalibrator::DropInvalidViews(const std::string& when) {
    std::vector<double> r;
    int dropped = 0;
    for (View& v : views_) {
        if (v.dropped) continue;
        bool ok = ViewResiduals(v, r);
        if (ok) {
            for (double e : r) ok = ok && std::isfinite(e);
        }
        if (!ok) {
            v.dropped = true;
            ++dropped;
        }
    }
    if (dropped) {
        std::cout << "[robocal] " << when << ": dropped " << dropped
                  << " view(s) whose points project behind a pinhole camera" << std::endl;
    }
    diag_.dropped_views += dropped;
}

void RobotCalibrator::BuildProblem(ceres::Problem& problem, const Flags& f) {
    std::set<int> arms_used, cams_used, arms_with_camera;
    std::set<std::pair<int, int>> frames_used;
    std::set<std::string> boards_used;
    for (View& v : views_) {
        if (v.dropped || (v.holdout && !f.include_holdout)) continue;
        if (static_cast<int>(v.obj.size()) < cfg_.solver.min_points_per_view) continue;
        const BoardState& b = boards_.at(v.board);
        if (f.only_static && (v.arm >= 0 || b.mounted || cams_[v.camera].handheld)) continue;
        std::vector<double*> blocks;
        std::vector<int> sizes;
        const ViewCost c = MakeViewCost(v, &blocks, &sizes, f.pose7);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<ViewCost, 4>(new ViewCost(c));
        for (int s : sizes) cost->AddParameterBlock(s);
        cost->SetNumResiduals(static_cast<int>(2 * v.obj.size()));
        // robust: view -> Huber on the view's summed error; corner -> per-corner IRLS weights (View.pw).
        ceres::LossFunction* loss = (f.use_loss && cfg_.solver.huber_px > 0 && cfg_.solver.robust == "view")
                                        ? new ceres::HuberLoss(cfg_.solver.huber_px / pixel_sigma_)
                                        : nullptr;
        int mult = 1;
        if (boot_mult_) {
            const auto it = boot_mult_->find(&v);
            mult = it == boot_mult_->end() ? 0 : it->second;
        }
        if (mult == 0) {
            delete cost;
            delete loss;
            continue;
        }
        problem.AddResidualBlock(cost, loss, blocks);
        for (int m = 1; m < mult; ++m) {  // bootstrap: a block drawn m times enters m times
            auto* c2 = new ceres::DynamicAutoDiffCostFunction<ViewCost, 4>(new ViewCost(c));
            for (int sz : sizes) c2->AddParameterBlock(sz);
            c2->SetNumResiduals(static_cast<int>(2 * v.obj.size()));
            problem.AddResidualBlock(c2, loss ? new ceres::HuberLoss(cfg_.solver.huber_px / pixel_sigma_) : nullptr, blocks);
        }
        if (v.arm >= 0) {
            arms_used.insert(v.arm);
            arms_with_camera.insert(v.arm);
        }
        if (b.mounted) arms_used.insert(b.arm);
        if (cams_[v.camera].handheld) frames_used.insert(std::make_pair(v.camera, v.frame_idx));
        cams_used.insert(v.camera);
        boards_used.insert(v.board);
    }

    auto quat = [&](double* q, bool constant) {
        problem.SetManifold(q, new ceres::EigenQuaternionManifold);
        if (constant) problem.SetParameterBlockConstant(q);
    };
    for (int ai : arms_used) {
        ArmState& a = arms_[ai];
        const ArmConfig& ac = cfg_.arms[ai];
        std::vector<int> fixed;
        for (int j = 0; j < a.chain.dof(); ++j) {
            if (!ac.dq_free[j]) fixed.push_back(j);
        }
        const bool dq_const = !f.free_dq || static_cast<int>(fixed.size()) == a.chain.dof();
        if (dq_const) {
            problem.SetParameterBlockConstant(a.dq.data());
        } else if (!fixed.empty()) {
            problem.SetManifold(a.dq.data(), new ceres::SubsetManifold(a.chain.dof(), fixed));
        }
        if (!dq_const && !f.no_priors) {
            DeltaQPrior prior{ac.dq_init, {}};
            for (double s : ac.dq_sigma_rad) prior.inv_sigma.push_back(s > 0 ? 1.0 / s : 0.0);
            auto* cost = new ceres::DynamicAutoDiffCostFunction<DeltaQPrior, 8>(new DeltaQPrior(prior));
            cost->AddParameterBlock(a.chain.dof());
            cost->SetNumResiduals(a.chain.dof());
            problem.AddResidualBlock(cost, nullptr, a.dq.data());
        }
        if (arms_with_camera.count(ai)) {
            quat(a.tip_cam.q, f.only_static);
            if (f.only_static) problem.SetParameterBlockConstant(a.tip_cam.t);
        }
        const bool base_const = ac.base_fixed || f.only_static;
        quat(a.world_base.q, base_const);
        if (base_const) {
            problem.SetParameterBlockConstant(a.world_base.t);
        } else if (!f.no_priors && ac.base_prior_trans_m > 0 && ac.base_prior_rot_deg > 0) {
            PosePrior prior{Eigen::Quaterniond(ac.T_world_base.linear()), ac.T_world_base.translation(),
                            1.0 / (ac.base_prior_rot_deg * kDeg), 1.0 / ac.base_prior_trans_m};
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PosePrior, 6, 4, 3>(new PosePrior(prior)),
                                     nullptr, a.world_base.q, a.world_base.t);
        }
    }
    const std::map<std::string, double> level = cfg_.LevelBoards();
    for (const std::string& name : boards_used) {
        BoardState& b = boards_.at(name);
        if (b.mounted) {
            quat(b.mount.q, b.mount_fixed || f.only_static);
            if (b.mount_fixed || f.only_static) problem.SetParameterBlockConstant(b.mount.t);
        } else {
            quat(b.pose.q, f.only_static || b.fixed);
            if (f.only_static || b.fixed) problem.SetParameterBlockConstant(b.pose.t);
            const auto lv = level.find(name);
            if (!f.only_static && !f.no_priors && lv != level.end()) {
                const double w = 1.0 / std::sin(lv->second * kDeg);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LevelPriorCost, 2, 4>(new LevelPriorCost{w}), nullptr, b.pose.q);
            }
        }
        const bool shape_free = b.fit_shape && f.free_dq && !f.only_static;
        if (!shape_free) {
            problem.SetParameterBlockConstant(b.shape);
        } else {
            if (!b.fit_scale) problem.SetManifold(b.shape, new ceres::SubsetManifold(5, {3, 4}));
            if (!f.no_priors) {
                ShapePrior prior{1.0 / b.shape_sigma_m, 1.0 / b.scale_sigma, {b.print_scale[0], b.print_scale[1]}};
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ShapePrior, 5, 5>(new ShapePrior(prior)),
                                         nullptr, b.shape);
            }
        }
    }
    // Per-point board heights: free with their prior once the main solve runs (the prior also
    // fixes their gauge against the board pose and shape).
    for (const std::string& name : boards_used) {
        BoardState& b = boards_.at(name);
        if (!b.fit_points || b.heights.empty() || !problem.HasParameterBlock(b.heights.data())) continue;
        if (!f.free_dq || f.only_static || f.no_priors) {
            problem.SetParameterBlockConstant(b.heights.data());
            continue;
        }
        const int n = static_cast<int>(b.heights.size());
        DeltaQPrior prior{std::vector<double>(n, 0.0), std::vector<double>(n, 1.0 / b.point_sigma_m)};
        auto* cost = new ceres::DynamicAutoDiffCostFunction<DeltaQPrior, 8>(new DeltaQPrior(prior));
        cost->AddParameterBlock(n);
        cost->SetNumResiduals(n);
        problem.AddResidualBlock(cost, nullptr, b.heights.data());
    }
    // In-plane print warp: same rule as the heights.
    for (const std::string& name : boards_used) {
        BoardState& b = boards_.at(name);
        if (!b.fit_warp || !problem.HasParameterBlock(b.warp)) continue;
        if (!f.free_dq || f.only_static || f.no_priors) {
            problem.SetParameterBlockConstant(b.warp);
            continue;
        }
        DeltaQPrior prior{std::vector<double>(8, 0.0), std::vector<double>(8, 1.0 / b.warp_sigma_m)};
        auto* cost = new ceres::DynamicAutoDiffCostFunction<DeltaQPrior, 8>(new DeltaQPrior(prior));
        cost->AddParameterBlock(8);
        cost->SetNumResiduals(8);
        problem.AddResidualBlock(cost, nullptr, b.warp);
    }
    for (const auto& key : frames_used) {
        if (f.pose7) problem.SetManifold(frame_poses_.at(key).q, new Pose7Manifold);
        else quat(frame_poses_.at(key).q, false);
    }
    // Drive model (c, h): free with priors once dq is free; entries without spread in their
    // regressor held (they would only trade with dq).
    if (cfg_.drive.model) {
        std::set<double*> done;
        for (int ai : arms_used) {
            double* d = DriveBlock(ai);
            if (!problem.HasParameterBlock(d) || !done.insert(d).second) continue;
            const int n = arms_[ai].chain.dof();
            if (!f.free_dq || f.no_priors || f.only_static) {
                problem.SetParameterBlockConstant(d);
                continue;
            }
            const std::vector<int>& held = arms_[cfg_.drive.shared ? 0 : ai].drv_const;
            if (static_cast<int>(held.size()) == 2 * n) {
                problem.SetParameterBlockConstant(d);
                continue;
            }
            if (!held.empty()) problem.SetManifold(d, new ceres::SubsetManifold(2 * n, held));
            if (!line_search_) {  // line-search minimisers take no bounds
                for (int j = 0; j < 2 * n; ++j) problem.SetParameterLowerBound(d, j, 0.0);
            }
            std::vector<double> mean(2 * n), inv(2 * n);
            for (int j = 0; j < n; ++j) {
                mean[j] = 0.0;
                inv[j] = 1.0 / (j < static_cast<int>(cfg_.drive.c_sigma.size()) ? cfg_.drive.c_sigma[j] : 2e-3);
                mean[n + j] = cfg_.drive.h_prior_deg * kDeg;
                inv[n + j] = 1.0 / (cfg_.drive.h_sigma_deg * kDeg);
            }
            DeltaQPrior prior{mean, inv};
            auto* cost = new ceres::DynamicAutoDiffCostFunction<DeltaQPrior, 8>(new DeltaQPrior(prior));
            cost->AddParameterBlock(2 * n);
            cost->SetNumResiduals(2 * n);
            problem.AddResidualBlock(cost, nullptr, d);
        }
    }
    // Encoder errors: free with their prior while dq is free; held at zero otherwise and in the
    // data-only observability problem (they are nuisance parameters, one set per frame).
    for (auto& kv : joint_eps_) {
        double* e = kv.second.data();
        if (!problem.HasParameterBlock(e)) continue;
        const ArmConfig& eac = cfg_.arms[kv.first.first];
        const double sigma = eac.joint_noise_rad;
        if (!f.free_dq || f.no_priors || f.only_static || sigma <= 0) {
            problem.SetParameterBlockConstant(e);
            continue;
        }
        // Per joint; inflated by the backlash where this frame's gear flank is uncertain.
        std::vector<double> inv(kv.second.size());
        const auto amb = frame_amb_.find(kv.first);
        const double* drv = cfg_.drive.model ? DriveBlock(kv.first.first) : nullptr;
        const size_t n = kv.second.size();
        for (size_t j = 0; j < n; ++j) {
            double s2 = std::pow(j < eac.joint_noise_joint_rad.size() ? eac.joint_noise_joint_rad[j] : sigma, 2);
            if (drv && amb != frame_amb_.end()) s2 += std::pow(amb->second[j] * drv[n + j], 2);
            inv[j] = 1.0 / std::sqrt(std::max(1e-16, s2));
        }
        DeltaQPrior prior{std::vector<double>(kv.second.size(), 0.0), inv};
        auto* cost = new ceres::DynamicAutoDiffCostFunction<DeltaQPrior, 8>(new DeltaQPrior(prior));
        cost->AddParameterBlock(static_cast<int>(kv.second.size()));
        cost->SetNumResiduals(static_cast<int>(kv.second.size()));
        problem.AddResidualBlock(cost, nullptr, e);
    }
    for (int ci : cams_used) {
        CamState& c = cams_[ci];
        if (c.is_static) quat(c.world_cam.q, false);
        const bool intr_free = f.free_intrinsics && cfg_.cameras[ci].optimize_intrinsics && !f.only_static;
        if (!intr_free) problem.SetParameterBlockConstant(c.intr.params.data());
    }
}

// Stage-1 problems (no arms, free-view cameras, held static boards, no warp) on the dedicated
// Schur LM (stage1_solver.hpp): same objective as BuildProblem, O(points) per view instead of
// Ceres' dense per-view heights block.
bool RobotCalibrator::RunStage1Solver(const Flags& f, const std::string& label, double* final_cost,
                                      std::map<const double*, Eigen::MatrixXd>* cov) {
    if (!cfg_.arms.empty() || f.only_static || f.no_priors || !f.pose7) return false;
    if (cfg_.solver.method != "lm" && cfg_.solver.method != "levenberg_marquardt" && !cfg_.solver.method.empty()) return false;
    if (cfg_.solver.linear_solver != "auto" && cfg_.solver.linear_solver != "dense_schur") return false;
    if (!cfg_.LevelBoards().empty() || cfg_.drive.model) return false;
    std::vector<S1Camera> cams(cams_.size());
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        if (!cams_[ci].free_views) return false;
        cams[ci].model = cams_[ci].intr.model;
        cams[ci].intr = cams_[ci].intr.params.data();
        cams[ci].n_intr = cams_[ci].intr.NumParams();
        cams[ci].free = f.free_intrinsics && cfg_.cameras[ci].optimize_intrinsics;
    }
    std::map<std::string, int> bidx;
    std::vector<S1Board> boards;
    std::vector<S1View> views;
    std::set<const double*> seen_pose;
    for (View& v : views_) {
        if (v.dropped || (v.holdout && !f.include_holdout)) continue;
        if (static_cast<int>(v.obj.size()) < cfg_.solver.min_points_per_view) continue;
        if (v.arm >= 0) return false;
        int mult = 1;
        if (boot_mult_) {
            const auto it = boot_mult_->find(&v);
            mult = it == boot_mult_->end() ? 0 : it->second;
        }
        if (mult == 0) continue;
        BoardState& b = boards_.at(v.board);
        if (b.mounted || b.fit_warp || !b.fixed) return false;
        auto it = bidx.find(v.board);
        if (it == bidx.end()) {
            S1Board sb;
            sb.pose = b.pose.ToIso();
            sb.geom = b.geom;
            sb.shape = b.shape;
            sb.shape_free = b.fit_shape && f.free_dq;
            sb.scale_free = b.fit_scale;
            sb.shape_prior = sb.shape_free;
            sb.inv_sigma_shape = 1.0 / b.shape_sigma_m;
            sb.inv_sigma_scale = 1.0 / b.scale_sigma;
            sb.scale0[0] = b.print_scale[0];
            sb.scale0[1] = b.print_scale[1];
            if (b.fit_points && !b.heights.empty()) {
                sb.heights = &b.heights;
                sb.heights_free = f.free_dq;
                sb.inv_sigma_height = 1.0 / b.point_sigma_m;
            }
            it = bidx.emplace(v.board, static_cast<int>(boards.size())).first;
            boards.push_back(sb);
        }
        PoseParam& fp = frame_poses_.at(std::make_pair(v.camera, v.frame_idx));
        if (!seen_pose.insert(fp.q).second) return false;  // one pose per view (free views)
        S1View sv;
        sv.view = &v;
        sv.pose = fp.q;
        sv.camera = v.camera;
        sv.board = it->second;
        sv.heights = b.fit_points && !b.heights.empty() && v.pids.size() == v.obj.size();
        sv.weight = 1.0 / (pixel_sigma_ * v.sigma_scale);
        sv.mult = mult;
        views.push_back(sv);
    }
    if (views.empty()) return false;
    S1Options o;
    o.max_iterations = cfg_.solver.max_iterations;
    o.huber = (f.use_loss && cfg_.solver.huber_px > 0 && cfg_.solver.robust == "view") ? cfg_.solver.huber_px / pixel_sigma_ : 0.0;
    o.threads = cfg_.solver.threads;
    o.verbose = cfg_.solver.verbose || std::getenv("ROBOCAL_S1_VERBOSE");
    o.covariance = cov;
    const S1Summary s = SolveStage1(cams, boards, views, o);
    if (cov) return !s.failed;
    if (s.failed) throw std::runtime_error(label + " failed: " + s.message);
    std::cout << "[robocal] " << label << ": cost " << std::scientific << std::setprecision(4) << s.initial_cost
              << " -> " << s.final_cost << std::fixed << " (" << s.iterations << " it, "
              << (s.converged ? "CONVERGENCE" : "NO_CONVERGENCE") << ", schur-lm)" << robocal::Took() << std::endl;
    if (std::getenv("ROBOCAL_CERES_REPORT")) {
        std::cout << "  schur-lm: " << s.message << ", linearize " << s.t_linearize << " s, solve " << s.t_solve
                  << " s, evaluate " << s.t_eval << " s" << std::endl;
    }
    *final_cost = s.final_cost;
    return true;
}

double RobotCalibrator::RunSolver(const Flags& f, const std::string& label) {
    const std::string& method = cfg_.solver.method;
    line_search_ = method == "lbfgs" || method == "bfgs" || method == "nlcg";
    {
        const char* env = std::getenv("ROBOCAL_BACKEND");  // A/B testing
        const std::string backend = env ? env : cfg_.solver.backend;
        double c = 0;
        if (backend != "ceres" && RunStage1Solver(f, label, &c)) return c;
        if (backend == "schur_lm") std::cout << "[robocal] " << label << ": not a stage-1 problem, Ceres used" << std::endl;
    }
    ceres::Problem problem;
    BuildProblem(problem, f);
    if (problem.NumResidualBlocks() == 0) return 0.0;
    ceres::Solver::Options opts;
    // Small problems: dense QR. Many hand-held frame poses: Schur elimination of the poses.
    int n_free = 0;
    {
        std::vector<double*> pbs;
        problem.GetParameterBlocks(&pbs);
        for (double* p : pbs) {
            if (!problem.IsParameterBlockConstant(p)) n_free += problem.ParameterBlockTangentSize(p);
        }
    }
    opts.linear_solver_type = n_free > 300 ? ceres::DENSE_SCHUR : ceres::DENSE_QR;
    const std::string& ls = cfg_.solver.linear_solver;
    if (ls == "dense_qr") opts.linear_solver_type = ceres::DENSE_QR;
    else if (ls == "dense_schur") opts.linear_solver_type = ceres::DENSE_SCHUR;
    else if (ls == "sparse_schur") opts.linear_solver_type = ceres::SPARSE_SCHUR;
    else if (ls == "iterative_schur") opts.linear_solver_type = ceres::ITERATIVE_SCHUR;
    else if (ls == "sparse_normal_cholesky") opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    else if (ls != "auto") throw std::runtime_error("solver.linear_solver: unknown " + ls);
    if (f.pose7 && (opts.linear_solver_type == ceres::DENSE_SCHUR || opts.linear_solver_type == ceres::SPARSE_SCHUR ||
                    opts.linear_solver_type == ceres::ITERATIVE_SCHUR)) {
        // Eliminate the hand-held frame poses (one 7-block each, no residual sees two of them).
        std::vector<double*> pbs;
        problem.GetParameterBlocks(&pbs);
        std::set<const double*> fr;
        for (const auto& kv : frame_poses_) fr.insert(kv.second.q);
        auto* ord = new ceres::ParameterBlockOrdering;
        int n0 = 0;
        for (double* p : pbs) {
            const bool e = fr.count(p) && !problem.IsParameterBlockConstant(p);
            ord->AddElementToGroup(p, e ? 0 : 1);
            n0 += e;
        }
        if (n0 > 0 && n0 < static_cast<int>(pbs.size())) opts.linear_solver_ordering.reset(ord);
        else delete ord;
    }
    opts.max_num_iterations = cfg_.solver.max_iterations;
    if (method == "dogleg" || method == "dogleg_subspace") {
        opts.trust_region_strategy_type = ceres::DOGLEG;
        opts.dogleg_type = method == "dogleg" ? ceres::TRADITIONAL_DOGLEG : ceres::SUBSPACE_DOGLEG;
    } else if (line_search_) {
        opts.minimizer_type = ceres::LINE_SEARCH;
        opts.line_search_direction_type =
            method == "lbfgs" ? ceres::LBFGS : (method == "bfgs" ? ceres::BFGS : ceres::NONLINEAR_CONJUGATE_GRADIENT);
        opts.max_num_iterations = std::max(cfg_.solver.max_iterations, 20000);
    }
    opts.num_threads = cfg_.solver.threads;
    opts.function_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.minimizer_progress_to_stdout = cfg_.solver.verbose;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << "[robocal] " << label << ": cost " << std::scientific << std::setprecision(4)
              << summary.initial_cost << " -> " << summary.final_cost << std::fixed << " ("
              << summary.iterations.size() << " it, " << ceres::TerminationTypeToString(summary.termination_type)
              << ")" << robocal::Took() << std::endl;
    if (summary.termination_type == ceres::FAILURE) {
        throw std::runtime_error(label + " failed: " + summary.message);
    }
    return summary.final_cost;
}

// How well a wrist-camera mount X makes the static boards' implied poses (base <- board =
// FK * X * T_cam_board) agree across that arm's frames: mean rotation spread (deg) + translation
// spread (cm), averaged over boards. Lower is better.
double RobotCalibrator::MountDispersion(int ai, const Iso3& X) const {
    const ArmState& a = arms_[ai];
    std::map<std::string, std::vector<Iso3>> per_board;
    for (const View& v : views_) {
        if (v.arm != ai || !v.has_T_cam_board || v.dropped || boards_.at(v.board).mounted) continue;
        std::vector<double> th(v.q.size());
        for (size_t j = 0; j < th.size(); ++j) th[j] = v.q[j] + a.dq[j];
        per_board[v.board].push_back(a.chain.ForwardIso(th) * X * v.T_cam_board);
    }
    double score = 0;
    int n = 0;
    for (const auto& kv : per_board) {
        if (kv.second.size() < 2) continue;
        const Iso3 mean = AveragePoses(kv.second);
        double r = 0, t = 0;
        for (const Iso3& G : kv.second) {
            r += RotationAngleDeg(G.linear(), mean.linear());
            t += (G.translation() - mean.translation()).norm() * 100.0;
        }
        score += (r + t) / kv.second.size();
        ++n;
    }
    return n ? score / n : std::numeric_limits<double>::infinity();
}

// Closed-form hand-eye start for each wrist camera (Park & Martin): for two frames i, j of one
// static board, A = FK_j^-1 FK_i (tip motion) and B = C_j C_i^-1 (camera motion, C = T_cam_board)
// satisfy A X = X B. Rotation: Procrustes on the rotation vectors (R_X b = a); translation: least
// squares on (R_A - I) t_X = R_X t_B - t_A. The CAD mount is kept when it explains the data at
// least as well (the FK still carries the nominal joint zeros).
void RobotCalibrator::InitMountsFromData() {
    if (cfg_.solver.mount_init == "cad") return;
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        ArmState& a = arms_[ai];
        if (a.camera < 0) continue;
        std::map<std::string, std::vector<std::pair<Iso3, Iso3>>> obs;  // board -> (FK, C)
        for (const View& v : views_) {
            if (v.arm != static_cast<int>(ai) || !v.has_T_cam_board || v.dropped || v.holdout) continue;
            if (boards_.at(v.board).mounted) continue;
            std::vector<double> th(v.q.size());
            for (size_t j = 0; j < th.size(); ++j) th[j] = v.q[j] + a.dq[j];
            obs[v.board].push_back({a.chain.ForwardIso(th), v.T_cam_board});
        }
        std::vector<std::pair<Iso3, Iso3>> AB;
        for (const auto& kv : obs) {
            const auto& o = kv.second;
            for (size_t i = 0; i < o.size(); ++i) {
                for (size_t j = i + 1; j < o.size(); ++j) {
                    const Iso3 A = o[j].first.inverse() * o[i].first;
                    if (RotationAngleDeg(A.linear(), Mat3::Identity()) < 10.0) continue;
                    AB.push_back({A, o[j].second * o[i].second.inverse()});
                }
            }
        }
        const Iso3 cad = a.tip_cam.ToIso();
        if (AB.size() < 3) {
            std::cout << "[robocal] " << cfg_.arms[ai].name << " mount: too few frame pairs for a hand-eye start, CAD kept"
                      << std::endl;
            continue;
        }
        Mat3 H = Mat3::Zero();
        for (const auto& p : AB) {
            const Eigen::AngleAxisd ra(p.first.linear()), rb(p.second.linear());
            H += (rb.axis() * rb.angle()) * (ra.axis() * ra.angle()).transpose();
        }
        Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3 D = Mat3::Identity();
        D(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0 ? -1.0 : 1.0;
        const Mat3 R = svd.matrixV() * D * svd.matrixU().transpose();
        Eigen::MatrixXd M(3 * AB.size(), 3);
        Eigen::VectorXd b(3 * AB.size());
        for (size_t k = 0; k < AB.size(); ++k) {
            M.block<3, 3>(3 * k, 0) = AB[k].first.linear() - Mat3::Identity();
            b.segment<3>(3 * k) = R * AB[k].second.translation() - AB[k].first.translation();
        }
        Iso3 data = Iso3::Identity();
        data.linear() = R;
        data.translation() = M.colPivHouseholderQr().solve(b);
        const double s_cad = MountDispersion(static_cast<int>(ai), cad), s_data = MountDispersion(static_cast<int>(ai), data);
        const bool use_data = cfg_.solver.mount_init == "data" || s_data < s_cad;
        std::cout << "[robocal] " << cfg_.arms[ai].name << " mount start: hand-eye from " << AB.size()
                  << " frame pairs is " << std::fixed << std::setprecision(1)
                  << RotationAngleDeg(data.linear(), cad.linear()) << " deg / "
                  << (data.translation() - cad.translation()).norm() * 1e3 << " mm from CAD; board consistency CAD "
                  << std::setprecision(2) << s_cad << ", data " << s_data << " -> " << (use_data ? "data" : "CAD") << std::endl;
        if (use_data) a.tip_cam = PoseParam::FromIso(data);
    }
}

// Random jump of the robot unknowns (joint zeros, camera mounts, free bases), scaled.
void RobotCalibrator::Perturb(std::mt19937& rng, double scale) {
    std::normal_distribution<double> N(0.0, 1.0);
    const double a = cfg_.solver.global_step_deg * kDeg * scale, m = cfg_.solver.global_step_m * scale;
    auto rot = [&](PoseParam& p) {
        const Vec3 w(N(rng) * a, N(rng) * a, N(rng) * a);
        Iso3 T = p.ToIso();
        T.linear() = Eigen::AngleAxisd(w.norm(), w.norm() > 0 ? Vec3(w.normalized()) : Vec3::UnitX()).toRotationMatrix() *
                     T.linear();
        T.translation() += Vec3(N(rng), N(rng), N(rng)) * m;
        p = PoseParam::FromIso(T);
    };
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        ArmState& as = arms_[ai];
        for (size_t j = 0; j < as.dq.size(); ++j) {
            if (cfg_.arms[ai].dq_free[j]) as.dq[j] += N(rng) * a;
        }
        if (as.camera >= 0) rot(as.tip_cam);
        if (!cfg_.arms[ai].base_fixed) rot(as.world_base);
    }
    for (auto& kv : joint_eps_) std::fill(kv.second.begin(), kv.second.end(), 0.0);
}

// Global search around pass 2. multistart: perturbed restarts from the pass-2 solution, the best
// kept. basin_hopping: a chain of perturb -> local solve, accepted by Metropolis at a temperature
// falling linearly to 0 (an annealed search over the local minima).
void RobotCalibrator::GlobalSearch(const Flags& f) {
    std::mt19937 rng(cfg_.solver.global_seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const int K = std::max(1, cfg_.solver.global_iterations);
    const bool annealed = cfg_.solver.global == "basin_hopping";
    Flags held = f;
    held.free_dq = false;
    auto local = [&](const std::string& label) {
        RunSolver(held, label + " (poses, dq held)");
        return RunSolver(f, label + " (poses + dq)");
    };
    // Reference cost of the current solution under the same flags.
    double cur = RunSolver(f, "global: reference");
    Snapshot best = TakeSnapshot(), current = best;
    double best_cost = cur;
    std::vector<double> costs;
    for (int k = 0; k < K; ++k) {
        Restore(annealed ? current : best);
        const double T = cfg_.solver.global_temperature * (1.0 - static_cast<double>(k) / K);
        Perturb(rng, annealed ? std::max(0.25, 1.0 - static_cast<double>(k) / K) : 1.0);
        const double c = local("global " + std::to_string(k + 1) + "/" + std::to_string(K));
        costs.push_back(c);
        if (c < best_cost) {
            best_cost = c;
            best = TakeSnapshot();
        }
        if (annealed) {
            const bool accept = c < cur || (T > 0 && U(rng) < std::exp(-(c - cur) / (T * std::max(1e-12, cur))));
            if (accept) {
                cur = c;
                current = TakeSnapshot();
            }
        }
    }
    Restore(best);
    std::sort(costs.begin(), costs.end());
    std::cout << "[robocal] global search (" << cfg_.solver.global << ", " << K << " local solves): best cost "
              << std::scientific << std::setprecision(4) << best_cost << ", restarts landed between " << costs.front()
              << " and " << costs.back() << std::fixed << std::endl;
}

// solver.robust: corner -- iteratively reweighted Huber per corner: w = 1 inside huber_px, huber_px / e
// outside (e = the corner's 2D error); View.pw holds sqrt(w). Training views only.
int RobotCalibrator::UpdateCornerWeights() {
    if (cfg_.solver.robust != "corner" || cfg_.solver.huber_px <= 0) return 0;
    std::vector<double> r;
    int down = 0, total = 0;
    for (View& v : views_) {
        v.pw.clear();
        if (v.dropped || v.holdout || !ViewResiduals(v, r)) continue;
        // ViewResiduals is in pixels. Threshold huber_px * sigma_scale: the same pixel threshold as
        // robust: view (Huber huber_px / pixel_sigma on residuals whitened by pixel_sigma * sigma_scale).
        std::vector<double> w(v.obj.size(), 1.0);
        for (size_t i = 0; i < v.obj.size(); ++i) {
            const double e = std::hypot(r[2 * i], r[2 * i + 1]) / v.sigma_scale;
            if (e > cfg_.solver.huber_px) {
                w[i] = std::sqrt(cfg_.solver.huber_px / e);
                ++down;
            }
        }
        total += static_cast<int>(w.size());
        v.pw.swap(w);
    }
    if (cfg_.solver.verbose || down > 0) {
        std::cout << "[robocal] corner weights: " << down << " of " << total << " corners beyond huber_px "
                  << cfg_.solver.huber_px << " px down-weighted" << std::endl;
    }
    return down;
}

int RobotCalibrator::RejectOutliers() {
    std::vector<double> norms;
    std::vector<double> r;
    for (const View& v : views_) {
        if (v.dropped || v.holdout || !ViewResiduals(v, r)) continue;
        for (size_t i = 0; i < v.obj.size(); ++i) norms.push_back(std::hypot(r[2 * i], r[2 * i + 1]));
    }
    if (norms.size() < 10) return 0;
    std::vector<double> sorted = norms;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2];
    std::vector<double> dev;
    for (double e : sorted) dev.push_back(std::fabs(e - med));
    std::sort(dev.begin(), dev.end());
    const double mad = dev[dev.size() / 2];
    const double thr = std::max(cfg_.solver.outlier_floor_px, med + cfg_.solver.outlier_mad_k * 1.4826 * mad);
    int removed = 0;
    for (View& v : views_) {
        if (v.dropped || v.holdout || !ViewResiduals(v, r)) continue;
        std::vector<Vec3> obj;
        std::vector<Vec2> img;
        std::vector<int> pids;
        const size_t first_guided = v.obj.size() - std::min(v.obj.size(), static_cast<size_t>(std::max(0, v.n_guided)));
        int guided_kept = 0;
        for (size_t i = 0; i < v.obj.size(); ++i) {
            if (std::hypot(r[2 * i], r[2 * i + 1]) <= thr) {
                if (i >= first_guided) ++guided_kept;
                obj.push_back(v.obj[i]);
                img.push_back(v.img[i]);
                if (i < v.pids.size()) pids.push_back(v.pids[i]);
            } else {
                ++removed;
            }
        }
        v.obj.swap(obj);
        v.img.swap(img);
        if (!v.pids.empty()) v.pids.swap(pids);  // per-corner heights are indexed by pid
        v.n_guided = guided_kept;                // guided corners stay at the end, in order
        v.pw.clear();
        if (static_cast<int>(v.obj.size()) < cfg_.solver.min_points_per_view) v.dropped = true;
    }
    std::cout << "[robocal] outliers: " << removed << " point(s) above " << std::setprecision(2) << thr
              << " px removed" << robocal::Took() << std::endl;
    return removed;
}

void RobotCalibrator::Solve() {
    if (!initialized_) Initialize();
    Flags f;
    f.free_dq = false;
    ComputeDriveRegressors(true);
    RunSolver(f, "pass 1 (poses, dq held)");
    f.free_dq = true;
    UpdateCornerWeights();
    RunSolver(f, "pass 2 (poses + dq)");
    if (cfg_.solver.global != "none") GlobalSearch(f);
    if (cfg_.drive.model) {
        // The regressors were built at the nominal zeros (gravity torques and flanks can be off by
        // degrees): rebuild them at the solved zeros / bases and solve again before the noise
        // estimate, which would otherwise absorb that mismatch.
        ComputeDriveRegressors(true);
        RunSolver(f, "pass 2b (drive regressors at the solved zeros)");
    }
    diag_.outlier_points = RejectOutliers();
    // Pixel noise: weights the images against the priors stated in absolute units (encoder noise,
    // level board, board shape, dq). Robust per-axis spread of the training residuals.
    if (cfg_.solver.pixel_sigma_px > 0) {
        pixel_sigma_ = cfg_.solver.pixel_sigma_px;
    } else {
        std::vector<double> e, r;
        for (const View& v : views_) {
            if (v.dropped || v.holdout || !ViewResiduals(v, r)) continue;
            for (double x : r) e.push_back(std::fabs(x));
        }
        if (e.size() > 20) {
            std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
            pixel_sigma_ = std::max(0.02, 1.4826 * e[e.size() / 2]);
        }
    }
    std::cout << "[robocal] pixel noise " << std::setprecision(3) << pixel_sigma_ << " px (1 sigma per axis)"
              << (cfg_.solver.pixel_sigma_px > 0 ? " (configured)" : " (estimated)") << robocal::Took() << std::endl;
    diag_.pixel_sigma = pixel_sigma_;
    UpdateCornerWeights();
    RunSolver(f, "pass 3 (weighted, after outliers)");
    for (int round = 0; round < cfg_.solver.view_weight_rounds; ++round) {
        // s_v^2 = sum r^2 / (2 n - 6) per training view; sigma_v = max(s_0, s_v).
        std::vector<double> r;
        int heavy = 0;
        double sq_all = 0, sq_heavy = 0;
        for (View& v : views_) {
            if (v.dropped || v.holdout || !ViewResiduals(v, r)) continue;
            double sq = 0;
            for (double x : r) sq += x * x;
            const double dof = std::max(1.0, static_cast<double>(r.size()) - 6.0);
            v.sigma_scale = std::max(1.0, std::sqrt(sq / dof) / pixel_sigma_);
            sq_all += sq;
            if (v.sigma_scale > 2.0) {
                ++heavy;
                sq_heavy += sq;
            }
        }
        std::cout << "[robocal] view weights (round " << round + 1 << "): " << heavy
                  << " view(s) above 2x the pixel noise carry " << std::setprecision(3)
                  << (sq_all > 0 ? 100.0 * sq_heavy / sq_all : 0.0) << " % of the squared error" << robocal::Took() << std::endl;
        UpdateCornerWeights();
        RunSolver(f, "pass 3 (view-weighted)");
    }
    bool any_intr = false;
    for (const CameraConfig& c : cfg_.cameras) any_intr = any_intr || c.optimize_intrinsics;
    if (any_intr) {
        f.free_intrinsics = true;
        UpdateCornerWeights();
        RunSolver(f, "pass 4 (+ intrinsics)");
        if (UpdateCornerWeights() > 0) RunSolver(f, "pass 4b (+ intrinsics, reweighted)");  // weights moved
    }
    final_free_intrinsics_ = f.free_intrinsics;
    final_flags_ = f;
    ComputeErrorStats();
    std::cout << "[robocal] error statistics + hold-out refit done" << robocal::Took() << std::endl;
    // Sanity gate: a robot solve whose residuals stay far above the detection noise landed in a
    // wrong minimum (or the data / CAD is inconsistent): it must not be used.
    if (!cfg_.arms.empty() && cfg_.solver.max_rms_px > 0) {
        for (const auto& kv : diag_.train) {
            if (std::isfinite(kv.second.rms_point) && kv.second.rms_point > cfg_.solver.max_rms_px) {
                diag_.solve_failed = true;
                std::ostringstream o;
                o << std::fixed << std::setprecision(2) << "camera " << kv.first << ": training RMS "
                  << kv.second.rms_point << " px > " << cfg_.solver.max_rms_px
                  << " px (wrong minimum or inconsistent data / CAD: check the camera mounts and joint signs)";
                diag_.fail_reason += (diag_.fail_reason.empty() ? "" : "; ") + o.str();
            }
        }
    }
    ComputeLensReports();
    std::cout << "[robocal] lens reports done" << robocal::Took() << std::endl;
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        if (cfg_.cameras[ci].optimize_intrinsics && cams_[ci].handheld) {
            BootstrapIntrinsics(static_cast<int>(ci));
            std::cout << "[robocal] " << cfg_.cameras[ci].name << " bootstrap done" << robocal::Took() << std::endl;
        }
    }
    ApplyGates();
    ComputeObservability(f);
    std::cout << "[robocal] gates + observability done" << robocal::Took() << std::endl;
}


// Lens report per camera: training corners binned by off-axis angle (radial / tangential error
// about the principal point), image coverage and the largest measured angle.
void RobotCalibrator::ComputeLensReports() {
    const double edges[] = {0, 15, 30, 45, 60, 70, 75, 80, 85, 90, 95, 180};
    const int nb = static_cast<int>(sizeof(edges) / sizeof(edges[0])) - 1;
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        const CamState& cam = cams_[ci];
        LensReport rep;
        std::vector<double> sr(nb, 0), st(nb, 0), sm(nb, 0);
        std::vector<int> n(nb, 0);
        const int GX = 16, GY = 9;
        std::vector<bool> hit(GX * GY, false);
        std::vector<double> r;
        for (const View& v : views_) {
            if (v.camera != static_cast<int>(ci) || v.dropped || v.holdout || !ViewResiduals(v, r)) continue;
            for (size_t i = 0; i < v.img.size(); ++i) {
                Vec3 b;
                if (!UnprojectPixel(cam.intr.model, cam.intr.params.data(), v.img[i], b)) continue;
                const double th = std::atan2(std::hypot(b.x(), b.y()), b.z()) / kDeg;
                Vec2 dir = v.img[i] - Vec2(cam.intr.params[2], cam.intr.params[3]);
                const double dn = dir.norm();
                dir = dn > 1e-9 ? Vec2(dir / dn) : Vec2(1, 0);
                const Vec2 e(r[2 * i], r[2 * i + 1]);
                const double er = e.dot(dir), et = dir.x() * e.y() - dir.y() * e.x();
                int k = 0;
                while (k < nb - 1 && th >= edges[k + 1]) ++k;
                sr[k] += er * er;
                st[k] += et * et;
                sm[k] += er;
                n[k]++;
                const int gx = std::min(GX - 1, std::max(0, static_cast<int>(v.img[i].x() * GX / std::max(1, cam.intr.width))));
                const int gy = std::min(GY - 1, std::max(0, static_cast<int>(v.img[i].y() * GY / std::max(1, cam.intr.height))));
                hit[gy * GX + gx] = true;
            }
        }
        // Measured angle: upper edge of the highest bin with enough corners to mean something.
        for (int k = 0; k < nb; ++k) {
            if (n[k] >= 20) rep.theta_max_deg = std::min(edges[k + 1], 100.0);
        }
        for (int k = 0; k < nb; ++k) {
            if (n[k] == 0) continue;
            ThetaBin tb;
            tb.lo_deg = edges[k];
            tb.hi_deg = edges[k + 1];
            tb.n = n[k];
            tb.rms_radial = std::sqrt(sr[k] / n[k]);
            tb.rms_tangential = std::sqrt(st[k] / n[k]);
            tb.mean_radial = sm[k] / n[k];
            rep.bins.push_back(tb);
        }
        // Coverage over the cells whose centre the lens sees within 95 deg.
        for (int gy = 0; gy < GY; ++gy) {
            for (int gx = 0; gx < GX; ++gx) {
                const Vec2 c((gx + 0.5) * cam.intr.width / GX, (gy + 0.5) * cam.intr.height / GY);
                Vec3 b;
                if (cam.intr.width <= 0 || !UnprojectPixel(cam.intr.model, cam.intr.params.data(), c, b)) continue;
                if (std::atan2(std::hypot(b.x(), b.y()), b.z()) > 95 * kDeg) continue;
                rep.cells_total++;
                if (hit[gy * GX + gx]) rep.cells_hit++;
            }
        }
        rep.coverage = rep.cells_total ? static_cast<double>(rep.cells_hit) / rep.cells_total : 0.0;
        diag_.lens[cfg_.cameras[ci].name] = rep;
    }
}

// Repeatability of a session's intrinsics: refit intrinsics + frame poses on frames resampled
// with replacement (boards and their shape held at the solution), spread of the lens curve.
void RobotCalibrator::BootstrapIntrinsics(int ci) {
    CamState& cam = cams_[ci];
    const int B = cfg_.solver.bootstrap;
    // Resample whole time blocks of source frames (solver.block_frames) with replacement and re-solve
    // the full problem (lens, view poses, board shapes / scales / heights with their priors): neighbouring
    // frames carry correlated errors, so resampling single views would understate the spread.
    if (!arms_.empty()) {
        std::cout << "[robocal] bootstrap skipped: block bootstrap is for intrinsics-only runs" << std::endl;
        return;
    }
    std::map<int, std::vector<const View*>> blocks;
    for (const View& v : views_) {
        if (v.camera == ci && !v.dropped && !v.holdout) blocks[BlockOf(v)].push_back(&v);
    }
    // Too few blocks for a bootstrap (short session or long blocks): halve the block length until
    // there are >= 10 (down to single source frames), and say so.
    for (int len = cfg_.solver.block_frames / 2; blocks.size() < 10 && len >= 1; len /= 2) {
        blocks.clear();
        for (const View& v : views_) {
            if (v.camera != ci || v.dropped || v.holdout) continue;
            const int f = v.src_frame >= 0 ? v.src_frame : v.frame_idx;
            blocks[f / len].push_back(&v);
        }
        if (blocks.size() >= 10 || len == 1) {
            std::cout << "[robocal] bootstrap: blocks of " << len << " frame(s) (" << blocks.size()
                      << "): too few blocks of " << cfg_.solver.block_frames << std::endl;
            break;
        }
    }
    if (B <= 1 || blocks.size() < 5) {
        if (B > 1) std::cout << "[robocal] bootstrap skipped: " << blocks.size() << " time block(s) < 5" << std::endl;
        return;
    }
    std::vector<int> keys;
    for (const auto& kv : blocks) keys.push_back(kv.first);
    std::vector<std::vector<double>> intr0;
    for (const CamState& c : cams_) intr0.push_back(c.intr.params);
    const std::map<std::pair<int, int>, PoseParam> poses0 = frame_poses_;
    const std::map<std::string, BoardState> boards0 = boards_;
    auto restore = [&]() {
        for (size_t k = 0; k < cams_.size(); ++k) cams_[k].intr.params = intr0[k];
        for (auto& kv : frame_poses_) kv.second = poses0.at(kv.first);
        for (auto& kv : boards_) kv.second = boards0.at(kv.first);
    };
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(keys.size()) - 1);
    std::vector<std::vector<double>> runs;
    const int iters0 = cfg_.solver.max_iterations;
    cfg_.solver.max_iterations = std::min(iters0, 30);  // warm start from the solution
    for (int b = 0; b < B; ++b) {
        restore();
        std::map<const View*, int> mult;
        for (const View& v : views_) {  // other cameras' views enter once
            if (v.camera != ci && !v.dropped && !v.holdout) mult[&v] = 1;
        }
        for (size_t s = 0; s < keys.size(); ++s) {
            for (const View* v : blocks[keys[pick(rng)]]) ++mult[v];
        }
        boot_mult_ = &mult;
        try {
            RunSolver(final_flags_, "bootstrap " + std::to_string(b + 1) + "/" + std::to_string(B) + " (" +
                                        std::to_string(keys.size()) + " time blocks)");
        } catch (...) {
            boot_mult_ = nullptr;
            cfg_.solver.max_iterations = iters0;
            restore();
            throw;
        }
        boot_mult_ = nullptr;
        runs.push_back(cam.intr.params);
    }
    cfg_.solver.max_iterations = iters0;
    restore();

    LensReport& rep = diag_.lens[cfg_.cameras[ci].name];
    rep.bootstrap_runs = static_cast<int>(runs.size());
    auto sd = [&](int k) {
        double m = 0, q = 0;
        for (const auto& p : runs) m += p[k];
        m /= runs.size();
        for (const auto& p : runs) q += (p[k] - m) * (p[k] - m);
        return std::sqrt(q / std::max<size_t>(1, runs.size() - 1));
    };
    rep.fx_sd = sd(0);
    rep.fy_sd = sd(1);
    rep.cx_sd = sd(2);
    rep.cy_sd = sd(3);
    // Lens curve: image radius (px, x focal) of a ray at theta, per run.
    rep.curve_sd.clear();
    rep.max_curve_sd = 0;
    for (double th_deg = 5; th_deg <= std::min(rep.theta_max_deg, 100.0) + 1e-9; th_deg += 5) {
        std::vector<double> rad;
        for (const auto& p : runs) {
            const double th = th_deg * kDeg;
            const Eigen::Matrix<double, 3, 1> X(std::sin(th), 0.0, std::cos(th));
            double uv[2];
            if (ProjectDirection(cam.intr.model, p.data(), X, uv)) rad.push_back(uv[0] - p[2]);
        }
        if (rad.size() < 2) continue;
        double m = 0, q = 0;
        for (double x : rad) m += x;
        m /= rad.size();
        for (double x : rad) q += (x - m) * (x - m);
        const double s_px = std::sqrt(q / (rad.size() - 1));
        rep.curve_sd.push_back({th_deg, s_px});
        rep.max_curve_sd = std::max(rep.max_curve_sd, s_px);
    }
}

void RobotCalibrator::ApplyGates() {
    const IntrinsicsGates& g = cfg_.gates;
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        const CameraConfig& cc = cfg_.cameras[ci];
        if (!cc.optimize_intrinsics || !(cams_[ci].handheld)) continue;
        LensReport& rep = diag_.lens[cc.name];
        rep.gated = true;
        rep.failures.clear();
        const ErrorStats& h = diag_.holdout[cc.name];
        auto fail = [&](const std::string& m) { rep.failures.push_back(m); };
        std::ostringstream o;
        o << std::fixed << std::setprecision(3);
        if (g.max_holdout_rms_px > 0 && !(h.n_points > 0 && h.rms_point <= g.max_holdout_rms_px)) {
            o.str("");
            o << "hold-out rms " << h.rms_point << " px > " << g.max_holdout_rms_px;
            fail(h.n_points ? o.str() : std::string("no hold-out frames"));
        }
        if (g.min_theta_deg > 0 && rep.theta_max_deg < g.min_theta_deg) {
            o.str("");
            o << "largest measured angle " << std::setprecision(1) << rep.theta_max_deg << " deg < " << g.min_theta_deg;
            fail(o.str());
        }
        if (g.min_coverage > 0 && rep.coverage < g.min_coverage) {
            o.str("");
            o << "coverage " << std::setprecision(2) << rep.coverage << " < " << g.min_coverage;
            fail(o.str());
        }
        if (g.max_curve_sd_px > 0 && rep.bootstrap_runs > 1 && rep.max_curve_sd > g.max_curve_sd_px) {
            o.str("");
            o << "lens curve spread " << rep.max_curve_sd << " px > " << g.max_curve_sd_px;
            fail(o.str());
        }
        // The fitted polynomial must stay one-to-one (increasing) through the measured angles and a
        // margin past them: beyond its fold a radius belongs to two angles.
        rep.fold_deg = KannalaBrandtFoldDeg(cams_[ci].intr.model, cams_[ci].intr.params.data());
        if (rep.fold_deg < rep.theta_max_deg + 5.0) {
            o.str("");
            o << "lens model folds back at " << std::setprecision(1) << rep.fold_deg << " deg (measured to "
              << rep.theta_max_deg << " deg)";
            fail(o.str());
        }
        for (const ThetaBin& tb : rep.bins) {
            if (g.max_mean_radial_px > 0 && tb.n >= 50 && std::fabs(tb.mean_radial) > g.max_mean_radial_px) {
                o.str("");
                o << "radial bias " << tb.mean_radial << " px at " << std::setprecision(0) << tb.lo_deg << "-" << tb.hi_deg << " deg";
                fail(o.str());
            }
        }
        rep.pass = rep.failures.empty();
    }
}

void RobotCalibrator::WriteCalibratedUrdf(const std::string& path) const {
    using namespace tinyxml2;
    auto fmt3 = [](const Vec3& v) {
        std::ostringstream o;
        o << std::setprecision(12) << v.x() << " " << v.y() << " " << v.z();
        return o.str();
    };
    XMLDocument out;
    out.InsertEndChild(out.NewDeclaration());
    XMLElement* robot = out.NewElement("robot");
    robot->SetAttribute("name", "calibrated_cell");
    out.InsertEndChild(robot);
    robot->InsertEndChild(out.NewComment(
        " calico-robocal: arm joint zero offsets are baked into the joint origins, so this model takes "
        "encoder angles (q_model = q_encoder + delta_q is already applied). "));
    std::set<std::string> links;
    auto add_link = [&](const std::string& name) {
        if (!links.insert(name).second) throw std::runtime_error("calibrated URDF: duplicate link " + name);
        XMLElement* l = out.NewElement("link");
        l->SetAttribute("name", name.c_str());
        robot->InsertEndChild(l);
    };
    auto add_fixed = [&](const std::string& name, const std::string& parent, const std::string& child, const Iso3& T) {
        XMLElement* j = out.NewElement("joint");
        j->SetAttribute("name", name.c_str());
        j->SetAttribute("type", "fixed");
        XMLElement* pe = out.NewElement("parent");
        pe->SetAttribute("link", parent.c_str());
        XMLElement* ce = out.NewElement("child");
        ce->SetAttribute("link", child.c_str());
        XMLElement* oe = out.NewElement("origin");
        oe->SetAttribute("xyz", fmt3(T.translation()).c_str());
        oe->SetAttribute("rpy", fmt3(RotationToRpy(T.linear())).c_str());
        j->InsertEndChild(pe);
        j->InsertEndChild(ce);
        j->InsertEndChild(oe);
        robot->InsertEndChild(j);
    };
    // Each distinct URDF is copied once (two arms may share one bimanual URDF); a joint gets the
    // offset of the arm that lists it. Links the URDFs already define (e.g. the world frame) are
    // not added again, and an arm base that already has a parent joint from the world frame gets
    // the calibrated pose on that joint instead of a second parent.
    auto read3 = [](const char* t, const Vec3& d) {
        if (t == nullptr) return d;
        std::istringstream in(t);
        Vec3 v;
        in >> v.x() >> v.y() >> v.z();
        return v;
    };
    std::vector<std::unique_ptr<XMLDocument>> srcs;
    std::map<std::string, int> src_of_key;
    std::vector<int> arm_src(arms_.size(), -1);
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        const ArmConfig& ac = cfg_.arms[ai];
        const std::string key = ac.urdf_xml.empty() ? "file:" + ac.urdf : "xml:" + ac.urdf_xml;
        auto it = src_of_key.find(key);
        if (it == src_of_key.end()) {
            auto doc = std::make_unique<XMLDocument>();
            const XMLError err = ac.urdf_xml.empty() ? doc->LoadFile(ac.urdf.c_str()) : doc->Parse(ac.urdf_xml.c_str());
            if (err != XML_SUCCESS || doc->FirstChildElement("robot") == nullptr) {
                throw std::runtime_error("calibrated URDF: cannot read the URDF of arm " + ac.name);
            }
            it = src_of_key.emplace(key, static_cast<int>(srcs.size())).first;
            srcs.push_back(std::move(doc));
        }
        arm_src[ai] = it->second;
    }
    std::map<std::string, std::pair<size_t, size_t>> joint_owner;  // joint -> (arm, index in the arm's chain)
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        for (size_t k = 0; k < cfg_.arms[ai].joints.size(); ++k) joint_owner[cfg_.arms[ai].joints[k]] = {ai, k};
    }
    std::set<std::string> src_links;
    std::map<std::string, XMLElement*> parent_joint;  // child link -> its joint in the output
    for (const auto& doc : srcs) {
        const XMLElement* srobot = doc->FirstChildElement("robot");
        for (const XMLElement* e = srobot->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
            if (std::string(e->Name()) == "link" && e->Attribute("name")) src_links.insert(e->Attribute("name"));
        }
    }
    if (!src_links.count(cfg_.world_frame)) add_link(cfg_.world_frame);
    for (const auto& doc : srcs) {
        const XMLElement* srobot = doc->FirstChildElement("robot");
        for (const XMLElement* e = srobot->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
            const std::string tag = e->Name();
            if (tag == "link") {
                if (!links.insert(e->Attribute("name")).second) {
                    throw std::runtime_error(std::string("calibrated URDF: link ") + e->Attribute("name") +
                                             " is defined by two different URDFs");
                }
            }
            XMLNode* copy = e->DeepClone(&out);
            if (tag == "joint") {
                XMLElement* j = copy->ToElement();
                const std::string jn = j->Attribute("name") ? j->Attribute("name") : "";
                if (const XMLElement* c = j->FirstChildElement("child")) {
                    if (c->Attribute("link")) parent_joint[c->Attribute("link")] = j;
                }
                const auto ow = joint_owner.find(jn);
                if (ow != joint_owner.end() && arms_[ow->second.first].dq[ow->second.second] != 0.0) {
                    const double dq = arms_[ow->second.first].dq[ow->second.second];
                    // origin * Rot(axis, q + dq) == (origin * Rot(axis, dq)) * Rot(axis, q)
                    XMLElement* o = j->FirstChildElement("origin");
                    if (o == nullptr) {
                        o = out.NewElement("origin");
                        j->InsertFirstChild(o);
                    }
                    const Vec3 xyz = read3(o->Attribute("xyz"), Vec3::Zero());
                    const Vec3 rpy = read3(o->Attribute("rpy"), Vec3::Zero());
                    const XMLElement* ax = j->FirstChildElement("axis");
                    const Vec3 axis = read3(ax ? ax->Attribute("xyz") : nullptr, Vec3::UnitX()).normalized();
                    const std::string type = j->Attribute("type") ? j->Attribute("type") : "";
                    Vec3 nxyz = xyz;
                    Mat3 R = RpyToRotation(rpy);
                    if (type == "prismatic") nxyz = xyz + R * axis * dq;
                    else R = R * Eigen::AngleAxisd(dq, axis).toRotationMatrix();
                    o->SetAttribute("xyz", fmt3(nxyz).c_str());
                    o->SetAttribute("rpy", fmt3(RotationToRpy(R)).c_str());
                }
            }
            robot->InsertEndChild(copy);
        }
    }
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        const ArmConfig& ac = cfg_.arms[ai];
        const ArmState& a = arms_[ai];
        const auto pj = parent_joint.find(ac.base_link);
        if (pj == parent_joint.end()) {
            add_fixed(cfg_.world_frame + "_to_" + ac.base_link, cfg_.world_frame, ac.base_link, a.world_base.ToIso());
        } else {
            XMLElement* j = pj->second;
            const XMLElement* pe = j->FirstChildElement("parent");
            const std::string parent = pe && pe->Attribute("link") ? pe->Attribute("link") : "";
            const std::string type = j->Attribute("type") ? j->Attribute("type") : "";
            if (parent != cfg_.world_frame || type != "fixed") {
                throw std::runtime_error("calibrated URDF: " + ac.base_link + " already has parent " + parent + " (" + type +
                                         " joint) in the URDF; the solved base pose is relative to " + cfg_.world_frame);
            }
            XMLElement* o = j->FirstChildElement("origin");
            if (o == nullptr) {
                o = out.NewElement("origin");
                j->InsertFirstChild(o);
            }
            o->SetAttribute("xyz", fmt3(a.world_base.ToIso().translation()).c_str());
            o->SetAttribute("rpy", fmt3(RotationToRpy(a.world_base.ToIso().linear())).c_str());
        }
        if (a.camera >= 0) {
            const std::string cam = cfg_.cameras[a.camera].name + "_optical_frame";
            add_link(cam);
            add_fixed(ac.tip_link + "_to_" + cam, ac.tip_link, cam, a.tip_cam.ToIso());
        }
    }
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        if (!cams_[ci].is_static) continue;
        const std::string cam = cfg_.cameras[ci].name + "_optical_frame";
        add_link(cam);
        add_fixed(cfg_.world_frame + "_to_" + cam, cfg_.world_frame, cam, cams_[ci].world_cam.ToIso());
    }
    if (out.SaveFile(path.c_str()) != XML_SUCCESS) throw std::runtime_error("cannot write " + path);
}

std::vector<std::string> RobotCalibrator::WriteCameraInfos(const std::string& dir) const {
    std::vector<std::string> written;
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        const CameraConfig& cc = cfg_.cameras[ci];
        if (!cc.optimize_intrinsics) continue;
        const CameraIntrinsics& in = cams_[ci].intr;
        const auto lr = diag_.lens.find(cc.name);
        const bool failed = lr != diag_.lens.end() && lr->second.gated && !lr->second.pass;
        // Intrinsics that failed their gates get a name no downstream config will load by accident.
        const std::string path = dir + "/" + cc.name + (failed ? ".camera_info.FAILED.yaml" : ".camera_info.yaml");
        std::ofstream f(path);
        if (!f) throw std::runtime_error("cannot write " + path);
        f << std::setprecision(15);
        const auto it = diag_.lens.find(cc.name);
        const ErrorStats& h = diag_.holdout.at(cc.name);
        f << "# " << cc.name << " intrinsics, calico-robocal (" << LensModelName(in.model) << ")\n";
        f << "# hold-out rms " << h.rms_point << " px per point (" << h.n_views << " views)";
        if (it != diag_.lens.end()) {
            const LensReport& L = it->second;
            f << ", measured to " << L.theta_max_deg << " deg, coverage " << L.coverage;
            if (L.bootstrap_runs > 1) f << ", lens curve spread " << L.max_curve_sd << " px";
            if (L.gated) f << ", gates " << (L.pass ? "PASS" : "FAIL");
        }
        f << "\n";
        f << "image_width: " << in.width << "\nimage_height: " << in.height << "\ncamera_name: " << cc.name << "\n";
        f << "width: " << in.width << "\nheight: " << in.height << "\n";
        f << "distortion_model: " << LensModelName(in.model) << "\n";
        f << "d: [";
        for (size_t k = 4; k < in.params.size(); ++k) f << (k > 4 ? ", " : "") << in.params[k];
        f << "]\n";
        const double fx = in.params[0], fy = in.params[1], cx = in.params[2], cy = in.params[3];
        f << "k: [" << fx << ", 0, " << cx << ", 0, " << fy << ", " << cy << ", 0, 0, 1]\n";
        f << "r: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
        f << "p: [" << fx << ", 0, " << cx << ", 0, 0, " << fy << ", " << cy << ", 0, 0, 0, 1, 0]\n";
        written.push_back(path);
        // Tools that only know ROS "equidistant" get the closest such model, with its error stated.
        if (in.model != LensModel::Equidistant && IsWideAngle(in.model)) {
            const double th = it != diag_.lens.end() && it->second.theta_max_deg > 0 ? it->second.theta_max_deg : 90.0;
            double rms = 0, mx = 0;
            // Non-central (kb4_pupil): the central lens that fits points at the camera's working distance.
            const bool np = in.model == LensModel::KannalaBrandtPupil;
            const double dw = cc.export_distance_m;
            const CameraIntrinsics eq = np ? FitModel(in, LensModel::Equidistant, th, &rms, &mx, dw)
                                           : FitEquidistant(in, th, &rms, &mx);
            const std::string p2 = dir + "/" + cc.name + (failed ? ".camera_info.equidistant.FAILED.yaml" : ".camera_info.equidistant.yaml");
            std::ofstream g(p2);
            g << std::setprecision(15);
            g << "# " << cc.name << ": equidistant approximation of the " << LensModelName(in.model)
              << " calibration (for ROS / tools without that model): rms " << rms << " px, max " << mx
              << " px over rays to " << th << " deg" << (np ? " for points at " + std::to_string(dw) + " m" : "") << "\n";
            if (np) {  // what the central file gets wrong at other distances
                for (double d2 : {0.12, 0.5, 1e6}) {
                    const IntrinsicsComparison cmp = CompareIntrinsicsDetailed(in, eq, std::min(th, 85.0), d2);
                    const double m2 = cmp.max[0];
                    g << "#   points at " << (d2 > 1e5 ? std::string("infinity") : std::to_string(d2) + " m")
                      << ": max error " << m2 << " px (rays to " << std::min(th, 85.0) << " deg)\n";
                }
                g << "#   pupil shift e1 = " << in.params[8] * 1e3 << " mm at 90 deg (z0 = e1 (theta / 90 deg)^2)\n";
            }
            g << "image_width: " << in.width << "\nimage_height: " << in.height << "\ncamera_name: " << cc.name << "\n";
            g << "width: " << in.width << "\nheight: " << in.height << "\n";
            g << "distortion_model: equidistant\nd: [" << eq.params[4] << ", " << eq.params[5] << ", " << eq.params[6]
              << ", " << eq.params[7] << "]\n";
            g << "k: [" << eq.params[0] << ", 0, " << eq.params[2] << ", 0, " << eq.params[1] << ", " << eq.params[3]
              << ", 0, 0, 1]\n";
            g << "r: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
            g << "p: [" << eq.params[0] << ", 0, " << eq.params[2] << ", 0, 0, " << eq.params[1] << ", " << eq.params[3]
              << ", 0, 0, 0, 1, 0]\n";
            written.push_back(p2);
        }
    }
    return written;
}

void RobotCalibrator::ComputeErrorStats() {
    RefitHeldOutFrames();
    std::map<int, std::vector<double>> train, hold;
    std::map<int, std::set<const View*>> train_v, hold_v;
    std::vector<double> r;
    for (const View& v : views_) {
        if (v.dropped || !ViewResiduals(v, r)) continue;
        auto& dst = v.holdout ? hold[v.camera] : train[v.camera];
        (v.holdout ? hold_v : train_v)[v.camera].insert(&v);
        for (size_t i = 0; i < v.obj.size(); ++i) dst.push_back(std::hypot(r[2 * i], r[2 * i + 1]));
    }
    diag_.train.clear();
    diag_.holdout.clear();
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        const std::string& name = cfg_.cameras[ci].name;
        diag_.train[name] = StatsOf(train[ci], static_cast<int>(train_v[ci].size()));
        diag_.holdout[name] = StatsOf(hold[ci], static_cast<int>(hold_v[ci].size()));
    }
}

void RobotCalibrator::ComputeObservability(const Flags& f_in) {
    Flags f = f_in;
    f.use_loss = false;
    f.no_priors = true;  // what the images alone determine
    f.pose7 = false;     // per-block names below (q and t separately)
    ceres::Problem problem;
    BuildProblem(problem, f);
    // Name every tangent dimension of every free block.
    struct BlockInfo {
        double* ptr;
        std::vector<std::string> names;
        enum Kind { DQ, ROT, TRANS, INTR, SHAPE } kind;
    };
    std::vector<BlockInfo> blocks;
    auto add = [&](double* p, BlockInfo::Kind kind, std::vector<std::string> names) {
        if (!problem.HasParameterBlock(p) || problem.IsParameterBlockConstant(p)) return;
        blocks.push_back({p, std::move(names), kind});
    };
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        ArmState& a = arms_[ai];
        const std::string& n = cfg_.arms[ai].name;
        std::vector<std::string> dq;
        for (int j = 0; j < a.chain.dof(); ++j) {
            if (cfg_.arms[ai].dq_free[j]) dq.push_back(n + ".dq[" + std::to_string(j + 1) + "]");
        }
        add(a.dq.data(), BlockInfo::DQ, dq);
        add(a.tip_cam.q, BlockInfo::ROT, {n + ".tip_cam.rx", n + ".tip_cam.ry", n + ".tip_cam.rz"});
        add(a.tip_cam.t, BlockInfo::TRANS, {n + ".tip_cam.x", n + ".tip_cam.y", n + ".tip_cam.z"});
        add(a.world_base.q, BlockInfo::ROT, {n + ".base.rx", n + ".base.ry", n + ".base.rz"});
        add(a.world_base.t, BlockInfo::TRANS, {n + ".base.x", n + ".base.y", n + ".base.z"});
    }
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        CamState& c = cams_[ci];
        const std::string& n = cfg_.cameras[ci].name;
        if (c.is_static) {
            add(c.world_cam.q, BlockInfo::ROT, {n + ".pose.rx", n + ".pose.ry", n + ".pose.rz"});
            add(c.world_cam.t, BlockInfo::TRANS, {n + ".pose.x", n + ".pose.y", n + ".pose.z"});
        }
        std::vector<std::string> in = {n + ".fx", n + ".fy", n + ".cx", n + ".cy"};
        for (int k = 0; k < DistortionCount(c.intr.model); ++k) in.push_back(n + ".d" + std::to_string(k));
        add(c.intr.params.data(), BlockInfo::INTR, in);
    }
    for (auto& kv : frame_poses_) {
        const std::string n = cfg_.cameras[kv.first.first].name + ".frame[" + std::to_string(kv.first.second) + "]";
        add(kv.second.q, BlockInfo::ROT, {n + ".rx", n + ".ry", n + ".rz"});
        add(kv.second.t, BlockInfo::TRANS, {n + ".x", n + ".y", n + ".z"});
    }
    for (auto& kv : boards_) {
        const std::string n = "board[" + kv.first + "]";
        BoardState& b = kv.second;
        if (b.mounted) {
            add(b.mount.q, BlockInfo::ROT, {n + ".mount.rx", n + ".mount.ry", n + ".mount.rz"});
            add(b.mount.t, BlockInfo::TRANS, {n + ".mount.x", n + ".mount.y", n + ".mount.z"});
        } else {
            add(b.pose.q, BlockInfo::ROT, {n + ".rx", n + ".ry", n + ".rz"});
            add(b.pose.t, BlockInfo::TRANS, {n + ".x", n + ".y", n + ".z"});
        }
        std::vector<std::string> sh = {n + ".bow_x", n + ".bow_y", n + ".twist"};
        if (b.fit_scale) {
            sh.push_back(n + ".scale_x");
            sh.push_back(n + ".scale_y");
        }
        add(b.shape, BlockInfo::SHAPE, sh);
    }
    std::vector<double*> ptrs;
    std::vector<std::string> names;
    for (const BlockInfo& b : blocks) {
        ptrs.push_back(b.ptr);
        if (static_cast<int>(b.names.size()) != problem.ParameterBlockTangentSize(b.ptr)) {
            throw std::runtime_error("internal: tangent size mismatch for " + b.names.front());
        }
        names.insert(names.end(), b.names.begin(), b.names.end());
    }
    ceres::Problem::EvaluateOptions eo;
    eo.parameter_blocks = ptrs;
    eo.apply_loss_function = false;
    double cost = 0.0;
    std::vector<double> res;
    ceres::CRSMatrix crs;
    problem.Evaluate(eo, &cost, &res, nullptr, &crs);
    const int m = crs.num_rows, n = crs.num_cols;
    if (n != static_cast<int>(names.size())) {
        throw std::runtime_error("internal: Jacobian has " + std::to_string(n) + " columns, expected " +
                                 std::to_string(names.size()));
    }
    diag_.n_free_params = n;
    diag_.n_residuals = m;
    diag_.final_cost = cost;
    const bool eigen = cfg_.solver.observability > 0 || (cfg_.solver.observability < 0 && !arms_.empty());
    diag_.observability_done = eigen;
    diag_.weak.clear();
    diag_.smallest_sigma_ratio.clear();
    diag_.zero_columns.clear();
    diag_.condition = std::numeric_limits<double>::quiet_NaN();
    if (eigen) {
    Eigen::MatrixXd JtJ = Eigen::MatrixXd::Zero(n, n);
    for (int r = 0; r < m; ++r) {
        for (int a = crs.rows[r]; a < crs.rows[r + 1]; ++a) {
            for (int b = crs.rows[r]; b < crs.rows[r + 1]; ++b) {
                JtJ(crs.cols[a], crs.cols[b]) += crs.values[a] * crs.values[b];
            }
        }
    }
    Eigen::VectorXd scale(n);
    diag_.zero_columns.clear();
    for (int j = 0; j < n; ++j) {
        const double s = std::sqrt(JtJ(j, j));
        scale[j] = s > 0 ? 1.0 / s : 0.0;
        if (s == 0) diag_.zero_columns.push_back(names[j]);
    }
    const Eigen::MatrixXd S = scale.asDiagonal() * JtJ * scale.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    const Eigen::VectorXd ev = es.eigenvalues().cwiseMax(0.0);
    const double smax = std::sqrt(ev.maxCoeff());
    diag_.smallest_sigma_ratio.clear();
    diag_.weak.clear();
    for (int i = 0; i < std::min(n, 6); ++i) diag_.smallest_sigma_ratio.push_back(std::sqrt(ev[i]) / smax);
    diag_.condition = ev[0] > 0 ? smax / std::sqrt(ev[0]) : std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const double ratio = std::sqrt(ev[i]) / smax;
        if (ratio > 1e-4) break;
        WeakDirection w;
        w.sigma_ratio = ratio;
        std::vector<std::pair<double, int>> comp;
        for (int j = 0; j < n; ++j) comp.push_back({std::fabs(es.eigenvectors()(j, i)), j});
        std::sort(comp.rbegin(), comp.rend());
        for (int k = 0; k < std::min(n, 4); ++k) {
            w.components.push_back({names[comp[k].second], es.eigenvectors()(comp[k].second, i)});
        }
        diag_.weak.push_back(w);
    }
    std::cout << "[robocal] observability (" << n << " params, dense eigen) done" << robocal::Took() << std::endl;
    }  // eigen

    // Marginal 1-sigma from the covariance (priors included), scaled by the residual variance.
    diag_.sigma.clear();
    Flags fp = f;
    fp.no_priors = false;
    ceres::Problem full;
    BuildProblem(full, fp);
    for (double* p : ptrs) {
        if (!full.HasParameterBlock(p) || full.IsParameterBlockConstant(p)) {
            throw std::runtime_error("internal: covariance block mismatch");
        }
    }
    {
        // Stage-1 problems: marginal covariance of the lens / shape blocks = blocks of S^-1 (reduced
        // Schur matrix, priors in, no loss) -- seconds instead of a sparse QR of the full Jacobian.
        const char* env = std::getenv("ROBOCAL_BACKEND");
        const std::string backend = env ? env : cfg_.solver.backend;
        std::map<const double*, Eigen::MatrixXd> s1;
        Flags fs = fp;
        fs.pose7 = true;
        double unused = 0;
        bool covered = backend != "ceres";
        if (covered) {
            std::set<const double*> fb;
            for (auto& kv : frame_poses_) {
                fb.insert(kv.second.q);
                fb.insert(kv.second.t);
            }
            covered = RunStage1Solver(fs, "covariance", &unused, &s1);
            for (const BlockInfo& b : blocks) {
                if (covered && !fb.count(b.ptr) &&
                    (!s1.count(b.ptr) || s1.at(b.ptr).rows() != full.ParameterBlockTangentSize(b.ptr))) {
                    covered = false;
                }
            }
            if (covered) {
                diag_.covariance_ok = true;
                std::cout << "[robocal] covariance (formal sigma, schur) done" << robocal::Took() << std::endl;
                const int dof = std::max(1, m - n);
                const double s2 = 2.0 * cost / dof;
                for (const BlockInfo& b : blocks) {
                    if (fb.count(b.ptr)) continue;
                    const Eigen::MatrixXd& C = s1.at(b.ptr);
                    for (int i = 0; i < C.rows(); ++i) {
                        double sd = std::sqrt(std::max(0.0, C(i, i) * s2));
                        if (b.kind == BlockInfo::SHAPE) sd *= (i < 3 ? 1e3 : 1e2);  // mm / %
                        diag_.sigma[b.names[i]] = sd;
                    }
                }
                return;
            }
        }
    }
    ceres::Covariance::Options co;
    int n_full = 0;
    {
        std::vector<double*> pbs;
        full.GetParameterBlocks(&pbs);
        for (double* p : pbs) {
            if (!full.IsParameterBlockConstant(p)) n_full += full.ParameterBlockTangentSize(p);
        }
    }
    co.algorithm_type = n_full > 400 ? ceres::SPARSE_QR : ceres::DENSE_SVD;
    co.null_space_rank = -1;
    co.min_reciprocal_condition_number = 1e-14;
    co.apply_loss_function = false;
    ceres::Covariance cov(co);
    std::vector<std::pair<const double*, const double*>> pairs;
    std::set<const double*> frame_blocks;  // per-frame poses: not reported, not requested
    for (auto& kv : frame_poses_) {
        frame_blocks.insert(kv.second.q);
        frame_blocks.insert(kv.second.t);
    }
    for (double* p : ptrs) {
        if (!frame_blocks.count(p)) pairs.push_back({p, p});
    }
    diag_.covariance_ok = cov.Compute(pairs, &full);
    std::cout << "[robocal] covariance (formal sigma) done" << robocal::Took() << std::endl;
    if (!diag_.covariance_ok) return;
    const int dof = std::max(1, m - n);
    const double s2 = 2.0 * cost / dof;
    for (const BlockInfo& b : blocks) {
        if (frame_blocks.count(b.ptr)) continue;
        const int k = full.ParameterBlockTangentSize(b.ptr);
        std::vector<double> C(k * k);
        if (!cov.GetCovarianceBlockInTangentSpace(b.ptr, b.ptr, C.data())) continue;
        for (int i = 0; i < k; ++i) {
            double sd = std::sqrt(std::max(0.0, C[i * k + i] * s2));
            switch (b.kind) {
                case BlockInfo::DQ: sd /= kDeg; break;                  // deg
                case BlockInfo::ROT: sd = 2.0 * sd / kDeg; break;       // quaternion tangent = half angle
                case BlockInfo::TRANS: sd *= 1e3; break;                // mm
                case BlockInfo::INTR: break;                            // native units
                case BlockInfo::SHAPE: sd *= (i < 3 ? 1e3 : 1e2); break; // mm / %
            }
            diag_.sigma[b.names[i]] = sd;
        }
    }
}

std::vector<double> RobotCalibrator::DeltaQ(int arm) const { return arms_.at(arm).dq; }
Iso3 RobotCalibrator::TipCam(int arm) const { return arms_.at(arm).tip_cam.ToIso(); }
Iso3 RobotCalibrator::WorldBase(int arm) const { return arms_.at(arm).world_base.ToIso(); }
Iso3 RobotCalibrator::WorldCamera(int cam) const { return cams_.at(cam).world_cam.ToIso(); }
Iso3 RobotCalibrator::FrameCamera(int cam, int frame) const { return frame_poses_.at({cam, frame}).ToIso(); }
Iso3 RobotCalibrator::WorldBoard(const std::string& b) const { return boards_.at(b).pose.ToIso(); }
Iso3 RobotCalibrator::LinkBoard(const std::string& b) const { return boards_.at(b).mount.ToIso(); }
std::vector<double> RobotCalibrator::BoardShape(const std::string& b) const {
    const BoardState& s = boards_.at(b);
    return std::vector<double>(s.shape, s.shape + 5);
}
const CameraIntrinsics& RobotCalibrator::Intrinsics(int cam) const { return cams_.at(cam).intr; }

std::string RobotCalibrator::Summary() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3);
    o << "== robocal result (RMS = sqrt(mean |e|^2) per point, px) ==\n";
    if (diag_.solve_failed) o << "  STATUS: FAILED -- " << diag_.fail_reason << "\n";
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        const std::string& n = cfg_.cameras[ci].name;
        const ErrorStats& t = diag_.train.at(n);
        const ErrorStats& h = diag_.holdout.at(n);
        o << "  " << n << ": train " << t.rms_point << " (" << t.n_views << " views, " << t.n_points
          << " pts, p95 " << t.p95 << ")  holdout " << h.rms_point << " (" << h.n_views << " views)\n";
    }
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        const std::string& n = cfg_.arms[ai].name;
        o << "  " << n << " dq deg:";
        for (size_t j = 0; j < arms_[ai].dq.size(); ++j) {
            o << " " << std::setprecision(3) << arms_[ai].dq[j] / kDeg;
            const auto it = diag_.sigma.find(n + ".dq[" + std::to_string(j + 1) + "]");
            if (it != diag_.sigma.end()) {
                o << "(+-" << std::setprecision(3) << it->second << ")";
                // Posterior close to the prior: the images do not determine this joint.
                if (it->second > 0.3 * cfg_.arms[ai].dq_sigma_rad[j] / kDeg) o << "[PRIOR-LIMITED]";
            }
            else if (!cfg_.arms[ai].dq_free[j]) o << "(fixed)";
        }
        o << "\n";
        if (cfg_.drive.model && (!cfg_.drive.shared || ai == 0)) {
            const ArmState& a = arms_[ai];
            const int nj = a.chain.dof();
            const std::vector<int>& held = a.drv_const;
            auto is_held = [&](int e) { return std::find(held.begin(), held.end(), e) != held.end(); };
            o << std::noshowpos << "  drive" << (cfg_.drive.shared ? " (all arms)" : " " + n)
              << ": stiffness N m/rad | backlash arcmin:";
            for (int j = 0; j < nj; ++j) {
                o << " J" << j + 1 << " ";
                if (is_held(j)) o << "-";
                else o << std::setprecision(0) << (a.drv[j] > 1e-9 ? 1.0 / a.drv[j] : 0.0);
                o << "|";
                if (is_held(nj + j)) o << "(" << std::setprecision(1) << 2 * a.drv[nj + j] / kDeg * 60 << " prior)";
                else o << std::setprecision(1) << 2 * a.drv[nj + j] / kDeg * 60;
            }
            o << "\n";
        }
    }
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        if (cfg_.arms[ai].joint_noise_rad <= 0) continue;
        double sq = 0;
        int n = 0;
        for (const auto& kv : joint_eps_) {
            if (kv.first.first != static_cast<int>(ai)) continue;
            for (double e : kv.second) {
                if (e != 0.0) {
                    sq += e * e;
                    ++n;
                }
            }
        }
        if (n) {
            o << "  " << cfg_.arms[ai].name << " encoder errors (fitted, per frame): rms " << std::setprecision(4)
              << std::sqrt(sq / n) / kDeg << " deg against the prior " << cfg_.arms[ai].joint_noise_rad / kDeg << " deg\n";
        }
    }
    if (arms_.size() >= 2) {
        const Iso3 rel = WorldBase(0).inverse() * WorldBase(1);
        o << "  base separation " << cfg_.arms[0].name << "->" << cfg_.arms[1].name << ": "
          << std::setprecision(1) << rel.translation().norm() * 1e3 << " mm\n";
    }
    for (const auto& kv : boards_) {
        const BoardState& b = kv.second;
        if (b.fit_shape) {
            o << "  board " << kv.first << " shape: bow x " << std::setprecision(2) << b.shape[0] * 1e3 << " mm, bow y "
              << b.shape[1] * 1e3 << " mm, twist " << b.shape[2] * 1e3 << " mm";
            if (b.fit_scale || b.print_scale[0] != 0 || b.print_scale[1] != 0) o << ", scale x " << std::setprecision(3) << b.shape[3] * 1e2 << " %, y " << b.shape[4] * 1e2 << " %";
            o << "\n";
        }
        if (b.fit_warp) {
            double sq = 0, mx = 0;
            for (double w : b.warp) sq += w * w, mx = std::max(mx, std::fabs(w));
            o << "  board " << kv.first << " in-plane warp: coefficients rms " << std::setprecision(3)
              << std::sqrt(sq / 8) * 1e3 << " mm, max " << mx * 1e3 << " mm\n";
        }
        if (b.fit_points && !b.heights.empty()) {
            double sq = 0, mx = 0;
            for (double h : b.heights) sq += h * h, mx = std::max(mx, std::fabs(h));
            o << "  board " << kv.first << " point heights: rms " << std::setprecision(3)
              << std::sqrt(sq / b.heights.size()) * 1e3 << " mm, max " << mx * 1e3 << " mm (" << b.heights.size()
              << " points)\n";
        }
        if (b.mounted) {
            const Vec3 t = b.mount.ToIso().translation();
            o << "  board " << kv.first << " on " << cfg_.arms[b.arm].name << (b.mount_fixed ? " (mount fixed)" : "")
              << ": link -> board " << std::setprecision(1) << t.x() * 1e3 << ", " << t.y() * 1e3 << ", " << t.z() * 1e3 << " mm\n";
        }
    }
    for (const auto& kv : diag_.lens) {
        const LensReport& L = kv.second;
        if (!L.gated) continue;
        o << "  lens " << kv.first << ": measured to " << std::setprecision(1) << L.theta_max_deg << " deg, coverage "
          << std::setprecision(2) << L.coverage << " (" << L.cells_hit << "/" << L.cells_total << " cells)"
          << ", model one-to-one to " << std::setprecision(1) << L.fold_deg << " deg";
        if (L.bootstrap_runs > 1) {
            o << ", bootstrap (" << L.bootstrap_runs << "): fx sd " << std::setprecision(3) << L.fx_sd << " px, c sd "
              << L.cx_sd << "/" << L.cy_sd << " px, lens curve sd max " << L.max_curve_sd << " px";
        }
        o << "\n    by angle (deg: n, radial rms / mean, tangential rms px):";
        for (const ThetaBin& tb : L.bins) {
            o << "\n      " << std::setprecision(0) << tb.lo_deg << "-" << tb.hi_deg << ": " << tb.n << ", "
              << std::setprecision(3) << tb.rms_radial << " / " << std::showpos << tb.mean_radial << std::noshowpos
              << ", " << tb.rms_tangential;
        }
        o << "\n    gates: " << (L.pass ? "PASS" : "FAIL");
        for (const std::string& f : L.failures) o << "\n      - " << f;
        o << "\n";
    }
    if (diag_.observability_done) {
        o << "  observability: " << diag_.n_free_params << " free params, condition " << std::scientific
          << std::setprecision(2) << diag_.condition << std::fixed << "\n";
    } else {
        o << "  observability: " << diag_.n_free_params
          << " free params, eigen analysis skipped (no arms; solver.observability: true to run it)\n";
    }
    for (const WeakDirection& w : diag_.weak) {
        o << "    WEAK (sigma ratio " << std::scientific << std::setprecision(1) << w.sigma_ratio << std::fixed << "):";
        for (const auto& c : w.components) o << " " << c.first << "=" << std::setprecision(2) << c.second;
        o << "\n";
    }
    for (const std::string& z : diag_.zero_columns) o << "    NO DATA: " << z << "\n";
    o << "  outlier points removed " << diag_.outlier_points << ", views dropped " << diag_.dropped_views << "\n";
    return o.str();
}

void RobotCalibrator::WriteYaml(const std::string& path) const {
    YAML::Emitter out;
    out.SetDoublePrecision(12);
    out << YAML::BeginMap;
    out << YAML::Key << "schema" << YAML::Value << "workcell_calib_v1";
    out << YAML::Key << "solver" << YAML::Value << "calico-robocal";
    out << YAML::Key << "status" << YAML::Value << (diag_.solve_failed ? "FAILED: " + diag_.fail_reason : "ok");
    out << YAML::Key << "world_frame" << YAML::Value << cfg_.world_frame;
    out << YAML::Key << "note" << YAML::Value
        << "Joint solve: q_model = q_encoder + delta_q. Poses are xyz/rpy (URDF fixed-axis) plus matrix. "
           "RMS is per point: sqrt(mean |e|^2) in px.";

    out << YAML::Key << "arms" << YAML::Value << YAML::BeginMap;
    for (size_t ai = 0; ai < arms_.size(); ++ai) {
        const ArmState& a = arms_[ai];
        const ArmConfig& ac = cfg_.arms[ai];
        out << YAML::Key << ac.name << YAML::Value << YAML::BeginMap;
        std::vector<double> deg;
        for (double d : a.dq) deg.push_back(d / kDeg);
        out << YAML::Key << "delta_q_rad" << YAML::Value << YAML::Flow << a.dq;
        out << YAML::Key << "delta_q_deg" << YAML::Value << YAML::Flow << deg;
        out << YAML::Key << "delta_q_free" << YAML::Value << YAML::Flow << ac.dq_free;
        if (cfg_.drive.model) {
            const int nj = a.chain.dof();
            const ArmState& da = arms_[cfg_.drive.shared ? 0 : ai];
            std::vector<double> c(da.drv.begin(), da.drv.begin() + nj), h(da.drv.begin() + nj, da.drv.end());
            std::vector<double> unloaded(nj);
            for (int j = 0; j < nj; ++j) unloaded[j] = a.dq[j] - c[j] * a.tau_ref[j] - h[j] * a.s_ref[j];
            std::vector<int> held(da.drv_const.begin(), da.drv_const.end());
            out << YAML::Key << "drive" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "note" << YAML::Value
                << "theta = q + delta_q + compliance (tau - tau_ref) + half_backlash (s - s_ref); tau: URDF gravity "
                   "torque (N m), s: gear flank (+-1). delta_q is the zero at the reference load.";
            out << YAML::Key << "approach" << YAML::Value << cfg_.drive.approach;
            out << YAML::Key << "compliance_rad_per_nm" << YAML::Value << YAML::Flow << c;
            out << YAML::Key << "half_backlash_rad" << YAML::Value << YAML::Flow << h;
            out << YAML::Key << "held_entries" << YAML::Value << YAML::Flow << held;
            out << YAML::Key << "tau_ref_nm" << YAML::Value << YAML::Flow << a.tau_ref;
            out << YAML::Key << "s_ref" << YAML::Value << YAML::Flow << a.s_ref;
            out << YAML::Key << "delta_q_unloaded_rad" << YAML::Value << YAML::Flow << unloaded;
            out << YAML::EndMap;
        }
        out << YAML::Key << "base_fixed" << YAML::Value << ac.base_fixed;
        out << YAML::Key << "base_to_base" << YAML::Value;
        EmitPose(out, a.world_base.ToIso());
        if (a.camera >= 0) {
            const Iso3 T_tip_opt = a.tip_cam.ToIso();
            out << YAML::Key << "T_tip_cam_optical" << YAML::Value;
            EmitPose(out, T_tip_opt);
            if (ac.has_optical_in_body) {
                Iso3 T_body_opt = Iso3::Identity();
                T_body_opt.linear() = RpyToRotation(ac.optical_in_body_rpy);
                out << YAML::Key << "camera_link6" << YAML::Value;
                Iso3 T_tip_body = T_tip_opt * T_body_opt.inverse();
                const Vec3 xyz = T_tip_body.translation();
                const Vec3 rpy = RotationToRpy(T_tip_body.linear());
                out << YAML::BeginMap << YAML::Key << "parent" << YAML::Value << ac.tip_link;
                out << YAML::Key << "xyz" << YAML::Value << YAML::Flow << std::vector<double>{xyz.x(), xyz.y(), xyz.z()};
                out << YAML::Key << "rpy" << YAML::Value << YAML::Flow << std::vector<double>{rpy.x(), rpy.y(), rpy.z()};
                out << YAML::EndMap;
            }
            if (a.has_ee) {
                out << YAML::Key << "T_EC" << YAML::Value;
                EmitPose(out, a.T_tip_ee.inverse() * T_tip_opt);
            }
            const std::string& cam = cfg_.cameras[a.camera].name;
            out << YAML::Key << "quality" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "train" << YAML::Value;
            EmitStats(out, diag_.train.at(cam));
            out << YAML::Key << "holdout" << YAML::Value;
            EmitStats(out, diag_.holdout.at(cam));
            out << YAML::EndMap;
        }
        // Errors-in-variables (joint_noise_deg): the fitted per-frame joint corrections, for
        // diagnostics (plot them against angle, load, approach direction).
        bool any_eps = false;
        for (const auto& kv : joint_eps_) any_eps = any_eps || kv.first.first == static_cast<int>(ai);
        if (any_eps) {
            out << YAML::Key << "joint_corrections_deg" << YAML::Value << YAML::BeginSeq;
            for (const auto& kv : joint_eps_) {
                if (kv.first.first != static_cast<int>(ai)) continue;
                std::string image;
                std::vector<double> q;
                for (const View& v : views_) {
                    if (v.arm == static_cast<int>(ai) && v.frame_idx == kv.first.second) {
                        image = v.image;
                        q = v.q;
                        break;
                    }
                }
                std::vector<double> e, qd;
                for (double x : kv.second) e.push_back(x / kDeg);
                for (double x : q) qd.push_back(x / kDeg);
                out << YAML::Flow << YAML::BeginMap << YAML::Key << "frame" << YAML::Value << kv.first.second << YAML::Key
                    << "image" << YAML::Value << image << YAML::Key << "q_deg" << YAML::Value << YAML::Flow << qd
                    << YAML::Key << "eps_deg" << YAML::Value << YAML::Flow << e << YAML::EndMap;
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndMap;
    }
    out << YAML::EndMap;

    out << YAML::Key << "boards" << YAML::Value << YAML::BeginMap;
    for (const auto& kv : boards_) {
        const BoardState& b = kv.second;
        out << YAML::Key << kv.first << YAML::Value << YAML::BeginMap;
        if (b.mounted) {
            out << YAML::Key << "mount_arm" << YAML::Value << cfg_.arms[b.arm].name;
            out << YAML::Key << "mount_fixed" << YAML::Value << b.mount_fixed;
            out << YAML::Key << "T_link_board" << YAML::Value;
            EmitPose(out, b.mount.ToIso());
        } else {
            out << YAML::Key << "T_world_board" << YAML::Value;
            EmitPose(out, b.pose.ToIso());
        }
        if (b.fit_shape) {
            out << YAML::Key << "shape" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "bow_x_mm" << YAML::Value << b.shape[0] * 1e3;
            out << YAML::Key << "bow_y_mm" << YAML::Value << b.shape[1] * 1e3;
            out << YAML::Key << "twist_mm" << YAML::Value << b.shape[2] * 1e3;
            out << YAML::Key << "scale_x_pct" << YAML::Value << b.shape[3] * 1e2;
            out << YAML::Key << "scale_y_pct" << YAML::Value << b.shape[4] * 1e2;
            out << YAML::EndMap;
        }
        if (b.fit_warp) {
            std::vector<double> mm;
            for (double w : b.warp) mm.push_back(w * 1e3);
            out << YAML::Key << "warp_mm" << YAML::Value << YAML::Flow << mm;  // a2 a3 a4 a5 b1 b3 b4 b5
        }
        if (b.fit_points && !b.heights.empty()) {
            std::vector<double> mm;
            for (double h : b.heights) mm.push_back(h * 1e3);
            out << YAML::Key << "point_heights_mm" << YAML::Value << YAML::Flow << mm;
        }
        out << YAML::EndMap;
    }
    out << YAML::EndMap;
    const std::map<std::string, double> level_boards = cfg_.LevelBoards();
    std::string ref = !level_boards.empty() ? level_boards.begin()->first : "";
    for (const auto& kv : boards_) {
        if (ref.empty() && !kv.second.mounted) ref = kv.first;
    }
    if (!ref.empty()) {
        out << YAML::Key << "T_base_board" << YAML::Value;
        EmitPose(out, boards_.at(ref).pose.ToIso());
    }

    out << YAML::Key << "sensors" << YAML::Value << YAML::BeginMap;
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        if (!cams_[ci].is_static) continue;
        const CameraConfig& cc = cfg_.cameras[ci];
        out << YAML::Key << cc.name << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "T_base_cam" << YAML::Value;
        EmitPose(out, cams_[ci].world_cam.ToIso());
        if (cc.has_optical_in_body) {
            Iso3 T_body_opt = Iso3::Identity();
            T_body_opt.linear() = RpyToRotation(cc.optical_in_body_rpy);
            out << YAML::Key << "T_base_cam_link" << YAML::Value;
            EmitPose(out, cams_[ci].world_cam.ToIso() * T_body_opt.inverse());
        }
        out << YAML::Key << "quality" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "train" << YAML::Value;
        EmitStats(out, diag_.train.at(cc.name));
        out << YAML::Key << "holdout" << YAML::Value;
        EmitStats(out, diag_.holdout.at(cc.name));
        out << YAML::EndMap << YAML::EndMap;
    }
    out << YAML::EndMap;

    out << YAML::Key << "intrinsics" << YAML::Value << YAML::BeginMap;
    for (size_t ci = 0; ci < cams_.size(); ++ci) {
        out << YAML::Key << cfg_.cameras[ci].name << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "model" << YAML::Value << LensModelName(cams_[ci].intr.model);
        out << YAML::Key << "optimized" << YAML::Value << (final_free_intrinsics_ && cfg_.cameras[ci].optimize_intrinsics);
        out << YAML::Key << "fx_fy_cx_cy_d" << YAML::Value << YAML::Flow << cams_[ci].intr.params;
        out << YAML::EndMap;
    }
    out << YAML::EndMap;

    out << YAML::Key << "lens_reports" << YAML::Value << YAML::BeginMap;
    for (const auto& kv : diag_.lens) {
        const LensReport& L = kv.second;
        out << YAML::Key << kv.first << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "theta_max_deg" << YAML::Value << L.theta_max_deg;
        out << YAML::Key << "coverage" << YAML::Value << L.coverage;
        out << YAML::Key << "bins" << YAML::Value << YAML::BeginSeq;
        for (const ThetaBin& tb : L.bins) {
            out << YAML::Flow << YAML::BeginMap << YAML::Key << "deg" << YAML::Value << YAML::Flow
                << std::vector<double>{tb.lo_deg, tb.hi_deg} << YAML::Key << "n" << YAML::Value << tb.n << YAML::Key
                << "radial_rms" << YAML::Value << tb.rms_radial << YAML::Key << "radial_mean" << YAML::Value
                << tb.mean_radial << YAML::Key << "tangential_rms" << YAML::Value << tb.rms_tangential << YAML::EndMap;
        }
        out << YAML::EndSeq;
        if (L.bootstrap_runs > 1) {
            out << YAML::Key << "bootstrap" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "runs" << YAML::Value << L.bootstrap_runs;
            out << YAML::Key << "fx_sd" << YAML::Value << L.fx_sd << YAML::Key << "fy_sd" << YAML::Value << L.fy_sd;
            out << YAML::Key << "cx_sd" << YAML::Value << L.cx_sd << YAML::Key << "cy_sd" << YAML::Value << L.cy_sd;
            out << YAML::Key << "curve_sd_px" << YAML::Value << YAML::BeginSeq;
            for (const auto& c : L.curve_sd) out << YAML::Flow << std::vector<double>{c.first, c.second};
            out << YAML::EndSeq << YAML::EndMap;
        }
        if (L.gated) {
            out << YAML::Key << "gates" << YAML::Value << (L.pass ? "PASS" : "FAIL");
            out << YAML::Key << "model_fold_deg" << YAML::Value << L.fold_deg;
            out << YAML::Key << "failures" << YAML::Value << YAML::Flow << L.failures;
        }
        out << YAML::EndMap;
    }
    out << YAML::EndMap;

    out << YAML::Key << "diagnostics" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "n_free_params" << YAML::Value << diag_.n_free_params;
    out << YAML::Key << "n_residuals" << YAML::Value << diag_.n_residuals;
    out << YAML::Key << "condition" << YAML::Value << diag_.condition;
    out << YAML::Key << "smallest_sigma_ratio" << YAML::Value << YAML::Flow << diag_.smallest_sigma_ratio;
    out << YAML::Key << "weak_directions" << YAML::Value << YAML::BeginSeq;
    for (const WeakDirection& w : diag_.weak) {
        out << YAML::BeginMap << YAML::Key << "sigma_ratio" << YAML::Value << w.sigma_ratio;
        out << YAML::Key << "components" << YAML::Value << YAML::BeginMap;
        for (const auto& c : w.components) out << YAML::Key << c.first << YAML::Value << c.second;
        out << YAML::EndMap << YAML::EndMap;
    }
    out << YAML::EndSeq;
    out << YAML::Key << "no_data_params" << YAML::Value << YAML::Flow << diag_.zero_columns;
    out << YAML::Key << "covariance_ok" << YAML::Value << diag_.covariance_ok;
    out << YAML::Key << "sigma_1" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "units" << YAML::Value << "dq, rotations: deg; translations: mm; intrinsics: px";
    for (const auto& kv : diag_.sigma) out << YAML::Key << kv.first << YAML::Value << kv.second;
    out << YAML::EndMap;
    out << YAML::Key << "outlier_points" << YAML::Value << diag_.outlier_points;
    out << YAML::Key << "dropped_views" << YAML::Value << diag_.dropped_views;
    out << YAML::EndMap;
    out << YAML::EndMap;

    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << out.c_str() << "\n";
}

}  // namespace robocal
