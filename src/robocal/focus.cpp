#include "focus.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace robocal {
namespace {

float Bilinear(const cv::Mat& img, double x, double y) {
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= img.cols || y0 + 1 >= img.rows) return std::numeric_limits<float>::quiet_NaN();
    const float a = static_cast<float>(x - x0), b = static_cast<float>(y - y0);
    return (1 - a) * (1 - b) * img.at<float>(y0, x0) + a * (1 - b) * img.at<float>(y0, x0 + 1) +
           (1 - a) * b * img.at<float>(y0 + 1, x0) + a * b * img.at<float>(y0 + 1, x0 + 1);
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// Board pose from the detected corners (shared virtual-pinhole IPPE, best candidate).
bool RoughBoardPose(const BoardSpec& board, const BoardDetection& det, const CameraIntrinsics& cam, Iso3& T_cb) {
    std::vector<Vec3> bearings, objs;
    for (size_t k = 0; k < det.ids.size(); ++k) {
        const auto obj = board.TagCorners(det.ids[k]);
        for (int c = 0; c < 4; ++c) {
            Vec3 b;
            if (!UnprojectPixel(cam.model, cam.params.data(), det.corners[k][c], b)) continue;
            bearings.push_back(b);
            objs.push_back(obj[c]);
        }
    }
    const std::vector<Iso3> cand = BoardPoseCandidates(bearings, objs);
    if (cand.empty()) return false;
    T_cb = cand.front();
    return true;
}

}  // namespace

double FitEdgeSigma(const cv::Mat& lin, const Vec2& p, const Vec2& n, double w) {
    const double step = 0.25;
    std::vector<double> xs, ys;
    for (double x = -w; x <= w + 1e-9; x += step) {
        const float v = Bilinear(lin, p.x() + x * n.x(), p.y() + x * n.y());
        if (!std::isfinite(v)) return -1.0;
        xs.push_back(x);
        ys.push_back(v);
    }
    const size_t N = xs.size(), q = std::max<size_t>(3, N / 8);
    double a = 0, hi = 0;
    for (size_t i = 0; i < q; ++i) {
        a += ys[i] / q;
        hi += ys[N - 1 - i] / q;
    }
    double b = hi - a;
    const double full = std::max(std::fabs(a), std::fabs(hi));
    if (std::fabs(b) < 0.06 || std::fabs(b) < 0.25 * full) return -1.0;  // no clean black/white edge
    double x0 = 0;
    for (size_t i = 1; i < N; ++i) {
        if ((ys[i - 1] - a - 0.5 * b) * (ys[i] - a - 0.5 * b) <= 0) {
            x0 = xs[i];
            break;
        }
    }
    double s = 1.0;
    // Levenberg-Marquardt on I(x) = a + b * Phi((x - x0) / s).
    double lambda = 1e-3;
    auto cost = [&](double A, double B, double X0, double S) {
        double c = 0;
        for (size_t i = 0; i < N; ++i) {
            const double z = (xs[i] - X0) / S;
            const double r = A + B * 0.5 * (1 + std::erf(z / std::sqrt(2.0))) - ys[i];
            c += r * r;
        }
        return c;
    };
    double c0 = cost(a, b, x0, s);
    for (int it = 0; it < 40; ++it) {
        Eigen::Matrix4d JtJ = Eigen::Matrix4d::Zero();
        Eigen::Vector4d Jtr = Eigen::Vector4d::Zero();
        for (size_t i = 0; i < N; ++i) {
            const double z = (xs[i] - x0) / s;
            const double Phi = 0.5 * (1 + std::erf(z / std::sqrt(2.0)));
            const double phi = std::exp(-0.5 * z * z) / std::sqrt(2 * M_PI);
            const double r = a + b * Phi - ys[i];
            const Eigen::Vector4d J(1.0, Phi, -b * phi / s, -b * phi * z / s);
            JtJ += J * J.transpose();
            Jtr += J * r;
        }
        Eigen::Matrix4d Aug = JtJ;
        for (int k = 0; k < 4; ++k) Aug(k, k) *= 1 + lambda;
        const Eigen::Vector4d d = Aug.ldlt().solve(-Jtr);
        const double na = a + d[0], nb = b + d[1], nx = x0 + d[2], ns = std::max(0.05, s + d[3]);
        const double c1 = cost(na, nb, nx, ns);
        if (c1 < c0) {
            a = na, b = nb, x0 = nx, s = ns;
            lambda = std::max(1e-7, lambda * 0.3);
            if (c0 - c1 < 1e-12 * (1 + c0)) break;
            c0 = c1;
        } else {
            lambda *= 10;
            if (lambda > 1e6) break;
        }
    }
    if (!(s > 0.12 && s < 0.45 * w && std::fabs(x0) < 0.5 * w)) return -1.0;
    return s;
}

FocusMeasure MeasureFocus(const cv::Mat& gray, const BoardSpec& board, const BoardDetection& det,
                          const CameraIntrinsics& rough, const FocusOptions& opts) {
    FocusMeasure m;
    if (det.ids.empty()) return m;
    // Linear intensities: the erf model holds before the camera's gamma.
    const cv::Mat lin = LinearizeGray(gray, opts.gamma);

    Iso3 T_cb = Iso3::Identity();
    const bool posed = RoughBoardPose(board, det, rough, T_cb);
    Vec3 centroid = Vec3::Zero();
    int n_pts = 0;
    std::vector<double> sig;
    for (size_t k = 0; k < det.ids.size(); ++k) {
        const std::array<Vec2, 4>& c = det.corners[k];
        const Vec2 mid = 0.25 * (c[0] + c[1] + c[2] + c[3]);
        Vec3 b;
        if (!UnprojectPixel(rough.model, rough.params.data(), mid, b)) continue;
        if (std::atan2(std::hypot(b.x(), b.y()), b.z()) > opts.max_off_axis_deg * M_PI / 180.0) continue;
        ++m.n_tags;
        if (posed) {
            for (const Vec3& p : board.TagCorners(det.ids[k])) {
                centroid += T_cb * p;
                ++n_pts;
            }
        }
        for (int e = 0; e < 4; ++e) {
            const Vec2 p0 = c[e], p1 = c[(e + 1) % 4];
            const double L = (p1 - p0).norm();
            if (L < 12.0) continue;
            const Vec2 dir = (p1 - p0) / L;
            const Vec2 nrm(-dir.y(), dir.x());
            const double w = std::clamp(0.16 * L, 2.5, 8.0);
            for (double u : {0.3, 0.42, 0.58, 0.7}) {
                const double s = FitEdgeSigma(lin, p0 + u * (p1 - p0), nrm, w);
                if (s > 0) sig.push_back(s);
            }
        }
    }
    m.n_edges = static_cast<int>(sig.size());
    if (sig.size() < 8) return m;
    m.blur_sigma_px = Median(sig);
    std::vector<double> dev;
    for (double s : sig) dev.push_back(std::fabs(s - m.blur_sigma_px));
    m.blur_spread_px = Median(dev);
    if (n_pts == 0) return m;  // blur without a board pose: no distance to classify the frame by
    centroid /= n_pts;
    m.distance_m = centroid.norm();
    m.off_axis_deg = std::atan2(std::hypot(centroid.x(), centroid.y()), centroid.z()) * 180.0 / M_PI;
    m.ok = true;
    return m;
}

}  // namespace robocal
