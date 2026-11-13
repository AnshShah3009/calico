// Camera projection models, templated on the scalar so Ceres can differentiate them.
//
// Parameter layout: fx, fy, cx, cy, then distortion in ROS camera_info order.
//   equidistant          Kannala-Brandt k1..k4. The off-axis angle is atan2(rho, z), so rays
//                        past 90 deg (and behind the camera) project where the lens puts them
//                        instead of wrapping to the mirrored side as cv2.fisheye does.
//   kb_rt                Kannala-Brandt k1..k4 + decentering p1, p2 on the distorted
//                        normalised coordinates (lens elements off-centre / tilted).
//   double_sphere        xi, alpha (Usenko et al. 2018): two unit spheres; closed-form
//                        inverse; valid past 90 deg.
//   plumb_bob            k1, k2, p1, p2, k3                   (pinhole, z > 0 only)
//   rational_polynomial  k1, k2, p1, p2, k3, k4, k5, k6       (pinhole, z > 0 only)
#ifndef ROBOCAL_CAMERA_MODELS_HPP_
#define ROBOCAL_CAMERA_MODELS_HPP_

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace robocal {

// KannalaBrandtPupil ("kb4_pupil"): KB4 (k1..k4) plus an entrance pupil that moves along the optical
// axis with the field angle, z0(theta) = e1 (theta / 90 deg)^2 (metres, towards the scene): a
// non-central camera. Very wide lenses have it (real 200 deg M12: e1 ~ 1.5 mm); a central model then
// fits close and far boards with different lenses. d = [k1, k2, k3, k4, e1].
enum class LensModel { Equidistant, PlumbBob, Rational, KannalaBrandtRT, DoubleSphere, KannalaBrandtPupil };

inline int DistortionCount(LensModel m) {
    switch (m) {
        case LensModel::Equidistant: return 4;
        case LensModel::KannalaBrandtRT: return 6;
        case LensModel::DoubleSphere: return 2;
        case LensModel::PlumbBob: return 5;
        case LensModel::Rational: return 8;
        case LensModel::KannalaBrandtPupil: return 5;
    }
    return 0;
}

inline std::string LensModelName(LensModel m) {
    switch (m) {
        case LensModel::Equidistant: return "equidistant";
        case LensModel::PlumbBob: return "plumb_bob";
        case LensModel::Rational: return "rational_polynomial";
        case LensModel::KannalaBrandtRT: return "kb_rt";
        case LensModel::DoubleSphere: return "double_sphere";
        case LensModel::KannalaBrandtPupil: return "kb4_pupil";
    }
    return "?";
}

// Wide-angle models (valid past 90 deg off axis).
inline bool IsWideAngle(LensModel m) {
    return m == LensModel::Equidistant || m == LensModel::KannalaBrandtRT || m == LensModel::DoubleSphere ||
           m == LensModel::KannalaBrandtPupil;
}

// Kannala-Brandt family (k1..k4 in d[0..3], theta = atan2(rho, z)).
inline bool IsKannalaBrandt(LensModel m) {
    return m == LensModel::Equidistant || m == LensModel::KannalaBrandtRT || m == LensModel::KannalaBrandtPupil;
}

// Distortion values a solve starts from (double_sphere: alpha 0.5 = stereographic-like, which
// already covers > 180 deg; the others: zero).
inline std::vector<double> DefaultDistortion(LensModel m) {
    if (m == LensModel::DoubleSphere) return {0.0, 0.5};
    return std::vector<double>(static_cast<size_t>(DistortionCount(m)), 0.0);
}

inline LensModel LensModelFromName(const std::string& name, size_t n_coeffs) {
    if (name == "equidistant" || name == "fisheye" || name == "kannala_brandt") {
        return LensModel::Equidistant;
    }
    if (name == "rational_polynomial") return LensModel::Rational;
    if (name == "kb_rt" || name == "kannala_brandt_rt") return LensModel::KannalaBrandtRT;
    if (name == "double_sphere" || name == "ds") return LensModel::DoubleSphere;
    if (name == "kb4_pupil" || name == "equidistant_pupil") return LensModel::KannalaBrandtPupil;
    if (name == "plumb_bob" || name.empty()) {
        return n_coeffs > 5 ? LensModel::Rational : LensModel::PlumbBob;
    }
    throw std::runtime_error("unsupported distortion_model '" + name + "'");
}

struct CameraIntrinsics {
    std::string name;
    LensModel model = LensModel::Equidistant;
    int width = 0;
    int height = 0;
    std::vector<double> params;  // fx, fy, cx, cy, distortion...

    int NumParams() const { return 4 + DistortionCount(model); }
};

// Projects a camera-frame point. Returns false where the model is undefined
// (pinhole models behind the image plane).
template <typename T>
bool ProjectPoint(LensModel model, const T* p, const Eigen::Matrix<T, 3, 1>& X, T* uv) {
    using std::atan2;
    using std::sqrt;
    const T& fx = p[0];
    const T& fy = p[1];
    const T& cx = p[2];
    const T& cy = p[3];
    const T* d = p + 4;
    const T x = X[0];
    const T y = X[1];
    const T z = X[2];
    if (model == LensModel::Equidistant || model == LensModel::KannalaBrandtRT || model == LensModel::KannalaBrandtPupil) {
        const T r2 = x * x + y * y;
        T scale;
        if (r2 < T(1e-20)) {
            if (z <= T(0)) return false;
            scale = T(1) / z;  // on the optical axis: theta_d / rho -> 1 / z
        } else {
            const T rho = sqrt(r2);
            T th = atan2(rho, z);
            if (model == LensModel::KannalaBrandtPupil) {
                // Ray from the pupil at (0, 0, z0(theta)): theta = atan2(rho, z - z0(theta)), fixed point
                // (contraction ~ z0' / distance, < 0.05 at 0.05 m); unrolled so autodiff carries the
                // implicit derivative.
                const T kq = T(4.0 / (M_PI * M_PI));
                for (int it = 0; it < 6; ++it) th = atan2(rho, z - d[4] * kq * th * th);
            }
            const T th2 = th * th;
            const T thd = th * (T(1) + th2 * (d[0] + th2 * (d[1] + th2 * (d[2] + th2 * d[3]))));
            scale = thd / rho;
        }
        T xd = scale * x, yd = scale * y;
        if (model == LensModel::KannalaBrandtRT) {
            const T rd2 = xd * xd + yd * yd;
            const T tx = T(2) * d[4] * xd * yd + d[5] * (rd2 + T(2) * xd * xd);
            const T ty = d[4] * (rd2 + T(2) * yd * yd) + T(2) * d[5] * xd * yd;
            xd += tx;
            yd += ty;
        }
        uv[0] = fx * xd + cx;
        uv[1] = fy * yd + cy;
        return true;
    }
    if (model == LensModel::DoubleSphere) {
        const T xi = d[0], alpha = d[1];
        const T d1 = sqrt(x * x + y * y + z * z);
        const T zz = xi * d1 + z;
        const T d2 = sqrt(x * x + y * y + zz * zz);
        const T den = alpha * d2 + (T(1) - alpha) * zz;
        if (den <= T(1e-12)) return false;
        uv[0] = fx * x / den + cx;
        uv[1] = fy * y / den + cy;
        return true;
    }
    if (z <= T(1e-6)) return false;
    const T xp = x / z;
    const T yp = y / z;
    const T r2 = xp * xp + yp * yp;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T k1 = d[0], k2 = d[1], p1 = d[2], p2 = d[3], k3 = d[4];
    T num = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
    T den = T(1);
    if (model == LensModel::Rational) den = T(1) + d[5] * r2 + d[6] * r4 + d[7] * r6;
    const T radial = num / den;
    const T xpp = xp * radial + T(2) * p1 * xp * yp + p2 * (r2 + T(2) * xp * xp);
    const T ypp = yp * radial + p1 * (r2 + T(2) * yp * yp) + T(2) * p2 * xp * yp;
    uv[0] = fx * xpp + cx;
    uv[1] = fy * ypp + cy;
    return true;
}

// Projects a DIRECTION (a point at infinity) from the camera origin: for central models the same as
// ProjectPoint; for kb4_pupil the pupil offset is ignored (rays at infinity). Use it wherever a unit
// bearing stands for a direction (ray fans, rectified-view remapping, curve comparisons); use
// ProjectPoint for real 3D points.
template <typename T>
bool ProjectDirection(LensModel model, const T* p, const Eigen::Matrix<T, 3, 1>& X, T* uv) {
    if (model == LensModel::KannalaBrandtPupil) return ProjectPoint(LensModel::Equidistant, p, X, uv);
    return ProjectPoint(model, p, X, uv);
}

// Pupil position on the optical axis (m) for a ray at theta (rad): 0 for central models.
inline double PupilOffset(LensModel model, const double* p, double theta) {
    if (model != LensModel::KannalaBrandtPupil) return 0.0;
    return p[8] * 4.0 / (M_PI * M_PI) * theta * theta;
}

// Pixel -> unit bearing in the camera frame (inverse of ProjectPoint). Valid past 90 deg for
// Kannala-Brandt models: the first angle (deg) at which theta_d(theta) stops increasing, i.e.
// where the fitted polynomial folds back (180 if never). Projection is only one-to-one below it.
inline double KannalaBrandtFoldDeg(LensModel model, const double* p) {
    if (!IsKannalaBrandt(model)) return 180.0;
    const double* d = p + 4;
    for (double deg = 0.25; deg < 180.0; deg += 0.25) {
        const double th = deg * M_PI / 180.0, t2 = th * th;
        if (1 + t2 * (3 * d[0] + t2 * (5 * d[1] + t2 * (7 * d[2] + 9 * t2 * d[3]))) <= 0) return deg;
    }
    return 180.0;
}

// the equidistant model. Returns false if the iteration does not converge.
inline bool UnprojectPixel(LensModel model, const double* p, const Eigen::Vector2d& uv,
                           Eigen::Vector3d& bearing) {
    const double* d = p + 4;
    const double my = (uv.y() - p[3]) / p[1];
    const double mx = (uv.x() - p[2]) / p[0];
    if (model == LensModel::DoubleSphere) {
        const double xi = d[0], alpha = d[1];
        const double r2 = mx * mx + my * my;
        const double q = 1.0 - (2.0 * alpha - 1.0) * r2;
        if (q < 0) return false;
        const double mz = (1.0 - alpha * alpha * r2) / (alpha * std::sqrt(q) + 1.0 - alpha);
        const double disc = mz * mz + (1.0 - xi * xi) * r2;
        if (disc < 0) return false;
        const double k = (mz * xi + std::sqrt(disc)) / (mz * mz + r2);
        bearing = Eigen::Vector3d(k * mx, k * my, k * mz - xi).normalized();
        return std::isfinite(bearing.x());
    }
    double ux = mx, uy = my;
    if (model == LensModel::KannalaBrandtRT) {  // remove the decentering first (fixed point)
        double xd = mx, yd = my;
        for (int it = 0; it < 30; ++it) {
            const double rd2 = xd * xd + yd * yd;
            const double tx = 2 * d[4] * xd * yd + d[5] * (rd2 + 2 * xd * xd);
            const double ty = d[4] * (rd2 + 2 * yd * yd) + 2 * d[5] * xd * yd;
            xd = mx - tx;
            yd = my - ty;
        }
        ux = xd;
        uy = yd;
    }
    // kb4_pupil: the bearing of the ray; it starts at (0, 0, PupilOffset(theta)), not at the origin.
    // The direction (and so the off-axis angle) is exact; callers that treat it as a ray through the
    // origin (the PnP initialisation) are off by the pupil shift only until the pose is refined
    // through ProjectPoint, which models the pupil.
    if (IsKannalaBrandt(model)) {
        const double mx = ux, my = uy;
        const double rd = std::hypot(mx, my);
        if (rd < 1e-12) {
            bearing = Eigen::Vector3d(0, 0, 1);
            return true;
        }
        double th = rd;
        for (int it = 0; it < 30; ++it) {
            const double t2 = th * th;
            const double f = th * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3])))) - rd;
            const double df = 1 + t2 * (3 * d[0] + t2 * (5 * d[1] + t2 * (7 * d[2] + 9 * t2 * d[3])));
            if (std::fabs(df) < 1e-12) return false;
            const double step = f / df;
            th -= step;
            if (std::fabs(step) < 1e-14) break;
        }
        if (!(th >= 0.0 && th < M_PI)) return false;
        // The polynomial must still be increasing here: past its turning point a radius has two
        // angles and Newton may return the wrong one.
        {
            const double t2 = th * th;
            if (1 + t2 * (3 * d[0] + t2 * (5 * d[1] + t2 * (7 * d[2] + 9 * t2 * d[3]))) <= 0) return false;
        }
        const double s = std::sin(th) / rd;
        bearing = Eigen::Vector3d(mx * s, my * s, std::cos(th));
        return true;
    }
    // Pinhole: fixed-point undistortion (OpenCV's scheme), then the ray through z = 1.
    double x = mx, y = my;
    for (int it = 0; it < 50; ++it) {
        const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
        double radial = 1 + d[0] * r2 + d[1] * r4 + d[4] * r6;
        if (model == LensModel::Rational) radial /= (1 + d[5] * r2 + d[6] * r4 + d[7] * r6);
        const double dx = 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x);
        const double dy = d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y;
        if (std::fabs(radial) < 1e-12) return false;
        x = (mx - dx) / radial;
        y = (my - dy) / radial;
    }
    bearing = Eigen::Vector3d(x, y, 1.0).normalized();
    return std::isfinite(bearing.x()) && std::isfinite(bearing.y());
}

}  // namespace robocal

#endif  // ROBOCAL_CAMERA_MODELS_HPP_
