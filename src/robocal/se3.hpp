// SE(3) helpers shared by the robot calibration module (robocal).
#ifndef ROBOCAL_SE3_HPP_
#define ROBOCAL_SE3_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <vector>

namespace robocal {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Iso3 = Eigen::Isometry3d;

// URDF / ROS fixed-axis roll-pitch-yaw: R = Rz(yaw) * Ry(pitch) * Rx(roll).
inline Mat3 RpyToRotation(const Vec3& rpy) {
    return (Eigen::AngleAxisd(rpy.z(), Vec3::UnitZ()) *
            Eigen::AngleAxisd(rpy.y(), Vec3::UnitY()) *
            Eigen::AngleAxisd(rpy.x(), Vec3::UnitX()))
        .toRotationMatrix();
}

inline Vec3 RotationToRpy(const Mat3& R) {
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return Vec3(roll, pitch, yaw);
}

inline Iso3 MakeIso(const Vec3& xyz, const Vec3& rpy) {
    Iso3 T = Iso3::Identity();
    T.linear() = RpyToRotation(rpy);
    T.translation() = xyz;
    return T;
}

// Pose parameter storage for Ceres: Eigen quaternion order (x, y, z, w) + translation.
struct PoseParam {
    double q[4] = {0.0, 0.0, 0.0, 1.0};
    double t[3] = {0.0, 0.0, 0.0};

    static PoseParam FromIso(const Iso3& T) {
        PoseParam p;
        Eigen::Quaterniond Q(T.linear());
        Q.normalize();
        p.q[0] = Q.x();
        p.q[1] = Q.y();
        p.q[2] = Q.z();
        p.q[3] = Q.w();
        for (int i = 0; i < 3; ++i) p.t[i] = T.translation()[i];
        return p;
    }

    Iso3 ToIso() const {
        Eigen::Quaterniond Q(q[3], q[0], q[1], q[2]);
        Q.normalize();
        Iso3 T = Iso3::Identity();
        T.linear() = Q.toRotationMatrix();
        T.translation() = Vec3(t[0], t[1], t[2]);
        return T;
    }
};

// Chordal mean of rotations (sign-aligned quaternion average) + mean translation.
inline Iso3 AveragePoses(const std::vector<Iso3>& poses) {
    Iso3 out = Iso3::Identity();
    if (poses.empty()) return out;
    Eigen::Quaterniond ref(poses.front().linear());
    Eigen::Vector4d acc = Eigen::Vector4d::Zero();
    Vec3 t = Vec3::Zero();
    for (const Iso3& T : poses) {
        Eigen::Quaterniond Q(T.linear());
        if (Q.dot(ref) < 0.0) Q.coeffs() *= -1.0;
        acc += Q.coeffs();
        t += T.translation();
    }
    Eigen::Quaterniond Qm;
    Qm.coeffs() = acc.normalized();
    out.linear() = Qm.toRotationMatrix();
    out.translation() = t / static_cast<double>(poses.size());
    return out;
}

inline double RotationAngleDeg(const Mat3& A, const Mat3& B) {
    const double c = std::max(-1.0, std::min(1.0, ((A.transpose() * B).trace() - 1.0) * 0.5));
    return std::acos(c) * 180.0 / M_PI;
}

}  // namespace robocal

#endif  // ROBOCAL_SE3_HPP_
