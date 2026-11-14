// Serial kinematic chain read from a URDF, with forward kinematics templated on the
// scalar type so Ceres can differentiate through it.
#ifndef ROBOCAL_URDF_CHAIN_HPP_
#define ROBOCAL_URDF_CHAIN_HPP_

#include "se3.hpp"

#include <ceres/rotation.h>

#include <string>
#include <vector>

namespace robocal {

struct ChainJoint {
    enum class Type { Fixed, Revolute, Prismatic };
    std::string name;
    Type type = Type::Fixed;
    Mat3 R_origin = Mat3::Identity();
    Vec3 t_origin = Vec3::Zero();
    Vec3 axis = Vec3::UnitX();
    Vec3 perp1 = Vec3::UnitY(), perp2 = Vec3::UnitZ();  // unit vectors perpendicular to the axis (geometry terms)
    int active = -1;  // index into the chain's joint vector; -1 = held at 0 / fixed
    // Inertial of the child link plus every off-chain link hanging from it (gripper, fingers,
    // camera ...; their joints at 0): mass (kg) and centre of mass in the child frame.
    double mass = 0.0;
    Vec3 com = Vec3::Zero();
};

class UrdfChain {
public:
    // Chain base_link -> tip_link. ``joint_names`` lists the active joints in the order the
    // joint vector uses; any movable joint on the chain that is not listed is held at 0.
    static UrdfChain FromFile(const std::string& urdf_path, const std::string& base_link,
                              const std::string& tip_link,
                              const std::vector<std::string>& joint_names);
    static UrdfChain FromString(const std::string& urdf_xml, const std::string& base_link,
                                const std::string& tip_link,
                                const std::vector<std::string>& joint_names);

    int dof() const { return dof_; }
    const std::vector<ChainJoint>& joints() const { return joints_; }
    const std::string& base_link() const { return base_link_; }
    const std::string& tip_link() const { return tip_link_; }

    // theta: dof() joint angles (encoder value + zero offset). Pose of tip in base.
    // geo (optional, 4 per active joint, minimal geometric calibration after Lu 2025 / Tobuschat 2026):
    // [tilt about perp1, tilt about perp2 (rad), shift along perp1, shift along perp2 (m)], applied in the
    // joint frame before the joint motion. The shift along / tilt about the axis itself are omitted
    // (they duplicate the next joint's terms or the zero offset).
    template <typename T>
    void Forward(const T* theta, Eigen::Matrix<T, 3, 3>& R, Eigen::Matrix<T, 3, 1>& t, const T* geo = nullptr) const {
        R.setIdentity();
        t.setZero();
        for (const ChainJoint& j : joints_) {
            t = t + R * j.t_origin.cast<T>();
            R = R * j.R_origin.cast<T>();
            if (j.active < 0) continue;
            if (geo != nullptr && j.type != ChainJoint::Type::Prismatic) {
                const T* g = geo + 4 * j.active;
                t = t + R * (j.perp1.cast<T>() * g[2] + j.perp2.cast<T>() * g[3]);
                const Eigen::Matrix<T, 3, 1> w = j.perp1.cast<T>() * g[0] + j.perp2.cast<T>() * g[1];
                Eigen::Matrix<T, 3, 3> dR;
                ceres::AngleAxisToRotationMatrix(w.data(), dR.data());  // column major, safe at zero
                R = R * dR;
            }
            if (j.type == ChainJoint::Type::Prismatic) {
                t = t + R * (j.axis.cast<T>() * theta[j.active]);
            } else {
                R = R * Eigen::AngleAxis<T>(theta[j.active], j.axis.cast<T>()).toRotationMatrix();
            }
        }
    }

    Iso3 ForwardIso(const std::vector<double>& theta) const;
    // Static gravity torque about each active joint (N m, dof() entries; prismatic: force, N),
    // from the URDF inertials: tau_j = axis_j . sum_{links after j} (com - p_j) x (m g).
    // R_world_base: base orientation in the world; g_world: gravity (m/s^2) in the world.
    std::vector<double> GravityTorques(const std::vector<double>& theta, const Mat3& R_world_base,
                                       const Vec3& g_world = Vec3(0, 0, -9.81)) const;
    double TotalMass() const;
    // Pose of every joint's child frame along the chain (base -> ... -> tip), for drawing.
    std::vector<Iso3> FrameChain(const std::vector<double>& theta) const;

private:
    std::vector<ChainJoint> joints_;
    std::string base_link_;
    std::string tip_link_;
    int dof_ = 0;
};

}  // namespace robocal

#endif  // ROBOCAL_URDF_CHAIN_HPP_
