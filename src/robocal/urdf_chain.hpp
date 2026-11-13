// Serial kinematic chain read from a URDF, with forward kinematics templated on the
// scalar type so Ceres can differentiate through it.
#ifndef ROBOCAL_URDF_CHAIN_HPP_
#define ROBOCAL_URDF_CHAIN_HPP_

#include "se3.hpp"

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
    template <typename T>
    void Forward(const T* theta, Eigen::Matrix<T, 3, 3>& R, Eigen::Matrix<T, 3, 1>& t) const {
        R.setIdentity();
        t.setZero();
        for (const ChainJoint& j : joints_) {
            t = t + R * j.t_origin.cast<T>();
            R = R * j.R_origin.cast<T>();
            if (j.active < 0) continue;
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
