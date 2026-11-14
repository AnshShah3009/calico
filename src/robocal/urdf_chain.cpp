#include "urdf_chain.hpp"

#include <tinyxml2.h>

#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace robocal {
namespace {

Vec3 ParseVec3(const char* text, const Vec3& fallback) {
    if (text == nullptr) return fallback;
    std::istringstream in(text);
    Vec3 v;
    if (!(in >> v.x() >> v.y() >> v.z())) {
        throw std::runtime_error(std::string("URDF: bad 3-vector '") + text + "'");
    }
    return v;
}

struct RawJoint {
    ChainJoint joint;
    std::string parent;
    std::string child;
};

}  // namespace

UrdfChain UrdfChain::FromFile(const std::string& urdf_path, const std::string& base_link,
                              const std::string& tip_link,
                              const std::vector<std::string>& joint_names) {
    std::ifstream in(urdf_path);
    if (!in) throw std::runtime_error("URDF not found: " + urdf_path);
    std::stringstream buf;
    buf << in.rdbuf();
    return FromString(buf.str(), base_link, tip_link, joint_names);
}

UrdfChain UrdfChain::FromString(const std::string& urdf_xml, const std::string& base_link,
                                const std::string& tip_link,
                                const std::vector<std::string>& joint_names) {
    tinyxml2::XMLDocument doc;
    if (doc.Parse(urdf_xml.c_str()) != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error(std::string("URDF parse error: ") + doc.ErrorStr());
    }
    const tinyxml2::XMLElement* robot = doc.FirstChildElement("robot");
    if (robot == nullptr) throw std::runtime_error("URDF: no <robot> element");

    // Link inertials (mass, centre of mass in the link frame).
    std::map<std::string, std::pair<double, Vec3>> inertial;
    for (const tinyxml2::XMLElement* l = robot->FirstChildElement("link"); l != nullptr;
         l = l->NextSiblingElement("link")) {
        const tinyxml2::XMLElement* in = l->FirstChildElement("inertial");
        if (in == nullptr || l->Attribute("name") == nullptr) continue;
        const tinyxml2::XMLElement* m = in->FirstChildElement("mass");
        const tinyxml2::XMLElement* o = in->FirstChildElement("origin");
        const double mass = m && m->Attribute("value") ? m->DoubleAttribute("value") : 0.0;
        inertial[l->Attribute("name")] = {mass, ParseVec3(o ? o->Attribute("xyz") : nullptr, Vec3::Zero())};
    }
    std::map<std::string, RawJoint> by_child;
    for (const tinyxml2::XMLElement* e = robot->FirstChildElement("joint"); e != nullptr;
         e = e->NextSiblingElement("joint")) {
        RawJoint raw;
        raw.joint.name = e->Attribute("name") ? e->Attribute("name") : "";
        const std::string type = e->Attribute("type") ? e->Attribute("type") : "fixed";
        if (type == "revolute" || type == "continuous") {
            raw.joint.type = ChainJoint::Type::Revolute;
        } else if (type == "prismatic") {
            raw.joint.type = ChainJoint::Type::Prismatic;
        } else if (type == "fixed") {
            raw.joint.type = ChainJoint::Type::Fixed;
        } else {
            throw std::runtime_error("URDF: unsupported joint type '" + type + "' (" +
                                     raw.joint.name + ")");
        }
        const tinyxml2::XMLElement* parent = e->FirstChildElement("parent");
        const tinyxml2::XMLElement* child = e->FirstChildElement("child");
        if (parent == nullptr || child == nullptr || parent->Attribute("link") == nullptr ||
            child->Attribute("link") == nullptr) {
            throw std::runtime_error("URDF: joint without parent/child link: " + raw.joint.name);
        }
        raw.parent = parent->Attribute("link");
        raw.child = child->Attribute("link");
        const tinyxml2::XMLElement* origin = e->FirstChildElement("origin");
        const Vec3 xyz = ParseVec3(origin ? origin->Attribute("xyz") : nullptr, Vec3::Zero());
        const Vec3 rpy = ParseVec3(origin ? origin->Attribute("rpy") : nullptr, Vec3::Zero());
        raw.joint.t_origin = xyz;
        raw.joint.R_origin = RpyToRotation(rpy);
        const tinyxml2::XMLElement* axis = e->FirstChildElement("axis");
        Vec3 a = ParseVec3(axis ? axis->Attribute("xyz") : nullptr, Vec3::UnitX());
        if (a.norm() < 1e-12) throw std::runtime_error("URDF: zero axis on " + raw.joint.name);
        raw.joint.axis = a.normalized();
        if (by_child.count(raw.child)) {
            throw std::runtime_error("URDF: link '" + raw.child + "' has two parent joints");
        }
        by_child[raw.child] = raw;
    }

    std::vector<ChainJoint> chain;
    std::set<std::string> seen;
    std::string link = tip_link;
    while (link != base_link) {
        auto it = by_child.find(link);
        if (it == by_child.end()) {
            throw std::runtime_error("URDF: no path from '" + base_link + "' to '" + tip_link +
                                     "' (stopped at '" + link + "')");
        }
        if (!seen.insert(link).second) throw std::runtime_error("URDF: cycle at " + link);
        chain.push_back(it->second.joint);
        link = it->second.parent;
    }
    // Off-chain subtrees (joints at 0) lumped into their chain link.
    std::map<std::string, std::vector<const RawJoint*>> children;
    for (const auto& kv : by_child) children[kv.second.parent].push_back(&kv.second);
    std::set<std::string> on_chain(seen.begin(), seen.end());
    on_chain.insert(base_link);
    auto lump = [&](const std::string& root) {
        double m = 0;
        Vec3 mc = Vec3::Zero();
        std::vector<std::pair<std::string, Iso3>> stack = {{root, Iso3::Identity()}};
        while (!stack.empty()) {
            const auto [link, T] = stack.back();
            stack.pop_back();
            auto it = inertial.find(link);
            if (it != inertial.end()) {
                m += it->second.first;
                mc += it->second.first * (T * it->second.second);
            }
            for (const RawJoint* rj : children[link]) {
                if (on_chain.count(rj->child)) continue;
                Iso3 o = Iso3::Identity();
                o.linear() = rj->joint.R_origin;
                o.translation() = rj->joint.t_origin;
                stack.push_back({rj->child, T * o});
            }
        }
        return std::make_pair(m, m > 0 ? Vec3(mc / m) : Vec3::Zero());
    };
    std::vector<std::string> chain_child;  // child link of each chain joint (base -> tip)
    {
        std::string l = tip_link;
        while (l != base_link) {
            chain_child.push_back(l);
            l = by_child.at(l).parent;
        }
        std::reverse(chain_child.begin(), chain_child.end());
    }
    UrdfChain out;
    out.joints_.assign(chain.rbegin(), chain.rend());
    for (size_t i = 0; i < out.joints_.size(); ++i) {
        const auto mc = lump(chain_child[i]);
        out.joints_[i].mass = mc.first;
        out.joints_[i].com = mc.second;
    }
    for (ChainJoint& j : out.joints_) {
        const Vec3 a = j.axis.normalized();
        const Vec3 ref = std::fabs(a.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY();
        j.perp1 = a.cross(ref).normalized();
        j.perp2 = a.cross(j.perp1).normalized();
    }
    out.base_link_ = base_link;
    out.tip_link_ = tip_link;
    out.dof_ = static_cast<int>(joint_names.size());

    std::set<std::string> found;
    for (ChainJoint& j : out.joints_) {
        if (j.type == ChainJoint::Type::Fixed) continue;
        for (size_t k = 0; k < joint_names.size(); ++k) {
            if (joint_names[k] == j.name) {
                j.active = static_cast<int>(k);
                found.insert(j.name);
            }
        }
    }
    for (const std::string& n : joint_names) {
        if (!found.count(n)) {
            throw std::runtime_error("URDF: joint '" + n + "' is not on the chain " + base_link +
                                     " -> " + tip_link);
        }
    }
    return out;
}

Iso3 UrdfChain::ForwardIso(const std::vector<double>& theta) const {
    if (static_cast<int>(theta.size()) != dof_) {
        throw std::runtime_error("ForwardIso: expected " + std::to_string(dof_) + " joints");
    }
    Mat3 R;
    Vec3 t;
    Forward(theta.data(), R, t);
    Iso3 T = Iso3::Identity();
    T.linear() = R;
    T.translation() = t;
    return T;
}

std::vector<Iso3> UrdfChain::FrameChain(const std::vector<double>& theta) const {
    std::vector<Iso3> out;
    Iso3 T = Iso3::Identity();
    for (const ChainJoint& j : joints_) {
        Iso3 o = Iso3::Identity();
        o.linear() = j.R_origin;
        o.translation() = j.t_origin;
        T = T * o;
        if (j.active >= 0) {
            Iso3 m = Iso3::Identity();
            if (j.type == ChainJoint::Type::Prismatic) m.translation() = j.axis * theta.at(j.active);
            else m.linear() = Eigen::AngleAxisd(theta.at(j.active), j.axis).toRotationMatrix();
            T = T * m;
        }
        out.push_back(T);
    }
    return out;
}

std::vector<double> UrdfChain::GravityTorques(const std::vector<double>& theta, const Mat3& R_world_base,
                                              const Vec3& g_world) const {
    const std::vector<Iso3> F = FrameChain(theta);
    const Vec3 g = R_world_base.transpose() * g_world;  // gravity in the base frame
    std::vector<double> tau(dof_, 0.0);
    for (size_t j = 0; j < joints_.size(); ++j) {
        const ChainJoint& J = joints_[j];
        if (J.active < 0) continue;
        const Vec3 axis = F[j].linear() * J.axis, p = F[j].translation();
        Vec3 sum = Vec3::Zero();
        double force = 0;
        for (size_t i = j; i < joints_.size(); ++i) {
            if (joints_[i].mass <= 0) continue;
            const Vec3 c = F[i] * joints_[i].com;
            sum += (c - p).cross(joints_[i].mass * g);
            force += axis.dot(joints_[i].mass * g);
        }
        tau[J.active] = J.type == ChainJoint::Type::Prismatic ? force : axis.dot(sum);
    }
    return tau;
}

double UrdfChain::TotalMass() const {
    double m = 0;
    for (const ChainJoint& j : joints_) m += j.mass;
    return m;
}

}  // namespace robocal
