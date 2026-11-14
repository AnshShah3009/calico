// One board view and its reprojection residual, for any combination of
//   camera: wrist (on an arm: FK(q + dq) * T_tip_cam)  or static (world pose),
//   board:  static (world pose)  or mounted on an arm link (FK(q + dq) * T_tip_link * T_link_board),
// with an optional board-shape model (bow, twist, x / y print scale).
#ifndef ROBOCAL_VIEW_COST_HPP_
#define ROBOCAL_VIEW_COST_HPP_

#include "camera_models.hpp"
#include "se3.hpp"
#include "urdf_chain.hpp"

#include <limits>
#include <map>
#include <string>
#include <vector>

namespace robocal {

struct View {
    int arm = -1;     // >= 0: wrist view of this arm
    int camera = -1;  // camera index
    std::string board = "0";
    int frame_idx = 0;
    std::vector<double> q;                    // encoder joints of the wrist camera's arm
    std::map<int, std::vector<double>> arm_q;  // encoder joints of every arm in this frame
    std::vector<Vec3> obj;                    // board points (board frame, m)
    std::vector<Vec2> img;                    // pixels
    std::vector<int> pids;                    // BoardSpec::Point ids of obj / img (detected boards)
    int n_guided = 0;                         // points added by the guided pass
    std::map<int, std::vector<double>> travel;  // per arm: signed final travel per joint (rad), if logged
    std::vector<double> dtau, ds, amb;        // drive regressors of the wrist arm (per joint)
    bool has_T_cam_board = false;
    Iso3 T_cam_board = Iso3::Identity();  // PnP, used only to initialize
    bool holdout = false;
    bool dropped = false;
    double sigma_scale = 1.0;  // view noise / pixel noise (solver.view_weights), >= 1
    int src_frame = -1;        // index of the source frame (frames list order): time blocks
    std::vector<double> pw;    // per-corner robust weight, sqrt applied (solver.robust: corner); empty = 1
    std::vector<double> dw;    // per-corner distance weight (solver.distance_sigma_mm), relative to the pixel noise; empty = 1
    std::string image;  // source frame (manifest input)
};

// Board shape: shape[0] bow along x, shape[1] bow along y, shape[2] twist (all m, at the
// board centre / corners), shape[3] / shape[4] relative print scale along x / y.
// u, v in [-1, 1] over the board: z += bx (1 - u^2) + by (1 - v^2) + tw u v.
struct BoardGeometry {
    double cx = 0, cy = 0, hw = 1, hh = 1;
};

template <typename T>
using M3T = Eigen::Matrix<T, 3, 3>;
template <typename T>
using V3T = Eigen::Matrix<T, 3, 1>;

// Parameter blocks, in order:
//   camera  wrist:  dq_c, [eps_c], [drive_c], tip_q, tip_t, base_q, base_t      static: cam_q, cam_t
//   board   static: board_q, board_t                         mounted: dq_b, [eps_b], mount_q, mount_t, base_q, base_t
// eps: this frame's encoder errors (errors-in-variables), theta = q + dq + eps.
//   shape (5), [warp (8)], [point heights (board points)], intrinsics.
struct ViewCost {
    const View* view = nullptr;
    LensModel model = LensModel::Equidistant;
    const UrdfChain* cam_chain = nullptr;    // wrist camera's arm (nullptr: static camera)
    const UrdfChain* board_chain = nullptr;  // mounting arm (nullptr: static board)
    std::vector<double> board_q;             // mounting arm's joints in this frame
    bool cam_eiv = false;                    // per-frame encoder error block follows dq_c
    bool board_eiv = false;                  // per-frame encoder error block follows dq_b
    bool cam_drive = false;                  // drive block (c, h) follows [eps_c]
    bool cam_geo = false;                    // geometry block (4 per joint) follows [drive]
    bool board_warp = false;                 // in-plane warp block (8) follows the shape block
    bool board_heights = false;              // per-point board heights follow the shape [/ warp] block
    bool cam_pose7 = false;                  // static / hand-held camera pose is one block (q[4], t[3])
    Iso3 T_tip_link = Iso3::Identity();      // URDF fixed transform tip -> mount link
    BoardGeometry geom;
    double weight = 1.0;  // 1 / pixel noise sigma: residuals in units of the detection noise
    bool corner_weights = true;  // apply View.pw (solve); false: plain residuals (reports, outlier tests)

    template <typename T>
    bool operator()(T const* const* p, T* r) const {
        int k = 0;
        M3T<T> Rwc;
        V3T<T> twc;
        if (cam_chain != nullptr) {
            const T* dq = p[k++];
            const T* eps = cam_eiv ? p[k++] : nullptr;
            const T* drv = cam_drive ? p[k++] : nullptr;
            const T* geo = cam_geo ? p[k++] : nullptr;
            const T* tq = p[k++];
            const T* tt = p[k++];
            const T* bq = p[k++];
            const T* bt = p[k++];
            std::vector<T> th(cam_chain->dof());
            const int n = cam_chain->dof();
            for (int i = 0; i < n; ++i) {
                th[i] = T(view->q[i]) + dq[i] + (eps ? eps[i] : T(0));
                if (drv) th[i] += drv[i] * T(view->dtau[i]) + drv[n + i] * T(view->ds[i]);
            }
            M3T<T> Rbt;
            V3T<T> tbt;
            cam_chain->Forward(th.data(), Rbt, tbt, geo);
            const M3T<T> Rwb = Eigen::Map<const Eigen::Quaternion<T>>(bq).normalized().toRotationMatrix();
            const M3T<T> Rtc = Eigen::Map<const Eigen::Quaternion<T>>(tq).normalized().toRotationMatrix();
            Rwc = Rwb * Rbt * Rtc;
            twc = Rwb * (Rbt * Eigen::Map<const V3T<T>>(tt) + tbt) + Eigen::Map<const V3T<T>>(bt);
        } else {
            const T* cq = p[k++];
            const T* ct = cam_pose7 ? cq + 4 : p[k++];
            Rwc = Eigen::Map<const Eigen::Quaternion<T>>(cq).normalized().toRotationMatrix();
            twc = Eigen::Map<const V3T<T>>(ct);
        }
        M3T<T> Rwo;
        V3T<T> two;
        if (board_chain != nullptr) {
            const T* dq = p[k++];
            const T* eps = board_eiv ? p[k++] : nullptr;
            const T* mq = p[k++];
            const T* mt = p[k++];
            const T* bq = p[k++];
            const T* bt = p[k++];
            std::vector<T> th(board_chain->dof());
            for (int i = 0; i < board_chain->dof(); ++i) th[i] = T(board_q[i]) + dq[i] + (eps ? eps[i] : T(0));
            M3T<T> Rbt;
            V3T<T> tbt;
            board_chain->Forward(th.data(), Rbt, tbt);
            const M3T<T> Rtl = T_tip_link.linear().cast<T>();
            const V3T<T> ttl = T_tip_link.translation().cast<T>();
            const M3T<T> Rwb = Eigen::Map<const Eigen::Quaternion<T>>(bq).normalized().toRotationMatrix();
            const M3T<T> Rlm = Eigen::Map<const Eigen::Quaternion<T>>(mq).normalized().toRotationMatrix();
            Rwo = Rwb * Rbt * Rtl * Rlm;
            two = Rwb * (Rbt * (Rtl * Eigen::Map<const V3T<T>>(mt) + ttl) + tbt) + Eigen::Map<const V3T<T>>(bt);
        } else {
            const T* oq = p[k++];
            const T* ot = p[k++];
            Rwo = Eigen::Map<const Eigen::Quaternion<T>>(oq).normalized().toRotationMatrix();
            two = Eigen::Map<const V3T<T>>(ot);
        }
        const T* shape = p[k++];
        const T* wp = board_warp ? p[k++] : nullptr;
        const T* hz = board_heights ? p[k++] : nullptr;
        const T* intr = p[k++];
        const M3T<T> Rcw = Rwc.transpose();
        const M3T<T> Rco = Rcw * Rwo;
        const V3T<T> tco = Rcw * (two - twc);
        for (size_t i = 0; i < view->obj.size(); ++i) {
            const Vec3& o = view->obj[i];
            const double u = (o.x() - geom.cx) / geom.hw;
            const double v = (o.y() - geom.cy) / geom.hh;
            V3T<T> X;
            X[0] = T(geom.cx) + T(o.x() - geom.cx) * (T(1) + shape[3]);
            X[1] = T(geom.cy) + T(o.y() - geom.cy) * (T(1) + shape[4]);
            X[2] = T(o.z()) + shape[0] * T(1 - u * u) + shape[1] * T(1 - v * v) + shape[2] * T(u * v);
            if (wp) {
                X[0] += wp[0] * T(v) + wp[1] * T(u * u) + wp[2] * T(u * v) + wp[3] * T(v * v);
                X[1] += wp[4] * T(u) + wp[5] * T(u * u) + wp[6] * T(u * v) + wp[7] * T(v * v);
            }
            if (hz) X[2] += hz[view->pids[i]];
            const V3T<T> Xc = Rco * X + tco;
            T uv[2];
            if (!ProjectPoint(model, intr, Xc, uv)) return false;
            double wi = corner_weights && i < view->pw.size() ? weight * view->pw[i] : weight;
            if (corner_weights && i < view->dw.size()) wi *= view->dw[i];
            r[2 * i] = (uv[0] - T(view->img[i].x())) * T(wi);
            r[2 * i + 1] = (uv[1] - T(view->img[i].y())) * T(wi);
        }
        return true;
    }
};

}  // namespace robocal

#endif  // ROBOCAL_VIEW_COST_HPP_
