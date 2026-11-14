#include "stage1_solver.hpp"

#include <ceres/jet.h>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace robocal {
namespace {

constexpr int kMaxIntr = 13;
using Jet = ceres::Jet<double, 3 + kMaxIntr>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using MatX = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

Mat3 Skew(const Vec3& v) {
    Mat3 S;
    S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return S;
}

// Linearization of one view: pose block p (6), dense F part d (lens [+ shape]), heights h (one per point).
struct ViewLin {
    int n = 0, d = 0;
    std::vector<int> didx;  // global F index of the dense columns
    std::vector<int> hidx;  // global F index of each point's height (-1: held / none)
    Mat6 U = Mat6::Zero();
    MatX Wd, Vdd, Vdh;  // 6 x d, d x d, d x n
    MatX Wh;            // 6 x n
    VecX Vhh, gd, gh;   // n, d, n
    Vec6 gp = Vec6::Zero();
    double cost = 0;
};

struct Layout {
    int nf = 0;
    std::vector<int> cam_off, shape_off, shape_n, h_off, h_n;
};

// Board point (board frame, with shape and heights) and its derivatives w.r.t. the 5 shape values.
inline Vec3 BoardPoint(const S1Board& b, const View& v, size_t i, bool heights, Eigen::Matrix<double, 3, 5>* dS) {
    const Vec3& o = v.obj[i];
    const BoardGeometry& g = b.geom;
    const double u = (o.x() - g.cx) / g.hw, w = (o.y() - g.cy) / g.hh;
    const double* s = b.shape;
    Vec3 X(g.cx + (o.x() - g.cx) * (1 + s[3]), g.cy + (o.y() - g.cy) * (1 + s[4]),
           o.z() + s[0] * (1 - u * u) + s[1] * (1 - w * w) + s[2] * (u * w));
    if (heights) X.z() += (*b.heights)[v.pids[i]];
    if (dS) {
        dS->setZero();
        (*dS)(2, 0) = 1 - u * u;
        (*dS)(2, 1) = 1 - w * w;
        (*dS)(2, 2) = u * w;
        (*dS)(0, 3) = o.x() - g.cx;
        (*dS)(1, 4) = o.y() - g.cy;
    }
    return X;
}

inline void Robust(double s, double a, double& rho, double& rho1) {
    if (a <= 0 || s <= a * a) {
        rho = s;
        rho1 = 1;
    } else {
        const double r = std::sqrt(s);
        rho = 2 * a * r - a * a;
        rho1 = std::max(std::numeric_limits<double>::min(), a / r);
    }
}

// Cost of one view; false if a point does not project.
bool ViewCostValue(const S1Camera& cam, const S1Board& b, const S1View& sv, double huber, double& cost) {
    const View& v = *sv.view;
    const Eigen::Quaterniond Q(sv.pose[3], sv.pose[0], sv.pose[1], sv.pose[2]);
    const Mat3 Rcw = Q.normalized().toRotationMatrix().transpose();
    const Vec3 twc(sv.pose[4], sv.pose[5], sv.pose[6]);
    const Mat3 Rco = Rcw * b.pose.linear();
    const Vec3 tco = Rcw * (b.pose.translation() - twc);
    double s = 0;
    for (size_t i = 0; i < v.obj.size(); ++i) {
        const Vec3 Xc = Rco * BoardPoint(b, v, i, sv.heights, nullptr) + tco;
        double uv[2];
        if (!ProjectPoint(cam.model, cam.intr, Eigen::Matrix<double, 3, 1>(Xc), uv)) return false;
        const double wi = i < v.pw.size() ? sv.weight * v.pw[i] : sv.weight;
        const double rx = (uv[0] - v.img[i].x()) * wi, ry = (uv[1] - v.img[i].y()) * wi;
        if (!std::isfinite(rx) || !std::isfinite(ry)) return false;
        s += rx * rx + ry * ry;
    }
    double rho, rho1;
    Robust(s, huber, rho, rho1);
    cost = 0.5 * rho * sv.mult;
    return true;
}

double PriorCost(const std::vector<S1Board>& boards) {
    double c = 0;
    for (const S1Board& b : boards) {
        if (b.shape_prior) {
            for (int k = 0; k < 3; ++k) c += std::pow(b.shape[k] * b.inv_sigma_shape, 2);
            for (int k = 3; k < 5; ++k) c += std::pow((b.shape[k] - b.scale0[k - 3]) * b.inv_sigma_scale, 2);
        }
        if (b.heights && b.heights_free) {
            for (double h : *b.heights) c += std::pow(h * b.inv_sigma_height, 2);
        }
    }
    return 0.5 * c;
}

bool TotalCost(const std::vector<S1Camera>& cams, const std::vector<S1Board>& boards, const std::vector<S1View>& views,
               const S1Options& o, double& cost) {
    double c = 0;
    bool ok = true;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : c) reduction(&& : ok) num_threads(o.threads)
    for (int k = 0; k < static_cast<int>(views.size()); ++k) {
        double cv = 0;
        if (!ViewCostValue(cams[views[k].camera], boards[views[k].board], views[k], o.huber, cv)) ok = false;
        c += cv;
    }
    cost = c + PriorCost(boards);
    return ok && std::isfinite(cost);
}

bool Linearize(const S1Camera& cam, const S1Board& b, const S1View& sv, const Layout& L, double huber, ViewLin& vl) {
    const View& v = *sv.view;
    const int n = static_cast<int>(v.obj.size());
    const int ni = cam.free ? cam.n_intr : 0;
    const int bi = sv.board;
    const int ns = L.shape_n[bi];
    vl.n = n;
    vl.d = ni + ns;
    const int d = vl.d;
    vl.didx.resize(d);
    for (int k = 0; k < ni; ++k) vl.didx[k] = L.cam_off[sv.camera] + k;
    for (int k = 0; k < ns; ++k) vl.didx[ni + k] = L.shape_off[bi] + k;
    vl.hidx.assign(n, -1);
    const bool hfree = sv.heights && b.heights_free;
    if (hfree) {
        for (int i = 0; i < n; ++i) vl.hidx[i] = L.h_off[bi] + v.pids[i];
    }
    const Eigen::Quaterniond Q(sv.pose[3], sv.pose[0], sv.pose[1], sv.pose[2]);
    const Mat3 Rwc = Q.normalized().toRotationMatrix();
    const Mat3 Rcw = Rwc.transpose();
    const Vec3 twc(sv.pose[4], sv.pose[5], sv.pose[6]);
    const Mat3 Rco = Rcw * b.pose.linear();
    const Vec3 tco = Rcw * (b.pose.translation() - twc);
    // Free shape columns (SubsetManifold: x / y scale held when !scale_free).
    int scol[5], nsc = 0;
    if (ns > 0) {
        for (int k = 0; k < 5; ++k) {
            if (k < 3 || b.scale_free) scol[nsc++] = k;
        }
    }
    Eigen::Matrix<double, 2, Eigen::Dynamic> Jp(2, 6), Jd(2, d);
    Eigen::Vector2d Jh;
    MatX Jall(2 * n, 6 + d);  // pose | dense
    MatX Jhs(2 * n, 1);
    VecX r(2 * n);
    Jet intrJ[kMaxIntr + 4];
    for (int k = 0; k < cam.n_intr; ++k) {
        intrJ[k] = Jet(cam.intr[k]);
        if (k < kMaxIntr) intrJ[k].v[3 + k] = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix<double, 3, 5> dS;
        const Vec3 Xb = BoardPoint(b, v, i, sv.heights, &dS);
        const Vec3 Xc = Rco * Xb + tco;
        Eigen::Matrix<Jet, 3, 1> XcJ;
        for (int a = 0; a < 3; ++a) XcJ[a] = Jet(Xc[a], a);
        Jet uv[2];
        if (!ProjectPoint(cam.model, intrJ, XcJ, uv)) return false;
        Eigen::Matrix<double, 2, 3> A;
        for (int a = 0; a < 3; ++a) {
            A(0, a) = uv[0].v[a];
            A(1, a) = uv[1].v[a];
        }
        const double w = static_cast<size_t>(i) < v.pw.size() ? sv.weight * v.pw[i] : sv.weight;
        r[2 * i] = (uv[0].a - v.img[i].x()) * w;
        r[2 * i + 1] = (uv[1].a - v.img[i].y()) * w;
        // Pose: world <- camera, left perturbation R <- Exp(dr) R, t <- t + dt.
        const Vec3 Xw = b.pose * Xb;
        const Eigen::Matrix<double, 2, 3> ARcw = A * Rcw;
        Jp.leftCols<3>() = ARcw * Skew(Xw - twc) * w;
        Jp.rightCols<3>() = -ARcw * w;
        for (int k = 0; k < ni; ++k) {
            Jd(0, k) = uv[0].v[3 + k] * w;
            Jd(1, k) = uv[1].v[3 + k] * w;
        }
        const Eigen::Matrix<double, 2, 3> ARco = A * Rco;
        for (int k = 0; k < nsc; ++k) Jd.col(ni + k) = ARco * dS.col(scol[k]) * w;
        Jh = ARco.col(2) * w;
        Jall.block(2 * i, 0, 2, 6) = Jp;
        if (d) Jall.block(2 * i, 6, 2, d) = Jd;
        Jhs.block(2 * i, 0, 2, 1) = Jh;
    }
    if (!r.allFinite() || !Jall.allFinite()) return false;
    double rho, rho1;
    Robust(r.squaredNorm(), huber, rho, rho1);
    rho1 *= sv.mult;  // bootstrap multiplicity: the view's terms enter mult times
    vl.cost = 0.5 * rho * sv.mult;
    // Normal-equation pieces, scaled by rho' (Ceres' corrector for Huber: alpha = 0).
    const MatX JtJ = Jall.transpose() * Jall;  // (6+d)^2
    const VecX Jtr = Jall.transpose() * r;
    vl.U = rho1 * JtJ.topLeftCorner<6, 6>();
    vl.Wd = rho1 * JtJ.topRightCorner(6, d);
    vl.Vdd = rho1 * JtJ.bottomRightCorner(d, d);
    vl.gp = rho1 * Jtr.head<6>();
    vl.gd = rho1 * Jtr.tail(d);
    if (hfree) {
        vl.Wh.resize(6, n);
        vl.Vdh.resize(d, n);
        vl.Vhh.resize(n);
        vl.gh.resize(n);
        for (int i = 0; i < n; ++i) {
            const auto Ji = Jall.block(2 * i, 0, 2, 6 + d);
            const Eigen::Vector2d jh = Jhs.block(2 * i, 0, 2, 1);
            const VecX c = Ji.transpose() * jh;  // 6 + d
            vl.Wh.col(i) = rho1 * c.head<6>();
            if (d) vl.Vdh.col(i) = rho1 * c.tail(d);
            vl.Vhh[i] = rho1 * jh.squaredNorm();
            vl.gh[i] = rho1 * jh.dot(r.segment<2>(2 * i));
        }
    } else {
        vl.Wh.resize(6, 0);
        vl.Vdh.resize(d, 0);
        vl.Vhh.resize(0);
        vl.gh.resize(0);
    }
    return true;
}

inline double Clamp(double x) { return std::min(1e32, std::max(1e-6, x)); }

}  // namespace

S1Summary SolveStage1(std::vector<S1Camera>& cams, std::vector<S1Board>& boards, std::vector<S1View>& views,
                      const S1Options& o) {
    S1Summary sum;
    Layout L;
    L.cam_off.assign(cams.size(), -1);
    for (size_t c = 0; c < cams.size(); ++c) {
        if (cams[c].n_intr > kMaxIntr) {
            sum.failed = true;
            sum.message = "too many intrinsics";
            return sum;
        }
        if (cams[c].free) {
            L.cam_off[c] = L.nf;
            L.nf += cams[c].n_intr;
        }
    }
    L.shape_off.assign(boards.size(), -1);
    L.shape_n.assign(boards.size(), 0);
    L.h_off.assign(boards.size(), -1);
    L.h_n.assign(boards.size(), 0);
    for (size_t b = 0; b < boards.size(); ++b) {
        if (boards[b].shape_free) {
            L.shape_off[b] = L.nf;
            L.shape_n[b] = boards[b].scale_free ? 5 : 3;
            L.nf += L.shape_n[b];
        }
        if (boards[b].heights && boards[b].heights_free) {
            L.h_off[b] = L.nf;
            L.h_n[b] = static_cast<int>(boards[b].heights->size());
            L.nf += L.h_n[b];
        }
    }
    const int nv = static_cast<int>(views.size());
    const int nf = L.nf;
    // F values: read / write through these pointers.
    std::vector<double*> fptr(nf, nullptr);
    for (size_t c = 0; c < cams.size(); ++c) {
        for (int k = 0; cams[c].free && k < cams[c].n_intr; ++k) fptr[L.cam_off[c] + k] = cams[c].intr + k;
    }
    for (size_t b = 0; b < boards.size(); ++b) {
        int j = 0;
        for (int k = 0; k < 5 && L.shape_n[b] > 0; ++k) {
            if (k < 3 || boards[b].scale_free) fptr[L.shape_off[b] + j++] = boards[b].shape + k;
        }
        for (int k = 0; k < L.h_n[b]; ++k) fptr[L.h_off[b] + k] = &(*boards[b].heights)[k];
    }

    double cost = 0;
    if (!TotalCost(cams, boards, views, o, cost)) {
        sum.failed = true;
        sum.message = "initial cost not finite";
        return sum;
    }
    sum.initial_cost = cost;
    std::vector<ViewLin> lin(nv);
    double radius = 1e4, decrease = 2.0;
    std::vector<double> pose_bak(7 * nv), f_bak(nf);
    MatX S(nf, nf);
    VecX rhs(nf), gf(nf), Df(nf);
    std::vector<Mat6> Ui(nv);
    std::vector<Vec6> dp(nv);
    bool relinearize = true;
    const int T = std::max(1, o.threads);
    std::vector<MatX> S_t(T);
    std::vector<VecX> rhs_t(T);
    for (int it = 0; it < o.max_iterations; ++it) {
        sum.iterations = it + 1;
        double t0 = Now();
        if (relinearize) {
            bool ok = true;
#pragma omp parallel for schedule(dynamic, 8) reduction(&& : ok) num_threads(o.threads)
            for (int k = 0; k < nv; ++k) {
                if (!Linearize(cams[views[k].camera], boards[views[k].board], views[k], L, o.huber, lin[k])) ok = false;
            }
            if (!ok) {
                sum.failed = true;
                sum.message = "linearization failed";
                break;
            }
            // Gradient and Marquardt diagonal of the F part (data + priors).
            gf.setZero();
            Df.setZero();
            for (int k = 0; k < nv; ++k) {
                const ViewLin& vl = lin[k];
                for (int a = 0; a < vl.d; ++a) {
                    gf[vl.didx[a]] += vl.gd[a];
                    Df[vl.didx[a]] += vl.Vdd(a, a);
                }
                for (int i = 0; i < vl.gh.size(); ++i) {
                    gf[vl.hidx[i]] += vl.gh[i];
                    Df[vl.hidx[i]] += vl.Vhh[i];
                }
            }
            for (size_t b = 0; b < boards.size(); ++b) {
                const S1Board& B = boards[b];
                if (B.shape_prior && L.shape_n[b] > 0) {
                    int j = 0;
                    for (int k = 0; k < 5; ++k) {
                        if (!(k < 3 || B.scale_free)) continue;
                        const double is = k < 3 ? B.inv_sigma_shape : B.inv_sigma_scale;
                        const double r = k < 3 ? B.shape[k] * is : (B.shape[k] - B.scale0[k - 3]) * is;
                        gf[L.shape_off[b] + j] += is * r;
                        Df[L.shape_off[b] + j] += is * is;
                        ++j;
                    }
                }
                for (int k = 0; k < L.h_n[b]; ++k) {
                    const double is = B.inv_sigma_height;
                    gf[L.h_off[b] + k] += is * is * (*B.heights)[k];
                    Df[L.h_off[b] + k] += is * is;
                }
            }
            relinearize = false;
        }
        sum.t_linearize += Now() - t0;
        // Gradient test (max norm).
        double gmax = gf.size() ? gf.cwiseAbs().maxCoeff() : 0.0;
        for (int k = 0; k < nv; ++k) gmax = std::max(gmax, lin[k].gp.cwiseAbs().maxCoeff());
        if (!o.covariance && gmax <= o.gradient_tolerance) {
            sum.converged = true;
            sum.message = "gradient tolerance";
            break;
        }
        t0 = Now();
        const double mu = o.covariance ? 0.0 : 1.0 / radius;
        // Reduced system S df = rhs, assembled per thread.
#pragma omp parallel num_threads(T)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            MatX& St = S_t[tid];
            VecX& rt = rhs_t[tid];
            if (St.rows() != nf) St.resize(nf, nf);
            if (rt.size() != nf) rt.resize(nf);
            St.setZero();
            rt.setZero();
#pragma omp for schedule(dynamic, 8)
            for (int k = 0; k < nv; ++k) {
                const ViewLin& vl = lin[k];
                Mat6 Ud = vl.U;
                for (int a = 0; a < 6; ++a) Ud(a, a) += mu * Clamp(vl.U(a, a));
                Ui[k] = Ud.llt().solve(Mat6::Identity());
                const int d = vl.d, nh = static_cast<int>(vl.gh.size()), m = d + nh;
                MatX W(6, m);
                if (d) W.leftCols(d) = vl.Wd;
                if (nh) W.rightCols(nh) = vl.Wh;
                const MatX UiW = Ui[k] * W;          // 6 x m
                MatX M = -W.transpose() * UiW;       // m x m
                VecX rl = UiW.transpose() * vl.gp;   // W^T Ui gp
                if (d) {
                    M.topLeftCorner(d, d) += vl.Vdd;
                    rl.head(d) -= vl.gd;
                }
                if (nh) {
                    for (int i = 0; i < nh; ++i) M(d + i, d + i) += vl.Vhh[i];
                    if (d) {
                        M.topRightCorner(d, nh) += vl.Vdh;
                        M.bottomLeftCorner(nh, d) += vl.Vdh.transpose();
                    }
                    rl.tail(nh) -= vl.gh;
                }
                std::vector<int> gidx(m);
                for (int a = 0; a < d; ++a) gidx[a] = vl.didx[a];
                for (int i = 0; i < nh; ++i) gidx[d + i] = vl.hidx[i];
                for (int a = 0; a < m; ++a) {
                    rt[gidx[a]] += rl[a];
                    for (int c = 0; c < m; ++c) St(gidx[a], gidx[c]) += M(a, c);
                }
            }
        }
        S.setZero();
        rhs.setZero();
        for (int t = 0; t < T; ++t) {
            if (S_t[t].rows() == nf) {
                S += S_t[t];
                rhs += rhs_t[t];
            }
        }
        // Priors (diagonal) + damping.
        for (size_t b = 0; b < boards.size(); ++b) {
            const S1Board& B = boards[b];
            if (B.shape_prior && L.shape_n[b] > 0) {
                int j = 0;
                for (int k = 0; k < 5; ++k) {
                    if (!(k < 3 || B.scale_free)) continue;
                    const double is = k < 3 ? B.inv_sigma_shape : B.inv_sigma_scale;
                    S(L.shape_off[b] + j, L.shape_off[b] + j) += is * is;
                    ++j;
                }
            }
            for (int k = 0; k < L.h_n[b]; ++k) S(L.h_off[b] + k, L.h_off[b] + k) += B.inv_sigma_height * B.inv_sigma_height;
        }
        // rhs so far: sum W^T Ui gp - g_data; the prior gradients are in gf only.
        VecX gprior = gf;
        for (int k = 0; k < nv; ++k) {
            const ViewLin& vl = lin[k];
            for (int a = 0; a < vl.d; ++a) gprior[vl.didx[a]] -= vl.gd[a];
            for (int i = 0; i < vl.gh.size(); ++i) gprior[vl.hidx[i]] -= vl.gh[i];
        }
        rhs -= gprior;
        if (o.covariance) {
            Eigen::LLT<MatX> llt(S);
            if (nf == 0 || llt.info() != Eigen::Success) {
                sum.failed = true;
                sum.message = "covariance: reduced matrix not positive definite";
                return sum;
            }
            const MatX Sinv = llt.solve(MatX::Identity(nf, nf));
            for (size_t c = 0; c < cams.size(); ++c) {
                if (cams[c].free) (*o.covariance)[cams[c].intr] = Sinv.block(L.cam_off[c], L.cam_off[c], cams[c].n_intr, cams[c].n_intr);
            }
            for (size_t b = 0; b < boards.size(); ++b) {
                if (L.shape_n[b] > 0) {
                    (*o.covariance)[boards[b].shape] = Sinv.block(L.shape_off[b], L.shape_off[b], L.shape_n[b], L.shape_n[b]);
                }
            }
            sum.converged = true;
            sum.message = "covariance";
            sum.final_cost = cost;
            return sum;
        }
        for (int a = 0; a < nf; ++a) S(a, a) += mu * Clamp(Df[a]);
        VecX df = VecX::Zero(nf);
        bool solved = true;
        if (nf > 0) {
            Eigen::LLT<MatX> llt(S);
            if (llt.info() != Eigen::Success) solved = false;
            else df = llt.solve(rhs);
            if (!df.allFinite()) solved = false;
        }
        double pred = 0;  // model decrease: 1/2 (-g.d + mu d^T D d)
        double step_sq = 0, x_sq = 0;
        if (solved) {
            for (int k = 0; k < nv; ++k) {
                const ViewLin& vl = lin[k];
                Vec6 b = vl.gp;
                if (vl.d) b += vl.Wd * df(Eigen::VectorXi::Map(vl.didx.data(), vl.d));
                for (int i = 0; i < vl.gh.size(); ++i) b += vl.Wh.col(i) * df[vl.hidx[i]];
                dp[k] = -Ui[k] * b;
                pred += -vl.gp.dot(dp[k]);
                for (int a = 0; a < 6; ++a) pred += mu * Clamp(vl.U(a, a)) * dp[k][a] * dp[k][a];
                step_sq += dp[k].squaredNorm();
            }
            pred += -gf.dot(df);
            for (int a = 0; a < nf; ++a) pred += mu * Clamp(Df[a]) * df[a] * df[a];
            pred *= 0.5;
            step_sq += df.squaredNorm();
        }
        sum.t_solve += Now() - t0;
        t0 = Now();
        bool accept = false;
        double new_cost = cost;
        if (solved && pred > 0) {
            for (int k = 0; k < nv; ++k) std::copy(views[k].pose, views[k].pose + 7, pose_bak.begin() + 7 * k);
            for (int a = 0; a < nf; ++a) f_bak[a] = *fptr[a];
            for (int k = 0; k < nv; ++k) {
                double* p = views[k].pose;
                x_sq += p[4] * p[4] + p[5] * p[5] + p[6] * p[6] + 1.0;
                const Vec3 dr = dp[k].head<3>();
                const double th = dr.norm();
                Eigen::Quaterniond dq = th > 0 ? Eigen::Quaterniond(Eigen::AngleAxisd(th, dr / th)) : Eigen::Quaterniond::Identity();
                Eigen::Quaterniond q(p[3], p[0], p[1], p[2]);
                q = (dq * q.normalized()).normalized();
                p[0] = q.x();
                p[1] = q.y();
                p[2] = q.z();
                p[3] = q.w();
                for (int a = 0; a < 3; ++a) p[4 + a] += dp[k][3 + a];
            }
            for (int a = 0; a < nf; ++a) {
                x_sq += f_bak[a] * f_bak[a];
                *fptr[a] += df[a];
            }
            if (TotalCost(cams, boards, views, o, new_cost)) {
                const double rho = (cost - new_cost) / pred;
                accept = rho > 1e-3;
                if (accept) {
                    radius = radius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
                    decrease = 2.0;
                }
            }
            if (!accept) {
                for (int k = 0; k < nv; ++k) std::copy(pose_bak.begin() + 7 * k, pose_bak.begin() + 7 * k + 7, views[k].pose);
                for (int a = 0; a < nf; ++a) *fptr[a] = f_bak[a];
            }
        }
        sum.t_eval += Now() - t0;
        if (o.verbose) {
            std::printf("  s1 it %3d cost %.6e -> %.6e pred %.3e mu %.2e %s\n", it, cost, new_cost, pred, mu,
                        accept ? "ok" : "rej");
        }
        if (!accept) {
            radius /= decrease;
            decrease *= 2.0;
            if (radius < 1e-32) {
                sum.converged = true;
                sum.message = "trust region radius tolerance";
                break;
            }
            continue;
        }
        const double change = cost - new_cost;
        cost = new_cost;
        relinearize = true;
        if (std::sqrt(step_sq) <= o.parameter_tolerance * (std::sqrt(x_sq) + o.parameter_tolerance)) {
            sum.converged = true;
            sum.message = "parameter tolerance";
            break;
        }
        if (std::fabs(change) <= o.function_tolerance * cost) {
            sum.converged = true;
            sum.message = "function tolerance";
            break;
        }
    }
    sum.final_cost = cost;
    if (!sum.converged && !sum.failed) sum.message = "max iterations";
    return sum;
}

}  // namespace robocal
