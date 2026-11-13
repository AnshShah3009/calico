#include "detect.hpp"

#include "../opencv-compat.hpp"

// TagDetector.h first: the Tag*.h code tables need its TagCodes definition.
#include <apriltags/TagDetector.h>
#include <apriltags/Tag16h5.h>
#include <apriltags/Tag25h7.h>
#include <apriltags/Tag25h9.h>
#include <apriltags/Tag36h11.h>
#include <apriltags/Tag36h9.h>

extern "C" {
#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h11.h>
}

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>

namespace robocal {
namespace {

const AprilTags::TagCodes& CodesFor(const std::string& family) {
    if (family == "tag36h11") return AprilTags::tagCodes36h11;
    if (family == "tag25h9") return AprilTags::tagCodes25h9;
    if (family == "tag16h5") return AprilTags::tagCodes16h5;
    if (family == "tag36h9") return AprilTags::tagCodes36h9;
    if (family == "tag25h7") return AprilTags::tagCodes25h7;
    throw std::runtime_error("unsupported AprilTag family '" + family + "'");
}

// AprilTag 3 with Kalibr tags: the black border is two cells thick, so the quad the detector
// finds (the outer black square) is two cells wider than the standard family's and every data
// bit sits one cell further in.
struct Apriltag3Kalibr {
    apriltag_family_t* base = nullptr;
    apriltag_family_t fam{};
    std::vector<uint32_t> bx, by;
    apriltag_detector_t* det = nullptr;
    void (*destroy)(apriltag_family_t*) = nullptr;

    explicit Apriltag3Kalibr(const std::string& family) {
        if (family == "tag36h11") base = tag36h11_create(), destroy = tag36h11_destroy;
        else if (family == "tag25h9") base = tag25h9_create(), destroy = tag25h9_destroy;
        else if (family == "tag16h5") base = tag16h5_create(), destroy = tag16h5_destroy;
        else throw std::runtime_error("apriltag3: no family " + family + " (tag36h11, tag25h9, tag16h5)");
        fam = *base;
        fam.impl = nullptr;
        fam.width_at_border = base->width_at_border + 2;
        fam.total_width = base->total_width + 2;
        bx.assign(base->bit_x, base->bit_x + base->nbits);
        by.assign(base->bit_y, base->bit_y + base->nbits);
        for (uint32_t& v : bx) ++v;
        for (uint32_t& v : by) ++v;
        fam.bit_x = bx.data();
        fam.bit_y = by.data();
        det = apriltag_detector_create();
        det->quad_decimate = 1.0f;  // full resolution: small, compressed tags at the image edge
        det->quad_sigma = 0.0f;
        det->refine_edges = true;
        det->nthreads = 1;
        apriltag_detector_add_family_bits(det, &fam, 2);
    }
    ~Apriltag3Kalibr() {
        apriltag_detector_destroy(det);
        destroy(base);
    }
};

Apriltag3Kalibr& Apriltag3For(const std::string& family) {
    thread_local std::map<std::string, std::unique_ptr<Apriltag3Kalibr>> cache;
    auto it = cache.find(family);
    if (it == cache.end()) it = cache.emplace(family, std::make_unique<Apriltag3Kalibr>(family)).first;
    return *it->second;
}

using RawTag = TagCorners;  // BL, BR, TR, TL

std::vector<RawTag> DetectTags(const cv::Mat& gray, const std::string& family, const std::string& which) {
    std::vector<RawTag> out;
    if (which == "apriltag3") {
        Apriltag3Kalibr& a = Apriltag3For(family);
        cv::Mat g = gray.isContinuous() ? gray : gray.clone();
        image_u8_t im{g.cols, g.rows, static_cast<int32_t>(g.step), g.data};
        zarray_t* dets = apriltag_detector_detect(a.det, &im);
        for (int i = 0; i < zarray_size(dets); ++i) {
            apriltag_detection_t* d;
            zarray_get(dets, i, &d);
            RawTag t;
            t.id = d->id;
            for (int k = 0; k < 4; ++k) t.p[k] = Vec2(d->p[k][0], d->p[k][1]);
            out.push_back(t);
        }
        apriltag_detections_destroy(dets);
        return out;
    }
    if (which != "kaess") throw std::runtime_error("detection.apriltag: kaess or apriltag3, not " + which);
    AprilTags::TagDetector detector(CodesFor(family), 2);
    for (const AprilTags::TagDetection& d : detector.extractTags(gray)) {
        if (!d.good) continue;
        RawTag t;
        t.id = d.id;
        for (int k = 0; k < 4; ++k) t.p[k] = Vec2(d.p[k].first, d.p[k].second);
        out.push_back(t);
    }
    return out;
}

// cornerSubPix on X-junctions; keeps a point only if it moved less than max_move px.
void RefineSaddles(const cv::Mat& gray, std::vector<Vec2>& pts, const std::vector<int>& half_windows,
                   const std::vector<double>& max_move, std::vector<bool>& ok) {
    ok.assign(pts.size(), false);
    for (size_t i = 0; i < pts.size(); ++i) {
        const int h = half_windows[i];
        const Vec2 p0 = pts[i];
        if (!(p0.x() >= h + 1 && p0.y() >= h + 1 && p0.x() < gray.cols - h - 1 && p0.y() < gray.rows - h - 1)) continue;
        std::vector<cv::Point2f> v = {cv::Point2f(static_cast<float>(p0.x()), static_cast<float>(p0.y()))};
        cv::cornerSubPix(gray, v, cv::Size(h, h), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 50, 1e-3));
        const Vec2 r(v[0].x, v[0].y);
        if (std::isfinite(r.x()) && (r - p0).norm() <= max_move[i]) {
            pts[i] = r;
            ok[i] = true;
        }
    }
}

// ------------------------------------------------------------------------------ X-junction fit
// Linear intensities of an 8-bit frame (inverse gamma), via a lookup table.
cv::Mat Linearize(const cv::Mat& gray, double gamma) {
    cv::Mat lut(1, 256, CV_32F);
    for (int i = 0; i < 256; ++i) lut.at<float>(i) = static_cast<float>(std::pow(i / 255.0, gamma));
    cv::Mat g8, out;
    gray.convertTo(g8, CV_8U);
    out.create(g8.size(), CV_32F);
    for (int r = 0; r < g8.rows; ++r) {
        const unsigned char* s = g8.ptr<unsigned char>(r);
        float* d = out.ptr<float>(r);
        for (int c = 0; c < g8.cols; ++c) d[c] = lut.at<float>(s[c]);
    }
    return out;
}

}  // namespace

// I(p) = c + a erf(l1 / (sqrt2 s)) erf(l2 / (sqrt2 s)), l_k = n_k . (p - p0), n_k = (cos phi_k, sin phi_k):
// two blurred straight edges crossing at p0 at any angle (foreshortened chessboard corner).
XJunctionFit FitXJunctionImpl(const cv::Mat& lin, const Vec2& p_init, const Eigen::Matrix2d& A, double module_m,
                              int black_xy, const XJunctionOptions& o);

XJunctionFit FitXJunction(const cv::Mat& lin, const Vec2& p_init, const Eigen::Matrix2d& A, double module_m,
                          int black_xy, const XJunctionOptions& o) {
    XJunctionFit f = FitXJunctionImpl(lin, p_init, A, module_m, black_xy, o);
    if (std::getenv("ROBOCAL_XFIT_DEBUG")) {
        static std::map<std::string, int> counts;
        static int n = 0;
#pragma omp critical
        {
            counts[f.ok ? "ok" : f.reason]++;
            if (++n % 100 == 0) {
                std::fprintf(stderr, "[xfit]");
                for (const auto& kv : counts) std::fprintf(stderr, " %s=%d", kv.first.c_str(), kv.second);
                std::fprintf(stderr, "\n");
            }
        }
    }
    return f;
}

XJunctionFit FitXJunctionImpl(const cv::Mat& lin, const Vec2& p_init, const Eigen::Matrix2d& A, double module_m,
                              int black_xy, const XJunctionOptions& o) {
    XJunctionFit f;
    f.p = p_init;
    const Eigen::Matrix2d Ainv = A.inverse();
    if (!std::isfinite(Ainv(0, 0))) return f;
    const double rho = o.rho_modules * module_m, cut = o.cut_modules * module_m;
    // Window: an ellipse in the image (a disc on the board), weights Gaussian in board units.
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(A);
    const double short_px = svd.singularValues()(1) * cut, long_px = svd.singularValues()(0) * cut;
    if (short_px < o.min_short_axis_px) {
        f.reason = "window";
        return f;
    }
    const int R = static_cast<int>(std::ceil(long_px)) + 1;
    const int x0 = static_cast<int>(std::floor(p_init.x())), y0 = static_cast<int>(std::floor(p_init.y()));
    if (x0 - R < 0 || y0 - R < 0 || x0 + R >= lin.cols || y0 + R >= lin.rows) {
        f.reason = "border";
        return f;
    }
    struct Px {
        double x, y, w, v;
    };
    std::vector<Px> px;
    int saturated = 0;
    for (int y = y0 - R; y <= y0 + R; ++y) {
        for (int x = x0 - R; x <= x0 + R; ++x) {
            const Vec2 d(x - p_init.x(), y - p_init.y());
            const Vec2 b = Ainv * d;
            const double q = b.norm();
            if (q > cut) continue;
            const float v = lin.at<float>(y, x);
            saturated += v >= o.saturation;
            px.push_back({static_cast<double>(x), static_cast<double>(y), std::exp(-0.5 * q * q / (rho * rho)), v});
        }
    }
    if (px.size() < 20 || saturated > o.max_saturated * px.size()) {
        f.reason = px.size() < 20 ? "window" : "saturated";
        return f;
    }
    // Initial lines: along the board axes through the corner. n1 crosses the board x-line toward
    // board +y, n2 crosses the y-line toward board +x.
    const Vec2 ex = (A.col(0)).normalized(), ey = (A.col(1)).normalized();
    Vec2 n1i(-ex.y(), ex.x()), n2i(-ey.y(), ey.x());
    if (n1i.dot(ey) < 0) n1i = -n1i;
    if (n2i.dot(ex) < 0) n2i = -n2i;
    double x[7] = {p_init.x(), p_init.y(), std::atan2(n1i.y(), n1i.x()), std::atan2(n2i.y(), n2i.x()), 1.0, 0.0, 0.0};
    const double k = 1.0 / std::sqrt(2.0);
    auto eval = [&](const double* P, std::vector<double>* res, Eigen::Matrix<double, Eigen::Dynamic, 7>* J) {
        const Vec2 n1(std::cos(P[2]), std::sin(P[2])), n2(std::cos(P[3]), std::sin(P[3]));
        const Vec2 t1(-n1.y(), n1.x()), t2(-n2.y(), n2.x());
        const double s = P[4];
        double cost = 0;
        if (res) res->resize(px.size());
        if (J) J->resize(static_cast<Eigen::Index>(px.size()), 7);
        for (size_t i = 0; i < px.size(); ++i) {
            const Vec2 d(px[i].x - P[0], px[i].y - P[1]);
            const double l1 = n1.dot(d), l2 = n2.dot(d);
            const double E1 = std::erf(k * l1 / s), E2 = std::erf(k * l2 / s);
            const double g1 = std::sqrt(2.0 / M_PI) / s * std::exp(-0.5 * l1 * l1 / (s * s));
            const double g2 = std::sqrt(2.0 / M_PI) / s * std::exp(-0.5 * l2 * l2 / (s * s));
            const double sw = std::sqrt(px[i].w);
            const double r = sw * (P[5] + P[6] * E1 * E2 - px[i].v);
            cost += r * r;
            if (res) (*res)[i] = r;
            if (J) {
                const double a = P[6];
                (*J)(i, 0) = sw * a * (E2 * g1 * (-n1.x()) + E1 * g2 * (-n2.x()));
                (*J)(i, 1) = sw * a * (E2 * g1 * (-n1.y()) + E1 * g2 * (-n2.y()));
                (*J)(i, 2) = sw * a * E2 * g1 * t1.dot(d);
                (*J)(i, 3) = sw * a * E1 * g2 * t2.dot(d);
                (*J)(i, 4) = sw * a * (-E2 * g1 * l1 / s - E1 * g2 * l2 / s);
                (*J)(i, 5) = sw;
                (*J)(i, 6) = sw * E1 * E2;
            }
        }
        return cost;
    };
    // c, a by weighted linear least squares at the initial geometry.
    {
        double sw = 0, se = 0, see = 0, sv = 0, sev = 0;
        const Vec2 n1(std::cos(x[2]), std::sin(x[2])), n2(std::cos(x[3]), std::sin(x[3]));
        for (const Px& p : px) {
            const Vec2 d(p.x - x[0], p.y - x[1]);
            const double e = std::erf(k * n1.dot(d) / x[4]) * std::erf(k * n2.dot(d) / x[4]);
            sw += p.w, se += p.w * e, see += p.w * e * e, sv += p.w * p.v, sev += p.w * e * p.v;
        }
        const double det = sw * see - se * se;
        if (std::fabs(det) < 1e-12) {
            f.reason = "flat";
            return f;
        }
        x[5] = (see * sv - se * sev) / det;
        x[6] = (sw * sev - se * sv) / det;
    }
    double lambda = 1e-3;
    std::vector<double> res;
    Eigen::Matrix<double, Eigen::Dynamic, 7> J;
    double cost = eval(x, &res, &J);
    bool converged = false;
    for (int it = 0; it < o.max_iterations; ++it) {
        Eigen::Map<const Eigen::VectorXd> r(res.data(), static_cast<Eigen::Index>(res.size()));
        const Eigen::Matrix<double, 7, 7> JtJ = J.transpose() * J;
        const Eigen::Matrix<double, 7, 1> g = J.transpose() * r;
        Eigen::Matrix<double, 7, 7> M = JtJ;
        for (int d = 0; d < 7; ++d) M(d, d) += lambda * std::max(1e-9, JtJ(d, d));
        const Eigen::Matrix<double, 7, 1> step = M.ldlt().solve(-g);
        double xn[7];
        for (int d = 0; d < 7; ++d) xn[d] = x[d] + step(d);
        xn[4] = std::max(0.15, xn[4]);
        std::vector<double> rn;
        Eigen::Matrix<double, Eigen::Dynamic, 7> Jn;
        const double cn = eval(xn, &rn, &Jn);
        if (cn < cost) {
            std::copy(xn, xn + 7, x);
            res.swap(rn);
            J.swap(Jn);
            const bool small = std::hypot(step(0), step(1)) < 1e-4;
            cost = cn;
            lambda = std::max(1e-9, lambda * 0.3);
            if (small) {
                converged = true;
                break;
            }
        } else {
            lambda *= 10;
            if (lambda > 1e8) {
                converged = true;  // no further decrease: at the minimum
                break;
            }
        }
    }
    f.p = Vec2(x[0], x[1]);
    f.blur_px = x[4];
    f.contrast = std::fabs(x[6]);
    double wsum = 0;
    for (const Px& p : px) wsum += p.w;
    f.residual = std::sqrt(cost / std::max(1e-12, wsum)) / std::max(1e-6, f.contrast);
    const double dof = std::max(1.0, static_cast<double>(px.size()) - 7.0);
    const double sigma2 = cost / dof;
    const Eigen::Matrix<double, 7, 7> JtJ = J.transpose() * J;
    const Eigen::Matrix<double, 7, 7> C = JtJ.inverse() * sigma2;
    f.cov = C.block<2, 2>(0, 0);
    // Gates.
    const Vec2 n1(std::cos(x[2]), std::sin(x[2])), n2(std::cos(x[3]), std::sin(x[3]));
    const double moved_modules = (Ainv * (f.p - p_init)).norm() / module_m;
    if (!converged || !std::isfinite(f.p.x())) f.reason = "diverged";
    else if (moved_modules > o.max_move_modules) f.reason = "moved";
    else if (std::fabs(n1.dot(n1i)) < 0.8 || std::fabs(n2.dot(n2i)) < 0.8) f.reason = "angles";
    else if (f.contrast < o.min_contrast) f.reason = "contrast";
    // Polarity: the black quadrants must be where the board has them (sign of a per the
    // initial orientation of the normals).
    else if ((x[6] * (n1.dot(n1i) > 0 ? 1 : -1) * (n2.dot(n2i) > 0 ? 1 : -1)) * black_xy > 0) f.reason = "polarity";
    else if (f.blur_px > 0.45 * short_px) f.reason = "blur";
    else if (f.residual > o.max_residual) f.reason = "residual";
    else if (!(f.cov(0, 0) >= 0 && f.cov(1, 1) >= 0)) f.reason = "covariance";
    else f.ok = true;
    return f;
}

namespace {

// Board -> image Jacobian (px per m) of a homography at board point (x, y).
Eigen::Matrix2d HomographyJacobian(const cv::Mat& H, double x, double y) {
    const double u = H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2);
    const double v = H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2);
    const double w = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);
    Eigen::Matrix2d J;
    for (int c = 0; c < 2; ++c) {
        J(0, c) = (H.at<double>(0, c) * w - u * H.at<double>(2, c)) / (w * w);
        J(1, c) = (H.at<double>(1, c) * w - v * H.at<double>(2, c)) / (w * w);
    }
    return J;
}

// Saddle polarity: +1 if the board quadrants (+x, +y) / (-x, -y) around the point are black.
int BlackXYImpl(const BoardSpec& b, int pid) {
    if (b.type == BoardType::AprilGrid) return (pid % 4) % 2 == 0 ? 1 : -1;  // BL, TR: tag / spacer diagonal
    const int i = pid % (b.cols - 1), j = pid / (b.cols - 1);
    return b.SquareBlack(j, i + 1) ? 1 : -1;  // the square up-right of corner (i, j) (rows from the top)
}

// Module (smallest printed cell) around a saddle, in board metres: the scale the window is sized in.
double SaddleModuleImpl(const BoardSpec& b) {
    if (b.type == BoardType::AprilGrid) {
        const int tw = b.family == "tag16h5" ? 8 : (b.family == "tag25h9" || b.family == "tag25h7") ? 9 : 10;
        return b.tag_size / tw;
    }
    if (b.type == BoardType::Charuco) return 0.5 * (b.square_size - b.marker_size) / 1.5;  // window cut = the gap
    return b.square_size / 3.0;
}

// ---------------------------------------------------------------------------------- AprilGrid
void DetectAprilGrids(const cv::Mat& gray, const cv::Mat& lin, const std::vector<BoardSpec>& boards,
                      const DetectOptions& opts, std::vector<BoardDetection>& out,
                      const std::map<std::string, std::vector<RawTag>>* given = nullptr) {
    std::set<std::string> families;
    for (const BoardSpec& b : boards) {
        if (b.type == BoardType::AprilGrid) families.insert(b.family);
    }
    for (const std::string& fam : families) {
        std::set<int> seen, dup;
        std::vector<RawTag> tags;
        if (given) {
            auto it = given->find(fam);
            if (it != given->end()) tags = it->second;
        } else {
            tags = DetectTags(gray, fam, opts.apriltag);
        }
        for (const RawTag& t : tags) {
            if (!seen.insert(t.id).second) dup.insert(t.id);
        }
        for (const RawTag& t : tags) {
            if (dup.count(t.id)) continue;  // an id seen twice in one frame: ambiguous, dropped
            for (size_t bi = 0; bi < boards.size(); ++bi) {
                const BoardSpec& b = boards[bi];
                if (b.type != BoardType::AprilGrid || b.family != fam || !b.HasId(t.id)) continue;
                std::array<Vec2, 4> c = t.p;
                std::array<bool, 4> keep = {true, true, true, true};
                std::array<Eigen::Matrix2d, 4> cov;
                for (auto& m : cov) m.setZero();
                if (opts.subpixel) {
                    // Every Kalibr tag corner is an X-junction (tag + spacer square). cornerSubPix
                    // (window: a tenth of the shortest tag edge) as the start, then the X-junction
                    // model fit. A corner whose refinement fails is dropped, never kept raw.
                    double shortest = 1e9;
                    for (int k = 0; k < 4; ++k) shortest = std::min(shortest, (c[k] - c[(k + 1) % 4]).norm());
                    const int half = std::max(3, std::min(9, static_cast<int>(std::lround(shortest / 10.0))));
                    std::vector<Vec2> pts(c.begin(), c.end());
                    std::vector<bool> ok;
                    RefineSaddles(gray, pts, std::vector<int>(4, half),
                                  std::vector<double>(4, opts.max_subpixel_move * shortest), ok);
                    // Local board -> image map from the tag's own corners.
                    const std::array<Vec3, 4> obj = b.TagCorners(t.id);
                    std::vector<cv::Point2f> src, dst;
                    for (int k = 0; k < 4; ++k) {
                        src.emplace_back(static_cast<float>(obj[k].x()), static_cast<float>(obj[k].y()));
                        dst.emplace_back(static_cast<float>(t.p[k].x()), static_cast<float>(t.p[k].y()));
                    }
                    const cv::Mat H = cv::getPerspectiveTransform(src, dst);
                    for (int k = 0; k < 4; ++k) {
                        keep[k] = ok[k];
                        if (opts.refine != "xfit") {
                            c[k] = pts[k];
                            continue;
                        }
                        // The model fit starts from cornerSubPix, or from the detector's corner
                        // where cornerSubPix wandered off (small or grazing tags).
                        const Vec2 start = ok[k] ? pts[k] : t.p[k];
                        const XJunctionFit f = FitXJunction(lin, start, HomographyJacobian(H, obj[k].x(), obj[k].y()),
                                                            SaddleModuleImpl(b), BlackXYImpl(b, 4 * (t.id - b.first_id) + k),
                                                            opts.xfit);
                        keep[k] = f.ok;
                        c[k] = f.ok ? f.p : pts[k];
                        cov[k] = f.cov;
                    }
                }
                if (std::getenv("ROBOCAL_DETECT_DEBUG")) {
                    static std::map<std::string, int> cnt;
#pragma omp critical
                    {
                        for (int k = 0; k < 4; ++k) cnt[keep[k] ? "kept" : "dropped_refine"]++;
                        if (++cnt["tags"] % 500 == 0) {
                            for (const auto& kv : cnt) std::fprintf(stderr, " %s=%d", kv.first.c_str(), kv.second);
                            std::fprintf(stderr, "\n");
                        }
                    }
                }
                BoardDetection& d = out[bi];
                d.ids.push_back(t.id);
                d.corners.push_back(c);
                for (int k = 0; k < 4; ++k) {
                    if (!keep[k]) continue;
                    d.pids.push_back(4 * (t.id - b.first_id) + k);
                    d.pts.push_back(c[k]);
                    d.covs.push_back(cov[k]);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------ ChArUco
}  // namespace

int ArucoDictionaryCode(const std::string& name) {
    static const std::map<std::string, int> codes = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},       {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},     {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},       {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},     {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},       {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},     {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},       {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},     {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
        {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5}, {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10}, {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
    };
    auto it = codes.find(name);
    if (it == codes.end()) throw std::runtime_error("unknown ArUco dictionary '" + name + "'");
    return it->second;
}

namespace {

void DetectCharucos(const cv::Mat& gray, const cv::Mat& lin, const std::vector<BoardSpec>& boards,
                    const DetectOptions& opts, std::vector<BoardDetection>& out) {
    std::set<std::string> dicts;
    for (const BoardSpec& b : boards) {
        if (b.type == BoardType::Charuco) dicts.insert(b.family);
    }
    for (const std::string& dname : dicts) {
        const cv::Ptr<cv::aruco::Dictionary> dict = CalicoGetDictionary(ArucoDictionaryCode(dname));
        cv::Ptr<cv::aruco::DetectorParameters> params = CalicoCreateDetectorParams();
        // Markers only give ids and a local homography; the chessboard corners are refined.
        params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
        params->adaptiveThreshWinSizeMin = 3;
        params->adaptiveThreshWinSizeMax = 63;
        params->adaptiveThreshWinSizeStep = 10;
        params->minMarkerPerimeterRate = 0.01;  // small markers far off-axis on a 1920 px image
        params->polygonalApproxAccuracyRate = 0.06;  // fisheye: marker edges are slightly curved
        params->perspectiveRemovePixelPerCell = 8;
        std::vector<std::vector<cv::Point2f>> mc, rejected;
        std::vector<int> mid;
        CalicoDetectMarkers(gray, dict, mc, mid, params, rejected);
        std::map<int, int> count;
        for (int id : mid) ++count[id];
        for (size_t bi = 0; bi < boards.size(); ++bi) {
            const BoardSpec& b = boards[bi];
            if (b.type != BoardType::Charuco || b.family != dname) continue;
            // Marker homographies (board xy -> image) by square.
            std::map<std::pair<int, int>, cv::Mat> H;
            for (size_t k = 0; k < mid.size(); ++k) {
                int r, c;
                if (count[mid[k]] != 1 || !b.MarkerSquare(mid[k], r, c)) continue;
                const std::array<Vec3, 4> obj = b.MarkerCorners(mid[k]);
                std::vector<cv::Point2f> src, dst;
                for (int q = 0; q < 4; ++q) {
                    src.emplace_back(static_cast<float>(obj[q].x()), static_cast<float>(obj[q].y()));
                    dst.push_back(mc[k][q]);
                }
                H[{r, c}] = cv::getPerspectiveTransform(src, dst);
            }
            if (static_cast<int>(H.size()) < opts.min_tags_per_board) continue;
            auto apply = [](const cv::Mat& h, double x, double y) {
                const double w = h.at<double>(2, 0) * x + h.at<double>(2, 1) * y + h.at<double>(2, 2);
                return Vec2((h.at<double>(0, 0) * x + h.at<double>(0, 1) * y + h.at<double>(0, 2)) / w,
                            (h.at<double>(1, 0) * x + h.at<double>(1, 1) * y + h.at<double>(1, 2)) / w);
            };
            std::vector<int> pids, half;
            std::vector<Vec2> pts;
            std::vector<double> scale;
            std::vector<Eigen::Matrix2d> jac;
            const double sq = b.square_size;
            for (int j = 0; j + 1 < b.rows; ++j) {
                for (int i = 0; i + 1 < b.cols; ++i) {
                    const int pid = j * (b.cols - 1) + i;
                    const Vec3 X = b.Point(pid);
                    // Predict the corner from each adjacent marker (local homographies: valid
                    // under strong lens distortion, unlike one homography for the whole board).
                    std::vector<Vec2> pred;
                    double s_px = 0;
                    Eigen::Matrix2d Jm = Eigen::Matrix2d::Zero();
                    for (int dr = 0; dr < 2; ++dr) {
                        for (int dc = 0; dc < 2; ++dc) {
                            auto it = H.find({j + dr, i + dc});
                            if (it == H.end()) continue;
                            pred.push_back(apply(it->second, X.x(), X.y()));
                            Jm += HomographyJacobian(it->second, X.x(), X.y());
                            s_px += 0.5 * ((apply(it->second, X.x() + sq, X.y()) - pred.back()).norm() +
                                           (apply(it->second, X.x(), X.y() + sq) - pred.back()).norm());
                        }
                    }
                    if (pred.empty()) continue;
                    s_px /= pred.size();
                    Jm /= static_cast<double>(pred.size());
                    Vec2 m = Vec2::Zero();
                    for (const Vec2& p : pred) m += p / pred.size();
                    bool consistent = true;
                    for (const Vec2& p : pred) consistent = consistent && (p - m).norm() < 0.12 * s_px;
                    if (!consistent || s_px < 6.0) continue;
                    // Window: inside the four squares, short of the markers' nearest corners.
                    const double gap_px = 0.5 * (sq - b.marker_size) / sq * s_px;
                    half.push_back(std::max(2, std::min(12, static_cast<int>(std::floor(gap_px - 0.5)))));
                    pids.push_back(pid);
                    pts.push_back(m);
                    scale.push_back(s_px);
                    jac.push_back(Jm);
                }
            }
            std::vector<bool> ok(pts.size(), true);
            std::vector<Eigen::Matrix2d> cov(pts.size(), Eigen::Matrix2d::Zero());
            if (opts.subpixel) {
                std::vector<double> move;
                for (double sc : scale) move.push_back(std::max(1.0, opts.max_subpixel_move * sc));
                const std::vector<Vec2> predicted = pts;
                RefineSaddles(gray, pts, half, move, ok);
                if (opts.refine == "xfit") {
                    for (size_t k = 0; k < pts.size(); ++k) {
                        const Vec2 start = ok[k] ? pts[k] : predicted[k];
                        const XJunctionFit f = FitXJunction(lin, start, jac[k], SaddleModuleImpl(b), BlackXYImpl(b, pids[k]), opts.xfit);
                        ok[k] = f.ok;
                        pts[k] = f.p;
                        cov[k] = f.cov;
                    }
                }
            }
            for (size_t k = 0; k < pts.size(); ++k) {
                if (!ok[k]) continue;
                out[bi].pids.push_back(pids[k]);
                out[bi].pts.push_back(pts[k]);
                out[bi].covs.push_back(cov[k]);
            }
        }
    }
}

// ------------------------------------------------------------------------------- checkerboard
void DetectCheckerboards(const cv::Mat& gray, const cv::Mat& lin, const std::vector<BoardSpec>& boards,
                         const DetectOptions& opts, std::vector<BoardDetection>& out) {
    for (size_t bi = 0; bi < boards.size(); ++bi) {
        const BoardSpec& b = boards[bi];
        if (b.type != BoardType::Checkerboard) continue;
        const int nx = b.cols - 1, ny = b.rows - 1;
        std::vector<cv::Point2f> c;
        if (!cv::findChessboardCornersSB(gray, cv::Size(nx, ny), c,
                                         cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY | cv::CALIB_CB_NORMALIZE_IMAGE) ||
            static_cast<int>(c.size()) != nx * ny) {
            continue;
        }
        // Orientation: the square outside corner 0 must be the black top-left square (odd x even
        // boards: the 180 deg turn puts a white square there).
        auto sample = [&](const cv::Point2f& p) {
            const int x = cvRound(p.x), y = cvRound(p.y);
            if (x < 1 || y < 1 || x >= gray.cols - 1 || y >= gray.rows - 1) return -1.0;
            return cv::mean(gray(cv::Rect(x - 1, y - 1, 3, 3)))[0];
        };
        const cv::Point2f ex = c[1] - c[0], ey = c[nx] - c[0];
        const double out_v = sample(c[0] - 0.5f * (ex + ey)), side = sample(c[0] + 0.5f * ex - 0.5f * ey);
        if (out_v < 0 || side < 0) {
            // Outer square not visible: decide from the far end instead.
            const cv::Point2f fx = c[nx * ny - 2] - c[nx * ny - 1], fy = c[nx * ny - 1 - nx] - c[nx * ny - 1];
            const double o2 = sample(c[nx * ny - 1] - 0.5f * (fx + fy)), s2 = sample(c[nx * ny - 1] + 0.5f * fx - 0.5f * fy);
            if (o2 < 0 || s2 < 0) continue;
            // The far corner's outer square is white on an odd x even board.
            if (o2 < s2) std::reverse(c.begin(), c.end());
        } else if (out_v > side) {
            std::reverse(c.begin(), c.end());
        }
        const cv::Point2f ax = c[1] - c[0], ay = c[nx] - c[0];
        if (ax.x * ay.y - ax.y * ay.x <= 0) continue;  // mirrored ordering: not a front view, dropped
        for (int k = 0; k < nx * ny; ++k) {
            Vec2 p(c[k].x, c[k].y);
            Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
            if (opts.subpixel && opts.refine == "xfit") {
                // Local board -> image map from the neighbouring corners (rows run down = board -y).
                const int i = k % nx, j = k / nx;
                const int il = std::max(0, i - 1), ir = std::min(nx - 1, i + 1), ju = std::max(0, j - 1), jd = std::min(ny - 1, j + 1);
                Eigen::Matrix2d A;
                const cv::Point2f dx = (c[j * nx + ir] - c[j * nx + il]) * (1.0f / ((ir - il) * b.square_size));
                const cv::Point2f dy = (c[ju * nx + i] - c[jd * nx + i]) * (1.0f / ((jd - ju) * b.square_size));
                A << dx.x, dy.x, dx.y, dy.y;
                const XJunctionFit f = FitXJunction(lin, p, A, SaddleModuleImpl(b), BlackXYImpl(b, k), opts.xfit);
                if (!f.ok) continue;
                p = f.p;
                cov = f.cov;
            }
            out[bi].pids.push_back(k);
            out[bi].pts.push_back(p);
            out[bi].covs.push_back(cov);
        }
    }
}

// Kalibr AprilGrid: the four saddles around one spacer square belong to four tags and form a
// small parallelogram (affine-exact at this scale): p_bl(tag i+1, j+1) + p_tr(tag i, j) ==
// p_br(tag i, j+1) + p_tl(tag i+1, j). A point that breaks most of the tests it is in goes.
void CheckSpacers(const BoardSpec& b, BoardDetection& d) {
    std::map<int, size_t> at;
    for (size_t k = 0; k < d.pids.size(); ++k) at[d.pids[k]] = k;
    std::map<int, int> tests, fails;
    auto pid = [&](int col, int row, int corner) { return 4 * (row * b.cols + col) + corner; };
    for (int row = 0; row + 1 < b.rows; ++row) {
        for (int col = 0; col + 1 < b.cols; ++col) {
            // Spacer between tags (col,row) [its TR], (col+1,row) [TL], (col,row+1) [BR], (col+1,row+1) [BL].
            const int q[4] = {pid(col, row, 2), pid(col + 1, row, 3), pid(col, row + 1, 1), pid(col + 1, row + 1, 0)};
            bool all = true;
            for (int v : q) all = all && at.count(v);
            if (!all) continue;
            const Vec2 e = d.pts[at[q[0]]] + d.pts[at[q[3]]] - d.pts[at[q[1]]] - d.pts[at[q[2]]];
            const bool bad = e.norm() > 0.3;
            for (int v : q) {
                ++tests[v];
                fails[v] += bad;
            }
        }
    }
    if (std::getenv("ROBOCAL_DETECT_DEBUG")) {
        int bad = 0, tot = 0;
        for (const auto& kv : tests) {
            tot += kv.second > 0;
            bad += kv.second > 0 && 2 * fails[kv.first] > kv.second;
        }
        static int sb = 0, st = 0;
#pragma omp critical
        {
            sb += bad;
            st += tot;
            std::fprintf(stderr, "[spacers] dropped %d of %d tested points (total %d / %d)\n", bad, tot, sb, st);
        }
    }
    BoardDetection kept = d;
    kept.pids.clear();
    kept.pts.clear();
    kept.covs.clear();
    for (size_t k = 0; k < d.pids.size(); ++k) {
        const int v = d.pids[k];
        if (tests[v] > 0 && 2 * fails[v] > tests[v]) continue;
        kept.pids.push_back(v);
        kept.pts.push_back(d.pts[k]);
        if (k < d.covs.size()) kept.covs.push_back(d.covs[k]);
    }
    d = kept;
}

}  // namespace

// ------------------------------------------------------------------------- rectified re-detection
RectifiedDetector::RectifiedDetector(const CameraIntrinsics& intr, int size, double fov_deg) : intr_(intr), size_(size) {
    f_ = 0.5 * size / std::tan(0.5 * fov_deg * M_PI / 180.0);
    // Virtual pinhole views: the centre, a ring at 50 deg and a ring at 80 deg off axis (the
    // image edge of a 200 deg lens), overlapping so every tag is near the middle of one view.
    std::vector<std::pair<double, double>> dirs = {{0.0, 0.0}};
    for (int k = 0; k < 6; ++k) dirs.push_back({50.0, 60.0 * k});
    for (int k = 0; k < 10; ++k) dirs.push_back({80.0, 36.0 * k + 18.0});
    for (const auto& d : dirs) {
        const double th = d.first * M_PI / 180.0, ph = d.second * M_PI / 180.0;
        const Vec3 z(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
        Vec3 x = Vec3::UnitY().cross(z);
        if (x.norm() < 1e-6) x = Vec3::UnitX();
        x.normalize();
        View v;
        v.R.col(0) = x;
        v.R.col(1) = z.cross(x);
        v.R.col(2) = z;
        v.mapx.create(size, size, CV_32F);
        v.mapy.create(size, size, CV_32F);
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                const Vec3 b = v.R * Vec3((c - 0.5 * size) / f_, (r - 0.5 * size) / f_, 1.0);
                double uv[2] = {-1, -1};
                if (!ProjectDirection(intr.model, intr.params.data(), Eigen::Matrix<double, 3, 1>(b), uv) ||
                    uv[0] < 0 || uv[1] < 0 || uv[0] > intr.width - 1 || uv[1] > intr.height - 1) {
                    uv[0] = uv[1] = -1;
                }
                v.mapx.at<float>(r, c) = static_cast<float>(uv[0]);
                v.mapy.at<float>(r, c) = static_cast<float>(uv[1]);
            }
        }
        views_.push_back(v);
    }
}

std::vector<BoardDetection> RectifiedDetector::Detect(const cv::Mat& gray_in, const std::vector<BoardSpec>& boards,
                                                      const DetectOptions& opts) const {
    cv::Mat gray;
    if (gray_in.channels() == 3) cv::cvtColor(gray_in, gray, cv::COLOR_BGR2GRAY);
    else gray = gray_in;
    return DetectGiven(gray, boards, opts, FindTags(gray, boards, opts));
}

TagsByFamily RectifiedDetector::FindTags(const cv::Mat& gray_in, const std::vector<BoardSpec>& boards,
                                         const DetectOptions& opts) const {
    cv::Mat gray;
    if (gray_in.channels() == 3) cv::cvtColor(gray_in, gray, cv::COLOR_BGR2GRAY);
    else gray = gray_in;
    std::set<std::string> families;
    for (const BoardSpec& b : boards) {
        if (b.type == BoardType::AprilGrid) families.insert(b.family);
    }
    std::map<std::string, std::vector<RawTag>> given;
    for (const std::string& fam : families) {
        std::map<int, std::pair<RawTag, double>> best;  // id -> tag in original pixels, distance to its view centre
        std::map<int, int> views_seen;
        for (const View& v : views_) {
            cv::Mat virt;
            cv::remap(gray, virt, v.mapx, v.mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            for (const RawTag& t : DetectTags(virt, fam, opts.apriltag)) {
                RawTag o;
                o.id = t.id;
                Vec2 mid = Vec2::Zero();
                bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    mid += t.p[k] / 4.0;
                    const Vec3 b = v.R * Vec3((t.p[k].x() - 0.5 * size_) / f_, (t.p[k].y() - 0.5 * size_) / f_, 1.0);
                    double uv[2];
                    ok = ok && ProjectDirection(intr_.model, intr_.params.data(), Eigen::Matrix<double, 3, 1>(b), uv);
                    o.p[k] = Vec2(uv[0], uv[1]);
                }
                if (!ok) continue;
                const double dc = (mid - Vec2(0.5 * size_, 0.5 * size_)).norm();
                auto it = best.find(t.id);
                if (it == best.end() || dc < it->second.second) best[t.id] = {o, dc};
            }
        }
        for (const auto& kv : best) given[fam].push_back(kv.second.first);
    }
    return given;
}

std::vector<BoardDetection> RectifiedDetector::DetectGiven(const cv::Mat& gray_in, const std::vector<BoardSpec>& boards,
                                                           const DetectOptions& opts, const TagsByFamily& given) {
    cv::Mat gray;
    if (gray_in.channels() == 3) cv::cvtColor(gray_in, gray, cv::COLOR_BGR2GRAY);
    else gray = gray_in;
    std::vector<BoardDetection> out(boards.size());
    for (size_t i = 0; i < boards.size(); ++i) out[i].board = static_cast<int>(i);
    const cv::Mat lin = Linearize(gray, opts.gamma);
    DetectAprilGrids(gray, lin, boards, opts, out, &given);
    std::vector<BoardDetection> kept;
    for (size_t i = 0; i < out.size(); ++i) {
        if (boards[i].type != BoardType::AprilGrid) continue;
        if (static_cast<int>(out[i].ids.size()) < opts.min_tags_per_board) continue;
        if (static_cast<int>(out[i].pts.size()) < opts.min_points_per_board) continue;
        kept.push_back(std::move(out[i]));
    }
    return kept;
}

cv::Mat LinearizeGray(const cv::Mat& gray, double gamma) { return Linearize(gray, gamma); }

std::vector<Iso3> BoardPoseCandidates(const std::vector<Vec3>& bearings, const std::vector<Vec3>& objs) {
    std::vector<Iso3> out;
    if (bearings.size() < 4 || bearings.size() != objs.size()) return out;
    Vec3 m = Vec3::Zero();
    for (const Vec3& b : bearings) m += b;
    m.normalize();
    // Virtual camera: z along m. R_cv maps virtual -> camera.
    Vec3 x = Vec3::UnitY().cross(m);
    if (x.norm() < 1e-6) x = Vec3::UnitX().cross(m);
    x.normalize();
    Mat3 R_cv;
    R_cv.col(0) = x;
    R_cv.col(1) = m.cross(x);
    R_cv.col(2) = m;
    std::vector<cv::Point3f> o;
    std::vector<cv::Point2f> p;
    for (size_t i = 0; i < bearings.size(); ++i) {
        const Vec3 bv = R_cv.transpose() * bearings[i];
        if (bv.z() < 0.05) continue;
        o.emplace_back(static_cast<float>(objs[i].x()), static_cast<float>(objs[i].y()), static_cast<float>(objs[i].z()));
        p.emplace_back(static_cast<float>(bv.x() / bv.z()), static_cast<float>(bv.y() / bv.z()));
    }
    if (o.size() < 4) return out;
    std::vector<cv::Mat> rv, tv;
    std::vector<double> err;
    try {
        cv::solvePnPGeneric(o, p, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), rv, tv, false, cv::SOLVEPNP_IPPE,
                            cv::noArray(), cv::noArray(), err);
    } catch (const cv::Exception&) {
        return out;
    }
    std::vector<size_t> order(rv.size());
    for (size_t s = 0; s < order.size(); ++s) order[s] = s;
    if (err.size() == rv.size()) std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return err[a] < err[b]; });
    Iso3 T_cv = Iso3::Identity();
    T_cv.linear() = R_cv;
    for (size_t s : order) {
        cv::Mat R;
        cv::Rodrigues(rv[s], R);
        Iso3 T_vb = Iso3::Identity();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) T_vb.linear()(r, c) = R.at<double>(r, c);
            T_vb.translation()[r] = tv[s].at<double>(r);
        }
        out.push_back(T_cv * T_vb);
    }
    return out;
}
int SaddlePolarity(const BoardSpec& b, int pid) { return BlackXYImpl(b, pid); }
double SaddleModule(const BoardSpec& b) { return SaddleModuleImpl(b); }

bool RefinePredictedSaddle(const cv::Mat& gray, const cv::Mat& lin, const BoardSpec& b, int pid, Vec2& p,
                           const Eigen::Matrix2d& A, const DetectOptions& opts, double max_move_modules,
                           Eigen::Matrix2d* cov) {
    const double module = SaddleModuleImpl(b);
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(A);
    const double module_px = svd.singularValues()(1) * module;  // short axis
    if (module_px < 1.2) return false;
    const Vec2 pred = p;
    std::vector<Vec2> pts = {p};
    std::vector<bool> ok;
    const int half = std::max(2, std::min(9, static_cast<int>(std::lround(1.2 * module_px))));
    RefineSaddles(gray, pts, {half}, {max_move_modules * svd.singularValues()(0) * module}, ok);
    if (!ok[0]) return false;
    XJunctionOptions xo = opts.xfit;
    xo.max_move_modules = max_move_modules;
    const XJunctionFit f = FitXJunction(lin, pts[0], A, module, BlackXYImpl(b, pid), xo);
    if (!f.ok || (A.inverse() * (f.p - pred)).norm() > max_move_modules * module) return false;
    p = f.p;
    if (cov) *cov = f.cov;
    return true;
}

std::vector<BoardDetection> DetectBoards(const cv::Mat& gray_in, const std::vector<BoardSpec>& boards,
                                         const DetectOptions& opts) {
    cv::Mat gray;
    if (gray_in.channels() == 3) cv::cvtColor(gray_in, gray, cv::COLOR_BGR2GRAY);
    else gray = gray_in;
    std::vector<BoardDetection> out(boards.size());
    for (size_t i = 0; i < boards.size(); ++i) out[i].board = static_cast<int>(i);
    const cv::Mat lin = Linearize(gray, opts.gamma);
    DetectAprilGrids(gray, lin, boards, opts, out);
    DetectCharucos(gray, lin, boards, opts, out);
    DetectCheckerboards(gray, lin, boards, opts, out);
    if (opts.check_spacers) {
        for (size_t i = 0; i < out.size(); ++i) {
            if (boards[i].type == BoardType::AprilGrid) CheckSpacers(boards[i], out[i]);
        }
    }
    std::vector<BoardDetection> kept;
    for (size_t i = 0; i < out.size(); ++i) {
        BoardDetection& d = out[i];
        if (boards[i].type == BoardType::AprilGrid && static_cast<int>(d.ids.size()) < opts.min_tags_per_board) continue;
        if (static_cast<int>(d.pts.size()) < opts.min_points_per_board) continue;
        kept.push_back(std::move(d));
    }
    return kept;
}

}  // namespace robocal
