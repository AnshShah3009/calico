// Synthetic rendering for robocal tests and the end-to-end simulator: Kalibr-style AprilGrid
// prints (optionally bowed) seen through a camera model by ray casting, 2x2 supersampled.
#ifndef ROBOCAL_RENDER_HPP_
#define ROBOCAL_RENDER_HPP_

#include "robocal.hpp"
#include "../opencv-compat.hpp"

#include <apriltag/apriltag.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace robocal {
namespace sim {

constexpr double kRenderDeg = M_PI / 180.0;

// Kalibr-style AprilGrid print: 2-bit black border tags + black spacer squares at the corners.
struct BoardTexture {
    BoardSpec spec;
    Iso3 T_world_board;
    static double& DefaultRes() {
        static double r = 0.0002;
        return r;
    }
    double res = DefaultRes();  // m per texel
    double gloss = 0.0;   // laminated print: specular strength (RenderReal)
    double bow_x = 0.0;   // true print bow along x (m, centre vs edges), rendered
    double x0 = 0, y0 = 0;  // board coords of texel (0, rows-1) bottom-left
    cv::Mat tex;           // CV_32F, 0..1, row 0 = top (max y)

    // Laser / toner print defects, applied when the print is rendered. Values are print
    // reflectance in 0 (black) .. 1 (paper).
    struct Toner {
        bool on = false;
        double black = 0.08;            // toner black is not black (reflectance relative to paper)
        double banding = 0.03;          // +- toner density along the paper feed, ~8 mm period
        double dot_gain_m = 40e-6;      // black areas grow by this much
        double edge_rough_m = 20e-6;    // ragged edges (1 sigma)
        double paper_grain = 0.02;
        double specks_per_dm2 = 25;     // dark toner specks on the white
        double voids_per_dm2 = 15;      // pale dropouts in the black
        int smudges = 2;                // soft grey smudges (thumb, roller)
        double streak_prob = 0.5;       // one faint streak along the feed
        unsigned seed = 1;
    };
    static Toner& DefaultToner() {
        static Toner t;
        return t;
    }
    Toner toner = DefaultToner();

    void ApplyToner() {
        if (!toner.on || tex.empty()) return;
        std::mt19937 rng(toner.seed * 7919u + static_cast<unsigned>(spec.first_id) * 131u +
                         static_cast<unsigned>(std::hash<std::string>()(spec.name)));
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> N(0.0, 1.0);
        const int H = tex.rows, W = tex.cols;
        // Dot gain + ragged edges: blur the binary print, perturb, threshold above mid-grey.
        cv::Mat soft;
        const double sig_px = std::max(0.6, 2.0 * toner.edge_rough_m / res + 0.5);
        cv::GaussianBlur(tex, soft, cv::Size(0, 0), sig_px);
        cv::Mat rough(H, W, CV_32F);
        cv::randn(rough, 0.0, 1.0);
        cv::GaussianBlur(rough, rough, cv::Size(0, 0), std::max(0.5, 0.5 * toner.edge_rough_m / res + 0.5));
        const double gain_level = 0.5 + 0.5 * std::min(0.9, toner.dot_gain_m / (sig_px * res));
        cv::Mat out(H, W, CV_32F);
        for (int r = 0; r < H; ++r) {
            const double band = 1.0 + toner.banding * std::sin(2 * M_PI * r * res / 0.008 + 1.3);
            for (int c = 0; c < W; ++c) {
                const double v = soft.at<float>(r, c) + 0.08 * rough.at<float>(r, c);
                // smooth threshold over ~1 texel
                const double ink = std::clamp((gain_level - v) / 0.08 + 0.5, 0.0, 1.0);
                const double black = std::clamp(toner.black * band, 0.0, 1.0);
                out.at<float>(r, c) = static_cast<float>(1.0 - ink * (1.0 - black));
            }
        }
        // Paper grain.
        cv::Mat grain(H, W, CV_32F);
        cv::randn(grain, 0.0, toner.paper_grain);
        cv::GaussianBlur(grain, grain, cv::Size(0, 0), 1.5);
        out += grain.mul(out);
        const double area_dm2 = (W * res) * (H * res) * 100.0;
        auto dot = [&](double cx, double cy, double rad_m, double target, double strength) {
            const double rp = rad_m / res;
            const int r0 = std::max(0, static_cast<int>(cy - 3 * rp)), r1 = std::min(H - 1, static_cast<int>(cy + 3 * rp));
            const int c0 = std::max(0, static_cast<int>(cx - 3 * rp)), c1 = std::min(W - 1, static_cast<int>(cx + 3 * rp));
            for (int r = r0; r <= r1; ++r) {
                for (int c = c0; c <= c1; ++c) {
                    const double d2 = ((r - cy) * (r - cy) + (c - cx) * (c - cx)) / std::max(1e-9, rp * rp);
                    const double w = strength * std::exp(-0.5 * d2);
                    float& px = out.at<float>(r, c);
                    px = static_cast<float>((1 - w) * px + w * target);
                }
            }
        };
        const int n_specks = static_cast<int>(toner.specks_per_dm2 * area_dm2), n_voids = static_cast<int>(toner.voids_per_dm2 * area_dm2);
        for (int k = 0; k < n_specks; ++k) {  // on white paper only
            const double cx = U(rng) * W, cy = U(rng) * H;
            if (tex.at<float>(static_cast<int>(cy), static_cast<int>(cx)) < 0.5f) continue;
            dot(cx, cy, (30 + 120 * U(rng)) * 1e-6, toner.black, 0.4 + 0.5 * U(rng));
        }
        for (int k = 0; k < n_voids; ++k) {  // in the black only
            const double cx = U(rng) * W, cy = U(rng) * H;
            if (tex.at<float>(static_cast<int>(cy), static_cast<int>(cx)) > 0.5f) continue;
            dot(cx, cy, (40 + 150 * U(rng)) * 1e-6, 1.0, 0.3 + 0.5 * U(rng));
        }
        for (int k = 0; k < toner.smudges; ++k) {  // soft grey smudges: darken paper, little effect on black
            const double cx = U(rng) * W, cy = U(rng) * H, rad = (0.004 + 0.012 * U(rng)) / res;
            const double sx = rad * (0.6 + 0.8 * U(rng)), sy = rad * (0.6 + 0.8 * U(rng)), dark = 0.08 + 0.15 * U(rng);
            for (int r = std::max(0, static_cast<int>(cy - 3 * sy)); r < std::min(H, static_cast<int>(cy + 3 * sy)); ++r) {
                for (int c = std::max(0, static_cast<int>(cx - 3 * sx)); c < std::min(W, static_cast<int>(cx + 3 * sx)); ++c) {
                    const double g = std::exp(-0.5 * ((r - cy) * (r - cy) / (sy * sy) + (c - cx) * (c - cx) / (sx * sx)));
                    out.at<float>(r, c) *= static_cast<float>(1.0 - dark * g);
                }
            }
        }
        if (U(rng) < toner.streak_prob) {  // faint streak along the feed (rows)
            const double cx = U(rng) * W, half = (0.0002 + 0.0006 * U(rng)) / res, dark = 0.05 + 0.15 * U(rng);
            for (int r = 0; r < H; ++r) {
                for (int c = std::max(0, static_cast<int>(cx - 3 * half)); c < std::min(W, static_cast<int>(cx + 3 * half)); ++c) {
                    out.at<float>(r, c) *= static_cast<float>(1.0 - dark * std::exp(-0.5 * (c - cx) * (c - cx) / (half * half)));
                }
            }
        }
        cv::max(out, 0.0, out);
        cv::min(out, 1.0, out);
        tex = out;
    }

    // ChArUco / checkerboard print: black top-left square, markers in the white squares.
    void RenderChess() {
        const double sq = spec.square_size, margin = 0.6 * sq, Hb = spec.Height();
        x0 = -margin;
        y0 = -margin;
        const double w = spec.Width() + 2 * margin, h = Hb + 2 * margin;
        const int W = static_cast<int>(std::ceil(w / res)), H = static_cast<int>(std::ceil(h / res));
        tex = cv::Mat(H, W, CV_32F, cv::Scalar(1.0f));
        auto fill = [&](double xa, double ya, double xb, double yb, float val) {
            const int c0 = static_cast<int>(std::lround((xa - x0) / res)), c1 = static_cast<int>(std::lround((xb - x0) / res));
            const int r0 = static_cast<int>(std::lround((y0 + h - yb) / res)), r1 = static_cast<int>(std::lround((y0 + h - ya) / res));
            cv::rectangle(tex, cv::Rect(c0, r0, c1 - c0, r1 - r0), cv::Scalar(val), cv::FILLED);
        };
        for (int r = 0; r < spec.rows; ++r) {
            for (int c = 0; c < spec.cols; ++c) {
                if (spec.SquareBlack(r, c)) fill(c * sq, Hb - (r + 1) * sq, (c + 1) * sq, Hb - r * sq, 0.0f);
            }
        }
        if (spec.type != BoardType::Charuco) {
            ApplyToner();
            return;
        }
        const cv::Ptr<cv::aruco::Dictionary> dict = CalicoGetDictionary(ArucoDictionaryCode(spec.family));
        for (int id = spec.first_id; id < spec.first_id + spec.NumTags(); ++id) {
            const std::array<Vec3, 4> mc = spec.MarkerCorners(id);  // TL, TR, BR, BL
            const int n = dict->markerSize + 2;
            cv::Mat m;
#if CALICO_ARUCO_MODERN
            cv::aruco::generateImageMarker(*dict, id, n, m, 1);
#else
            cv::aruco::drawMarker(dict, id, n, m, 1);
#endif
            const double cell = spec.marker_size / n;
            for (int r = 0; r < n; ++r) {
                for (int c = 0; c < n; ++c) {
                    const double xa = mc[0].x() + c * cell, yb = mc[0].y() - r * cell;
                    fill(xa, yb - cell, xa + cell, yb, m.at<unsigned char>(r, c) ? 1.0f : 0.0f);
                }
            }
        }
        ApplyToner();
    }

    void Render(apriltag_family_t* fam) {
        if (spec.type != BoardType::AprilGrid) {
            RenderChess();
            return;
        }
        const double g = spec.tag_size * spec.tag_spacing;
        const double margin = 2 * g;
        x0 = -g - margin;
        y0 = -g - margin;
        const double w = spec.Width() + 2 * (g + margin), h = spec.Height() + 2 * (g + margin);
        const int W = static_cast<int>(std::ceil(w / res)), H = static_cast<int>(std::ceil(h / res));
        tex = cv::Mat(H, W, CV_32F, cv::Scalar(1.0f));
        auto fill = [&](double xa, double ya, double xb, double yb, float val) {  // board coords, y up
            const int c0 = static_cast<int>(std::lround((xa - x0) / res)), c1 = static_cast<int>(std::lround((xb - x0) / res));
            const int r0 = static_cast<int>(std::lround((y0 + h - yb) / res)), r1 = static_cast<int>(std::lround((y0 + h - ya) / res));
            cv::rectangle(tex, cv::Rect(c0, r0, c1 - c0, r1 - r0), cv::Scalar(val), cv::FILLED);
        };
        const double P = spec.Pitch();
        for (int i = 0; i <= spec.cols; ++i) {
            for (int j = 0; j <= spec.rows; ++j) fill(i * P - g, j * P - g, i * P, j * P, 0.0f);
        }
        for (int k = 0; k < spec.NumTags(); ++k) {
            const int id = spec.first_id + k;
            const double tx = (k % spec.cols) * P, ty = (k / spec.cols) * P;
            const int tw = static_cast<int>(fam->total_width);
            const int bs = (fam->total_width - fam->width_at_border) / 2;
            std::vector<unsigned char> cells(tw * tw, 0);
            for (unsigned b = 0; b < fam->nbits; ++b) {
                if (fam->codes[id] & (uint64_t(1) << (fam->nbits - b - 1))) {
                    cells[(fam->bit_y[b] + bs) * tw + fam->bit_x[b] + bs] = 255;
                }
            }
            const double cell = spec.tag_size / tw;
            for (int r = 0; r < tw; ++r) {
                for (int c = 0; c < tw; ++c) {  // tag image row 0 = top of the tag = max y
                    fill(tx + c * cell, ty + (tw - 1 - r) * cell, tx + (c + 1) * cell, ty + (tw - r) * cell,
                         cells[r * tw + c] ? 1.0f : 0.0f);
                }
            }
        }
        ApplyToner();
    }

    // Board-frame point -> intensity, or -1 outside the print.
    float Sample(double x, double y) const {
        const double cf = (x - x0) / res - 0.5, rf = (y0 + tex.rows * res - y) / res - 0.5;
        const int c = static_cast<int>(std::floor(cf)), r = static_cast<int>(std::floor(rf));
        if (c < 0 || r < 0 || c + 1 >= tex.cols || r + 1 >= tex.rows) return -1.0f;
        const float a = static_cast<float>(cf - c), b = static_cast<float>(rf - r);
        return (1 - a) * (1 - b) * tex.at<float>(r, c) + a * (1 - b) * tex.at<float>(r, c + 1) +
               (1 - a) * b * tex.at<float>(r + 1, c) + a * b * tex.at<float>(r + 1, c + 1);
    }
};

inline Iso3 LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 z = (target - eye).normalized();
    Vec3 x = up.cross(z);
    if (x.norm() < 1e-6) x = Vec3::UnitX().cross(z);
    x.normalize();
    Iso3 T = Iso3::Identity();
    T.linear().col(0) = x;
    T.linear().col(1) = z.cross(x);
    T.linear().col(2) = z;
    T.translation() = eye;
    return T;
}

// A robot link: segment a-b with radius r (shade 0..1, material id >= 2 for figures).
struct Capsule {
    Vec3 a, b;
    double r = 0.03;
    double shade = 0.75;
    int material = 2;
};

struct SceneExtras {
    std::vector<Capsule> capsules;
    bool table = false;
    double table_z = 0.0;
    Vec2 table_min = Vec2(-0.45, -0.85), table_max = Vec2(1.05, 0.85);
};

// Ray (unit d) vs capsule; nearest t > 0 and the outward normal.
inline bool HitCapsule(const Vec3& o, const Vec3& d, const Capsule& c, double& t, Vec3& n) {
    const Vec3 ba = c.b - c.a, oa = o - c.a;
    const double baba = ba.dot(ba), bard = ba.dot(d), baoa = ba.dot(oa), rdoa = d.dot(oa), oaoa = oa.dot(oa);
    const double A = baba - bard * bard, B = baba * rdoa - baoa * bard, C = baba * oaoa - baoa * baoa - c.r * c.r * baba;
    double h = B * B - A * C;
    if (h >= 0.0 && A > 1e-12) {
        t = (-B - std::sqrt(h)) / A;
        const double y = baoa + t * bard;
        if (t > 0 && y > 0 && y < baba) {
            n = (oa + t * d - ba * (y / baba)) / c.r;
            return true;
        }
        const Vec3 oc = (y <= 0) ? oa : Vec3(o - c.b);
        const double b2 = d.dot(oc), c2 = oc.dot(oc) - c.r * c.r;
        h = b2 * b2 - c2;
        if (h > 0) {
            t = -b2 - std::sqrt(h);
            if (t > 0) {
                n = (oc + t * d) / c.r;
                return true;
            }
        }
    }
    return false;
}

// Optics of a real (fixed-focus, wide-angle) lens, beyond its projection model.
struct Lens {
    double fov_deg = 0.0;       // image circle (field stop); rays past fov/2 are black. 0: none
    double vignette = 0.0;      // relative illumination loss at the circle edge (0..1)
    double base_blur_px = 0.5;  // optical blur sigma on axis
    double edge_blur_px = 0.0;  // extra blur sigma at the circle edge (MTF fall-off)
    double focus_m = 0.0;       // distance the lens is focused at (0: no defocus model)
    double defocus_px = 0.0;    // blur-circle diameter (px) per dioptre of defocus = aperture * f_px
    double decenter_px = 0.0;   // unmodelled decentering distortion at the circle edge (px)
    double decenter_dir = 0.6;  // its direction (rad)
    double pupil_e1_m = 0.0;    // non-central lens: rays start at z0 = e1 (theta / 90 deg)^2 on the axis
    double PupilZ(double th) const { return pupil_e1_m * 4.0 / (M_PI * M_PI) * th * th; }

    Lens Scaled(double k) const {  // the same lens on a sensor resampled by k
        Lens l = *this;
        l.base_blur_px *= k;
        l.edge_blur_px *= k;
        l.defocus_px *= k;
        l.decenter_px *= k;
        return l;
    }
};

// Scene lighting: one lamp + ambient, shadows of the robot links.
struct Lighting {
    Vec3 lamp = Vec3(0.35, -0.45, 1.30);
    double ambient = 0.40;
    double direct = 0.80;   // lamp irradiance at range_m, facing it
    double range_m = 1.3;
    bool shadows = true;
    double background = 0.20;  // the room behind the cell
};

// One exposure.
struct Shot {
    Vec3 v = Vec3::Zero();  // camera body velocity, camera frame (m/s)
    Vec3 w = Vec3::Zero();  // camera body angular velocity, camera frame (rad/s)
    double exposure_s = 0.008;
    double readout_s = 0.0;   // rolling shutter: first row to last row (0: global shutter)
    double gain = 0.0;        // 0: auto exposure on the image circle
    double ae_target = 0.30;  // auto-exposure target mean (linear)
    double read_noise = 0.004, shot_noise = 0.012;  // linear units: var = read^2 + shot^2 * I
    double gamma = 1.0 / 2.2;
    int jpeg_quality = 0;     // > 0: MJPEG-style compression round trip
};

struct Hit {
    double t = 1e9;
    unsigned char material = 0;  // 0 nothing, 1 board, 2+ capsule material, 9 table
    float value = 0.0f;          // board print 0..1 / capsule shade / table value
    Vec3 n = Vec3::Zero();
    double gloss = 0.0;
};

struct Renderer {
    CameraIntrinsics cam;
    int ss = 2;
    Lens lens;
    std::vector<Eigen::Vector3f> rays;  // supersampled bearings, camera frame
    double circle_px = 0.0;             // image-circle radius (px), for the decentering field

    // Unmodelled decentering distortion (px) at a pixel: tangential, grows with r^2.
    Vec2 Decenter(const Vec2& uv) const {
        if (lens.decenter_px == 0.0 || circle_px <= 0.0) return Vec2::Zero();
        const double x = (uv.x() - cam.params[2]) / circle_px, y = (uv.y() - cam.params[3]) / circle_px;
        const double r2 = x * x + y * y, k = lens.decenter_px / 3.0;
        const double p1 = k * std::cos(lens.decenter_dir), p2 = k * std::sin(lens.decenter_dir);
        return Vec2(2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);
    }

    // Where the real lens images a camera-frame point (projection model + decentering).
    bool TrueProject(const Vec3& X, double uv[2]) const {
        ProjectPoint(cam.model, cam.params.data(), Eigen::Matrix<double, 3, 1>(X), uv);
        const Vec2 d = Decenter(Vec2(uv[0], uv[1]));
        uv[0] += d.x();
        uv[1] += d.y();
        return std::isfinite(uv[0]) && std::isfinite(uv[1]);
    }

    // Image-circle radius (decentering field); Init() does this too.
    void InitCircle() {
        const double th_edge = (lens.fov_deg > 0 ? 0.5 * lens.fov_deg : 90.0) * kRenderDeg;
        if (IsWideAngle(cam.model)) {
            double e[2];
            ProjectPoint(cam.model, cam.params.data(),
                         Eigen::Matrix<double, 3, 1>(std::sin(th_edge), 0.0, std::cos(th_edge)), e);
            circle_px = std::fabs(e[0] - cam.params[2]);
        }
    }

    void Init() {
        InitCircle();
        const int W = cam.width * ss, H = cam.height * ss;
        rays.resize(static_cast<size_t>(W) * H);
#pragma omp parallel for
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                Vec3 b;
                Vec2 uv((c + 0.5) / ss - 0.5, (r + 0.5) / ss - 0.5);
                uv -= Decenter(uv);
                if (!UnprojectPixel(cam.model, cam.params.data(), uv, b)) b = Vec3(0, 0, 0);
                rays[static_cast<size_t>(r) * W + c] = b.cast<float>();
            }
        }
    }

    struct PlaneRef {
        Vec3 n, p;
        Mat3 Rt;
        const BoardTexture* tex;
    };
    static std::vector<PlaneRef> Planes(const std::vector<BoardTexture>& boards, const std::vector<Iso3>& poses) {
        std::vector<PlaneRef> planes;
        for (size_t i = 0; i < boards.size(); ++i) {
            planes.push_back({poses[i].linear().col(2), poses[i].translation(), poses[i].linear().transpose(), &boards[i]});
        }
        return planes;
    }

    // Nearest surface along a ray: boards (bowed prints), robot links, table.
    static Hit Trace(const Vec3& o, const Vec3& d, const std::vector<PlaneRef>& planes, const SceneExtras& extras) {
        Hit h;
        for (const PlaneRef& pl : planes) {
            const double den = pl.n.dot(d);
            if (std::fabs(den) < 1e-9) continue;
            double t = pl.n.dot(pl.p - o) / den;
            Vec3 pb = pl.Rt * (o + t * d - pl.p);
            if (pl.tex->bow_x != 0.0) {  // curved print: z = bow * (1 - u^2)
                const double hw = 0.5 * pl.tex->spec.Width();
                for (int it = 0; it < 4; ++it) {
                    const double u = (pb.x() - hw) / hw;
                    t = (pl.tex->bow_x * (1 - u * u) + pl.n.dot(pl.p - o)) / den;
                    pb = pl.Rt * (o + t * d - pl.p);
                }
            }
            if (t <= 0 || t >= h.t) continue;
            const float s = pl.tex->Sample(pb.x(), pb.y());
            if (s < 0) continue;
            h.t = t;
            h.material = 1;
            h.value = s;
            h.n = pl.n;
            h.gloss = pl.tex->gloss;
        }
        for (const Capsule& cap : extras.capsules) {
            double t;
            Vec3 n;
            if (!HitCapsule(o, d, cap, t, n) || t >= h.t) continue;
            h.t = t;
            h.material = static_cast<unsigned char>(cap.material);
            h.value = static_cast<float>(cap.shade);
            h.n = n;
            h.gloss = 0.0;
        }
        if (extras.table && std::fabs(d.z()) > 1e-9) {
            const double t = (extras.table_z - o.z()) / d.z();
            if (t > 0 && t < h.t) {
                const Vec3 p = o + t * d;
                if (p.x() > extras.table_min.x() && p.x() < extras.table_max.x() && p.y() > extras.table_min.y() &&
                    p.y() < extras.table_max.y()) {
                    // Matte table top: faint 5 cm grid and a slow grain.
                    const double gx = std::fabs(std::remainder(p.x(), 0.05));
                    const double gy = std::fabs(std::remainder(p.y(), 0.05));
                    const double grain = 0.03 * std::sin(p.x() * 37.0 + 3.0 * std::sin(p.y() * 11.0));
                    h.t = t;
                    h.material = 9;
                    h.value = static_cast<float>(0.64 + grain - ((gx < 0.0008 || gy < 0.0008) ? 0.08 : 0.0));
                    h.n = Vec3::UnitZ();
                    h.gloss = 0.0;
                }
            }
        }
        return h;
    }

    // Linear radiance of a hit under a lamp (Lambert + optional gloss), with link shadows.
    static float Shade(const Hit& h, const Vec3& o, const Vec3& d, const Lighting& L, const SceneExtras& x) {
        if (h.material == 0) return static_cast<float>(L.background * (0.75 + 0.5 * std::max(0.0, -d.z())));
        const Vec3 p = o + h.t * d;
        Vec3 n = h.n;
        if (n.dot(d) > 0) n = -n;
        Vec3 l = L.lamp - p;
        const double dist = l.norm();
        l /= dist;
        double direct = L.direct * std::max(0.0, n.dot(l)) * (L.range_m * L.range_m) / (dist * dist);
        if (direct > 0 && L.shadows) {
            const Vec3 ps = p + 1e-4 * n;
            for (const Capsule& c : x.capsules) {
                double t;
                Vec3 nn;
                if (HitCapsule(ps, l, c, t, nn) && t < dist) {
                    direct = 0;
                    break;
                }
            }
        }
        double albedo;
        if (h.material == 1) {
            albedo = 0.86 * std::max(0.04 / 0.86, static_cast<double>(h.value));  // print reflectance x paper
        } else if (h.material == 9) {
            albedo = 0.55 * h.value;
        } else {
            albedo = 0.6 * h.value;
        }
        double rad = albedo * (L.ambient + direct);
        if (h.gloss > 0 && direct > 0) {  // laminated print: specular lobe (washes the tags out)
            const Vec3 r = 2.0 * n.dot(l) * n - l;
            rad += h.gloss * direct * std::pow(std::max(0.0, r.dot(-d)), 80.0);
        }
        return static_cast<float>(rad);
    }

    cv::Mat Render(const Iso3& T_wc, const std::vector<BoardTexture>& boards, std::mt19937& rng) const {
        std::vector<Iso3> poses;
        for (const BoardTexture& b : boards) poses.push_back(b.T_world_board);
        return Render(T_wc, boards, poses, rng);
    }

    cv::Mat Render(const Iso3& T_wc, const std::vector<BoardTexture>& boards, const std::vector<Iso3>& poses,
                   std::mt19937& rng) const {
        return Render(T_wc, boards, poses, SceneExtras(), rng, nullptr);
    }

    // Idealised render (tests, observer views): boards + robot links + table, flat lighting,
    // light noise. Optionally the material of each output pixel.
    cv::Mat Render(const Iso3& T_wc, const std::vector<BoardTexture>& boards, const std::vector<Iso3>& poses,
                   const SceneExtras& extras, std::mt19937& rng, cv::Mat* material, bool noise = true) const {
        const int W = cam.width * ss, H = cam.height * ss;
        cv::Mat hi(H, W, CV_32F);
        cv::Mat mat_hi(H, W, CV_8U, cv::Scalar(0));
        const Mat3 R = T_wc.linear();
        const Vec3 o = T_wc.translation();
        const std::vector<PlaneRef> planes = Planes(boards, poses);
        const Vec3 light = Vec3(0.3, -0.45, 1.0).normalized();
#pragma omp parallel for
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const Eigen::Vector3f& bf = rays[static_cast<size_t>(r) * W + c];
                float val = 0.42f + 0.12f * static_cast<float>(r) / H;  // background
                unsigned char m = 0;
                if (bf.squaredNorm() > 0.5f) {
                    const Vec3 d = (R * bf.cast<double>()).normalized();
                    const double thp = std::atan2(std::hypot(bf.x(), bf.y()), bf.z());
                    const Vec3 op = o + R.col(2) * lens.PupilZ(thp);
                    const Hit h = Trace(op, d, planes, extras);
                    m = h.material;
                    if (h.material == 1) {
                        val = 0.08f + 0.84f * h.value;
                    } else if (h.material == 9) {
                        val = h.value;
                    } else if (h.material != 0) {
                        val = static_cast<float>(h.value * (0.30 + 0.70 * std::max(0.0, h.n.dot(light))));
                    }
                }
                hi.at<float>(r, c) = val;
                mat_hi.at<unsigned char>(r, c) = m;
            }
        }
        cv::Mat lo;
        cv::resize(hi, lo, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_AREA);
        cv::GaussianBlur(lo, lo, cv::Size(0, 0), 0.6);
        if (noise) {
            cv::Mat nz(lo.size(), CV_32F);
            cv::randn(nz, 0.0, 0.008);
            lo += nz;
        }
        if (material) cv::resize(mat_hi, *material, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_NEAREST);
        cv::Mat out;
        lo.convertTo(out, CV_8U, 255.0);
        (void)rng;
        return out;
    }

    // Physically motivated capture: lamp + shadows + gloss, image circle + vignetting, motion
    // blur and rolling shutter from the camera velocity, field- and depth-dependent optical blur
    // (fixed focus), auto exposure, shot + read noise, gamma, 8 bit, optional JPEG.
    cv::Mat RenderReal(const Iso3& T_wc, const std::vector<BoardTexture>& boards, const std::vector<Iso3>& poses,
                       const SceneExtras& extras, const Lighting& L, const Shot& S, std::mt19937& rng,
                       cv::Mat* material = nullptr) const {
        const int W = cam.width * ss, H = cam.height * ss;
        const std::vector<PlaneRef> planes = Planes(boards, poses);
        // Enough time samples that the image moves < 0.5 px between two of them.
        const double fpx = cam.params[0];
        const double motion_px = fpx * (S.w.norm() + S.v.norm() / 0.2) * (S.exposure_s + 0.0);
        const int K = std::clamp(static_cast<int>(std::ceil(motion_px / 0.5)), 1, 24);
        std::vector<Mat3> Rk(static_cast<size_t>(H) * K);
        std::vector<Vec3> ok(static_cast<size_t>(H) * K);
        for (int r = 0; r < H; ++r) {
            for (int k = 0; k < K; ++k) {
                const double t = ((r + 0.5) / H - 0.5) * S.readout_s + ((k + 0.5) / K - 0.5) * S.exposure_s;
                Mat3 dR = Mat3::Identity();
                if (S.w.norm() > 0) dR = Eigen::AngleAxisd(S.w.norm() * t, S.w.normalized()).toRotationMatrix();
                Rk[static_cast<size_t>(r) * K + k] = T_wc.linear() * dR;
                ok[static_cast<size_t>(r) * K + k] = T_wc.translation() + T_wc.linear() * (S.v * t);
            }
        }
        const double th_max = (lens.fov_deg > 0 ? 0.5 * lens.fov_deg : 180.0) * kRenderDeg;
        const double soft = 0.4 * kRenderDeg;
        cv::Mat hi(H, W, CV_32F), depth_hi(H, W, CV_32F), theta_hi(H, W, CV_32F);
        cv::Mat mat_hi(H, W, CV_8U, cv::Scalar(0));
#pragma omp parallel for
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const Eigen::Vector3f& bf = rays[static_cast<size_t>(r) * W + c];
                float val = 0.0f, dep = 10.0f, th = static_cast<float>(th_max);
                unsigned char m = 0;
                if (bf.squaredNorm() > 0.5f) {
                    const Vec3 b = bf.cast<double>().normalized();
                    th = static_cast<float>(std::atan2(std::hypot(b.x(), b.y()), b.z()));
                    double illum = 1.0;
                    if (lens.fov_deg > 0) illum = std::clamp((th_max - th) / soft, 0.0, 1.0);
                    if (lens.vignette > 0) illum *= 1.0 - lens.vignette * std::pow(std::min(1.0, th / th_max), 2.0);
                    if (illum > 0) {
                        double acc = 0.0;
                        for (int k = 0; k < K; ++k) {
                            const size_t i = static_cast<size_t>(r) * K + k;
                            const Vec3 d = Rk[i] * b;
                            const Vec3 op = ok[i] + Rk[i].col(2) * lens.PupilZ(th);
                            const Hit h = Trace(op, d, planes, extras);
                            acc += Shade(h, op, d, L, extras);
                            if (k == K / 2) {
                                m = h.material;
                                dep = static_cast<float>(h.material ? h.t : 10.0);
                            }
                        }
                        val = static_cast<float>(illum * acc / K);
                    }
                }
                hi.at<float>(r, c) = val;
                depth_hi.at<float>(r, c) = dep;
                theta_hi.at<float>(r, c) = th;
                mat_hi.at<unsigned char>(r, c) = m;
            }
        }
        cv::Mat lo, dep_lo, th_lo;
        cv::resize(hi, lo, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_AREA);  // pixel integration
        cv::resize(depth_hi, dep_lo, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_NEAREST);
        cv::resize(theta_hi, th_lo, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_NEAREST);
        // Spatially varying optical blur: blend a stack of Gaussian blurs by the local sigma.
        cv::Mat sig(lo.size(), CV_32F);
        for (int r = 0; r < lo.rows; ++r) {
            for (int c = 0; c < lo.cols; ++c) {
                const double fr = std::min(1.0, th_lo.at<float>(r, c) / th_max);
                double s2 = lens.base_blur_px * lens.base_blur_px;
                s2 += std::pow(lens.edge_blur_px * fr * fr, 2);
                if (lens.focus_m > 0 && lens.defocus_px > 0) {
                    const double z = std::max(0.02, static_cast<double>(dep_lo.at<float>(r, c)));
                    const double c_px = lens.defocus_px * std::fabs(1.0 / z - 1.0 / lens.focus_m);
                    s2 += std::pow(c_px / 4.0, 2);  // uniform disc of diameter c: sigma = c / 4
                }
                sig.at<float>(r, c) = static_cast<float>(std::sqrt(s2));
            }
        }
        static const double levels[] = {0.3, 0.5, 0.75, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0};
        constexpr int NL = sizeof(levels) / sizeof(levels[0]);
        std::vector<cv::Mat> stack(NL);
        for (int l = 0; l < NL; ++l) cv::GaussianBlur(lo, stack[l], cv::Size(0, 0), levels[l]);
        cv::Mat blurred(lo.size(), CV_32F);
        for (int r = 0; r < lo.rows; ++r) {
            for (int c = 0; c < lo.cols; ++c) {
                const double s = std::clamp(static_cast<double>(sig.at<float>(r, c)), levels[0], levels[NL - 1]);
                int l = 0;
                while (l + 2 < NL && levels[l + 1] < s) ++l;
                const double a = (s - levels[l]) / (levels[l + 1] - levels[l]);
                blurred.at<float>(r, c) =
                    static_cast<float>((1 - a) * stack[l].at<float>(r, c) + a * stack[l + 1].at<float>(r, c));
            }
        }
        // Auto exposure on the image circle, then sensor noise, gamma, 8 bit.
        double gain = S.gain;
        if (gain <= 0) {
            double sum = 0;
            int n = 0;
            for (int r = 0; r < lo.rows; r += 4) {
                for (int c = 0; c < lo.cols; c += 4) {
                    if (th_lo.at<float>(r, c) < th_max - soft) {
                        sum += blurred.at<float>(r, c);
                        ++n;
                    }
                }
            }
            gain = n > 0 && sum > 0 ? std::clamp(S.ae_target * n / sum, 0.25, 8.0) : 1.0;
        }
        std::normal_distribution<float> N01(0.0f, 1.0f);
        cv::Mat out(lo.size(), CV_8U);
        for (int r = 0; r < lo.rows; ++r) {
            for (int c = 0; c < lo.cols; ++c) {
                const double I = gain * blurred.at<float>(r, c);
                const double sd = std::sqrt(S.read_noise * S.read_noise + S.shot_noise * S.shot_noise * std::max(0.0, I));
                const double v = std::clamp(I + sd * N01(rng), 0.0, 1.0);
                out.at<unsigned char>(r, c) = cv::saturate_cast<unsigned char>(255.0 * std::pow(v, S.gamma));
            }
        }
        if (S.jpeg_quality > 0) {
            std::vector<unsigned char> buf;
            cv::imencode(".jpg", out, buf, {cv::IMWRITE_JPEG_QUALITY, S.jpeg_quality});
            out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
        }
        if (material) cv::resize(mat_hi, *material, cv::Size(cam.width, cam.height), 0, 0, cv::INTER_NEAREST);
        return out;
    }
};

inline std::string PoseYaml(const Iso3& T) {
    std::ostringstream o;
    const Vec3 x = T.translation(), r = RotationToRpy(T.linear());
    o << std::setprecision(17) << "{xyz: [" << x.x() << ", " << x.y() << ", " << x.z() << "], rpy: [" << r.x() << ", "
      << r.y() << ", " << r.z() << "]}";
    return o.str();
}

inline bool BoardInView(const Iso3& T_wc, const BoardTexture& b, const CameraIntrinsics& cam, double max_deg = 75.0) {
    const Vec3 centre = b.T_world_board * Vec3(b.spec.Width() / 2, b.spec.Height() / 2, 0);
    if ((T_wc.translation() - centre).dot(b.T_world_board.linear().col(2)) < 0.05) return false;
    const Vec3 Xc = T_wc.inverse() * centre;
    if (std::atan2(std::hypot(Xc.x(), Xc.y()), Xc.z()) > max_deg * kRenderDeg) return false;
    double uv[2];
    ProjectPoint(cam.model, cam.params.data(), Eigen::Matrix<double, 3, 1>(Xc), uv);
    return uv[0] > 40 && uv[1] > 40 && uv[0] < cam.width - 40 && uv[1] < cam.height - 40 && Xc.norm() < 0.75;
}


inline std::string CameraInfoYaml(const CameraIntrinsics& kb) {
    std::ostringstream ci;
    ci << std::setprecision(17) << "width: " << kb.width << "\nheight: " << kb.height
       << "\ndistortion_model: equidistant\nd: [" << kb.params[4] << ", " << kb.params[5] << ", " << kb.params[6]
       << ", " << kb.params[7] << "]\nk: [" << kb.params[0] << ", 0, " << kb.params[2] << ", 0, " << kb.params[1]
       << ", " << kb.params[3] << ", 0, 0, 1]\n";
    return ci.str();
}

inline Iso3 Place(const BoardSpec& s, const Vec3& centre, const Vec3& rpy) {
    Iso3 T = MakeIso(centre, rpy);
    T.translation() = centre - T.linear() * Vec3(s.Width() / 2, s.Height() / 2, 0);
    return T;
}

}  // namespace sim
}  // namespace robocal

#endif  // ROBOCAL_RENDER_HPP_
