// Corner refinement tests: model math, synthetic blurred X- / L-corners with known sub-pixel
// truth (gamma 1/2.2 encoding, noise), end-to-end synthetic boards (ArUco grid, AprilTag grid,
// ChArUco) through the real detectors, and CPU / GPU agreement.
#include "test_framework.hpp"
#include "../corner-refine.hpp"
#include "../apriltag-file.hpp"
#include "../cuda-detect.hpp"

#include <chrono>
#include <cstdio>
#include <random>

using namespace calico_refine;

namespace {

struct Stats {
    double bx = 0, by = 0, median = 0, p95 = 0, rms = 0;
    int n = 0, total = 0;
};

Stats Summarize(const std::vector<cv::Point2d>& err, int total) {
    Stats s;
    s.n = static_cast<int>(err.size());
    s.total = total;
    if (err.empty()) return s;
    std::vector<double> m;
    for (const auto& e : err) {
        s.bx += e.x / s.n;
        s.by += e.y / s.n;
        m.push_back(std::hypot(e.x, e.y));
        s.rms += e.x * e.x + e.y * e.y;
    }
    s.rms = std::sqrt(s.rms / s.n);
    std::sort(m.begin(), m.end());
    s.median = m[m.size() / 2];
    s.p95 = m[std::min(m.size() - 1, static_cast<size_t>(0.95 * m.size()))];
    return s;
}

void Print(const char* name, const Stats& s) {
    std::printf("    %-34s n=%4d/%-4d bias=(%+.4f, %+.4f) median=%.4f p95=%.4f rms=%.4f px\n", name, s.n, s.total,
                s.bx, s.by, s.median, s.p95, s.rms);
}

// Linear irradiance -> 8-bit with gamma 1/g encoding and Gaussian read noise (DN).
cv::Mat Encode(const cv::Mat& lin, double gamma, double noise_dn, std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, noise_dn);
    cv::Mat out(lin.size(), CV_8UC1);
    for (int r = 0; r < lin.rows; ++r) {
        for (int c = 0; c < lin.cols; ++c) {
            const double v = std::pow(std::min(1.0, std::max(0.0, static_cast<double>(lin.at<float>(r, c)))), 1.0 / gamma);
            out.at<unsigned char>(r, c) = cv::saturate_cast<unsigned char>(255.0 * v + nd(rng));
        }
    }
    return out;
}

// One synthetic corner (64 x 64 frame). u, v: unit edge directions (image); the dark set is
// {alpha > 0, beta > 0} (L) or {alpha beta > 0} (X) with q = alpha u + beta v. Rendered at 8x,
// Gaussian-blurred, area-averaged (pixel integration), encoded with gamma.
cv::Mat RenderCorner(int type, cv::Point2d p0, cv::Point2d u, cv::Point2d v, double blur, double black, double white,
                     double gamma, double noise, std::mt19937& rng) {
    const int N = 64, S = 8;
    cv::Mat hi(N * S, N * S, CV_32F);
    const double det = u.x * v.y - u.y * v.x;
    for (int Y = 0; Y < hi.rows; ++Y) {
        for (int X = 0; X < hi.cols; ++X) {
            const double qx = (X + 0.5) / S - 0.5 - p0.x, qy = (Y + 0.5) / S - 0.5 - p0.y;
            const double a = (qx * v.y - qy * v.x) / det, b = (u.x * qy - u.y * qx) / det;
            const bool dark = type == kLCorner ? (a > 0 && b > 0) : (a * b > 0);
            hi.at<float>(Y, X) = static_cast<float>(dark ? black : white);
        }
    }
    cv::GaussianBlur(hi, hi, cv::Size(0, 0), blur * S);
    cv::Mat lo;
    cv::resize(hi, lo, cv::Size(N, N), 0, 0, cv::INTER_AREA);
    return Encode(lo, gamma, noise, rng);
}

struct CornerCase {
    cv::Mat img;
    cv::Point2d truth, start;
    FitTask task;
};

std::vector<CornerCase> MakeCorners(int type, int count, unsigned seed, double gamma = 2.2) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<CornerCase> out;
    const double module_px = 8.0;
    for (int i = 0; i < count; ++i) {
        CornerCase c;
        c.truth = cv::Point2d(31.0 + U(rng), 31.0 + U(rng));
        const double th = 2 * CV_PI * U(rng);
        const double open = (60.0 + 60.0 * U(rng)) * CV_PI / 180.0;  // 60..120 deg between the edges
        const cv::Point2d u(std::cos(th), std::sin(th)), v(std::cos(th + open), std::sin(th + open));
        const double blur = 0.7 + 1.3 * U(rng);
        const double black = 0.02 + 0.08 * U(rng), white = 0.5 + 0.45 * U(rng);
        c.img = RenderCorner(type, c.truth, u, v, blur, black, white, gamma, 1.5, rng);
        c.start = c.truth + cv::Point2d(U(rng) - 0.5, U(rng) - 0.5) * 1.2;
        FitTask& t = c.task;
        t.x = c.start.x;
        t.y = c.start.y;
        // Board units = modules: A columns = edge directions times the module size.
        t.A[0] = u.x * module_px, t.A[1] = v.x * module_px, t.A[2] = u.y * module_px, t.A[3] = v.y * module_px;
        t.rho = 0.7;
        t.cut = type == kLCorner ? 1.4 : 1.5;
        t.max_move = 0.5;
        t.type = type;
        t.polarity = 1;
        out.push_back(c);
    }
    return out;
}

cv::Point2d CornerSubPixFrom(const cv::Mat& img, cv::Point2d p, int half) {
    std::vector<cv::Point2f> v = {cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y))};
    cv::cornerSubPix(img, v, cv::Size(half, half), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 50, 1e-3));
    return cv::Point2d(v[0].x, v[0].y);
}

void CompareOnCorners(int type, const char* label, Stats* model_out, Stats* subpix_out) {
    const std::vector<CornerCase> cases = MakeCorners(type, 300, type == kLCorner ? 7u : 3u);
    std::vector<cv::Point2d> e_model, e_sub, e_lin1;
    for (const CornerCase& c : cases) {
        // cornerSubPix (window half 5 = 0.6 module) from the same start, as calico's legacy path.
        const cv::Point2d s = CornerSubPixFrom(c.img, c.start, 5);
        e_sub.push_back(s - c.truth);
        // Model fit from the cornerSubPix result (the pipeline order).
        FitTask t = c.task;
        if (cv::norm(s - c.start) < 4.0) t.x = s.x, t.y = s.y;
        const FitResult f = FitCornerCPU(Linearize(c.img, 2.2), t, DefaultFitParams());
        if (f.ok) e_model.push_back(cv::Point2d(f.x, f.y) - c.truth);
        // Same fit without undoing the gamma (shows what linearisation buys).
        const FitResult f1 = FitCornerCPU(Linearize(c.img, 1.0), t, DefaultFitParams());
        if (f1.ok) e_lin1.push_back(cv::Point2d(f1.x, f1.y) - c.truth);
    }
    std::printf("  %s (300 corners, blur 0.7-2.0 px, edges 60-120 deg, gamma 1/2.2, noise 1.5 DN):\n", label);
    const Stats sm = Summarize(e_model, 300), ss = Summarize(e_sub, 300), s1 = Summarize(e_lin1, 300);
    Print("cornerSubPix", ss);
    Print("model fit, --gamma=2.2", sm);
    Print("model fit, --gamma=1 (no linearise)", s1);
    *model_out = sm;
    *subpix_out = ss;
}

}  // namespace

TEST_CASE(TestPhi2KnownValues) {
    // Phi2(0, 0; r) = 1/4 + asin(r) / (2 pi); independence at r = 0.
    for (double r : {-0.9, -0.5, 0.0, 0.3, 0.74, 0.76, 0.9}) {
        ASSERT_NEAR(Phi2(0, 0, r), 0.25 + std::asin(r) / (2 * CV_PI), 1e-12);
    }
    ASSERT_NEAR(Phi2(0.7, -1.2, 0.0), NormCdf(0.7) * NormCdf(-1.2), 1e-15);
    // r -> 1: Phi2(h, k) -> Phi(min(h, k)); at 0.9 still close for well-separated arguments.
    ASSERT_NEAR(Phi2(-3, 3, 0.9), NormCdf(-3), 1e-4);
    // dPhi2/dh against a central difference.
    const double h = 0.4, k = -0.3, r = 0.6, e = 1e-6;
    double M, Mh, Mk, Mr;
    ModelEval(kLCorner, h, k, r, &M, &Mh, &Mk, &Mr);
    ASSERT_NEAR(Mh, (Phi2(h + e, k, r) - Phi2(h - e, k, r)) / (2 * e), 1e-8);
    ASSERT_NEAR(Mk, (Phi2(h, k + e, r) - Phi2(h, k - e, r)) / (2 * e), 1e-8);
    ASSERT_NEAR(Mr, (Phi2(h, k, r + e) - Phi2(h, k, r - e)) / (2 * e), 1e-7);
}

TEST_CASE(TestModelJacobianMatchesFiniteDifferences) {
    for (int type : {kXJunction, kLCorner}) {
        const double P[7] = {10.3, 9.7, 0.4, 2.1, 1.3, 0.2, -0.6};
        const ModelState m = MakeState(P);
        double J[7];
        const double r0 = PixelResidual(type, P, m, 11.0, 8.0, 0.8f, 0.3f, J);
        (void)r0;
        for (int d = 0; d < 7; ++d) {
            double Pp[7], Pm[7];
            std::copy(P, P + 7, Pp);
            std::copy(P, P + 7, Pm);
            Pp[d] += 1e-6;
            Pm[d] -= 1e-6;
            const double rp = PixelResidual(type, Pp, MakeState(Pp), 11.0, 8.0, 0.8f, 0.3f, nullptr);
            const double rm = PixelResidual(type, Pm, MakeState(Pm), 11.0, 8.0, 0.8f, 0.3f, nullptr);
            ASSERT_NEAR(J[d], (rp - rm) / 2e-6, 1e-6);
        }
    }
}

TEST_CASE(TestXJunctionFitBeatsCornerSubPix) {
    Stats m, s;
    CompareOnCorners(kXJunction, "X-junction", &m, &s);
    ASSERT_TRUE(m.n >= 270);                 // >= 90% accepted
    ASSERT_TRUE(m.median < s.median);
    ASSERT_TRUE(m.p95 < s.p95);
    ASSERT_TRUE(m.p95 < 0.06);
    ASSERT_TRUE(std::fabs(m.bx) < 0.01 && std::fabs(m.by) < 0.01);
}

TEST_CASE(TestLCornerFitBeatsCornerSubPix) {
    Stats m, s;
    CompareOnCorners(kLCorner, "L-corner", &m, &s);
    ASSERT_TRUE(m.n >= 270);
    ASSERT_TRUE(m.median < s.median);
    ASSERT_TRUE(m.p95 < s.p95);
    ASSERT_TRUE(m.p95 < 0.1);
}

TEST_CASE(TestFitRejectsWrongPolarityAndFlat) {
    std::vector<CornerCase> cases = MakeCorners(kXJunction, 10, 11u);
    int polarity_rejects = 0;
    for (CornerCase& c : cases) {
        c.task.polarity = -1;  // the board says the other diagonal is black
        const FitResult f = FitCornerCPU(Linearize(c.img, 2.2), c.task, DefaultFitParams());
        polarity_rejects += !f.ok;
    }
    ASSERT_EQ(polarity_rejects, 10);
    cv::Mat flat(64, 64, CV_8UC1, cv::Scalar(128));
    const FitResult f = FitCornerCPU(Linearize(flat, 2.2), cases[0].task, DefaultFitParams());
    ASSERT_FALSE(f.ok);
    // Window leaving the image.
    FitTask t = cases[0].task;
    t.x = 2.0;
    ASSERT_EQ(FitCornerCPU(Linearize(cases[0].img, 2.2), t, DefaultFitParams()).reason, static_cast<int>(kBorder));
}

TEST_CASE(TestDropDuplicateIds) {
    std::vector<AprilTags::TagDetection> d(4);
    d[0].id = 1, d[1].id = 2, d[2].id = 1, d[3].id = 3;
    ASSERT_EQ(DropDuplicateIds(d), 2);
    ASSERT_EQ(static_cast<int>(d.size()), 2);
    ASSERT_EQ(d[0].id, 2);
    ASSERT_EQ(d[1].id, 3);
}

// ------------------------------------------------------------------------- synthetic boards
namespace {

// Camera: pattern pixel coords -> H -> ideal image -> division-model distortion around c.
struct SynthCamera {
    cv::Mat H;     // pattern -> undistorted image
    double k;      // division model: x_u = c + (x_d - c) / (1 + k r_d^2)
    cv::Point2d c;
    cv::Point2d Project(cv::Point2d q) const {
        const double w = H.at<double>(2, 0) * q.x + H.at<double>(2, 1) * q.y + H.at<double>(2, 2);
        const cv::Point2d u((H.at<double>(0, 0) * q.x + H.at<double>(0, 1) * q.y + H.at<double>(0, 2)) / w,
                            (H.at<double>(1, 0) * q.x + H.at<double>(1, 1) * q.y + H.at<double>(1, 2)) / w);
        const cv::Point2d d = u - c;
        const double ru = std::hypot(d.x, d.y);
        if (k == 0 || ru < 1e-9) return u;
        const double rd = (1.0 - std::sqrt(1.0 - 4.0 * k * ru * ru)) / (2.0 * k * ru);
        return c + d * (rd / ru);
    }
};

// Renders the binary pattern (0 / 255) through the camera: 4x supersampled remap in linear light,
// Gaussian blur, area-averaged, then exposure + gamma encoding + noise.
cv::Mat RenderBoard(const cv::Mat& pattern, const SynthCamera& cam, cv::Size size, double blur, double black,
                    double white, double gamma, double noise, unsigned seed) {
    const int S = 4;
    cv::Mat mapx(size.height * S, size.width * S, CV_32F), mapy(mapx.size(), CV_32F);
    const cv::Mat Hinv = cam.H.inv();
    for (int Y = 0; Y < mapx.rows; ++Y) {
        for (int X = 0; X < mapx.cols; ++X) {
            const cv::Point2d xd((X + 0.5) / S - 0.5, (Y + 0.5) / S - 0.5);
            const cv::Point2d d = xd - cam.c;
            const double f = 1.0 / (1.0 + cam.k * (d.x * d.x + d.y * d.y));
            const cv::Point2d xu = cam.c + d * f;
            const double w = Hinv.at<double>(2, 0) * xu.x + Hinv.at<double>(2, 1) * xu.y + Hinv.at<double>(2, 2);
            mapx.at<float>(Y, X) = static_cast<float>((Hinv.at<double>(0, 0) * xu.x + Hinv.at<double>(0, 1) * xu.y + Hinv.at<double>(0, 2)) / w);
            mapy.at<float>(Y, X) = static_cast<float>((Hinv.at<double>(1, 0) * xu.x + Hinv.at<double>(1, 1) * xu.y + Hinv.at<double>(1, 2)) / w);
        }
    }
    cv::Mat lin_pattern;
    pattern.convertTo(lin_pattern, CV_32F, (white - black) / 255.0, black);
    cv::Mat hi;
    cv::remap(lin_pattern, hi, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.5 * (black + white)));
    cv::GaussianBlur(hi, hi, cv::Size(0, 0), blur * S);
    cv::Mat lo;
    cv::resize(hi, lo, size, 0, 0, cv::INTER_AREA);
    std::mt19937 rng(seed);
    return Encode(lo, gamma, noise, rng);
}

std::vector<SynthCamera> Cameras(cv::Size pattern, cv::Size image) {
    // Three views: frontal, tilted about x, tilted about y (board fills ~70% of the frame),
    // with barrel distortion (~10% at the corners).
    std::vector<SynthCamera> cams;
    const double k = -0.4 / (image.width * image.width);
    const cv::Point2d c(image.width / 2.0, image.height / 2.0);
    const double sx = 0.7 * image.width / pattern.width, sy = 0.7 * image.height / pattern.height;
    const double s = std::min(sx, sy);
    const double tilts[3][2] = {{0, 0}, {0.35, 0}, {0, -0.35}};
    for (const auto& tl : tilts) {
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1000, 0, c.x, 0, 1000, c.y, 0, 0, 1);
        cv::Mat R;
        cv::Rodrigues(cv::Vec3d(tl[0], tl[1], 0.05), R);
        // Board plane: pattern px * (s / 1000) metres-ish at depth 1, centred.
        cv::Mat B = (cv::Mat_<double>(3, 3) << s / 1000, 0, -s / 1000 * pattern.width / 2.0, 0, s / 1000,
                     -s / 1000 * pattern.height / 2.0, 0, 0, 1);
        cv::Mat M = (cv::Mat_<double>(3, 3) << R.at<double>(0, 0), R.at<double>(0, 1), 0, R.at<double>(1, 0),
                     R.at<double>(1, 1), 0, R.at<double>(2, 0), R.at<double>(2, 1), 1.0);
        SynthCamera cam;
        cam.H = K * M * B;
        cam.k = k;
        cam.c = c;
        cams.push_back(cam);
    }
    return cams;
}

const cv::Size kSceneImage(1280, 800);
const double kSceneBlur = 1.1, kSceneBlack = 0.04, kSceneWhite = 0.75, kSceneGamma = 2.2, kSceneNoise = 1.5;

// Tag grid (AprilTag or ArUco) in calico's layout: tag (x, y) at x0 = m + x (sq + gap),
// y0 = H - sq - m - y (sq + gap). Returns the pattern and the true corners (pattern px, pixel-centre
// convention) per id, BL, BR, TR, TL.
cv::Mat TagGrid(int nx, int ny, int sq, int gap, int margin, const std::function<cv::Mat(int)>& marker,
                std::map<int, std::array<cv::Point2d, 4> >& truth) {
    const int W = nx * (sq + gap) - gap + 2 * margin, Hh = ny * (sq + gap) - gap + 2 * margin;
    cv::Mat img(Hh, W, CV_8UC1, cv::Scalar(255));
    int id = 0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x, ++id) {
            const int x0 = margin + x * (sq + gap), y0 = Hh - sq - margin - y * (sq + gap);
            marker(id).copyTo(img(cv::Rect(x0, y0, sq, sq)));
            const double l = x0 - 0.5, r = x0 + sq - 0.5, t = y0 - 0.5, b = y0 + sq - 0.5;
            truth[id] = {cv::Point2d(l, b), cv::Point2d(r, b), cv::Point2d(r, t), cv::Point2d(l, t)};
        }
    }
    return img;
}

// Runs one tag-grid detector over the three views and the refinement modes.
void EvaluateTagGrid(const char* label, const cv::Mat& pattern, const std::map<int, std::array<cv::Point2d, 4> >& truth,
                     int cells, double gap_tags, double margin_tags,
                     const std::function<void(const cv::Mat&, std::vector<AprilTags::TagDetection>&)>& detect,
                     Stats* raw_out, Stats* sub_out, Stats* model_out) {
    std::vector<cv::Point2d> e[3];
    int total = 0;
    const char* names[3] = {"raw detector corners (legacy none)", "cornerSubPix (--corner-refine=subpix)",
                            "L-corner model (--corner-refine=model)"};
    const auto cams = Cameras(pattern.size(), kSceneImage);
    for (size_t v = 0; v < cams.size(); ++v) {
        const cv::Mat img = RenderBoard(pattern, cams[v], kSceneImage, kSceneBlur, kSceneBlack, kSceneWhite, kSceneGamma, kSceneNoise, 100 + v);
        std::vector<AprilTags::TagDetection> base;
        detect(img, base);
        DropDuplicateIds(base);
        total += 4 * static_cast<int>(base.size());
        const RefineMode modes[3] = {RefineMode::kNone, RefineMode::kSubpix, RefineMode::kModel};
        for (int mi = 0; mi < 3; ++mi) {
            std::vector<AprilTags::TagDetection> d = base;
            std::vector<std::array<bool, 4> > keep;
            RefineConfig cfg;
            cfg.mode = modes[mi];
            cfg.gamma = kSceneGamma;
            RefineTagCorners(img, d, cells, gap_tags, margin_tags, keep, cfg);
            for (size_t j = 0; j < d.size(); ++j) {
                auto it = truth.find(d[j].id);
                if (it == truth.end()) continue;
                for (int k = 0; k < 4; ++k) {
                    if (!keep[j][k]) continue;
                    const cv::Point2d p(d[j].p[k].first, d[j].p[k].second);
                    double best = 1e9;
                    cv::Point2d be;
                    for (const auto& q : it->second) {
                        const cv::Point2d err = p - cams[v].Project(q);
                        if (cv::norm(err) < best) best = cv::norm(err), be = err;
                    }
                    e[mi].push_back(be);
                }
            }
        }
    }
    std::printf("  %s end-to-end (3 views, distortion, blur %.1f px, gamma 1/2.2, noise %.1f DN):\n", label, kSceneBlur, kSceneNoise);
    Stats s[3];
    for (int mi = 0; mi < 3; ++mi) {
        s[mi] = Summarize(e[mi], total);
        Print(names[mi], s[mi]);
    }
    *raw_out = s[0], *sub_out = s[1], *model_out = s[2];
}

}  // namespace

TEST_CASE(TestArucoGridEndToEnd) {
#if CALICO_ARUCO_MODERN
    const int arc = 10;  // DICT_6X6_250
    cv::Ptr<cv::aruco::Dictionary> dict = CalicoGetDictionary(arc);
    std::map<int, std::array<cv::Point2d, 4> > truth;
    const int sq = 160, gap = 32, margin = 40;
    const cv::Mat pattern = TagGrid(6, 4, sq, gap, margin, [&](int id) {
        cv::Mat m;
        dict->generateImageMarker(id, sq, m, 1);
        return m;
    }, truth);
    Stats raw, sub, model;
    EvaluateTagGrid("ArUco 6x4 DICT_6X6_250", pattern, truth, ArucoCells(arc), double(gap) / sq, double(margin) / sq,
                    [&](const cv::Mat& g, std::vector<AprilTags::TagDetection>& d) { DetectArucoMarkerGrid(g, arc, d); },
                    &raw, &sub, &model);
    ASSERT_TRUE(model.n > 0.9 * raw.n);
    ASSERT_TRUE(model.median < raw.median);
    ASSERT_TRUE(model.p95 < raw.p95);
#endif
}

TEST_CASE(TestAprilGridEndToEnd) {
    apriltag_family_t* tf = tag36h11_create();
    std::map<int, std::array<cv::Point2d, 4> > truth;
    const int sq = 160, gap = 32, margin = 40;
    const cv::Mat pattern = TagGrid(6, 4, sq, gap, margin, [&](int id) {
        return RenderAprilTagMarker("tag36h11", id, sq, tf);
    }, truth);
    AprilTags::TagDetector kaess(AprilTags::tagCodes36h11);
    Stats raw, sub, model;
    EvaluateTagGrid("AprilTag 6x4 tag36h11 (Kaess)", pattern, truth, AprilTagCells("tag36h11", true),
                    double(gap) / sq, double(margin) / sq,
                    [&](const cv::Mat& g, std::vector<AprilTags::TagDetection>& d) {
                        DetectAprilTagGrid(g, "tag36h11", &kaess, d);
                    },
                    &raw, &sub, &model);
    tag36h11_destroy(tf);
    ASSERT_TRUE(model.n > 0.9 * raw.n);
    ASSERT_TRUE(model.median < raw.median);
    ASSERT_TRUE(model.p95 < raw.p95);
}

TEST_CASE(TestCharucoEndToEnd) {
    const int sqpx = 160, mkpx = 120, margin = 40, nx = 7, ny = 5;
    cv::Ptr<cv::aruco::Dictionary> dict = CalicoGetDictionary(10);
    cv::Ptr<cv::aruco::CharucoBoard> board = CalicoCreateCharucoBoard(nx, ny, sqpx, mkpx, dict);
    cv::Mat pattern;
    CalicoGenerateBoardImage(board, cv::Size(nx * sqpx + 2 * margin, ny * sqpx + 2 * margin), pattern, margin, 1);
    const std::vector<cv::Point3f> chess = CharucoChessboardCorners(board);
    cv::Ptr<cv::aruco::DetectorParameters> params = CalicoCreateDetectorParams();
    std::vector<cv::Point2d> e_legacy, e_model;
    int total = 0;
    const auto cams = Cameras(pattern.size(), kSceneImage);
    for (size_t v = 0; v < cams.size(); ++v) {
        const cv::Mat img = RenderBoard(pattern, cams[v], kSceneImage, kSceneBlur, kSceneBlack, kSceneWhite, kSceneGamma, kSceneNoise, 200 + v);
        std::vector<std::vector<cv::Point2f> > corners, rejected;
        std::vector<int> ids;
        CalicoDetectMarkers(img, dict, corners, ids, params, rejected);
        // Board y runs down the generated image on OpenCV >= 4.6 and up on older releases.
        const double board_h = static_cast<double>(ny) * sqpx;
        auto truth_y = [&](int id, bool flip) {
            const double y = flip ? board_h - chess[id].y : chess[id].y;
            return cams[v].Project(cv::Point2d(margin - 0.5 + chess[id].x, margin - 0.5 + y));
        };
        bool flip = false;
        if (!ids.empty()) {
            // Pick the convention under which marker 0's first corner lands nearest its detection.
            const std::vector<std::vector<cv::Point3f> > obj = CharucoObjPoints(board);
            const std::vector<int> bids = CharucoIds(board);
            const size_t bi = std::find(bids.begin(), bids.end(), ids[0]) - bids.begin();
            if (bi < bids.size()) {
                double best = 1e18;
                for (bool f : {false, true}) {
                    double dsum = 0;
                    for (int q = 0; q < 4; ++q) {
                        const double y = f ? board_h - obj[bi][q].y : obj[bi][q].y;
                        dsum += cv::norm(cams[v].Project(cv::Point2d(margin - 0.5 + obj[bi][q].x, margin - 0.5 + y)) -
                                         cv::Point2d(corners[0][q]));
                    }
                    if (dsum < best) best = dsum, flip = f;
                }
            }
        }
        auto truth = [&](int id) { return truth_y(id, flip); };
        {
            std::vector<std::vector<cv::Point2f> > c2 = corners, r2 = rejected;
            std::vector<int> i2 = ids;
            std::vector<cv::Point2f> cc;
            std::vector<int> ci;
#if CALICO_ARUCO_MODERN
            CalicoInterpolateCharuco(img, board, c2, i2, r2, cc, ci, params);
#else
            cv::aruco::interpolateCornersCharuco(c2, i2, img, board, cc, ci);
#endif
            total += static_cast<int>(chess.size());
            for (size_t k = 0; k < ci.size(); ++k) e_legacy.push_back(cv::Point2d(cc[k]) - truth(ci[k]));
        }
        {
            std::vector<cv::Point2f> cc;
            std::vector<int> ci;
            RefineConfig cfg;
            cfg.gamma = kSceneGamma;
            RefineCharucoCorners(img, board, corners, ids, cc, ci, cfg);
            for (size_t k = 0; k < ci.size(); ++k) e_model.push_back(cv::Point2d(cc[k]) - truth(ci[k]));
        }
    }
    std::printf("  ChArUco 7x5 end-to-end (3 views, distortion, blur %.1f px, gamma 1/2.2, noise %.1f DN):\n", kSceneBlur, kSceneNoise);
    const Stats sl = Summarize(e_legacy, total), sm = Summarize(e_model, total);
    Print("OpenCV interpolation + cornerSubPix", sl);
    Print("local homography + X-junction model", sm);
    ASSERT_TRUE(sm.n > 0.9 * sl.n);
    ASSERT_TRUE(sm.median < sl.median);
    ASSERT_TRUE(sm.p95 < sl.p95);
}

TEST_CASE(TestGpuMatchesCpu) {
    if (!calico_cuda::CompiledWithCuda() || !calico_cuda::GpuAvailable()) {
        std::printf("  (skipped: %s)\n", calico_cuda::CompiledWithCuda() ? "no GPU visible" : "built without CUDA");
        return;
    }
    // A 1920 x 1200 frame tiled with the synthetic corners (X and L), 2400 fits.
    const int per_side_x = 30, per_side_y = 18;
    cv::Mat frame(per_side_y * 64, per_side_x * 64, CV_8UC1);
    std::vector<FitTask> tasks;
    std::vector<cv::Point2d> truth;
    const std::vector<CornerCase> xs = MakeCorners(kXJunction, per_side_x * per_side_y / 2, 21u);
    const std::vector<CornerCase> ls = MakeCorners(kLCorner, per_side_x * per_side_y / 2, 22u);
    for (int i = 0; i < per_side_x * per_side_y; ++i) {
        const CornerCase& c = (i % 2 == 0) ? xs[i / 2] : ls[i / 2];
        const int ox = (i % per_side_x) * 64, oy = (i / per_side_x) * 64;
        c.img.copyTo(frame(cv::Rect(ox, oy, 64, 64)));
        FitTask t = c.task;
        t.x += ox;
        t.y += oy;
        tasks.push_back(t);
        truth.push_back(c.truth + cv::Point2d(ox, oy));
    }
    // Repeat to 2400 tasks (fits are independent; repeats measure throughput).
    const size_t base = tasks.size();
    while (tasks.size() < 2400) tasks.push_back(tasks[tasks.size() % base]);
    RefineConfig cfg;
    cfg.gamma = 2.2;
    std::vector<FitResult> gpu, cpu;
    cfg.use_gpu = true;
    ASSERT_TRUE(FitCornersGPU(frame, tasks, gpu, cfg));  // warm-up (context creation)
    auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(FitCornersGPU(frame, tasks, gpu, cfg));
    auto t1 = std::chrono::steady_clock::now();
    cfg.use_gpu = false;
    FitCorners(frame, tasks, cpu, cfg);
    auto t2 = std::chrono::steady_clock::now();
    std::vector<FitResult> cpu1(tasks.size());
    const cv::Mat lin = Linearize(frame, 2.2);
    auto t3 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tasks.size(); ++i) cpu1[i] = FitCornerCPU(lin, tasks[i], cfg.fit);
    auto t4 = std::chrono::steady_clock::now();
    double maxd = 0;
    int flag_diff = 0, ok = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        flag_diff += gpu[i].ok != cpu[i].ok;
        if (gpu[i].ok && cpu[i].ok) {
            ++ok;
            maxd = std::max(maxd, std::hypot(gpu[i].x - cpu[i].x, gpu[i].y - cpu[i].y));
        }
    }
    const double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_cpu = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double ms_cpu1 = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::printf("  GPU (%s) vs CPU on %zu fits, %d both ok: max |dp| = %.3g px, ok-flag mismatches = %d\n",
                calico_cuda::DeviceName().c_str(), tasks.size(), ok, maxd, flag_diff);
    std::printf("  time: GPU %.1f ms (incl. upload + linearise), CPU OpenMP (%d threads) %.1f ms, CPU 1 thread %.1f ms\n",
                ms_gpu, omp_get_max_threads(), ms_cpu, ms_cpu1);
    // Device linearisation is the same LUT.
    ASSERT_TRUE(flag_diff <= static_cast<int>(tasks.size() / 500));  // LM accept/reject ties may differ in rare cases
    ASSERT_TRUE(maxd < 1e-6);
}
