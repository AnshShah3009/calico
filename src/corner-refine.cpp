#include "corner-refine.hpp"
#include "cuda-detect.hpp"

#include <atomic>
#include <map>

#ifdef CALICO_USE_CUDA
extern "C" int calico_cuda_fit_corners(const unsigned char* gray, int cols, int rows, size_t step,
        double gamma, const calico_refine::FitTask* tasks, int n,
        const calico_refine::FitParams* params, calico_refine::FitResult* out);
#endif

namespace calico_refine {

namespace {
std::atomic<long> g_counts[kNumReasons];
std::atomic<long> g_dropped_duplicates{0};
std::atomic<long> g_subpix_fallbacks{0};

void Count(const FitResult& r) {
    if (r.reason >= 0 && r.reason < kNumReasons) g_counts[r.reason]++;
}
}  // namespace

RefineConfig& Config() {
    static RefineConfig cfg;
    return cfg;
}

bool ParseMode(const std::string& s, RefineMode* mode) {
    if (s == "model" || s == "xfit") *mode = RefineMode::kModel;
    else if (s == "subpix" || s == "cornersubpix") *mode = RefineMode::kSubpix;
    else if (s == "none" || s == "off" || s == "legacy") *mode = RefineMode::kNone;
    else return false;
    return true;
}

const char* ModeName(RefineMode m) {
    switch (m) {
    case RefineMode::kModel: return "model";
    case RefineMode::kSubpix: return "subpix";
    default: return "none";
    }
}

const char* ReasonName(int reason) {
    static const char* names[kNumReasons] = {"ok", "window", "border", "saturated", "flat",
            "diverged", "moved", "angles", "contrast", "polarity", "blur", "residual", "covariance"};
    return (reason >= 0 && reason < kNumReasons) ? names[reason] : "?";
}

cv::Mat Linearize(const cv::Mat& gray, double gamma) {
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<float>(std::pow(i / 255.0, gamma));
    cv::Mat g8;
    if (gray.type() == CV_8UC1) g8 = gray;
    else if (gray.channels() == 3) cv::cvtColor(gray, g8, cv::COLOR_BGR2GRAY);
    else gray.convertTo(g8, CV_8U);
    cv::Mat out(g8.size(), CV_32F);
    for (int r = 0; r < g8.rows; ++r) {
        const unsigned char* s = g8.ptr<unsigned char>(r);
        float* d = out.ptr<float>(r);
        for (int c = 0; c < g8.cols; ++c) d[c] = lut[s[c]];
    }
    return out;
}

FitResult FitCornerCPU(const cv::Mat& lin, const FitTask& t, const FitParams& fp) {
    FitResult f;
    std::memset(&f, 0, sizeof(f));
    f.x = t.x;
    f.y = t.y;
    const WindowGeom g = ComputeWindow(t, fp, lin.cols, lin.rows);
    if (g.reason != kOk) {
        f.reason = g.reason;
        return f;
    }
    // Window pixels (box order, row-major), weights and values in float as on the GPU.
    std::vector<int> px, py;
    std::vector<float> pw, pv;
    int saturated = 0;
    for (int y = g.y0 - g.R; y <= g.y0 + g.R; ++y) {
        for (int x = g.x0 - g.R; x <= g.x0 + g.R; ++x) {
            float w;
            if (!PixelWeight(t, g, x, y, &w)) continue;
            const float v = lin.at<float>(y, x);
            saturated += v >= fp.saturation;
            px.push_back(x);
            py.push_back(y);
            pw.push_back(w);
            pv.push_back(v);
        }
    }
    const int n = static_cast<int>(px.size());
    if (n < 20) {
        f.reason = kWindow;
        return f;
    }
    if (saturated > MaxSaturated(fp, t.type) * n) {
        f.reason = kSaturated;
        return f;
    }
    double P[kNumParams];
    P[0] = t.x;
    P[1] = t.y;
    InitialAngles(t, &P[2], &P[3]);
    P[4] = 1.0;
    P[5] = P[6] = 0;
    double wsum = 0;
    {
        // c, a by weighted linear least squares at the initial geometry.
        const ModelState m = MakeState(P);
        double sw = 0, se = 0, see = 0, sv = 0, sev = 0;
        for (int i = 0; i < n; ++i) {
            const double dx = px[i] - P[0], dy = py[i] - P[1];
            double M, Mh, Mk, Mr;
            ModelEval(t.type, (m.n1x * dx + m.n1y * dy) / m.s, (m.n2x * dx + m.n2y * dy) / m.s, m.r, &M, &Mh, &Mk, &Mr);
            const double w = pw[i], v = pv[i];
            sw += w, se += w * M, see += w * M * M, sv += w * v, sev += w * M * v;
        }
        wsum = sw;
        const double det = sw * see - se * se;
        if (!(std::fabs(det) > 1e-12)) {
            f.reason = kFlat;
            return f;
        }
        P[5] = (see * sv - se * sev) / det;
        P[6] = (sw * sev - se * sv) / det;
    }
    auto accumulate = [&](const double* Q, NormalEq* e) {
        ZeroNormalEq(e);
        const ModelState m = MakeState(Q);
        double J[kNumParams];
        for (int i = 0; i < n; ++i) {
            const double r = PixelResidual(t.type, Q, m, px[i], py[i], pw[i], pv[i], J);
            AddPixel(e, r, J);
        }
    };
    NormalEq ne, nn;
    accumulate(P, &ne);
    double lambda = 1e-3;
    bool converged = false;
    int it = 0;
    for (; it < fp.max_iterations; ++it) {
        double step[kNumParams];
        if (!LmStep(ne, lambda, P, step)) {
            lambda *= 10;
            if (lambda > 1e8) break;
            continue;
        }
        double Pn[kNumParams];
        for (int d = 0; d < kNumParams; ++d) Pn[d] = P[d] + step[d];
        Pn[4] = std::max(kMinBlur, Pn[4]);
        accumulate(Pn, &nn);
        if (nn.cost < ne.cost) {
            const bool small = StepConverged(step, ne.cost, nn.cost);
            std::copy(Pn, Pn + kNumParams, P);
            ne = nn;
            lambda = std::max(1e-9, lambda * 0.3);
            if (small) {
                converged = true;
                break;
            }
        } else {
            lambda *= 10;
            if (lambda > 1e8 || RejectedStepConverged(step)) {
                converged = true;  // no further decrease: at the minimum
                break;
            }
        }
    }
    f.iterations = it;
    FinishFit(t, g, fp, P, ne, n, wsum, converged, &f);
    return f;
}

bool FitCornersGPU(const cv::Mat& gray, const std::vector<FitTask>& tasks, std::vector<FitResult>& out,
                   const RefineConfig& cfg) {
#ifdef CALICO_USE_CUDA
    if (tasks.empty()) {
        out.clear();
        return true;
    }
    if (!calico_cuda::GpuAvailable() || gray.type() != CV_8UC1) return false;
    out.resize(tasks.size());
    return calico_cuda_fit_corners(gray.ptr<unsigned char>(), gray.cols, gray.rows, gray.step, cfg.gamma,
                                   tasks.data(), static_cast<int>(tasks.size()), &cfg.fit, out.data()) == 0;
#else
    (void)gray;
    (void)tasks;
    (void)out;
    (void)cfg;
    return false;
#endif
}

void FitCorners(const cv::Mat& gray_in, const std::vector<FitTask>& tasks, std::vector<FitResult>& out,
                const RefineConfig& cfg, bool* used_gpu) {
    if (used_gpu) *used_gpu = false;
    out.assign(tasks.size(), FitResult());
    if (tasks.empty()) return;
    cv::Mat gray;
    if (gray_in.type() == CV_8UC1) gray = gray_in;
    else if (gray_in.channels() == 3) cv::cvtColor(gray_in, gray, cv::COLOR_BGR2GRAY);
    else gray_in.convertTo(gray, CV_8U);
    bool done = false;
    if (cfg.use_gpu) {
        done = FitCornersGPU(gray, tasks, out, cfg);
        if (used_gpu) *used_gpu = done;
    }
    if (!done) {
        const cv::Mat lin = Linearize(gray, cfg.gamma);
#pragma omp parallel for schedule(dynamic, 4) if (!omp_in_parallel() && tasks.size() > 16)
        for (int i = 0; i < static_cast<int>(tasks.size()); ++i) out[i] = FitCornerCPU(lin, tasks[i], cfg.fit);
    }
    for (const FitResult& r : out) Count(r);
    if (std::getenv("CALICO_REFINE_DEBUG")) {
        int c[kNumReasons] = {0};
        for (const FitResult& r : out) c[r.reason]++;
#pragma omp critical(calico_refine_debug)
        {
            std::cerr << "[corner-refine] " << tasks.size() << " fits (" << (used_gpu && *used_gpu ? "GPU" : "CPU") << "):";
            for (int i = 0; i < kNumReasons; ++i) {
                if (c[i]) std::cerr << " " << ReasonName(i) << "=" << c[i];
            }
            std::cerr << std::endl;
        }
    }
}

void HomographyJacobian(const cv::Mat& H, double u, double v, double A[4]) {
    const double X = H.at<double>(0, 0) * u + H.at<double>(0, 1) * v + H.at<double>(0, 2);
    const double Y = H.at<double>(1, 0) * u + H.at<double>(1, 1) * v + H.at<double>(1, 2);
    const double W = H.at<double>(2, 0) * u + H.at<double>(2, 1) * v + H.at<double>(2, 2);
    for (int c = 0; c < 2; ++c) {
        A[0 * 2 + c] = (H.at<double>(0, c) * W - X * H.at<double>(2, c)) / (W * W);
        A[1 * 2 + c] = (H.at<double>(1, c) * W - Y * H.at<double>(2, c)) / (W * W);
    }
}

static cv::Point2d ApplyH(const cv::Mat& H, double u, double v) {
    const double W = H.at<double>(2, 0) * u + H.at<double>(2, 1) * v + H.at<double>(2, 2);
    return cv::Point2d((H.at<double>(0, 0) * u + H.at<double>(0, 1) * v + H.at<double>(0, 2)) / W,
                       (H.at<double>(1, 0) * u + H.at<double>(1, 1) * v + H.at<double>(1, 2)) / W);
}

int AprilTagCells(const std::string& family, bool rendered_by_apriltag_lib) {
    // AprilRobotics-rendered tags (calico --create-patterns for tag36h11 / tag25h9 / tag16h5)
    // span total_width cells with the outer ring painted black: 10 / 9 / 8 cells.
    // OpenCV-rendered DICT_APRILTAG_* markers span markerSize + 2 cells.
    if (family.find("16h5") != std::string::npos) return rendered_by_apriltag_lib ? 8 : 6;
    if (family.find("25h") != std::string::npos) return rendered_by_apriltag_lib ? 9 : 7;
    if (family.find("36h10") != std::string::npos) return 8;
    return rendered_by_apriltag_lib ? 10 : 8;  // 36h11 / 36h9
}

int ArucoCells(int arc_code) {
    cv::Ptr<cv::aruco::Dictionary> d = CalicoGetDictionary(arc_code);
    return d->markerSize + 2;  // borderBits = 1
}

int DropDuplicateIds(std::vector<AprilTags::TagDetection>& dets) {
    std::map<int, int> count;
    for (const auto& d : dets) ++count[d.id];
    std::vector<AprilTags::TagDetection> kept;
    for (const auto& d : dets) {
        if (count[d.id] == 1) kept.push_back(d);
    }
    const int removed = static_cast<int>(dets.size() - kept.size());
    g_dropped_duplicates += removed;
    dets.swap(kept);
    return removed;
}

int DropDuplicateIds(std::vector<std::vector<cv::Point2f> >& corners, std::vector<int>& ids) {
    std::map<int, int> count;
    for (int id : ids) ++count[id];
    std::vector<std::vector<cv::Point2f> > kc;
    std::vector<int> ki;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (count[ids[i]] != 1) continue;
        kc.push_back(corners[i]);
        ki.push_back(ids[i]);
    }
    const int removed = static_cast<int>(ids.size() - ki.size());
    g_dropped_duplicates += removed;
    corners.swap(kc);
    ids.swap(ki);
    return removed;
}

// cornerSubPix of one point; false if it moved more than max_move or left the image.
static bool SubPix(const cv::Mat& gray, cv::Point2d& p, int half, double max_move) {
    if (!(p.x >= half + 1 && p.y >= half + 1 && p.x < gray.cols - half - 1 && p.y < gray.rows - half - 1)) return false;
    std::vector<cv::Point2f> v = {cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y))};
    cv::cornerSubPix(gray, v, cv::Size(half, half), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 50, 1e-3));
    const cv::Point2d r(v[0].x, v[0].y);
    if (!std::isfinite(r.x) || !std::isfinite(r.y) || cv::norm(r - p) > max_move) return false;
    p = r;
    return true;
}

static void ToGray8(const cv::Mat& in, cv::Mat& out) {
    if (in.type() == CV_8UC1) out = in;
    else if (in.channels() == 3) cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    else in.convertTo(out, CV_8U);
}

void RefineTagCorners(const cv::Mat& gray_in, std::vector<AprilTags::TagDetection>& dets, int cells,
                      double gap_tags, double margin_tags, std::vector<std::array<bool, 4> >& keep,
                      const RefineConfig& cfg) {
    keep.assign(dets.size(), {true, true, true, true});
    if (cfg.mode == RefineMode::kNone || dets.empty()) return;
    cv::Mat gray;
    ToGray8(gray_in, gray);
    const double m = 1.0 / std::max(1, cells);  // module (tag sides)
    // Window radius: inside the black border (>= 1 module deep, the data bits start beyond
    // sqrt2 modules on the diagonal) and inside the white gap / margin around the tag.
    double cut = 1.4 * m;
    if (gap_tags > 0) cut = std::min(cut, 0.9 * gap_tags);
    if (margin_tags > 0) cut = std::min(cut, 0.9 * margin_tags);
    const double uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};  // BL, BR, TR, TL (u right, v up)
    std::vector<FitTask> tasks;
    std::vector<std::pair<int, int> > where;
    for (size_t j = 0; j < dets.size(); ++j) {
        std::vector<cv::Point2f> src, dst;
        for (int k = 0; k < 4; ++k) {
            src.emplace_back(static_cast<float>(uv[k][0]), static_cast<float>(uv[k][1]));
            dst.emplace_back(dets[j].p[k].first, dets[j].p[k].second);
        }
        const cv::Mat H = cv::getPerspectiveTransform(src, dst);
        for (int k = 0; k < 4; ++k) {
            double J[4];
            HomographyJacobian(H, uv[k][0], uv[k][1], J);
            // Dark quadrant = toward the tag centre: flip the axes that point outward.
            const double su = uv[k][0] == 0 ? 1 : -1, sv = uv[k][1] == 0 ? 1 : -1;
            FitTask t;
            t.A[0] = J[0] * su, t.A[1] = J[1] * sv, t.A[2] = J[2] * su, t.A[3] = J[3] * sv;
            double smax, smin;
            SingularValues2(t.A, &smax, &smin);
            const cv::Point2d raw(dets[j].p[k].first, dets[j].p[k].second);
            cv::Point2d p = raw;
            const int half = std::max(2, std::min(9, static_cast<int>(std::lround(smin * m))));
            const bool sub_ok = SubPix(gray, p, half, std::max(1.0, 0.5 * smax * m));
            if (cfg.mode == RefineMode::kSubpix) {
                keep[j][k] = sub_ok;
                if (sub_ok) dets[j].p[k] = std::make_pair(static_cast<float>(p.x), static_cast<float>(p.y));
                continue;
            }
            if (!sub_ok) {
                p = raw;  // cornerSubPix wandered off: start the model fit from the detector corner
                g_subpix_fallbacks++;
            }
            t.x = p.x;
            t.y = p.y;
            t.rho = 0.5 * cut;
            t.cut = cut;
            t.max_move = 0.5 * m;
            t.type = kLCorner;
            t.polarity = 1;
            tasks.push_back(t);
            where.emplace_back(static_cast<int>(j), k);
        }
    }
    if (cfg.mode != RefineMode::kModel) return;
    std::vector<FitResult> res;
    FitCorners(gray, tasks, res, cfg);
    for (size_t i = 0; i < tasks.size(); ++i) {
        const int j = where[i].first, k = where[i].second;
        keep[j][k] = res[i].ok != 0;
        if (res[i].ok) dets[j].p[k] = std::make_pair(static_cast<float>(res[i].x), static_cast<float>(res[i].y));
    }
}

int RefineCharucoCorners(const cv::Mat& gray_in, const cv::Ptr<cv::aruco::CharucoBoard>& board,
                         const std::vector<std::vector<cv::Point2f> >& marker_corners,
                         const std::vector<int>& marker_ids, std::vector<cv::Point2f>& corners_out,
                         std::vector<int>& ids_out, const RefineConfig& cfg) {
    corners_out.clear();
    ids_out.clear();
    if (marker_ids.empty()) return 0;
    cv::Mat gray;
    ToGray8(gray_in, gray);
    const std::vector<int> board_ids = CharucoIds(board);
    const std::vector<std::vector<cv::Point3f> > obj = CharucoObjPoints(board);
    const std::vector<cv::Point3f> chess = CharucoChessboardCorners(board);
    const std::vector<std::vector<int> > nearest = CharucoNearestMarkerIdx(board);
    if (chess.empty() || obj.empty()) return 0;
    // Printed geometry (board units) from the layout itself.
    const double marker_len = cv::norm(obj[0][1] - obj[0][0]);
    double sq = 1e300;
    for (size_t i = 1; i < chess.size(); ++i) sq = std::min(sq, static_cast<double>(cv::norm(chess[i] - chess[0])));
    if (!(sq < 1e300) || sq <= marker_len) return 0;
    const double module = 0.5 * (sq - marker_len) / 1.5;  // window cut (1.5 modules) = the gap to the markers

    // Marker homographies (board xy -> image), unique ids only.
    std::map<int, int> count;
    for (int id : marker_ids) ++count[id];
    std::map<int, cv::Mat> H;  // board marker index -> homography
    for (size_t k = 0; k < marker_ids.size(); ++k) {
        if (count[marker_ids[k]] != 1 || marker_corners[k].size() != 4) continue;
        const auto it = std::find(board_ids.begin(), board_ids.end(), marker_ids[k]);
        if (it == board_ids.end()) continue;
        const int bidx = static_cast<int>(it - board_ids.begin());
        std::vector<cv::Point2f> src;
        for (int q = 0; q < 4; ++q) src.emplace_back(obj[bidx][q].x, obj[bidx][q].y);
        H[bidx] = cv::getPerspectiveTransform(src, marker_corners[k]);
    }
    if (H.empty()) return 0;

    std::vector<FitTask> tasks;
    std::vector<int> task_ids;
    std::vector<cv::Point2d> predicted;
    std::vector<int> halfw;
    std::vector<double> scale;
    for (size_t i = 0; i < chess.size(); ++i) {
        const double X = chess[i].x, Y = chess[i].y;
        std::vector<cv::Point2d> pred;
        double s_px = 0;
        double Am[4] = {0, 0, 0, 0};
        for (int midx : nearest[i]) {
            const auto it = H.find(midx);
            if (it == H.end()) continue;
            const cv::Point2d p = ApplyH(it->second, X, Y);
            pred.push_back(p);
            double J[4];
            HomographyJacobian(it->second, X, Y, J);
            for (int q = 0; q < 4; ++q) Am[q] += J[q];
            s_px += 0.5 * (cv::norm(ApplyH(it->second, X + sq, Y) - p) + cv::norm(ApplyH(it->second, X, Y + sq) - p));
        }
        if (pred.empty()) continue;
        const double np = static_cast<double>(pred.size());
        s_px /= np;
        for (double& a : Am) a /= np;
        cv::Point2d mean(0, 0);
        for (const auto& p : pred) mean += p * (1.0 / np);
        bool consistent = true;
        for (const auto& p : pred) consistent = consistent && cv::norm(p - mean) < 0.12 * s_px;
        if (!consistent || s_px < 6.0) continue;
        if (!(mean.x >= 0 && mean.y >= 0 && mean.x < gray.cols && mean.y < gray.rows)) continue;
        // Polarity from the layout: markers sit in the white squares.
        int polarity = 1;
        if (!nearest[i].empty()) {
            cv::Point3f c(0, 0, 0);
            for (const auto& q : obj[nearest[i][0]]) c += q * 0.25f;
            if ((c.x - X) * (c.y - Y) > 0) polarity = -1;  // (+u,+v) / (-u,-v) white -> the others black
        }
        FitTask t;
        std::memcpy(t.A, Am, sizeof(Am));
        t.rho = 0.7 * module;
        t.cut = 1.5 * module;
        t.max_move = 0.5 * module;
        t.type = kXJunction;
        t.polarity = polarity;
        t.x = mean.x;
        t.y = mean.y;
        const double gap_px = 0.5 * (sq - marker_len) / sq * s_px;
        halfw.push_back(std::max(2, std::min(12, static_cast<int>(std::floor(gap_px - 0.5)))));
        scale.push_back(s_px);
        predicted.push_back(mean);
        tasks.push_back(t);
        task_ids.push_back(static_cast<int>(i));
    }
    std::vector<bool> ok(tasks.size(), false);
    for (size_t k = 0; k < tasks.size(); ++k) {
        cv::Point2d p = predicted[k];
        ok[k] = SubPix(gray, p, halfw[k], std::max(1.0, 0.25 * scale[k]));
        if (ok[k]) {
            tasks[k].x = p.x;
            tasks[k].y = p.y;
        } else {
            g_subpix_fallbacks++;
        }
    }
    if (cfg.mode == RefineMode::kModel) {
        std::vector<FitResult> res;
        FitCorners(gray, tasks, res, cfg);
        for (size_t k = 0; k < tasks.size(); ++k) {
            ok[k] = res[k].ok != 0;
            tasks[k].x = res[k].x;
            tasks[k].y = res[k].y;
        }
    }
    for (size_t k = 0; k < tasks.size(); ++k) {
        if (!ok[k]) continue;
        ids_out.push_back(task_ids[k]);
        corners_out.emplace_back(static_cast<float>(tasks[k].x), static_cast<float>(tasks[k].y));
    }
    return static_cast<int>(ids_out.size());
}

void PrintStats(std::ostream& os) {
    long total = 0;
    for (int i = 0; i < kNumReasons; ++i) total += g_counts[i];
    os << "Corner refinement (" << ModeName(Config().mode) << ", gamma " << Config().gamma << "): " << total
       << " model fits";
    for (int i = 0; i < kNumReasons; ++i) {
        if (g_counts[i] > 0) os << ", " << ReasonName(i) << "=" << g_counts[i];
    }
    os << "; cornerSubPix fallbacks=" << g_subpix_fallbacks << ", duplicate-id drops=" << g_dropped_duplicates
       << std::endl;
}

}  // namespace calico_refine
