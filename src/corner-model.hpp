// Model-based corner refinement: shared host / device math.
//
// Two corner types, both fitted by Levenberg-Marquardt on LINEAR (gamma-undone) intensities:
//
//   X-junction (chessboard saddle):  I(p) = c + a * S(h, k; rho)
//   L-corner  (one dark quadrant):   I(p) = c + a * Phi2(h, k; rho)
//
//   h = n1 . (p - p0) / s,  k = n2 . (p - p0) / s,  n_i = (cos t_i, sin t_i),  rho = n1 . n2
//
// The ideal pattern is the indicator of the wedge {n1.q > 0, n2.q > 0} (L) or the product of
// the two edge signs (X); blurring with an isotropic Gaussian of sigma s gives exactly the
// bivariate normal CDF Phi2 with correlation rho = n1.n2 (and S = 4 Phi2 - 2 Phi(h) - 2 Phi(k)
// + 1 = E[sgn sgn]). For perpendicular edges (rho = 0) these reduce to Phi(h) Phi(k) and
// erf(h/sqrt2) erf(k/sqrt2); for foreshortened (skewed) corners they stay exact, which matters
// for L-corners (an approximate product model biases a skewed L-corner, a saddle stays symmetric).
//
// 7 parameters: p0 (x, y), t1, t2, s, c, a. This header is compiled both by the C++ compiler
// (CPU path) and by nvcc (GPU path), so the CPU and GPU fits run the same arithmetic.
#ifndef CORNER_MODEL_HPP_
#define CORNER_MODEL_HPP_

#include <math.h>

#ifdef __CUDACC__
#define CALICO_HD __host__ __device__ __forceinline__
#else
#define CALICO_HD inline
#endif

namespace calico_refine {

enum CornerType { kXJunction = 0, kLCorner = 1 };

enum FitReason {
    kOk = 0,
    kWindow = 1,      // window too small / too few pixels
    kBorder = 2,      // window leaves the image
    kSaturated = 3,   // most of the window clipped (glare)
    kFlat = 4,        // no contrast for the initial linear solve
    kDiverged = 5,
    kMoved = 6,       // left the prediction by more than max_move
    kAngles = 7,      // edge directions far from the prediction
    kContrast = 8,
    kPolarity = 9,
    kBlur = 10,       // blur comparable to the window
    kResidual = 11,
    kCovariance = 12,
    kNumReasons = 13
};

// One corner to refine. Units: "board units" are whatever the caller's local board frame uses
// (tag sides, board millimetres, ...); A maps them to pixels.
struct FitTask {
    double x, y;        // start (pixels)
    double A[4];        // board -> image Jacobian at the corner, row-major [dx/du dx/dv; dy/du dy/dv].
                        // The quadrant (+u, +v) is: L-corner -> the dark one; X -> the polarity reference.
    double rho;         // Gaussian window sigma (board units)
    double cut;         // window radius (board units); the next printed structure must lie beyond it
    double max_move;    // allowed move from (x, y) (board units)
    int type;           // CornerType
    int polarity;       // X only: +1 if quadrants (+u,+v) / (-u,-v) are black, -1 otherwise
};

struct FitResult {
    int ok;
    int reason;         // FitReason
    double x, y;        // refined corner (pixels)
    double cov[3];      // position covariance xx, xy, yy (pixels^2)
    double blur;        // fitted sigma (pixels)
    double contrast;    // |a| (linear)
    double residual;    // weighted rms / |a|
    int iterations;
};

struct FitParams {
    double saturation;     // linear level counted as clipped
    double max_saturated;  // X: clipped share of the window tolerated (whites alone, ~1/2, do not bias a saddle; glare does)
    double max_saturated_l;  // L: same for L-corners, whose window is ~3/4 white
    double min_contrast;   // |a| (linear)
    double max_residual;   // weighted rms / |a|
    double min_short_px;   // shortest window radius (pixels)
    double max_window_px;  // longest window radius (pixels); larger windows are scaled down
    double max_blur_frac;  // blur gate: s <= max_blur_frac * short window radius
    int max_iterations;
};

CALICO_HD FitParams DefaultFitParams() {
    FitParams p;
    p.saturation = 0.97;
    p.max_saturated = 0.6;
    p.max_saturated_l = 0.85;
    p.min_contrast = 0.03;
    p.max_residual = 0.25;
    p.min_short_px = 3.0;
    p.max_window_px = 30.0;
    p.max_blur_frac = 0.45;
    p.max_iterations = 40;
    return p;
}

CALICO_HD double MaxSaturated(const FitParams& p, int type) {
    return type == kLCorner ? p.max_saturated_l : p.max_saturated;
}

// Window radius cap in pixels -> half box size. Keeps GPU shared memory bounded:
// (2 * kMaxHalfBox + 1)^2 <= kMaxWindowPixels.
constexpr int kMaxHalfBox = 31;
constexpr int kMaxWindowPixels = (2 * kMaxHalfBox + 1) * (2 * kMaxHalfBox + 1);
constexpr int kNumParams = 7;
constexpr int kNumJtJ = 28;   // upper triangle of 7x7

// Geometry of one task's window, derived on host or device from the task alone.
struct WindowGeom {
    double Ainv[4];
    double rho, cut;        // possibly scaled down by max_window_px
    double short_px, long_px;
    int R, x0, y0;          // box [x0-R, x0+R] x [y0-R, y0+R]
    int reason;             // kOk, kWindow or kBorder
};

CALICO_HD void SingularValues2(const double* A, double* smax, double* smin) {
    // Singular values of a 2x2 matrix (closed form).
    const double a = A[0], b = A[1], c = A[2], d = A[3];
    const double s1 = a * a + b * b + c * c + d * d;
    const double det = a * d - b * c;
    const double disc = sqrt(fmax(0.0, s1 * s1 - 4.0 * det * det));
    *smax = sqrt(0.5 * (s1 + disc));
    *smin = sqrt(fmax(0.0, 0.5 * (s1 - disc)));
}

CALICO_HD WindowGeom ComputeWindow(const FitTask& t, const FitParams& p, int cols, int rows) {
    WindowGeom g;
    g.reason = kOk;
    g.R = g.x0 = g.y0 = 0;
    const double det = t.A[0] * t.A[3] - t.A[1] * t.A[2];
    if (!(fabs(det) > 1e-300) || !isfinite(det)) {
        g.reason = kWindow;
        return g;
    }
    g.Ainv[0] = t.A[3] / det;
    g.Ainv[1] = -t.A[1] / det;
    g.Ainv[2] = -t.A[2] / det;
    g.Ainv[3] = t.A[0] / det;
    double smax, smin;
    SingularValues2(t.A, &smax, &smin);
    g.rho = t.rho;
    g.cut = t.cut;
    double long_px = smax * g.cut;
    const double cap = fmin(p.max_window_px, (double)kMaxHalfBox - 1.0);
    if (long_px > cap) {
        const double f = cap / long_px;
        g.rho *= f;
        g.cut *= f;
        long_px = cap;
    }
    g.long_px = long_px;
    g.short_px = smin * g.cut;
    if (g.short_px < p.min_short_px) {
        g.reason = kWindow;
        return g;
    }
    g.R = (int)ceil(long_px) + 1;
    if (g.R > kMaxHalfBox) g.R = kMaxHalfBox;
    if (!isfinite(t.x) || !isfinite(t.y)) {
        g.reason = kBorder;
        return g;
    }
    g.x0 = (int)floor(t.x);
    g.y0 = (int)floor(t.y);
    if (g.x0 - g.R < 0 || g.y0 - g.R < 0 || g.x0 + g.R >= cols || g.y0 + g.R >= rows) g.reason = kBorder;
    return g;
}

// Window weight of pixel (x, y); returns false outside the elliptical cut.
CALICO_HD bool PixelWeight(const FitTask& t, const WindowGeom& g, int x, int y, float* w) {
    const double dx = x - t.x, dy = y - t.y;
    const double bu = g.Ainv[0] * dx + g.Ainv[1] * dy;
    const double bv = g.Ainv[2] * dx + g.Ainv[3] * dy;
    const double q2 = bu * bu + bv * bv;
    if (q2 > g.cut * g.cut) return false;
    *w = (float)exp(-0.5 * q2 / (g.rho * g.rho));
    return true;
}

// Initial normals from the board axes: n1 crosses the u-edge toward +v, n2 the v-edge toward +u.
CALICO_HD void InitialAngles(const FitTask& t, double* th1, double* th2) {
    double ux = t.A[0], uy = t.A[2], vx = t.A[1], vy = t.A[3];
    const double nu = sqrt(ux * ux + uy * uy), nv = sqrt(vx * vx + vy * vy);
    ux /= nu, uy /= nu, vx /= nv, vy /= nv;
    double n1x = -uy, n1y = ux, n2x = -vy, n2y = vx;
    if (n1x * vx + n1y * vy < 0) n1x = -n1x, n1y = -n1y;
    if (n2x * ux + n2y * uy < 0) n2x = -n2x, n2y = -n2y;
    *th1 = atan2(n1y, n1x);
    *th2 = atan2(n2y, n2x);
}

// ----------------------------------------------------------------------- bivariate normal
CALICO_HD double NormPdf(double x) { return 0.3989422804014327 * exp(-0.5 * x * x); }
CALICO_HD double NormCdf(double x) { return 0.5 * erfc(-x * 0.7071067811865476); }

// Phi2(h, k; r) = Phi(h) Phi(k) + 1/(2 pi) int_0^{asin r} exp(-(h^2 + k^2 - 2hk sin t) / (2 cos^2 t)) dt
// (Drezner-Wesolowsky / Genz). Gauss-Legendre: 12 nodes for |r| < 0.75, 20 up to the clamp 0.95.
CALICO_HD double Phi2(double h, double k, double r) {
    const double base = NormCdf(h) * NormCdf(k);
    if (fabs(r) < 1e-14) return base;
    // Gauss-Legendre nodes / weights on [-1, 1] (positive half; symmetric).
    const double x6[6] = {0.1252334085114689, 0.3678314989981802, 0.5873179542866175,
                          0.7699026741943047, 0.9041172563704749, 0.9815606342467192};
    const double w6[6] = {0.2491470458134028, 0.2334925365383548, 0.2031674267230659,
                          0.1600783285433462, 0.1069393259953184, 0.0471753363865118};
    const double x10[10] = {0.0765265211334973, 0.2277858511416451, 0.3737060887154195,
                            0.5108670019508271, 0.6360536807265150, 0.7463319064601508,
                            0.8391169718222188, 0.9122344282513259, 0.9639719272779138,
                            0.9931285991850949};
    const double w10[10] = {0.1527533871307258, 0.1491729864726037, 0.1420961093183820,
                            0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
                            0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
                            0.0176140071391521};
    const double asr = asin(r);
    const double half = 0.5 * asr;
    const double hk = h * k, hs = 0.5 * (h * h + k * k);
    double sum = 0;
    const bool small = fabs(r) < 0.75;
    const int n = small ? 6 : 10;
    for (int i = 0; i < n; ++i) {
        const double xi = small ? x6[i] : x10[i];
        const double wi = small ? w6[i] : w10[i];
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const double sn = sin(half * (1.0 + sgn * xi));
            sum += wi * exp((sn * hk - hs) / (1.0 - sn * sn));
        }
    }
    return base + sum * half * 0.15915494309189535;
}

// Model value M and its partials (dM/dh, dM/dk, dM/drho) for one pixel.
CALICO_HD void ModelEval(int type, double h, double k, double r, double* M, double* Mh, double* Mk, double* Mr) {
    const double q = sqrt(1.0 - r * r);
    const double P2 = Phi2(h, k, r);
    const double ph = NormPdf(h), pk = NormPdf(k);
    const double d2h = ph * NormCdf((k - r * h) / q);
    const double d2k = pk * NormCdf((h - r * k) / q);
    const double d2r = exp(-(h * h - 2.0 * r * h * k + k * k) / (2.0 * q * q)) * 0.15915494309189535 / q;
    if (type == kLCorner) {
        *M = P2;
        *Mh = d2h;
        *Mk = d2k;
        *Mr = d2r;
    } else {
        *M = 4.0 * P2 - 2.0 * NormCdf(h) - 2.0 * NormCdf(k) + 1.0;
        *Mh = 4.0 * d2h - 2.0 * ph;
        *Mk = 4.0 * d2k - 2.0 * pk;
        *Mr = 4.0 * d2r;
    }
}

constexpr double kMaxRho = 0.95;  // |n1.n2| clamp for the evaluation (edges >= ~18 deg apart)

// Per-parameter-vector quantities shared by every pixel.
struct ModelState {
    double n1x, n1y, n2x, n2y, t1x, t1y, t2x, t2y, r, dr1, dr2, s;
};

CALICO_HD ModelState MakeState(const double* P) {
    ModelState m;
    m.n1x = cos(P[2]), m.n1y = sin(P[2]);
    m.n2x = cos(P[3]), m.n2y = sin(P[3]);
    m.t1x = -m.n1y, m.t1y = m.n1x;
    m.t2x = -m.n2y, m.t2y = m.n2x;
    m.r = cos(P[2] - P[3]);
    const double sd = sin(P[2] - P[3]);
    m.dr1 = -sd;
    m.dr2 = sd;
    if (m.r > kMaxRho) m.r = kMaxRho, m.dr1 = m.dr2 = 0;
    if (m.r < -kMaxRho) m.r = -kMaxRho, m.dr1 = m.dr2 = 0;
    m.s = P[4];
    return m;
}

// Weighted residual r = sqrt(w) (model - v) and its Jacobian row J[7].
CALICO_HD double PixelResidual(int type, const double* P, const ModelState& m, double px, double py,
                               float w, float v, double* J) {
    const double dx = px - P[0], dy = py - P[1];
    const double h = (m.n1x * dx + m.n1y * dy) / m.s;
    const double k = (m.n2x * dx + m.n2y * dy) / m.s;
    double M, Mh, Mk, Mr;
    ModelEval(type, h, k, m.r, &M, &Mh, &Mk, &Mr);
    const double sw = sqrt((double)w);
    const double a = P[6];
    if (J) {
        J[0] = sw * a * (Mh * (-m.n1x) + Mk * (-m.n2x)) / m.s;
        J[1] = sw * a * (Mh * (-m.n1y) + Mk * (-m.n2y)) / m.s;
        J[2] = sw * a * (Mh * (m.t1x * dx + m.t1y * dy) / m.s + Mr * m.dr1);
        J[3] = sw * a * (Mk * (m.t2x * dx + m.t2y * dy) / m.s + Mr * m.dr2);
        J[4] = sw * a * (-Mh * h - Mk * k) / m.s;
        J[5] = sw;
        J[6] = sw * M;
    }
    return sw * (P[5] + a * M - (double)v);
}

// Accumulator for cost, J^T J (upper triangle, row-major) and J^T r.
struct NormalEq {
    double cost;
    double JtJ[kNumJtJ];
    double g[kNumParams];
};

CALICO_HD void ZeroNormalEq(NormalEq* e) {
    e->cost = 0;
    for (int i = 0; i < kNumJtJ; ++i) e->JtJ[i] = 0;
    for (int i = 0; i < kNumParams; ++i) e->g[i] = 0;
}

CALICO_HD void AddPixel(NormalEq* e, double r, const double* J) {
    e->cost += r * r;
    int q = 0;
    for (int i = 0; i < kNumParams; ++i) {
        e->g[i] += J[i] * r;
        for (int j = i; j < kNumParams; ++j) e->JtJ[q++] += J[i] * J[j];
    }
}

CALICO_HD double JtJAt(const double* u, int i, int j) {
    if (i > j) {
        const int tmp = i;
        i = j;
        j = tmp;
    }
    // Row i of the upper triangle starts at i*7 - i*(i-1)/2.
    return u[i * kNumParams - i * (i - 1) / 2 + (j - i)];
}

// Cholesky solve of (JtJ + lambda diag) x = b; false if not positive definite.
CALICO_HD bool SolveDamped(const double* JtJ, double lambda, const double* b, double* x) {
    double L[kNumParams][kNumParams];
    for (int i = 0; i < kNumParams; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = JtJAt(JtJ, i, j);
            if (i == j) s += lambda * fmax(1e-9, JtJAt(JtJ, i, i));
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (!(s > 0)) return false;
                L[i][i] = sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    double y[kNumParams];
    for (int i = 0; i < kNumParams; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    for (int i = kNumParams - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < kNumParams; ++k) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return true;
}

constexpr double kMinBlur = 0.25;  // lower bound of the blur sigma (pixels): ~ pixel-area integration

// Convergence tests. Accepted step: corner step below 1e-4 px, or below 5e-3 px with a relative
// cost decrease below 1e-4. Rejected step: a corner step below 1e-3 px (no better point that
// close). Very sharp (aliased) edges make the cost non-smooth at the 1e-3 px scale, where LM
// would otherwise zig-zag until max_iterations.
CALICO_HD bool StepConverged(const double* step, double cost_old, double cost_new) {
    const double dp = sqrt(step[0] * step[0] + step[1] * step[1]);
    return dp < 1e-4 || (dp < 5e-3 && cost_old - cost_new <= 1e-4 * cost_old);
}
CALICO_HD bool RejectedStepConverged(const double* step) {
    return sqrt(step[0] * step[0] + step[1] * step[1]) < 1e-3;
}

// One damped Gauss-Newton step. The blur sigma has a lower bound: when it sits on the bound and
// the step would push it further down, it is frozen for this step (active set), otherwise the
// clamped step never shrinks and LM stalls on very sharp (unblurred, synthetic) edges.
CALICO_HD bool LmStep(const NormalEq& ne, double lambda, const double* P, double* step) {
    double negg[kNumParams];
    for (int d = 0; d < kNumParams; ++d) negg[d] = -ne.g[d];
    if (!SolveDamped(ne.JtJ, lambda, negg, step)) return false;
    if (P[4] + step[4] >= kMinBlur) return true;
    if (P[4] > kMinBlur * (1.0 + 1e-9)) {
        step[4] = kMinBlur - P[4];  // reach the bound this step
        return true;
    }
    double J2[kNumJtJ];
    int q = 0;
    for (int i = 0; i < kNumParams; ++i) {
        for (int j = i; j < kNumParams; ++j, ++q) {
            J2[q] = (i == 4 || j == 4) ? (i == j ? 1.0 : 0.0) : ne.JtJ[q];
        }
    }
    negg[4] = 0;
    if (!SolveDamped(J2, lambda, negg, step)) return false;
    step[4] = 0;
    return true;
}

// Closes a fit: covariance and the gates. P: final parameters; ne: normal equations at P;
// npx: window pixel count; wsum: sum of weights; converged: LM stopped normally.
CALICO_HD void FinishFit(const FitTask& t, const WindowGeom& g, const FitParams& fp, const double* P,
                         const NormalEq& ne, int npx, double wsum, bool converged, FitResult* f) {
    f->x = P[0];
    f->y = P[1];
    f->blur = P[4];
    f->contrast = fabs(P[6]);
    f->residual = sqrt(ne.cost / fmax(1e-12, wsum)) / fmax(1e-6, f->contrast);
    const double dof = fmax(1.0, (double)npx - kNumParams);
    const double sigma2 = ne.cost / dof;
    double c0[kNumParams], c1[kNumParams], e0[kNumParams] = {1, 0, 0, 0, 0, 0, 0},
                                             e1[kNumParams] = {0, 1, 0, 0, 0, 0, 0};
    const bool inv_ok = SolveDamped(ne.JtJ, 0.0, e0, c0) && SolveDamped(ne.JtJ, 0.0, e1, c1);
    f->cov[0] = inv_ok ? c0[0] * sigma2 : -1;
    f->cov[1] = inv_ok ? c0[1] * sigma2 : 0;
    f->cov[2] = inv_ok ? c1[1] * sigma2 : -1;
    double th1, th2;
    InitialAngles(t, &th1, &th2);
    const double d1 = cos(P[2] - th1), d2 = cos(P[3] - th2);
    const double mx = P[0] - t.x, my = P[1] - t.y;
    const double bu = g.Ainv[0] * mx + g.Ainv[1] * my, bv = g.Ainv[2] * mx + g.Ainv[3] * my;
    const double moved = sqrt(bu * bu + bv * bv);
    f->ok = 0;
    if (!converged || !isfinite(P[0]) || !isfinite(P[1])) f->reason = kDiverged;
    else if (moved > t.max_move) f->reason = kMoved;
    else if ((t.type == kLCorner && (d1 < 0.8 || d2 < 0.8)) || fabs(d1) < 0.8 || fabs(d2) < 0.8) f->reason = kAngles;
    else if (fabs(cos(P[2] - P[3])) > 0.9) f->reason = kAngles;
    else if (f->contrast < fp.min_contrast) f->reason = kContrast;
    // L: the wedge must be the dark side (a < 0). X: the black quadrants must be where the
    // board has them (a's sign per the orientation of the normals relative to the start).
    else if (t.type == kLCorner ? P[6] >= 0 : (P[6] * (d1 > 0 ? 1 : -1) * (d2 > 0 ? 1 : -1)) * t.polarity > 0)
        f->reason = kPolarity;
    else if (f->blur > fp.max_blur_frac * g.short_px) f->reason = kBlur;
    else if (f->residual > fp.max_residual) f->reason = kResidual;
    else if (!(f->cov[0] >= 0 && f->cov[2] >= 0)) f->reason = kCovariance;
    else {
        f->ok = 1;
        f->reason = kOk;
    }
}

}  // namespace calico_refine

#endif  // CORNER_MODEL_HPP_
