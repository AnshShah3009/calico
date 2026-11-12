// Batched GPU corner refinement: gamma linearisation of the frame on the device, then one thread
// block per corner running the same Levenberg-Marquardt fit as FitCornerCPU (corner-model.hpp).
// The window (weights + linear values) is cached in shared memory; every LM iteration reduces
// the normal equations (cost, J^T J, J^T r: 36 doubles) over the block; thread 0 solves the
// damped 7x7 system and decides accept / reject. Same arithmetic as the CPU path, only the
// summation order differs, so results agree to rounding (see tests: < 1e-6 px).
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "corner-model.hpp"

using namespace calico_refine;

namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRed = 1 + kNumJtJ + kNumParams;  // cost, JtJ, g

__global__ void k_linearize(const unsigned char* gray, size_t pitch, float* lin, int cols, int rows,
                            const float* lut) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    lin[static_cast<size_t>(y) * cols + x] = lut[gray[static_cast<size_t>(y) * pitch + x]];
}

__device__ double WarpSum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum of `count` doubles per thread (vals), result in out[0..count) for every thread.
__device__ void BlockSum(double* vals, int count, double (*scratch)[kRed], double* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = 0; i < count; ++i) {
        const double s = WarpSum(vals[i]);
        if (lane == 0) scratch[warp][i] = s;
    }
    __syncthreads();
    if (threadIdx.x < count) {
        double s = 0;
        for (int w = 0; w < kWarps; ++w) s += scratch[w][threadIdx.x];
        out[threadIdx.x] = s;
    }
    __syncthreads();
}

__global__ void k_fit_corners(const float* lin, int cols, int rows, const FitTask* tasks, FitParams fp,
                              FitResult* results) {
    __shared__ float s_w[kMaxWindowPixels];
    __shared__ float s_v[kMaxWindowPixels];
    __shared__ double scratch[kWarps][kRed];
    __shared__ double red[kRed];
    __shared__ double P[kNumParams], Pn[kNumParams];
    __shared__ WindowGeom g;
    __shared__ int s_n, s_sat, s_stop, s_converged, s_it;
    __shared__ double s_wsum;

    const FitTask t = tasks[blockIdx.x];
    FitResult* f = &results[blockIdx.x];
    if (threadIdx.x == 0) {
        g = ComputeWindow(t, fp, cols, rows);
        s_n = 0;
        s_sat = 0;
        memset(f, 0, sizeof(FitResult));
        f->x = t.x;
        f->y = t.y;
        f->reason = g.reason;
    }
    __syncthreads();
    if (g.reason != kOk) return;

    const int side = 2 * g.R + 1, nbox = side * side;
    for (int i = threadIdx.x; i < nbox; i += kThreads) {
        const int x = g.x0 - g.R + i % side, y = g.y0 - g.R + i / side;
        float w;
        if (PixelWeight(t, g, x, y, &w)) {
            const float v = lin[static_cast<size_t>(y) * cols + x];
            s_w[i] = w;
            s_v[i] = v;
            atomicAdd(&s_n, 1);
            if (v >= fp.saturation) atomicAdd(&s_sat, 1);
        } else {
            s_w[i] = -1.0f;
        }
    }
    __syncthreads();
    const int n = s_n;
    if (n < 20 || s_sat > MaxSaturated(fp, t.type) * n) {
        if (threadIdx.x == 0) f->reason = n < 20 ? kWindow : kSaturated;
        return;
    }
    if (threadIdx.x == 0) {
        P[0] = t.x;
        P[1] = t.y;
        double a1, a2;
        InitialAngles(t, &a1, &a2);
        P[2] = a1;
        P[3] = a2;
        P[4] = 1.0;
        P[5] = P[6] = 0;
    }
    __syncthreads();

    // c, a by weighted linear least squares at the initial geometry.
    {
        double acc[5] = {0, 0, 0, 0, 0};
        double Pl[kNumParams];
        for (int d = 0; d < kNumParams; ++d) Pl[d] = P[d];
        const ModelState m = MakeState(Pl);
        for (int i = threadIdx.x; i < nbox; i += kThreads) {
            const float w = s_w[i];
            if (w < 0) continue;
            const double dx = (g.x0 - g.R + i % side) - Pl[0], dy = (g.y0 - g.R + i / side) - Pl[1];
            double M, Mh, Mk, Mr;
            ModelEval(t.type, (m.n1x * dx + m.n1y * dy) / m.s, (m.n2x * dx + m.n2y * dy) / m.s, m.r, &M, &Mh, &Mk, &Mr);
            const double v = s_v[i];
            acc[0] += w, acc[1] += w * M, acc[2] += w * M * M, acc[3] += w * v, acc[4] += w * M * v;
        }
        BlockSum(acc, 5, scratch, red);
        if (threadIdx.x == 0) {
            const double sw = red[0], se = red[1], see = red[2], sv = red[3], sev = red[4];
            s_wsum = sw;
            const double det = sw * see - se * se;
            s_stop = !(fabs(det) > 1e-12);
            if (!s_stop) {
                P[5] = (see * sv - se * sev) / det;
                P[6] = (sw * sev - se * sv) / det;
            } else {
                f->reason = kFlat;
            }
        }
        __syncthreads();
        if (s_stop) return;
    }

    // Normal equations at Q, reduced into red[] (every thread sees them after the call).
    auto accumulate = [&](const double* Qs) {
        double Q[kNumParams];
        for (int d = 0; d < kNumParams; ++d) Q[d] = Qs[d];
        const ModelState m = MakeState(Q);
        NormalEq e;
        ZeroNormalEq(&e);
        double J[kNumParams];
        for (int i = threadIdx.x; i < nbox; i += kThreads) {
            const float w = s_w[i];
            if (w < 0) continue;
            const double r = PixelResidual(t.type, Q, m, g.x0 - g.R + i % side, g.y0 - g.R + i / side, w, s_v[i], J);
            AddPixel(&e, r, J);
        }
        double vals[kRed];
        vals[0] = e.cost;
        for (int i = 0; i < kNumJtJ; ++i) vals[1 + i] = e.JtJ[i];
        for (int i = 0; i < kNumParams; ++i) vals[1 + kNumJtJ + i] = e.g[i];
        BlockSum(vals, kRed, scratch, red);
    };

    // Thread 0 keeps the accepted normal equations and lambda.
    NormalEq ne;
    double lambda = 1e-3;
    accumulate(P);
    if (threadIdx.x == 0) {
        ne.cost = red[0];
        for (int i = 0; i < kNumJtJ; ++i) ne.JtJ[i] = red[1 + i];
        for (int i = 0; i < kNumParams; ++i) ne.g[i] = red[1 + kNumJtJ + i];
        s_converged = 0;
        s_it = 0;
    }
    double step[kNumParams];
    for (int it = 0; it < fp.max_iterations; ++it) {
        __syncthreads();  // everyone has read s_stop of the previous iteration
        if (threadIdx.x == 0) {
            s_stop = 0;
            double Pc[kNumParams];
            for (int d = 0; d < kNumParams; ++d) Pc[d] = P[d];
            if (!LmStep(ne, lambda, Pc, step)) {
                lambda *= 10;
                s_stop = lambda > 1e8 ? 2 : 1;  // 2: give up, 1: retry with more damping
            } else {
                for (int d = 0; d < kNumParams; ++d) Pn[d] = P[d] + step[d];
                Pn[4] = fmax(kMinBlur, Pn[4]);
            }
            s_it = it + 1;
        }
        __syncthreads();
        if (s_stop == 2) break;
        if (s_stop == 1) continue;
        accumulate(Pn);
        if (threadIdx.x == 0) {
            if (red[0] < ne.cost) {
                const bool small = StepConverged(step, ne.cost, red[0]);
                for (int d = 0; d < kNumParams; ++d) P[d] = Pn[d];
                ne.cost = red[0];
                for (int i = 0; i < kNumJtJ; ++i) ne.JtJ[i] = red[1 + i];
                for (int i = 0; i < kNumParams; ++i) ne.g[i] = red[1 + kNumJtJ + i];
                lambda = fmax(1e-9, lambda * 0.3);
                if (small) {
                    s_converged = 1;
                    s_stop = 2;
                    s_it = it;
                }
            } else {
                lambda *= 10;
                if (lambda > 1e8 || RejectedStepConverged(step)) {
                    s_converged = 1;
                    s_stop = 2;
                    s_it = it;
                }
            }
        }
        __syncthreads();
        if (s_stop == 2) break;
    }
    if (threadIdx.x == 0) {
        double Pf[kNumParams];
        for (int d = 0; d < kNumParams; ++d) Pf[d] = P[d];
        FitResult r;
        memset(&r, 0, sizeof(r));
        FinishFit(t, g, fp, Pf, ne, n, s_wsum, s_converged != 0, &r);
        r.iterations = s_it;
        *f = r;
    }
}

}  // namespace

// Returns 0 on success, non-zero on any CUDA failure (caller falls back to the CPU).
extern "C" int calico_cuda_fit_corners(const unsigned char* gray, int cols, int rows, size_t step, double gamma,
                                       const FitTask* tasks, int n, const FitParams* params, FitResult* out) {
    if (gray == nullptr || cols <= 0 || rows <= 0 || n <= 0) return n == 0 ? 0 : -1;
    // Same LUT as calico_refine::Linearize (computed in double, stored as float).
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<float>(std::pow(i / 255.0, gamma));
    unsigned char* d_gray = nullptr;
    float* d_lin = nullptr;
    float* d_lut = nullptr;
    FitTask* d_tasks = nullptr;
    FitResult* d_out = nullptr;
    size_t pitch = 0;
    int rc = -1;
    do {
        if (cudaMallocPitch(&d_gray, &pitch, cols, rows) != cudaSuccess) break;
        if (cudaMemcpy2D(d_gray, pitch, gray, step, cols, rows, cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMalloc(&d_lin, sizeof(float) * static_cast<size_t>(cols) * rows) != cudaSuccess) break;
        if (cudaMalloc(&d_lut, sizeof(lut)) != cudaSuccess) break;
        if (cudaMemcpy(d_lut, lut, sizeof(lut), cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMalloc(&d_tasks, sizeof(FitTask) * n) != cudaSuccess) break;
        if (cudaMemcpy(d_tasks, tasks, sizeof(FitTask) * n, cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMalloc(&d_out, sizeof(FitResult) * n) != cudaSuccess) break;
        const dim3 b2(32, 8), g2((cols + 31) / 32, (rows + 7) / 8);
        k_linearize<<<g2, b2>>>(d_gray, pitch, d_lin, cols, rows, d_lut);
        if (cudaGetLastError() != cudaSuccess) break;
        k_fit_corners<<<n, kThreads>>>(d_lin, cols, rows, d_tasks, *params, d_out);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) break;
        if (cudaMemcpy(out, d_out, sizeof(FitResult) * n, cudaMemcpyDeviceToHost) != cudaSuccess) break;
        rc = 0;
    } while (false);
    cudaFree(d_gray);
    cudaFree(d_lin);
    cudaFree(d_lut);
    cudaFree(d_tasks);
    cudaFree(d_out);
    return rc;
}

// Device-side linearisation alone (exposed for tests): lin must hold cols * rows floats.
extern "C" int calico_cuda_linearize(const unsigned char* gray, int cols, int rows, size_t step, double gamma,
                                     float* lin) {
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<float>(std::pow(i / 255.0, gamma));
    unsigned char* d_gray = nullptr;
    float* d_lin = nullptr;
    float* d_lut = nullptr;
    size_t pitch = 0;
    int rc = -1;
    do {
        if (cudaMallocPitch(&d_gray, &pitch, cols, rows) != cudaSuccess) break;
        if (cudaMemcpy2D(d_gray, pitch, gray, step, cols, rows, cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMalloc(&d_lin, sizeof(float) * static_cast<size_t>(cols) * rows) != cudaSuccess) break;
        if (cudaMalloc(&d_lut, sizeof(lut)) != cudaSuccess) break;
        if (cudaMemcpy(d_lut, lut, sizeof(lut), cudaMemcpyHostToDevice) != cudaSuccess) break;
        const dim3 b2(32, 8), g2((cols + 31) / 32, (rows + 7) / 8);
        k_linearize<<<g2, b2>>>(d_gray, pitch, d_lin, cols, rows, d_lut);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) break;
        if (cudaMemcpy(lin, d_lin, sizeof(float) * static_cast<size_t>(cols) * rows, cudaMemcpyDeviceToHost) != cudaSuccess) break;
        rc = 0;
    } while (false);
    cudaFree(d_gray);
    cudaFree(d_lin);
    cudaFree(d_lut);
    return rc;
}
