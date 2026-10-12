#include "cuda-detect.hpp"
#include <cstring>

#ifdef CALICO_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#ifdef CALICO_HAVE_CUAPRILTAGS
#include <cuAprilTags.h>
#endif

int g_use_cuda = 0;

#ifdef CALICO_USE_CUDA
extern "C" int calico_cuda_bgr_to_gray(const unsigned char* bgr, unsigned char* gray,
        int width, int height);
#endif

namespace calico_cuda {

bool CompiledWithCuda() {
#ifdef CALICO_USE_CUDA
    return true;
#else
    return false;
#endif
}

bool GpuAvailable() {
#ifdef CALICO_USE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
#else
    return false;
#endif
}

string DeviceName() {
#ifdef CALICO_USE_CUDA
    if (!GpuAvailable()) {
        return "";
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "";
    }
    return string(prop.name);
#else
    return "";
#endif
}

#ifdef CALICO_USE_CUDA
static void LogGpuGrayOnce(const char* backend) {
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;
    cout << "GPU BGR→gray: " << backend << endl;
}
#endif

void BgrToGray(const cv::Mat& bgr, cv::Mat& gray) {
    if (bgr.type() != CV_8UC3) {
        if (bgr.channels() == 1) {
            gray = bgr;
        } else {
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        }
        return;
    }

#ifdef CALICO_USE_CUDA
    if (g_use_cuda && GpuAvailable()) {
#ifdef HAVE_OPENCV_CUDAIMGPROC
        try {
            cv::cuda::GpuMat d_bgr, d_gray;
            d_bgr.upload(bgr);
            cv::cuda::cvtColor(d_bgr, d_gray, cv::COLOR_BGR2GRAY);
            d_gray.download(gray);
            LogGpuGrayOnce("OpenCV cudaimgproc");
            return;
        } catch (const cv::Exception&) {
            // Fall through to CALICO kernel / CPU.
        }
#endif
        cv::Mat bgr_c = bgr.isContinuous() ? bgr : bgr.clone();
        gray.create(bgr.rows, bgr.cols, CV_8UC1);
        if (calico_cuda_bgr_to_gray(bgr_c.ptr<unsigned char>(), gray.ptr<unsigned char>(),
                bgr.cols, bgr.rows) == 0) {
            LogGpuGrayOnce("calico CUDA kernel");
            return;
        }
        gray.release();
    }
#endif
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
}

bool ProbeGpuGray() {
    cv::Mat bgr(32, 32, CV_8UC3, cv::Scalar(10, 20, 250));
    cv::Mat gray;
    BgrToGray(bgr, gray);
    return !gray.empty() && gray.rows == 32 && gray.cols == 32 && gray.channels() == 1;
}

#ifdef CALICO_HAVE_CUAPRILTAGS
static bool DetectWithCuAprilTags(const cv::Mat& gray, const string& family,
        vector<AprilTags::TagDetection>& detections) {
    if (!g_use_cuda || !GpuAvailable() || gray.empty()) {
        return false;
    }
    if (family != "tag36h11" && NormalizeAprilFamily(family) != "tag36h11") {
        cout << "cuAprilTags CUDA backend supports tag36h11; falling back to CPU for "
             << family << endl;
        return false;
    }

    // cuAprilTags expects device uchar3 RGB (pitch in bytes), not single-channel gray.
    cv::Mat rgb;
    if (gray.channels() == 1) {
        cv::cvtColor(gray, rgb, cv::COLOR_GRAY2RGB);
    } else if (gray.type() == CV_8UC3) {
        cv::cvtColor(gray, rgb, cv::COLOR_BGR2RGB);
    } else {
        return false;
    }
    cv::Mat contiguous = rgb.isContinuous() ? rgb : rgb.clone();

    cuAprilTagsHandle handle = nullptr;
    if (nvCreateAprilTagsDetector(&handle, contiguous.cols, contiguous.rows, 4,
            NVAT_TAG36H11, nullptr, 1.0f) != 0 || handle == nullptr) {
        return false;
    }

    const uint32_t max_tags = 256;
    vector<cuAprilTagsID_t> tags(max_tags);
    uint32_t num_tags = 0;
    cuAprilTagsImageInput_t input;
    memset(&input, 0, sizeof(input));
    input.width = static_cast<uint16_t>(contiguous.cols);
    input.height = static_cast<uint16_t>(contiguous.rows);

    int rc = -1;
#ifdef HAVE_OPENCV_CUDAIMGPROC
    cv::cuda::GpuMat d_rgb;
    d_rgb.upload(contiguous);
    input.pitch = d_rgb.step;
    input.dev_ptr = reinterpret_cast<uchar3*>(d_rgb.ptr());
    rc = cuAprilTagsDetect(handle, &input, tags.data(), &num_tags, max_tags, nullptr);
#elif defined(CALICO_USE_CUDA)
    unsigned char* d_rgb = nullptr;
    const size_t nbytes = static_cast<size_t>(contiguous.rows) *
            static_cast<size_t>(contiguous.cols) * 3u;
    if (cudaMalloc(&d_rgb, nbytes) == cudaSuccess &&
            cudaMemcpy(d_rgb, contiguous.ptr<unsigned char>(), nbytes,
                    cudaMemcpyHostToDevice) == cudaSuccess) {
        input.dev_ptr = reinterpret_cast<uchar3*>(d_rgb);
        input.pitch = static_cast<size_t>(contiguous.cols) * sizeof(uchar3);
        rc = cuAprilTagsDetect(handle, &input, tags.data(), &num_tags, max_tags, nullptr);
    }
    if (d_rgb != nullptr) {
        cudaFree(d_rgb);
    }
#else
    (void)input;
#endif
    cuAprilTagsDestroy(handle);
    if (rc != 0) {
        return false;
    }

    detections.clear();
    detections.reserve(num_tags);
    for (uint32_t i = 0; i < num_tags; i++) {
        // Reject tags with bit errors; Kaess/OpenCV paths only keep confident hits.
        if (tags[i].hamming_error > 0) {
            continue;
        }
        AprilTags::TagDetection d;
        d.id = static_cast<int>(tags[i].id);
        d.good = true;
        d.hammingDistance = static_cast<int>(tags[i].hamming_error);
        // cuAprilTags matches the AprilTag/Kaess corner convention used by
        // ConvertAprilMarkerIdIndexToGridPointIndex: BL, BR, TR, TL (CCW).
        // OpenCV DICT_APRILTAG_* uses TL-first and is remapped in DetectAprilTagsOpenCV.
        for (int k = 0; k < 4; k++) {
            d.p[k].first = tags[i].corners[k].x;
            d.p[k].second = tags[i].corners[k].y;
        }
        detections.push_back(d);
    }
    cout << "cuAprilTags GPU detector: " << detections.size() << " tags" << endl;
    return !detections.empty();
}
#endif

bool DetectAprilTags(const cv::Mat& gray, const string& family,
        vector<AprilTags::TagDetection>& detections) {
#ifdef CALICO_HAVE_CUAPRILTAGS
    return DetectWithCuAprilTags(gray, family, detections);
#else
    (void)gray;
    (void)family;
    (void)detections;
    return false;
#endif
}

void ApplyCeresOptions(ceres::Solver::Options& options, bool use_cuda) {
    if (!use_cuda) {
        return;
    }
#ifndef CALICO_USE_CUDA
    cout << "Ceres: --use-cuda set but this binary was built without CUDA." << endl;
    return;
#else
    if (!GpuAvailable()) {
        cout << "Ceres: --use-cuda set but no NVIDIA GPU is visible; using CPU." << endl;
        return;
    }
#ifdef CERES_USE_CUDA
    options.dense_linear_algebra_library_type = ceres::CUDA;
    cout << "Ceres dense linear algebra: CUDA" << endl;
#else
    cout << "Ceres: this Ceres build has no CUDA dense solver; using CPU." << endl;
    (void)options;
#endif
#endif
}

}
