#ifndef CUDA_DETECT_HPP_
#define CUDA_DETECT_HPP_

#include "Includes.hpp"

extern int g_use_cuda;

namespace calico_cuda {

bool CompiledWithCuda();
bool GpuAvailable();
std::string DeviceName();
bool ProbeGpuGray();

void BgrToGray(const cv::Mat& bgr, cv::Mat& gray);

// Returns true if a CUDA AprilTag backend filled detections.
// When false, callers should run the CPU Kaess/AprilRobotics detector on `gray`.
bool DetectAprilTags(const cv::Mat& gray, const std::string& family,
        std::vector<AprilTags::TagDetection>& detections);

void ApplyCeresOptions(ceres::Solver::Options& options, bool use_cuda);

}

#endif
