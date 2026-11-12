// Model-based fiducial corner refinement (--corner-refine, --gamma).
//
// ChArUco chessboard corners are X-junctions (saddles); AprilTag / ArUco grid corners are
// L-corners (outer corner of a dark tag on a light background). Both are fitted with an exact
// blurred-wedge model on gamma-linearised intensities (corner-model.hpp), started from
// cornerSubPix (or from the detector / predicted corner where cornerSubPix wanders off).
// A corner whose refinement fails is dropped, never kept unrefined.
#ifndef CORNER_REFINE_HPP_
#define CORNER_REFINE_HPP_

#include "Includes.hpp"
#include "corner-model.hpp"

#include <array>

namespace calico_refine {

enum class RefineMode {
    kModel,   // cornerSubPix start, then the X-junction / L-corner model fit (default)
    kSubpix,  // cornerSubPix only (ChArUco: OpenCV's own interpolation + cornerSubPix)
    kNone     // legacy: ChArUco as OpenCV interpolates it; tag grids keep raw detector corners
};

struct RefineConfig {
    RefineMode mode = RefineMode::kModel;
    double gamma = 2.2;   // camera encoding gamma undone before the fit (1 = linear sensor)
    bool use_gpu = false; // batched CUDA fit (needs ENABLE_CUDA and a visible GPU)
    FitParams fit = DefaultFitParams();
};

// Process-wide configuration (set once from the command line).
RefineConfig& Config();

bool ParseMode(const std::string& s, RefineMode* mode);
const char* ModeName(RefineMode m);
const char* ReasonName(int reason);

// Linear intensities of an 8-bit grey frame (inverse gamma via a lookup table), CV_32F in [0, 1].
cv::Mat Linearize(const cv::Mat& gray, double gamma);

// One fit on the CPU. lin: CV_32F from Linearize.
FitResult FitCornerCPU(const cv::Mat& lin, const FitTask& task, const FitParams& params);

// Batch of fits on one frame: GPU (device-side linearisation, one thread block per corner) when
// cfg.use_gpu and a GPU is available, else CPU (OpenMP over corners). gray: CV_8UC1.
// *used_gpu (optional) reports which path ran.
void FitCorners(const cv::Mat& gray, const std::vector<FitTask>& tasks, std::vector<FitResult>& out,
                const RefineConfig& cfg, bool* used_gpu = nullptr);

// Same on the GPU only; false if the binary has no CUDA or the launch failed.
bool FitCornersGPU(const cv::Mat& gray, const std::vector<FitTask>& tasks, std::vector<FitResult>& out,
                   const RefineConfig& cfg);

// Board -> image Jacobian (row-major, px per board unit) of homography H (CV_64F) at (u, v).
void HomographyJacobian(const cv::Mat& H, double u, double v, double A[4]);

// Printed cells across one tag (AprilTag family / ArUco dictionary): the L-corner window unit.
int AprilTagCells(const std::string& family, bool rendered_by_apriltag_lib);
int ArucoCells(int arc_code);

// Removes every detection whose id occurs more than once in the frame (ambiguous). Returns count removed.
int DropDuplicateIds(std::vector<AprilTags::TagDetection>& dets);
int DropDuplicateIds(std::vector<std::vector<cv::Point2f> >& corners, std::vector<int>& ids);

// L-corner refinement of tag-grid detections (BL, BR, TR, TL per tag), in place.
// cells: printed cells across a tag; gap_tags / margin_tags: tag spacing and board margin in tag
// sides (the window stays inside them). keep[j][k] false = corner dropped.
void RefineTagCorners(const cv::Mat& gray, std::vector<AprilTags::TagDetection>& dets, int cells,
                      double gap_tags, double margin_tags, std::vector<std::array<bool, 4> >& keep,
                      const RefineConfig& cfg);

// ChArUco: chessboard corners predicted from the adjacent markers' local homographies, then
// cornerSubPix + X-junction fit. Only refined corners are returned (ids ascending).
int RefineCharucoCorners(const cv::Mat& gray, const cv::Ptr<cv::aruco::CharucoBoard>& board,
                         const std::vector<std::vector<cv::Point2f> >& marker_corners,
                         const std::vector<int>& marker_ids, std::vector<cv::Point2f>& corners,
                         std::vector<int>& ids, const RefineConfig& cfg);

// Counters of fit outcomes since start (thread-safe), printed with PrintStats.
void PrintStats(std::ostream& os);

}  // namespace calico_refine

#endif  // CORNER_REFINE_HPP_
