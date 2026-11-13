// Board detection on raw frames: Kalibr AprilGrids (Kaess or AprilTag 3 with a Kalibr-border
// family), ChArUco (markers + local-homography corner prediction + saddle refinement) and plain
// checkerboards (whole board only). Every detector reports board points (BoardSpec::Point ids).
#ifndef ROBOCAL_DETECT_HPP_
#define ROBOCAL_DETECT_HPP_

#include "boards.hpp"
#include "camera_models.hpp"

#include <Eigen/Dense>

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace robocal {

struct BoardDetection {
    int board = -1;                            // index into the board list
    std::vector<int> pids;                     // BoardSpec::Point ids
    std::vector<Vec2> pts;                     // their image positions
    std::vector<Eigen::Matrix2d> covs;         // per-point covariance (px^2) when the X-junction fit ran
    // AprilGrid only: whole tags (focus measurement), BL, BR, TR, TL per tag.
    std::vector<int> ids;
    std::vector<std::array<Vec2, 4>> corners;
};

// Model-based X-junction (chessboard saddle) refinement:
//   I(p) = c + a erf(l1 / (sqrt2 s)) erf(l2 / (sqrt2 s)),  l_k = n_k . (p - p0)
// fitted by Levenberg-Marquardt on linearised intensities over an elliptical window sized in board
// units (the local board -> image Jacobian A). Centro-symmetric, so unbiased under blur, gamma and
// exposure; handles foreshortened (non-orthogonal) corners; gives a covariance and the blur.
struct XJunctionOptions {
    double rho_modules = 0.7;      // Gaussian window radius (board modules)
    double cut_modules = 1.5;      // hard cut (the next printed structure is >= 2 modules away)
    double min_short_axis_px = 3.0;
    double saturation = 0.97;      // linear level counted as saturated
    double max_saturated = 0.6;    // clipped share allowed: whites alone (~half) do not bias the saddle; glare does
    double max_move_modules = 0.5;
    double min_contrast = 0.03;    // |a| (linear)
    double max_residual = 0.25;    // weighted rms / |a| (gross failures only; motion blur raises it)
    int max_iterations = 40;
};

struct XJunctionFit {
    bool ok = false;
    std::string reason;          // why not ok
    Vec2 p = Vec2::Zero();
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    double blur_px = 0, contrast = 0, residual = 0;
};

// lin: CV_32F linear intensities. A: board -> image Jacobian at the corner (px per m; columns =
// board x, y). module_m: printed cell size around the saddle. black_xy: +1 if the board
// quadrants (+x, +y) / (-x, -y) around the corner are black, else -1 (polarity check).
XJunctionFit FitXJunction(const cv::Mat& lin, const Vec2& p_init, const Eigen::Matrix2d& A, double module_m,
                          int black_xy, const XJunctionOptions& o = XJunctionOptions());

struct DetectOptions {
    bool subpixel = true;      // corner refinement (cornerSubPix with a per-corner window)
    int min_tags_per_board = 2;      // aprilgrid: tags; charuco: markers
    int min_points_per_board = 6;    // any board: points after refinement
    double max_subpixel_move = 0.25;  // fraction of the local feature size
    std::string apriltag = "kaess";   // aprilgrid detector: kaess | apriltag3
    std::string refine = "xfit";      // corner refinement: xfit (model fit after cornerSubPix) | subpix
    double gamma = 2.2;               // camera encoding, undone before the X-junction fit
    // aprilgrid: spacer-square parallelogram test. Off by default: close to a fisheye the spacer
    // (3 modules, 20-40 px) is not affine -- its closure error reaches 0.5-1.6 px.
    bool check_spacers = false;
    bool guided = true;               // after the intrinsics prepass: refine every predicted corner
    bool rectified = true;            // guided pass also re-detects tags in virtual pinhole views
    std::string rectified_cache;      // jsonl of those tags per image: read if present, else written
    int guided_rounds = 3;            // fit -> add corners -> refit, until < 1 % new corners
    XJunctionOptions xfit;
};

// cv::aruco predefined dictionary code from its name (DICT_5X5_1000, DICT_APRILTAG_36h11, ...).
int ArucoDictionaryCode(const std::string& name);

// Linear intensities of an 8-bit frame (undoes the camera's gamma encoding).
cv::Mat LinearizeGray(const cv::Mat& gray, double gamma);
// Saddle polarity of board point pid: +1 if the board quadrants (+x, +y) / (-x, -y) are black.
int SaddlePolarity(const BoardSpec& b, int pid);
// Printed cell around a saddle (board metres): the unit the X-junction window is sized in.
double SaddleModule(const BoardSpec& b);
// Refines a predicted board point (e.g. projected through a fitted camera model) in a frame:
// cornerSubPix, then the X-junction fit; false unless it lands within max_move_modules of the
// prediction with the board's polarity. A: board -> image Jacobian at the point (px per m).
bool RefinePredictedSaddle(const cv::Mat& gray, const cv::Mat& lin, const BoardSpec& b, int pid, Vec2& p,
                           const Eigen::Matrix2d& A, const DetectOptions& opts, double max_move_modules,
                           Eigen::Matrix2d* cov = nullptr);

// TartanCalib-style re-detection with a fitted lens: the frame is resampled into virtual pinhole
// views (the centre, rings at 50 and 80 deg off axis) where tags squeezed at the edge of a fisheye
// decode again; their corners are mapped back and refined on the ORIGINAL pixels (no resampling
// bias). AprilGrid boards only. Build once per lens, reuse for every frame.
// A decoded tag: id and corners (BL, BR, TR, TL) in original image pixels.
struct TagCorners {
    int id;
    std::array<Vec2, 4> p;
};
using TagsByFamily = std::map<std::string, std::vector<TagCorners>>;

class RectifiedDetector {
public:
    explicit RectifiedDetector(const CameraIntrinsics& intr, int size = 640, double fov_deg = 90.0);
    std::vector<BoardDetection> Detect(const cv::Mat& gray, const std::vector<BoardSpec>& boards,
                                       const DetectOptions& opts) const;
    // The two halves of Detect: tags decoded in the virtual views (mapped back to the image), and
    // the board detection (corner refinement in the original image) seeded with them.
    TagsByFamily FindTags(const cv::Mat& gray, const std::vector<BoardSpec>& boards, const DetectOptions& opts) const;
    static std::vector<BoardDetection> DetectGiven(const cv::Mat& gray, const std::vector<BoardSpec>& boards,
                                                   const DetectOptions& opts, const TagsByFamily& given);

private:
    struct View {
        Mat3 R;  // virtual -> camera
        cv::Mat mapx, mapy;
    };
    CameraIntrinsics intr_;
    int size_;
    double f_;
    std::vector<View> views_;
};

// Detects every configured board in one grayscale frame.
std::vector<BoardDetection> DetectBoards(const cv::Mat& gray, const std::vector<BoardSpec>& boards,
                                         const DetectOptions& opts);

}  // namespace robocal

#endif  // ROBOCAL_DETECT_HPP_
