// Focus setting for fixed-focus (M12-style) lenses, done once before the intrinsics session.
//
// A lens focused at s blurs a point at distance d into a disc of diameter
//     c = A * f_px * |1/d - 1/s|      (A: aperture diameter, f_px: focal length in px)
// so over an operating range [near, far] the worst blur is smallest, and equal at both ends,
// when 1/s = (1/near + 1/far) / 2: s is the harmonic mean of near and far.
//
// Sharpness is measured on the AprilGrid itself: the black tag borders are straight
// black/white edges; an edge-spread function (erf) is fitted across them and its sigma (px)
// is the blur. Board distance comes from a rough PnP (focus is set before intrinsics exist,
// so a nominal focal length is enough: the distance only has to be right to a few percent).
#ifndef ROBOCAL_FOCUS_HPP_
#define ROBOCAL_FOCUS_HPP_

#include "boards.hpp"
#include "camera_models.hpp"
#include "detect.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace robocal {

// Focus distance with the smallest worst-case defocus over [near, far].
inline double FocusTarget(double near_m, double far_m) { return 2.0 * near_m * far_m / (near_m + far_m); }

struct FocusOptions {
    double max_off_axis_deg = 35.0;  // tags further off axis are not used (field blur, curvature)
    double gamma = 2.2;              // camera encoding; profiles are linearised before the fit
};

struct FocusMeasure {
    bool ok = false;
    int n_tags = 0;
    int n_edges = 0;
    double blur_sigma_px = 0.0;  // median edge-spread sigma over the fitted tag borders (px)
    double blur_spread_px = 0.0; // median absolute deviation of those sigmas
    double distance_m = 0.0;     // camera to the seen part of the board (rough intrinsics)
    double off_axis_deg = 0.0;   // that point's angle from the optical axis
};

// Blur and distance of one board in one frame. ``rough`` needs only a nominal focal length /
// centre for the lens model; ``det`` is that board's detection in ``gray``.
FocusMeasure MeasureFocus(const cv::Mat& gray, const BoardSpec& board, const BoardDetection& det,
                          const CameraIntrinsics& rough, const FocusOptions& opts = FocusOptions());

// Edge-spread sigma (px) of one straight black/white edge crossing ``p`` with unit normal
// ``n``; half window ``w`` px. < 0 when no clean edge is found. ``lin`` is the linearised image.
double FitEdgeSigma(const cv::Mat& lin, const Vec2& p, const Vec2& n, double w);

}  // namespace robocal

#endif  // ROBOCAL_FOCUS_HPP_
