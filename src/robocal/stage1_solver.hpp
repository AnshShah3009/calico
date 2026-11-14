// Structure-exploiting Levenberg-Marquardt for the stage-1 (intrinsics) problem:
//   free_views camera: one 6-DoF pose per board view (eliminated by Schur, 6x6 per view),
//   static boards with fixed poses, per-board shape (5) and per-point heights, shared lens.
// Same objective as the Ceres problem RobotCalibrator::BuildProblem builds for that case:
//   1/2 sum_v rho_v(|w (proj - obs)|^2) + 1/2 |shape prior|^2 + 1/2 |height prior|^2,
// rho = Huber on the whole view when solver.robust: view (Ceres' per-residual-block loss; its corrector
// reduces to a plain sqrt(rho') scaling for Huber), else identity with per-corner IRLS weights View.pw. Ceres forms every view's 192-wide heights Jacobian block
// densely (EBlockRowOuterProduct, O(points^2) per view); here a point touches only its own
// height, so the reduced system is assembled in O(points * (13 + points)) per view and solved
// densely (it is ~1400 x 1400 for 7 boards x 192 points + lens).
#ifndef ROBOCAL_STAGE1_SOLVER_HPP_
#define ROBOCAL_STAGE1_SOLVER_HPP_

#include "camera_models.hpp"
#include "view_cost.hpp"

#include <map>
#include <string>
#include <vector>

namespace robocal {

struct S1Camera {
    LensModel model = LensModel::Equidistant;
    double* intr = nullptr;
    int n_intr = 0;
    bool free = false;
};

struct S1Board {
    Iso3 pose = Iso3::Identity();  // world -> board, held
    BoardGeometry geom;
    double* shape = nullptr;
    bool shape_free = false;
    bool scale_free = true;  // false: shape[3], shape[4] held (SubsetManifold)
    bool shape_prior = false;
    double inv_sigma_shape = 0, inv_sigma_scale = 0, scale0[2] = {0, 0};
    std::vector<double>* heights = nullptr;  // nullptr: none
    bool heights_free = false;               // free => prior 1 / point sigma
    double inv_sigma_height = 0;
};

struct S1View {
    const View* view = nullptr;
    double* pose = nullptr;  // q[4] (x y z w), t[3]: world <- camera
    int camera = 0;
    int board = 0;
    bool heights = false;  // this view's points carry their board heights
    double weight = 1.0;   // 1 / pixel noise; corner i also carries view->pw[i] (IRLS, sqrt weight)
    double mult = 1.0;     // bootstrap: the view enters this many times
};

struct S1Options {
    int max_iterations = 100;
    double function_tolerance = 1e-12;
    double parameter_tolerance = 1e-12;
    double gradient_tolerance = 1e-12;
    double huber = 0;  // 0: no loss
    int threads = 8;
    bool verbose = false;
    // Covariance mode: no solve; the marginal covariance (tangent space, unscaled) of each free lens /
    // shape block = the matching block of S^-1, S = the reduced (Schur) matrix at the current point.
    std::map<const double*, Eigen::MatrixXd>* covariance = nullptr;
};

struct S1Summary {
    double initial_cost = 0, final_cost = 0;
    int iterations = 0;
    bool converged = false;
    bool failed = false;
    std::string message;
    double t_linearize = 0, t_solve = 0, t_eval = 0;
};

S1Summary SolveStage1(std::vector<S1Camera>& cams, std::vector<S1Board>& boards, std::vector<S1View>& views,
                      const S1Options& o);

}  // namespace robocal

#endif  // ROBOCAL_STAGE1_SOLVER_HPP_
