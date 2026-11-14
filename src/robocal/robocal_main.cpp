// calico-robocal: offline joint calibration of arm joint zeros, wrist / static cameras,
// arm bases and boards from a calibration run's detected board corners.
#include "robocal.hpp"
#include "log_time.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>

namespace {

void Usage() {
    std::cout
        << "Usage: calico-robocal --config=<robocal.yaml> [--out=<dir>] [--verbose]\n\n"
           "Solves, in one problem: per-joint zero offsets (q_model = q_encoder + delta_q),\n"
           "wrist camera mounts, static cameras, arm bases (one fixed as the world gauge),\n"
           "board poses, and optionally intrinsics (last pass).\n\n"
           "Writes <out>/robot_calib.yaml (workcell_calib_v1-style arms / sensors / boards plus\n"
           "diagnostics: per-camera train / hold-out RMS, weak parameter directions, 1-sigma).\n"
           "See configs/robocal/README.md for the config format.\n\n"
           "       calico-robocal --compare-intrinsics=a.camera_info.yaml,b.camera_info.yaml\n"
           "                      [--max-diff-px=0.30] [--max-rms-px=0.14] [--max-raw-px=0.60] [--distance-m=d]\n"
           "                      [--theta-deg=85] [--align=rotation|none]\n"
           "Repeatability of two intrinsics fits of one camera (two sessions, or two halves of one).\n"
           "Prints, per 5 deg ring, the pixel difference raw, after the best rotation of b onto a\n"
           "(a camera rotation is absorbed by the stage-2 mount, so it is not a lens difference) and\n"
           "with b's principal point set to a's. --align=rotation (default) gates the rotation-aligned\n"
           "max / rms and the raw max (--max-raw-px); --align=none gates the raw max only.\n"
           "Exit 3 if they differ too much.\n"
           "Exit codes: 3 intrinsics gates failed, 4 robot solve failed its sanity gate.\n";
}

std::string ArgValue(const std::string& arg, const std::string& key) {
    const std::string prefix = "--" + key + "=";
    return arg.rfind(prefix, 0) == 0 ? arg.substr(prefix.size()) : std::string();
}

}  // namespace

int main(int argc, char** argv) {
    robocal::LogClockStart();
    std::string config, out = ".", compare;
    double compare_max = 0.30, compare_rms = 0.14, compare_raw = 0.60, compare_theta = 85.0, compare_dist = 0.0;
    std::string compare_align = "rotation";
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            Usage();
            return 0;
        }
        if (a == "--verbose") verbose = true;
        else if (!ArgValue(a, "config").empty()) config = ArgValue(a, "config");
        else if (!ArgValue(a, "out").empty()) out = ArgValue(a, "out");
        else if (!ArgValue(a, "compare-intrinsics").empty()) compare = ArgValue(a, "compare-intrinsics");
        else if (!ArgValue(a, "max-diff-px").empty()) compare_max = std::stod(ArgValue(a, "max-diff-px"));
        else if (!ArgValue(a, "theta-deg").empty()) compare_theta = std::stod(ArgValue(a, "theta-deg"));
        else if (!ArgValue(a, "max-rms-px").empty()) compare_rms = std::stod(ArgValue(a, "max-rms-px"));
        else if (!ArgValue(a, "max-raw-px").empty()) compare_raw = std::stod(ArgValue(a, "max-raw-px"));
        else if (!ArgValue(a, "distance-m").empty()) compare_dist = std::stod(ArgValue(a, "distance-m"));
        else if (!ArgValue(a, "align").empty()) compare_align = ArgValue(a, "align");
        else {
            std::cerr << "unknown argument: " << a << "\n";
            Usage();
            return 2;
        }
    }
    if (!compare.empty()) {  // repeatability of two intrinsics sessions
        const size_t c = compare.find(',');
        if (c == std::string::npos) {
            std::cerr << "--compare-intrinsics=a.yaml,b.yaml\n";
            return 2;
        }
        try {
            const auto a = robocal::LoadCameraInfo(compare.substr(0, c), "a");
            const auto b = robocal::LoadCameraInfo(compare.substr(c + 1), "b");
            if (compare_align != "rotation" && compare_align != "none") {
                std::cerr << "--align: rotation or none\n";
                return 2;
            }
            const robocal::IntrinsicsComparison c = robocal::CompareIntrinsicsDetailed(a, b, compare_theta, compare_dist);
            std::cout << std::fixed << std::setprecision(3) << "b - a: fx " << c.d_params[0] << " fy " << c.d_params[1]
                      << " cx " << c.d_params[2] << " cy " << c.d_params[3] << " px; best rotation of b onto a "
                      << std::setprecision(4) << c.rotation_deg << " deg\n" << std::setprecision(3)
                      << "  ring (deg)   rays   raw max / rms   rotation-aligned max / rms   pp-matched max / rms\n";
            for (size_t i = 0; i < c.rings.size(); ++i) {
                const auto& g = c.rings[i];
                if (!g.n) continue;
                std::cout << "  " << std::setw(3) << 5 * i << "-" << std::setw(3) << 5 * (i + 1) << "  " << std::setw(6)
                          << g.n;
                for (int k = 0; k < 3; ++k) {
                    std::cout << "   " << std::setw(7) << g.max[k] << " / " << std::setw(5) << std::sqrt(g.sq[k] / g.n)
                              << (k == 1 ? "         " : "");
                }
                std::cout << "\n";
            }
            bool ok;
            if (compare_align == "rotation") {
                ok = c.max[1] <= compare_max && c.rms[1] <= compare_rms && c.max[0] <= compare_raw;
                std::cout << "intrinsics repeatability over rays to " << compare_theta << " deg: rotation-aligned max "
                          << c.max[1] << " px (limit " << compare_max << "), rms " << c.rms[1] << " px (limit "
                          << compare_rms << "); raw max " << c.max[0] << " px (limit " << compare_raw << "), rms "
                          << c.rms[0] << " px: " << (ok ? "PASS" : "FAIL") << std::endl;
            } else {
                ok = c.max[0] <= compare_max;
                std::cout << "intrinsics repeatability over rays to " << compare_theta << " deg: max " << c.max[0]
                          << " px, rms " << c.rms[0] << " px (limit " << compare_max << " px): " << (ok ? "PASS" : "FAIL")
                          << std::endl;
            }
            return ok ? 0 : 3;
        } catch (const std::exception& e) {
            std::cerr << "calico-robocal: " << e.what() << std::endl;
            return 1;
        }
    }
    if (config.empty()) {
        Usage();
        return 2;
    }
    try {
        robocal::Config cfg = robocal::Config::Load(config);
        if (verbose) cfg.solver.verbose = true;
        // model: auto -- fit each candidate lens model, keep the one with the lowest hold-out
        // error (held-out frames, poses refitted alone). equidistant (what ROS loads) wins unless
        // another model is clearly better (> 5 % and > 0.002 px).
        std::vector<size_t> auto_cams;
        for (size_t i = 0; i < cfg.cameras.size(); ++i) {
            if (cfg.cameras[i].model == "auto") auto_cams.push_back(i);
        }
        std::string chosen;
        std::unique_ptr<robocal::RobotCalibrator> final_cal;
        if (!auto_cams.empty()) {
            // equidistant first: every other candidate starts from it (fitted over the same rays),
            // not from a generic guess.
            std::vector<std::string> cands = {"equidistant"};
            for (const std::string& m : cfg.cameras[auto_cams[0]].models) {
                if (m != "equidistant") cands.push_back(m);
            }
            struct Cand {
                std::string model;
                double holdout, bic;
            };
            std::vector<Cand> table;
            // Every candidate's solve is kept: the chosen one is the final result (no rerun).
            std::map<std::string, std::unique_ptr<robocal::RobotCalibrator>> solved;
            std::map<size_t, std::pair<robocal::CameraIntrinsics, double>> eq_start;  // camera -> (intrinsics, theta)
            for (const std::string& m : cands) {
                robocal::Config c = cfg;
                for (size_t i : auto_cams) {
                    c.cameras[i].model = m;
                    auto it = eq_start.find(i);
                    if (m != "equidistant" && it != eq_start.end()) {
                        double rms = 0, mx = 0;
                        c.cameras[i].intrinsics = robocal::FitModel(
                            it->second.first, robocal::LensModelFromName(m, 0), it->second.second, &rms, &mx);
                        c.cameras[i].intrinsics_given = true;
                    }
                }
                double worst = 0, bic = 0;
                bool pass = true;
                try {
                    auto kp = std::make_unique<robocal::RobotCalibrator>(c);
                    robocal::RobotCalibrator& k = *kp;
                    k.Load();
                    k.Initialize();
                    k.Solve();
                    // Every candidate's lens, for inspection: <out>/candidates/<model>/.
                    mkdir(out.c_str(), 0755);
                    mkdir((out + "/candidates").c_str(), 0755);
                    mkdir((out + "/candidates/" + m).c_str(), 0755);
                    k.WriteCameraInfos(out + "/candidates/" + m);
                    for (size_t i : auto_cams) {
                        const auto& d = k.diagnostics();
                        worst = std::max(worst, d.holdout.at(c.cameras[i].name).rms_point);
                        // BIC on the training residuals: what the lens model explains, per parameter.
                        // (Hold-out frames refit their pose, which absorbs smooth lens errors such as
                        // decentering: hold-out RMS alone cannot tell the models apart.)
                        const auto& tr = d.train.at(c.cameras[i].name);
                        const double n_res = 2.0 * tr.n_points;
                        const int k_par = 4 + robocal::DistortionCount(robocal::LensModelFromName(m, 0));
                        bic += n_res * std::log(std::max(1e-12, tr.rms_axis * tr.rms_axis)) + k_par * std::log(n_res);
                        const auto it = d.lens.find(c.cameras[i].name);
                        if (it != d.lens.end() && it->second.gated && !it->second.pass) pass = false;
                        if (m == "equidistant") {
                            const double th = it != d.lens.end() && it->second.theta_max_deg > 0 ? it->second.theta_max_deg : 90.0;
                            eq_start[i] = {k.Intrinsics(static_cast<int>(i)), th};
                        }
                    }
                    solved[m] = std::move(kp);
                } catch (const std::exception& e) {
                    std::cout << "[robocal] model " << m << ": failed (" << e.what() << ")" << std::endl;
                    continue;
                }
                std::cout << "[robocal] model " << m << ": hold-out rms " << worst << " px, BIC " << std::fixed
                          << std::setprecision(1) << bic << (pass ? "" : " (gates FAIL)") << robocal::Took() << std::endl;
                if (pass) table.push_back({m, worst, bic});
                else solved.erase(m);  // never chosen: free its memory
            }
            if (table.empty()) throw std::runtime_error("model auto: no candidate model passed its gates");
            // Choice by the block hold-out error (whole time blocks of frames held out, poses refitted):
            // equidistant (KB4, what ROS loads) unless another model is clearly better (> 3 %). BIC is
            // printed but not used: it counts ~1e5 correlated residuals as independent, so any 1 % RMS gain
            // beats its parameter penalty (on real data it picked kb_rt, the least repeatable model).
            const Cand* eq = nullptr;
            const Cand* best = nullptr;
            for (const Cand& t : table) {
                if (t.model == "equidistant") eq = &t;
                if (!best || t.holdout < best->holdout) best = &t;
            }
            chosen = (eq && best->holdout > 0.97 * eq->holdout) ? std::string("equidistant") : best->model;
            std::cout << "[robocal] model auto -> " << chosen << std::endl;
            for (size_t i : auto_cams) cfg.cameras[i].model = chosen;
            final_cal = std::move(solved.at(chosen));
        }
        if (!final_cal) {
            final_cal = std::make_unique<robocal::RobotCalibrator>(cfg);
            final_cal->Load();
            final_cal->Initialize();
            final_cal->Solve();
        }
        robocal::RobotCalibrator& cal = *final_cal;
        mkdir(out.c_str(), 0755);
        // A robot solve that failed its sanity gate is written under names nothing loads by accident.
        const bool failed = cal.diagnostics().solve_failed;
        const std::string tag = failed ? ".FAILED" : "";
        const std::string path = out + "/robot_calib" + tag + ".yaml";
        cal.WriteYaml(path);
        std::cout << cal.Summary() << "wrote " << path << robocal::Took() << std::endl;
        for (const std::string& ci : cal.WriteCameraInfos(out)) std::cout << "wrote " << ci << std::endl;
        cal.WriteResiduals(out + "/residuals.csv");
        if (!cal.config().arms.empty()) {
            try {  // the calibration itself is already written: an export problem must not fail the run
                cal.WriteCalibratedUrdf(out + "/calibrated_cell" + tag + ".urdf");
                std::cout << "wrote " << out << "/calibrated_cell" << tag << ".urdf" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "calico-robocal: warning: calibrated URDF not written: " << e.what() << std::endl;
            }
        }
        if (failed) {
            std::cerr << "calico-robocal: robot solve FAILED its sanity gate: " << cal.diagnostics().fail_reason << std::endl;
            return 4;
        }
        // Intrinsics that failed their gates must not be used downstream: distinct exit code.
        for (const auto& kv : cal.diagnostics().lens) {
            if (kv.second.gated && !kv.second.pass) {
                std::cerr << "calico-robocal: intrinsics gates FAILED for " << kv.first << std::endl;
                return 3;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "calico-robocal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
