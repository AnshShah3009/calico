// calico-focus: set a fixed-focus lens for an operating range, before calibrating intrinsics.
//
// Focus changes the lens-to-sensor distance and with it the focal length (and centre), so it
// must be set and locked (glue / set screw) first; the intrinsics session comes after.
//
//   calico-focus --config=<session.yaml> (--images=<dir> | --device=<index|path>) [--camera=<name>]
//                [--out=<dir>] [--frames=<n>]
//
// session.yaml: the intrinsics-session config (boards, cameras: model + width/height + f_init,
// or a nominal camera_info) plus
//   focus: {near_m: 0.13, far_m: 0.30, max_off_axis_deg: 35}
// Intensities are linearised with detection.gamma (as the detector does); focus.gamma overrides it.
//
// Procedure (also printed by the tool):
//   1. Board centred in the image at the target distance s = harmonic mean of near and far
//      (the tool prints the measured distance). Camera and board on stands (no motion blur).
//   2. Turn the lens slowly through focus; the edge blur (px) drops, then rises: go back to
//      the minimum. Lock the lens.
//   3. Check: board at near, then at far. A correctly set lens blurs both equally.
#include "focus.hpp"
#include "robocal.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace robocal;

namespace {

std::string Arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    const std::string p = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind(p, 0) == 0) return a.substr(p.size());
    }
    return def;
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct FrameResult {
    std::string name;
    FocusMeasure m;
};

}  // namespace

int main(int argc, char** argv) {
    const std::string config = Arg(argc, argv, "config");
    const std::string images = Arg(argc, argv, "images");
    const std::string device = Arg(argc, argv, "device");
    if (config.empty() || (images.empty() && device.empty())) {
        std::cout << "usage: calico-focus --config=<session.yaml> (--images=<dir> | --device=<index|path>)\n"
                     "                    [--camera=<name>] [--out=<dir>] [--frames=<n>]\n"
                     "Sets a fixed-focus lens to the harmonic mean of focus: {near_m, far_m} in the config.\n";
        return 2;
    }
    try {
        const Config cfg = Config::Load(config);
        const YAML::Node fnode = YAML::LoadFile(config)["focus"];
        if (!fnode || !fnode["near_m"] || !fnode["far_m"]) throw std::runtime_error("config needs focus: {near_m, far_m}");
        const double near_m = fnode["near_m"].as<double>(), far_m = fnode["far_m"].as<double>();
        if (!(near_m > 0 && far_m > near_m)) throw std::runtime_error("focus: need 0 < near_m < far_m");
        FocusOptions fo;
        if (fnode["max_off_axis_deg"]) fo.max_off_axis_deg = fnode["max_off_axis_deg"].as<double>();
        fo.gamma = cfg.detect.gamma;  // the camera's encoding, as the detector uses it (detection.gamma)
        if (fnode["gamma"]) fo.gamma = fnode["gamma"].as<double>();
        const double target = FocusTarget(near_m, far_m);
        if (cfg.cameras.empty() || cfg.boards.empty()) throw std::runtime_error("config needs cameras and boards");
        const std::string cam_name = Arg(argc, argv, "camera", cfg.cameras[0].name);
        const CameraConfig* cc = nullptr;
        for (const CameraConfig& c : cfg.cameras) {
            if (c.name == cam_name) cc = &c;
        }
        if (!cc) throw std::runtime_error("no camera " + cam_name + " in the config");
        CameraIntrinsics rough;
        if (!cc->camera_info.empty()) {
            rough = LoadCameraInfo(cc->camera_info, cc->name);
        } else {
            if (cc->width <= 0 || cc->height <= 0 || cc->f_init <= 0) {
                throw std::runtime_error("camera " + cc->name + ": give camera_info, or width / height / f_init");
            }
            rough.model = LensModelFromName(cc->model, cc->model == "rational_polynomial" ? 8 : 5);
            rough.width = cc->width;
            rough.height = cc->height;
            rough.params = {cc->f_init, cc->f_init, 0.5 * (cc->width - 1), 0.5 * (cc->height - 1)};
            for (double v : DefaultDistortion(rough.model)) rough.params.push_back(v);
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[focus] operating range " << near_m << " - " << far_m << " m  ->  focus at the harmonic mean "
                  << target << " m\n"
                  << "[focus] 1) board centred, camera and board still, at " << target
                  << " m; turn the lens through focus and stop at the smallest blur\n"
                  << "[focus] 2) check: board at " << near_m << " m, then at " << far_m
                  << " m; both should show the same blur\n";
        std::cout << "frame                              dist_m   blur_px  (+-)   tags  edges  note\n";

        std::vector<FrameResult> results;
        double best_blur = 1e9;
        std::string best_name;
        auto process = [&](const cv::Mat& frame, const std::string& name) {
            cv::Mat gray;
            if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            else gray = frame;
            FrameResult fr;
            fr.name = name;
            for (const BoardDetection& d : DetectBoards(gray, cfg.boards, cfg.detect)) {
                const FocusMeasure m = MeasureFocus(gray, cfg.boards[d.board], d, rough, fo);
                if (m.ok && m.n_edges > fr.m.n_edges) fr.m = m;
            }
            std::string note;
            if (!fr.m.ok) {
                note = fr.m.n_edges >= 8 ? "no board pose (check the rough intrinsics)" : "no board near the image centre";
            } else {
                const double rel = fr.m.distance_m / target - 1.0;
                if (std::fabs(rel) <= 0.10) {
                    note = "at target";
                    if (fr.m.blur_sigma_px < best_blur) {
                        best_blur = fr.m.blur_sigma_px;
                        best_name = name;
                        note += "  * sharpest so far";
                    }
                } else if (fr.m.distance_m < near_m * 1.15) {
                    note = "near check";
                } else if (fr.m.distance_m > far_m * 0.87) {
                    note = "far check";
                } else {
                    note = rel < 0 ? "move the board away" : "move the board closer";
                }
                if (fr.m.off_axis_deg > 15) note += "  (centre the board)";
            }
            std::cout << std::left << std::setw(34) << name.substr(0, 33) << std::right << std::setw(7)
                      << fr.m.distance_m << std::setw(10) << fr.m.blur_sigma_px << std::setw(7) << fr.m.blur_spread_px
                      << std::setw(6) << fr.m.n_tags << std::setw(7) << fr.m.n_edges << "  " << note << "\n";
            results.push_back(fr);
        };

        if (!images.empty()) {
            std::vector<std::string> files;
            for (const auto& e : std::filesystem::directory_iterator(images)) {
                std::string ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                    files.push_back(e.path().string());
                }
            }
            std::sort(files.begin(), files.end());
            if (files.empty()) throw std::runtime_error("no images in " + images);
            for (const std::string& f : files) {
                const cv::Mat img = cv::imread(f, cv::IMREAD_UNCHANGED);
                if (img.empty()) continue;
                process(img, std::filesystem::path(f).filename().string());
            }
        } else {
            cv::VideoCapture cap;
            const bool numeric = !device.empty() && std::all_of(device.begin(), device.end(), ::isdigit);
            if (numeric) cap.open(std::stoi(device));
            else cap.open(device);
            if (!cap.isOpened()) throw std::runtime_error("cannot open device " + device);
            const int max_frames = std::stoi(Arg(argc, argv, "frames", "0"));
            cv::Mat frame;
            for (int k = 0; (max_frames <= 0 || k < max_frames) && cap.read(frame); ++k) {
                process(frame, "live_" + std::to_string(k));
            }
        }

        // Summary: sharpest frame at the target distance, near / far balance.
        std::vector<double> nb, fb;
        for (const FrameResult& r : results) {
            if (!r.m.ok) continue;
            if (r.m.distance_m < near_m * 1.15) nb.push_back(r.m.blur_sigma_px);
            if (r.m.distance_m > far_m * 0.87) fb.push_back(r.m.blur_sigma_px);
        }
        std::string verdict = "no near / far check frames";
        bool balanced = false;
        const double bn = Median(nb), bf = Median(fb);
        if (nb.size() >= 2 && fb.size() >= 2) {
            const double tol = 0.08 + 0.12 * std::min(bn, bf);
            if (std::fabs(bn - bf) <= tol) {
                verdict = "balanced: the lens is focused for the whole range";
                balanced = true;
            } else if (bn > bf) {
                verdict = "near end blurrier: the lens is focused too far, turn it toward close focus";
            } else {
                verdict = "far end blurrier: the lens is focused too close, turn it toward far focus";
            }
        }
        std::cout << "[focus] target " << target << " m: "
                  << (best_name.empty() ? std::string("no frame at the target distance")
                                        : "sharpest frame " + best_name + " (blur " + std::to_string(best_blur).substr(0, 5) + " px)")
                  << "\n[focus] near " << bn << " px (" << nb.size() << " frames), far " << bf << " px (" << fb.size()
                  << " frames): " << verdict << "\n";

        const std::string out = Arg(argc, argv, "out");
        if (!out.empty()) {
            std::filesystem::create_directories(out);
            YAML::Emitter e;
            e.SetDoublePrecision(6);
            e << YAML::BeginMap;
            e << YAML::Key << "near_m" << YAML::Value << near_m << YAML::Key << "far_m" << YAML::Value << far_m;
            e << YAML::Key << "target_m" << YAML::Value << target;
            e << YAML::Key << "best_frame" << YAML::Value << best_name;
            e << YAML::Key << "best_blur_px" << YAML::Value << (best_name.empty() ? 0.0 : best_blur);
            e << YAML::Key << "near_blur_px" << YAML::Value << bn << YAML::Key << "far_blur_px" << YAML::Value << bf;
            e << YAML::Key << "balanced" << YAML::Value << balanced << YAML::Key << "verdict" << YAML::Value << verdict;
            e << YAML::Key << "frames" << YAML::Value << YAML::BeginSeq;
            for (const FrameResult& r : results) {
                e << YAML::Flow << YAML::BeginMap << YAML::Key << "image" << YAML::Value << r.name;
                e << YAML::Key << "ok" << YAML::Value << r.m.ok;
                e << YAML::Key << "distance_m" << YAML::Value << r.m.distance_m;
                e << YAML::Key << "blur_px" << YAML::Value << r.m.blur_sigma_px;
                e << YAML::Key << "edges" << YAML::Value << r.m.n_edges << YAML::EndMap;
            }
            e << YAML::EndSeq << YAML::EndMap;
            std::ofstream(out + "/focus_report.yaml") << e.c_str() << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "calico-focus: " << ex.what() << "\n";
        return 1;
    }
}
