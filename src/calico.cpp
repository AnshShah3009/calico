//============================================================================
// Name        : calico.cpp
// Author      : Amy Tabb
// Version     :
// Copyright   : MIT
// Description :
//============================================================================

#include "calico.hpp"

#include "helper.hpp"
#include "DirectoryFunctions.hpp"
#include "camera-calibration.hpp"
#include "multicamera.hpp"
#include "solving-structure.hpp"
#include "pattern-parameters.hpp"
#include <ctime>
#include <unistd.h>
#include <fcntl.h>

int g_num_threads = omp_get_max_threads();


int main(int argc, char **argv){

    int print_help = 0;
    int calibrateGroupCameras = 0;
    int verbose = 0;
    int is_charuco = 0;
    int is_april = 0;
    int zero_tangent_dist = 1;
    int zero_k3 = 1;
    int read_camera_calibration = 0;
    int only_camera_calibration = 0;
    int max_internal_read  = -1;
    int max_internal_use = -1;
    int max_external_positions = -1;
    int selectedk = -1;
    int create_patterns_only = 0;
    int fix_principal_point = 0;
    int number_points_needed_to_count_pattern = -1;
    int synchronized_rotating_option = 0;
    int number_threads = omp_get_max_threads();
    float percentage_global_alg = 0.2;
    float percentage_global_rp = 0.5;
    int interleaved_solve = 1;
    int doProfile = 0;
    int dry_run = 0;
    int timestamp_output = 0;
    int no_overwrite = 0;
    int show_progress = 0;
    int checkpoint_enabled = 0;
    int quiet = 0;
    int summary = 0;
    int force = 0;
    int no_progress = 0;
    int json_output = 0;
    int auto_rename = 0;
    int print_version = 0;

    string input_dir = "";
    string output_dir = "";
    float initial_focal_px = -1;
    double camera_size = 40.0;
    float track_size = 0.5;
    string src_file = "../src/detector_params.yml";
    string camera_color_str = "";
    string pattern_color_str = "";
    string camera_names_str = "";
    string resume_dir = "";
    string config_file = "";
    string exclude_camera_str = "";
    string focus_camera_str = "";
    int max_images = -1;
    int min_boards = -1;
    int no_visualization = 0;
    int no_debug_images = 0;
    int detection_summary = 0;
    int per_camera_mse = 0;

    if (argc == 1){
        print_help = 1;
    }

    while (1)
    {
        static struct option long_options[] =
        {
                {"help",   no_argument,       &print_help, 1},
                {"calibrate", no_argument,       &calibrateGroupCameras, 1},
                {"create-patterns", no_argument, &create_patterns_only, 1},
                {"non-zero-tangent", no_argument,       &zero_tangent_dist, 0},
                {"non-zero-k3", no_argument,            &zero_k3, 0},
                {"fix-pp", no_argument, &fix_principal_point, 1},
                {"verbose", no_argument, &verbose, 1},
                {"charuco",   no_argument,       &is_charuco, 1},
                {"april",   no_argument,       &is_april, 1},
                {"dry-run", no_argument, &dry_run, 1},
                {"timestamp", no_argument, &timestamp_output, 1},
                {"no-overwrite", no_argument, &no_overwrite, 1},
                {"progress", no_argument, &show_progress, 1},
                {"checkpoint", no_argument, &checkpoint_enabled, 1},
                {"input",   required_argument, 0, 'a'},
                {"output",  required_argument, 0, 'b'},
                {"camera-size", required_argument, 0, 'c'},
                {"focal-px",  required_argument, 0, 'd'},
                {"src-dir", required_argument, 0, 'e'},
                {"max-internal-read", required_argument, 0, 'f'},
                {"max-internal-use", required_argument, 0, 'g'},
                {"max-external", required_argument, 0, 'h'},
                {"k", required_argument, 0, 'i'},
                {"track-size", required_argument, 0, 'j'},
                {"num-pattern", required_argument, 0, 'k'},
                {"num-threads", required_argument, 0, 'l'},
                {"perc-ae", required_argument, 0, 'm'},
                {"perc-rp", required_argument, 0, 'n'},
                {"camera-color", required_argument, 0, 'o'},
                {"pattern-color", required_argument, 0, 'p'},
                {"camera-names", required_argument, 0, 'q'},
                {"resume", required_argument, 0, 'r'},
                {"config", required_argument, 0, 's'},
                {"quiet", no_argument, &quiet, 1},
                {"summary", no_argument, &summary, 1},
                {"force", no_argument, &force, 1},
                {"no-progress", no_argument, &no_progress, 1},
                {"json", no_argument, &json_output, 1},
                {"auto-rename", no_argument, &auto_rename, 1},
                {"version", no_argument, &print_version, 1},
                {"exclude-camera", required_argument, 0, 't'},
                {"max-images", required_argument, 0, 'u'},
                {"no-visualization", no_argument, &no_visualization, 1},
                {"no-debug-images", no_argument, &no_debug_images, 1},
                {"detection-summary", no_argument, &detection_summary, 1},
                {"per-camera-mse", no_argument, &per_camera_mse, 1},
                {"focus-camera", required_argument, 0, 'v'},
                {"min-boards", required_argument, 0, 'w'},
        };

        if (argc == 1){ print_help = 1; }

        if (print_help == 1){
            cout << "Printing help for calico, Dec. 2023."<< endl;

            cout << "ESSENTIAL FUNCTIONALITY -------------------" << endl;
            cout << std::left << setw(30) << "--verbose" << "No arguments.  Writes additional information during the run." << endl;
            cout << std::left << setw(30) << "--create-patterns" << "No arguments, write charuco or april image patterns from a specification file." << endl;
            cout << std::left << setw(30) << "--calibrate" << "No arguments, indicates that this dataset is a camera network or ";
                                cout << "multicamera system." << endl;
            cout << std::left << setw(30) << "--num-threads "<< "Number of threads to use.  Default is # returned by ";
            cout << "omp_get_max_threads();, currently = " << number_threads << endl;
            cout << std::left << setw(30) << "--charuco" << "Using charuco patterns." << endl;
            cout << std::left << setw(30) << "--april" << "Using AprilTag patterns." << endl;

            cout << endl;
            cout << "DIRECTORIES AND PATHS ----------------------- " << endl;
            cout << std::left << setw(30) << "--input=[STRING] "<< "Mandatory, has to be a directory." << endl;
            cout << std::left << setw(30) << "--output=[STRING] " << "Mandatory, has to be a directory." << endl;
            cout << std::left << setw(30) << "--src-dir=[STRING] ";
            cout << "Directory where the source code resides relative to where the executable is being run. " << endl;
            cout << " Specifically, the location of 'detector_params.yml'  Default is ../src/ . " << endl;

            cout << endl;
            cout << "CAMERA CALIBRATION OPTIONS ---------------------------" << endl;
            cout << std::left << setw(30) << "--non-zero-tangent " << "No arguments. In the camera calibration part, sets the tangential components of radial distortion (p1, p2) to non-zero." << endl;
            cout << std::left << setw(30) << "--non-zero-k3 " << "No arguments. In the camera calibration part, sets the 3rd radial distortion k value to non-zero." << endl;
            cout << std::left << setw(30) << "--fix-pp " << "No arguments. In the camera calibration part, sets the principal point to the image center. " << endl;
            cout << std::left << setw(30) << "--focal-px=[float] " << "Initial focal length in pixels for the camera.  Default is max dimension * 1.2 " << endl;

            cout << endl;
            cout <<"OPTIONS ON NUMBER OF IMAGES READ/USED; NUMBER OF POINTS USED FOR NETWORK -----------" << endl;
            cout << std::left << setw(30) << "--max-internal-read=[int] " << "Integer argument. Sets the number of internal camera calibration images to read.";
            cout << " Default is the number of images in the directory 'internal'." << endl;
            cout << std::left << setw(30) << "--max-internal-use=[int] " << "Integer argument. Sets the number of images to use where the pattern is detected in the calibration, from the 'internal' directory. ";
            cout << "The default is the maximum number of patterns found in . " << endl;
            cout << std::left << setw(30) << "--max-external=[int] " << "Integer argument. Sets the number of images /time instants to read from ";
            cout << " the 'external' directory for each camera." << endl;
            cout << std::left << setw(30) << "--k=[int] " << "Specifies the number of points to use for the minimization of reprojection error, relevant only ";
            cout << " for the network case.  The default is 8." << endl;
            cout << std::left << setw(30) << "--num-pattern=[int] " << "Integer argument. Sets the number of points required to estimate the  ";
            cout << " pose for a pattern.  Default is >=10 for network, >= 4 for rotating." << endl;

            cout <<"OPTIONS HOW OFTEN TO RUN GLOBAL MINIMIZATION OF VARIABLES -----------" << endl;
            cout << std::left << setw(30) << "--perc-ae=[float] " << "float argument from (0,1]. Global LM optimization of variables for algebraic error occurs after this percentage ";
            cout << "of variables are solved for. Unspecified, the default is 0.2, so this step will be run 5 times." << endl;
            cout << std::left << setw(30) << "--perc-rp=[float] " << "float argument from (0,1]. Global LM optimization of variables for reprojection error is triggered when this percentage ";
            cout << "of constraints/equations are added to the model. Unspecified, the default is 0.5, so this step will be run 2 times." << endl;

            cout << endl;
            cout <<"DISPLAY -----------" << endl;
            cout << std::left << setw(30) << "--camera-size=[float] " << "Float argument.  Specifies the size of the cameras written, default is 40." << endl;
            cout << std::left << setw(30) << "--track-size=[float] " << "Float argument.  Specifies the size of the track size written, default is 0.5 ." << endl;

            cout << endl;
            cout <<"QUALITY OF LIFE -----------" << endl;
            cout << std::left << setw(30) << "--dry-run " << "Validate input and print summary without running calibration." << endl;
            cout << std::left << setw(30) << "--timestamp " << "Append timestamp to output directory to avoid overwrites." << endl;
            cout << std::left << setw(30) << "--no-overwrite " << "Refuse to run if output directory already has results." << endl;
            cout << std::left << setw(30) << "--progress " << "Show progress percentage and elapsed time during stages 4-5." << endl;
            cout << std::left << setw(30) << "--camera-color=[STR] " << "Per-camera colors as R,G,B|R,G,B|... for PLY output." << endl;
            cout << std::left << setw(30) << "--pattern-color=[STR] " << "Per-pattern colors as R,G,B|R,G,B|... for PLY output." << endl;
            cout << std::left << setw(30) << "--camera-names=[STR] " << "Comma-separated camera names for output." << endl;
            cout << std::left << setw(30) << "--checkpoint " << "Enable periodic checkpointing of solver state." << endl;
            cout << std::left << setw(30) << "--resume=[DIR] " << "Resume from a checkpoint directory." << endl;
            cout << std::left << setw(30) << "--config=[FILE] " << "Read options from YAML config file." << endl;
            cout << std::left << setw(30) << "--quiet " << "Suppress all output except errors and results." << endl;
            cout << std::left << setw(30) << "--summary " << "Print brief calibration summary at end." << endl;
            cout << std::left << setw(30) << "--force " << "Override --no-overwrite and run anyway." << endl;
            cout << std::left << setw(30) << "--no-progress " << "Disable progress output during stages 4-5." << endl;
            cout << std::left << setw(30) << "--version " << "Print version and exit." << endl;
            cout << std::left << setw(30) << "--json " << "Output results in JSON format." << endl;
            cout << std::left << setw(30) << "--auto-rename " << "Rename output directory based on input name." << endl;
            cout << std::left << setw(30) << "--exclude-camera=[STR] " << "Comma-separated camera directories to exclude." << endl;
            cout << std::left << setw(30) << "--max-images=[INT] " << "Unified max images cap for all cameras." << endl;
            cout << std::left << setw(30) << "--no-visualization " << "Skip writing PLY meshes and equation PNGs." << endl;
            cout << std::left << setw(30) << "--no-debug-images " << "Skip per-image detection visualization PNGs." << endl;
            cout << std::left << setw(30) << "--detection-summary " << "Print board visibility table per camera." << endl;
            cout << std::left << setw(30) << "--per-camera-mse " << "Append per-camera reprojection MSE to total_results.txt." << endl;
            cout << std::left << setw(30) << "--focus-camera=[STR] " << "Comma-separated cameras to calibrate (inverse of exclude)." << endl;
            cout << std::left << setw(30) << "--min-boards=[INT] " << "Auto-exclude cameras detecting fewer than N boards." << endl;

            cout << "All other arguments are ignored." << endl;
            cout << endl << endl;

            exit(1);
        }

        int option_index = 0;
        int opt_argument;

        opt_argument = getopt_long (argc, argv, "abcdefghijklmnopqrstuvw",
                long_options, &option_index);

        if (opt_argument == -1)
            break;

        switch (opt_argument)
        {
        case 0:
            if (long_options[option_index].flag != 0)
                break;
            printf ("option %s", long_options[option_index].name);
            if (optarg)
                printf (" with arg %s", optarg);
            printf ("\n");
            break;

        case 'a':
            input_dir = optarg;
            EnsureDirHasTrailingBackslash(input_dir);
            break;
        case 'b':
            output_dir = optarg;
            break;
        case 'c':
            camera_size = FromString<float>(optarg);
            break;
        case 'd':
            initial_focal_px = FromString<float>(optarg);
            break;
        case 'e':
            src_file = optarg;
            EnsureDirHasTrailingBackslash(src_file);
            src_file = src_file + "detector_params.yml";
            break;
        case 'f':
            max_internal_read  = FromString<int>(optarg);
            break;
        case 'g':
            max_internal_use  = FromString<int>(optarg);
            break;
        case 'h':
            max_external_positions  = FromString<int>(optarg);
            break;
        case 'i':
            selectedk  = FromString<int>(optarg);
            break;
        case 'j':
            track_size  = FromString<int>(optarg);
            break;
        case 'k':
            number_points_needed_to_count_pattern  = FromString<int>(optarg);
            break;
        case 'l':
            number_threads  = FromString<int>(optarg);
            omp_set_num_threads(number_threads);
            break;
        case 'm':{
            float temp = FromString<float>(optarg);
            if (temp <= 0 || temp > 1){
                cout << "Value for --perc-ae is bad.  Needs to be greater than 0 and less than or equal to 1." << endl;
                exit(1);
            }

            percentage_global_alg = temp;
        }
        break;
        case 'n':
        {
            float temp = FromString<float>(optarg);
            if (temp <= 0 || temp > 1){
                cout << "Value for --perc-rp is bad.  Needs to be greater than 0 and less than or equal to 1." << endl;
                exit(1);
            }

            percentage_global_rp = temp;
        }
        break;
        case 'o':
            camera_color_str = optarg;
            break;
        case 'p':
            pattern_color_str = optarg;
            break;
        case 'q':
            camera_names_str = optarg;
            break;
        case 'r':
            resume_dir = optarg;
            break;
        case 's':
            config_file = optarg;
            break;
        case 't':
            exclude_camera_str = optarg;
            break;
        case 'u':
            max_images = FromString<int>(optarg);
            break;
        case 'v':
            focus_camera_str = optarg;
            break;
        case 'w':
            min_boards = FromString<int>(optarg);
            break;

        default:{
            cout << "Argument not found " << optarg << endl;
            exit(1);
        }

        }

    }

    // Handle --version first, before any checks
    if (print_version) {
        cout << "calico-dec2023 (https://github.com/amy-tabb/calico)" << endl;
        cout << "Multi-camera calibration with charuco or april tags." << endl;
        exit(0);
    }

    // Save CLI values to restore after config loading (CLI overrides config)
    int cli_verbose = verbose;
    int cli_dry_run = dry_run;
    int cli_no_overwrite = no_overwrite;
    int cli_quiet = quiet;
    int cli_no_visualization = no_visualization;
    int cli_no_debug_images = no_debug_images;
    int cli_detection_summary = detection_summary;
    int cli_per_camera_mse = per_camera_mse;
    int cli_min_boards = min_boards;
    string cli_camera_color_str = camera_color_str;
    string cli_pattern_color_str = pattern_color_str;
    string cli_camera_names_str = camera_names_str;
    string cli_exclude_camera_str = exclude_camera_str;
    string cli_focus_camera_str = focus_camera_str;
    int cli_number_threads = number_threads;
    int cli_max_internal_read = max_internal_read;
    int cli_max_internal_use = max_internal_use;
    int cli_max_external_positions = max_external_positions;
    int cli_selectedk = selectedk;
    float cli_camera_size = camera_size;
    float cli_track_size = track_size;
    float cli_initial_focal_px = initial_focal_px;
    float cli_percentage_global_alg = percentage_global_alg;
    float cli_percentage_global_rp = percentage_global_rp;

    // Load config file (provides defaults that CLI overrides)
    if (config_file.size() > 0) {
        ifstream cfg(config_file.c_str());
        if (cfg.good()) {
            string line;
            while (getline(cfg, line)) {
                // Skip empty lines, comments, YAML header
                if (line.empty() || line[0] == '#' || line[0] == '%') continue;
                size_t colon = line.find(':');
                if (colon == string::npos) continue;
                string key = line.substr(0, colon);
                string val_str = line.substr(colon + 1);
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                val_str.erase(0, val_str.find_first_not_of(" \t"));
                val_str.erase(val_str.find_last_not_of(" \t") + 1);
                int ival = atoi(val_str.c_str());
                float fval = (float)atof(val_str.c_str());

                if (key == "num-threads" || key == "num_threads") {
                    if (ival > 0) { number_threads = ival; omp_set_num_threads(ival); }
                } else if (key == "max-internal-read" || key == "max_internal_read") { if (ival > 0) max_internal_read = ival;
                } else if (key == "max-internal-use" || key == "max_internal_use") { if (ival > 0) max_internal_use = ival;
                } else if (key == "max-external" || key == "max_external") { if (ival > 0) max_external_positions = ival;
                } else if (key == "k") { if (ival > 0) selectedk = ival;
                } else if (key == "num-pattern" || key == "num_pattern") { if (ival > 0) number_points_needed_to_count_pattern = ival;
                } else if (key == "verbose") { if (ival) verbose = 1;
                } else if (key == "dry-run" || key == "dry_run") { if (ival) dry_run = 1;
                } else if (key == "timestamp") { if (ival) timestamp_output = 1;
                } else if (key == "no-overwrite" || key == "no_overwrite") { if (ival) no_overwrite = 1;
                } else if (key == "progress") { if (ival) show_progress = 1;
                } else if (key == "checkpoint") { if (ival) checkpoint_enabled = 1;
                } else if (key == "quiet") { if (ival) quiet = 1;
                } else if (key == "summary") { if (ival) summary = 1;
                } else if (key == "force") { if (ival) force = 1;
                } else if (key == "json") { if (ival) json_output = 1;
                } else if (key == "no-progress" || key == "no_progress") { if (ival) no_progress = 1;
                } else if (key == "auto-rename" || key == "auto_rename") { if (ival) auto_rename = 1;
                } else if (key == "camera-color" || key == "camera_color") {
                    if (!val_str.empty()) camera_color_str = val_str;
                } else if (key == "pattern-color" || key == "pattern_color") {
                    if (!val_str.empty()) pattern_color_str = val_str;
                } else if (key == "camera-names" || key == "camera_names") {
                    if (!val_str.empty()) camera_names_str = val_str;
                } else if (key == "exclude-camera" || key == "exclude_camera") {
                    if (!val_str.empty()) exclude_camera_str = val_str;
                } else if (key == "camera-size" || key == "camera_size") { if (fval > 0) camera_size = fval;
                } else if (key == "track-size" || key == "track_size") { if (fval > 0) track_size = fval;
                } else if (key == "focal-px" || key == "focal_px") { if (fval > 0) initial_focal_px = fval;
                } else if (key == "perc-ae" || key == "perc_ae") { if (fval > 0) percentage_global_alg = fval;
                } else if (key == "perc-rp" || key == "perc_rp") { if (fval > 0) percentage_global_rp = fval;
                } else if (key == "no-visualization" || key == "no_visualization") { if (ival) no_visualization = 1;
                } else if (key == "no-debug-images" || key == "no_debug_images") { if (ival) no_debug_images = 1;
                } else if (key == "detection-summary" || key == "detection_summary") { if (ival) detection_summary = 1;
                } else if (key == "per-camera-mse" || key == "per_camera_mse") { if (ival) per_camera_mse = 1;
                } else if (key == "focus-camera" || key == "focus_camera") {
                    if (!val_str.empty()) focus_camera_str = val_str;
                } else if (key == "min-boards" || key == "min_boards") { if (ival > 0) min_boards = ival;
                }
            }
            cfg.close();
            cout << "Loaded config from " << config_file << endl;
        } else {
            cerr << "Warning: Could not open config file " << config_file << endl;
        }
    }

    // Restore CLI values (CLI overrides config; only if explicitly set)
    if (cli_verbose) verbose = 1;
    if (cli_dry_run) dry_run = 1;
    if (cli_no_overwrite) no_overwrite = 1;
    if (cli_quiet) quiet = 1;
    if (cli_no_visualization) no_visualization = 1;
    if (cli_no_debug_images) no_debug_images = 1;
    if (cli_detection_summary) detection_summary = 1;
    if (cli_per_camera_mse) per_camera_mse = 1;
    if (cli_min_boards >= 0) min_boards = cli_min_boards;
    if (cli_camera_color_str.size()) camera_color_str = cli_camera_color_str;
    if (cli_pattern_color_str.size()) pattern_color_str = cli_pattern_color_str;
    if (cli_camera_names_str.size()) camera_names_str = cli_camera_names_str;
    if (cli_exclude_camera_str.size()) exclude_camera_str = cli_exclude_camera_str;
    if (cli_focus_camera_str.size()) focus_camera_str = cli_focus_camera_str;
    if (cli_number_threads != (int)omp_get_max_threads()) number_threads = cli_number_threads;
    if (cli_max_internal_read >= 0) max_internal_read = cli_max_internal_read;
    if (cli_max_internal_use >= 0) max_internal_use = cli_max_internal_use;
    if (cli_max_external_positions >= 0) max_external_positions = cli_max_external_positions;
    if (cli_selectedk >= 0) selectedk = cli_selectedk;
    if (cli_camera_size != 40.0) camera_size = cli_camera_size;
    if (cli_track_size != 0.5) track_size = cli_track_size;
    if (cli_initial_focal_px >= 0) initial_focal_px = cli_initial_focal_px;
    if (cli_percentage_global_alg != 0.2) percentage_global_alg = cli_percentage_global_alg;
    if (cli_percentage_global_rp != 0.5) percentage_global_rp = cli_percentage_global_rp;
    omp_set_num_threads(number_threads);

    // Build CalicoOptions
    CalicoOptions opts;
    opts.dry_run = (dry_run == 1);
    opts.show_progress = (show_progress == 1);
    opts.timestamp_output = (timestamp_output == 1);
    opts.no_overwrite = (no_overwrite == 1);
    opts.checkpoint_enabled = (checkpoint_enabled == 1);
    opts.quiet = (quiet == 1);
    opts.summary = (summary == 1);
    opts.force = (force == 1);
    opts.no_progress = (no_progress == 1);
    opts.json_output = (json_output == 1);
    opts.auto_rename = (auto_rename == 1);
    opts.no_visualization = (no_visualization == 1);
    opts.no_debug_images = (no_debug_images == 1);
    opts.detection_summary = (detection_summary == 1);
    opts.per_camera_mse = (per_camera_mse == 1);
    opts.min_boards = min_boards;
    opts.num_threads = number_threads;
    opts.max_images = max_images;
    opts.resume_dir = resume_dir;
    opts.config_file = config_file;

    g_num_threads = number_threads;

    // Parse --focus-camera
    if (focus_camera_str.size() > 0) {
        size_t start = 0, end;
        while ((end = focus_camera_str.find(',', start)) != string::npos) {
            opts.focus_cameras.push_back(focus_camera_str.substr(start, end - start));
            start = end + 1;
        }
        if (start < focus_camera_str.size()) {
            opts.focus_cameras.push_back(focus_camera_str.substr(start));
        }
    }

    // Parse --exclude-camera
    if (exclude_camera_str.size() > 0) {
        size_t start = 0, end;
        while ((end = exclude_camera_str.find(',', start)) != string::npos) {
            opts.exclude_cameras.push_back(exclude_camera_str.substr(start, end - start));
            start = end + 1;
        }
        if (start < exclude_camera_str.size()) {
            opts.exclude_cameras.push_back(exclude_camera_str.substr(start));
        }
    }

    if (camera_color_str.size() > 0) {
        opts.camera_colors = ParseColorList(camera_color_str);
    }
    if (pattern_color_str.size() > 0) {
        opts.pattern_colors = ParseColorList(pattern_color_str);
    }
    if (camera_names_str.size() > 0) {
        size_t start = 0, end;
        while ((end = camera_names_str.find(',', start)) != string::npos) {
            opts.camera_names_override.push_back(camera_names_str.substr(start, end - start));
            start = end + 1;
        }
        if (start < camera_names_str.size()) {
            opts.camera_names_override.push_back(camera_names_str.substr(start));
        }
    }

    // Handle --no-overwrite (unless --force)
    if (opts.no_overwrite && !opts.force) {
        string results_file = output_dir + "total_results.txt";
        ifstream test(results_file.c_str());
        if (test.good()) {
            cerr << "ERROR: output directory already contains results. "
                 << "Use --timestamp, --force, or remove old results." << endl;
            exit(1);
        }
    }

    EnsureDirHasTrailingBackslash(output_dir);
    mkdir(output_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    // Handle --auto-rename: add input basename to output dir
    if (opts.auto_rename) {
        string in = input_dir;
        if (!in.empty() && in[in.size()-1] == '/') in = in.substr(0, in.size()-1);
        size_t slash = in.rfind('/');
        string base = (slash != string::npos) ? in.substr(slash + 1) : in;
        string renamed = output_dir + base + "/";
        mkdir(renamed.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        output_dir = renamed;
    }

    // Handle --timestamp
    if (opts.timestamp_output) {
        time_t now = time(0);
        struct tm tstruct;
        tstruct = *localtime(&now);
        char buf[80];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tstruct);
        string ts(buf);
        opts.no_overwrite = false; // timestamped dir is always new
        string ts_dir = output_dir + "calico_" + ts + "/";
        mkdir(ts_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        output_dir = ts_dir;
    }

    ofstream out;
    string filename = output_dir + "arguments-calico.txt";
    out.open(filename.c_str());
    out << "arguments: " << endl << argv[0] << " ";

    if ( (is_charuco && is_april) ){
        cout << "You cannot select both --charuco and --april.  It is exclusive or.  Quitting ... " << endl;
        exit(1);
    }

    if ((!is_april && !is_charuco)){
        cout << "You need to select either --charuco or --april.  Quitting ... " << endl;
        exit(1);
    }

    if (is_charuco){
        out << "--charuco " ;
    }

    if (is_april){
        out << "--april ";
    }

    if (calibrateGroupCameras){
        out << "--calibrate ";

        if (selectedk == -1){
            selectedk = 8;
        }

        if (selectedk != -1){
            out << "--k=" << selectedk << " ";
        }

        if (selectedk < 4){
            cout << "argument --k has to be greater than or equal to 4. Currently is it set to " << selectedk << endl;
            cout << "Quitting " << selectedk << endl;
            exit(1);
        }

        if (number_points_needed_to_count_pattern == -1){
            number_points_needed_to_count_pattern = 10;
        }

        synchronized_rotating_option = 1;
    }

    if (create_patterns_only){
        out << "--create-patterns " << endl;

        if (input_dir.size() == 0|| output_dir.size() == 0 ){
            cout << "Error!  --create-patterns flag  used, but either :" << endl;
            cout << "1) input directory is empty" << endl;
            cout << "2) output directory is empty" << endl;
            cout << "none of these can occur with --create-patterns.  Fix and run again." << endl;
            exit(1);
        }
    }

    if (zero_tangent_dist == 0){
        out << "--non-zero-tangent ";
    }
    if (zero_k3 == 0){
        out << "--non-zero-k3  ";
    }

    out << "--perc-ae=" << percentage_global_alg << " --perc-rp=" << percentage_global_rp << " ";

    out << "--num-threads "<< number_threads << " ";

    out << "--input=" << input_dir << " " ;
    out << "--output=" << output_dir << " ";

    if (max_internal_use > max_internal_read){
        cout << "You entered --max-internal-read values < --max-internal-use values ... " << max_internal_read << ", "
                << max_internal_use << endl;
        cout << "I am correcting this error, check the argument file --   " << endl;

        max_internal_use = max_internal_read;
    }

    out << "--camera-size="<< camera_size << " " ;
    out << "--track-size="<< track_size << " " ;

    out << "--num-pattern=" << number_points_needed_to_count_pattern << " ";

    if (fix_principal_point){
        out << "--fix-pp ";
    }

    if (initial_focal_px > 0){
        out << "--focal-px="<< initial_focal_px << " ";
    }

    if (max_internal_read >= 0){
        out << "--max-internal-read=" << max_internal_read << " ";
    }

    if (max_internal_use >= 0) {
        out << "--max-internal-use=" << max_internal_use << " ";
    }

    if (max_external_positions >=  0){
        out << "--max-external=" << max_external_positions << " ";
    }

    if (verbose > 0 ){
        out << "--verbose  ";
    }

    if ((!calibrateGroupCameras && !create_patterns_only)){
        cout << "You need to select either --calibrate or --create-patterns.  Quitting ... " << endl;
        exit(1);
    }

    if (!interleaved_solve){
        out << "--non-incremental ";
    }

    out << endl << endl << "OpenCV Version " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << endl;

    out.close();

    // Handle --dry-run
    if (opts.dry_run) {
        if (!create_patterns_only) {
            if (opts.quiet) {
            fflush(stdout);
            int dr_saved = dup(STDOUT_FILENO);
            int dr_null = open("/dev/null", O_WRONLY);
            if (dr_null >= 0) dup2(dr_null, STDOUT_FILENO);
            DryRunValidation(input_dir, output_dir, src_file, is_charuco == 1);
            fflush(stdout);
            if (dr_saved >= 0) { dup2(dr_saved, STDOUT_FILENO); close(dr_saved); }
            if (dr_null >= 0) close(dr_null);
        } else {
            DryRunValidation(input_dir, output_dir, src_file, is_charuco == 1);
        }
        } else {
            cout << "Dry-run: would generate patterns from " << input_dir << endl;
        }
        exit(0);
    }

    if (!create_patterns_only){
        MultipleCameraCalibration(input_dir, output_dir, src_file, camera_size, track_size,
                initial_focal_px, zero_tangent_dist,
                zero_k3, fix_principal_point, verbose, read_camera_calibration, only_camera_calibration, max_internal_read, max_internal_use,
                max_external_positions, selectedk, number_points_needed_to_count_pattern, synchronized_rotating_option,
                percentage_global_alg, percentage_global_rp, is_charuco, is_april, interleaved_solve, opts);
    }   else {
        CreatePatterns(input_dir, output_dir, src_file, is_charuco);
    }

    return 0;

}



void CreatePatterns(const string& input_dir, const string& output_dir,
        const string& src_file, bool is_charuco){

    string pattern_dir = output_dir + "patterns/";

    mkdir(pattern_dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    PatternsCreated P_Class(input_dir, pattern_dir, src_file, false, is_charuco, true);

}

void MultipleCameraCalibration(const string& input_dir, const string& output_dir,
        const string& src_file, float camera_size, float track_size,  float initial_focal_px,
        int zero_tangent_dist, int zero_k3, int fix_principal_point,
        bool verbose, bool read_camera_calibration, bool only_camera_calibration,
        int max_internal_read, int max_internal_use, int max_external_positions, int selectedk,
        int number_points_needed_to_count_pattern, int synchronized_rotating_option, float percentage_global_alg,
        float percentage_global_rp,
        const bool is_charuco, const bool is_april, const bool interleavedSolve,
        const CalicoOptions& options){

    // Quiet mode: redirect stdout at fd level (avoids streambuf heap corruption)
    int saved_stdout_fd = -1;
    int null_fd = -1;
    bool quiet = options.quiet;
    if (quiet) {
        fflush(stdout);
        saved_stdout_fd = dup(STDOUT_FILENO);
        null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) dup2(null_fd, STDOUT_FILENO);
    }
    bool show_progress = options.show_progress && !options.no_progress;

    cout << "is charuco, is april " << is_charuco << ", " << is_april << endl;

    int number_not_in = 0;

    string logfile = output_dir + "logfile.txt";
    google::InitGoogleLogging(logfile.c_str());

    bool write_internals = false;

    auto start_timer = std::chrono::high_resolution_clock::now();

    vector<string> camera_write_dirs;
    string command;

    string filename = output_dir + "trace.txt";
    ofstream out_trace; out_trace.open(filename.c_str());

    out_trace << "Max external " << max_external_positions << endl;
    out_trace << "Max internal read " << max_internal_read << endl;
    out_trace << "Max internal use " << max_internal_use << endl;
    out_trace << "Write internal " << write_internals << endl;

    string pattern_dir = output_dir + "patterns/";

    mkdir(pattern_dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    PatternsCreated* P_Class = 0;

    P_Class = new PatternsCreated(input_dir, pattern_dir, src_file, false, is_charuco, false);

    string temp;  int number_cameras;
    string id_camera_dir;

    ifstream in;
    ofstream out;  ofstream ceres_out;

    vector<string> camera_directories;
    ReadDirectory(string(input_dir + "data"), camera_directories);

    if (camera_directories.size() == 0){
        cout << "No camera directories, abort." << endl;
        exit(1);
    }

    // Filter to focus cameras only (if specified)
    if (options.focus_cameras.size() > 0) {
        for (auto it = camera_directories.begin(); it != camera_directories.end(); ) {
            if (std::find(options.focus_cameras.begin(), options.focus_cameras.end(), *it)
                    == options.focus_cameras.end()) {
                cout << "Excluding camera (not in --focus-camera): " << *it << endl;
                it = camera_directories.erase(it);
            } else {
                ++it;
            }
        }
        if (camera_directories.size() == 0) {
            cout << "No cameras match --focus-camera list, abort." << endl;
            exit(1);
        }
    }

    // Filter excluded cameras
    if (options.exclude_cameras.size() > 0) {
        for (auto it = camera_directories.begin(); it != camera_directories.end(); ) {
            if (std::find(options.exclude_cameras.begin(), options.exclude_cameras.end(), *it)
                    != options.exclude_cameras.end()) {
                cout << "Excluding camera: " << *it << endl;
                it = camera_directories.erase(it);
            } else {
                ++it;
            }
        }
        if (camera_directories.size() == 0) {
            cout << "All cameras excluded, abort." << endl;
            exit(1);
        }
    }

    // Save original directory names for file I/O
    vector<string> original_camera_dirs = camera_directories;

    // Apply camera name overrides if provided (for display/output only)
    if (options.camera_names_override.size() == camera_directories.size()) {
        camera_directories = options.camera_names_override;
        cout << "Using custom camera names." << endl;
    } else if (options.camera_names_override.size() > 0) {
        cout << "Warning: --camera-names count (" << options.camera_names_override.size()
             << ") != camera count (" << camera_directories.size() << "). Ignoring." << endl;
    }

    if (!read_camera_calibration){

        mkdir((output_dir + "data/").c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }

    string verbose_dir = output_dir + "verbose_output/";
    if (verbose){

        mkdir(verbose_dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }

    string reconstructed_patterns_dir = output_dir + "reconstructed-patterns/";

    mkdir(reconstructed_patterns_dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    ////// Stage 1. Calibrate cameras wrt visible cameras at all time instants

    number_cameras = camera_directories.size();
    out_trace << "After read " << number_cameras << endl;

    vector<CameraCali*> CCV(number_cameras, 0);
    camera_write_dirs.resize(number_cameras);

    cout << "Calibrating " << number_cameras << " camera(s) ..." << endl;

#pragma omp parallel for private(id_camera_dir) if(number_cameras > 1)
    for (int i = 0; i < number_cameras; i++){

        string actual_cam_dir = original_camera_dirs[i];

        id_camera_dir = input_dir + "data/" + actual_cam_dir;

        CameraCali* C = new CameraCali(id_camera_dir, P_Class,
                max_external_positions, max_internal_read,
                max_internal_use);

#pragma omp critical
        {
            out_trace << "After create new camera " << i << " " << camera_directories[i] << endl;
        }

        CCV[i] = C;
        CCV[i]->no_debug_images = options.no_debug_images;

        id_camera_dir = output_dir + "data/" + camera_directories[i] + "/";
        camera_write_dirs[i] = id_camera_dir;

#pragma omp critical
        {
            cout << id_camera_dir << endl;
        }

        if (!read_camera_calibration){

            mkdir(id_camera_dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

            CCV[i]->FindCorners(id_camera_dir);

            CCV[i]->CalibrateBasic(initial_focal_px, zero_tangent_dist, zero_k3,
                    fix_principal_point, id_camera_dir, number_points_needed_to_count_pattern);

        }

    }

    // --min-boards: auto-exclude cameras detecting fewer than N distinct boards
    if (options.min_boards > 0) {
        for (int i = number_cameras - 1; i >= 0; i--) {
            int board_count = 0;
            for (int j = 0; j < (int)CCV[i]->boards_detected.size(); j++) {
                for (size_t k = 0; k < CCV[i]->boards_detected[j].size(); k++) {
                    if (CCV[i]->boards_detected[j][k]) { board_count++; break; }
                }
            }
            if (board_count < options.min_boards) {
                cout << "Excluding camera (only " << board_count << " boards < --min-boards="
                     << options.min_boards << "): " << camera_directories[i] << endl;
                delete CCV[i];
                CCV.erase(CCV.begin() + i);
                camera_directories.erase(camera_directories.begin() + i);
                original_camera_dirs.erase(original_camera_dirs.begin() + i);
                camera_write_dirs.erase(camera_write_dirs.begin() + i);
                number_cameras--;
            }
        }
        if (number_cameras == 0) {
            cout << "All cameras excluded by --min-boards, abort." << endl;
            exit(1);
        }
    }

    // --detection-summary: print board visibility per camera
    if (options.detection_summary) {
        PrintDetectionSummary(CCV, camera_directories, P_Class->pp.numberBoards);
    }

    auto end_find_id_calibrate = std::chrono::high_resolution_clock::now();

    if (only_camera_calibration){
        out << "Times, seconds" << endl;
        out << "    Load, internal/external cali ... ";
        out << std::chrono::duration_cast<std::chrono::seconds>(end_find_id_calibrate - start_timer).count() << endl;
        out.close();

        exit(1);
    }

    double* camera_params = new double[CCV.size()*12];

    CopyFromCalibration(CCV, camera_params);

    /////// Stage 2. Generation of single and double foundational relationships

    filename = output_dir + "multicamera_detail.txt";
    out.open(filename.c_str());
    filename = output_dir + "multicamera_detail_ceres.txt";
    ceres_out.open(filename.c_str());

    filename = output_dir + "rae_detail_ceres.txt";
    ofstream rae_ceres_out;
    rae_ceres_out.open(filename.c_str());

    for (uint i = 0; i < CCV.size(); i++){
        ceres_out << "Camera " << i << endl;
        for (int j = 0; j < 12; j++){
            ceres_out <<  camera_params[12*i + j] << " ";
        }
        ceres_out << endl;
    }

    MCcali MC(CCV, *P_Class, max_external_positions, out, synchronized_rotating_option);

    number_not_in = MC.BuildCostFunctionAndGraphWithTestAndDegenerateInitialize(CCV, out, verbose_dir, verbose);

    out << "Number not in the connected component = " << number_not_in << endl;
    out << "Number in the connected component = " << MC.NumberSingles() - number_not_in << endl;


    if (number_not_in > 0){
        cout << "Cannot solve, cc component number greater than 1 " << endl;
        cout << "Quitting " << endl;

        exit(1);
    }

    /////// Stage 3. Substitute exemplar pattern and time
    vector<int> variable_order;

    MC.SubstitutePTstar(variable_order, verbose_dir, verbose);

    /////// Stage 4. Iteratively find the best solutions for individual variables

    bool has_some_to_solve  = true;

    auto start_stage_4 = std::chrono::high_resolution_clock::now();
    auto end_stage_4 = std::chrono::high_resolution_clock::now();

    for (uint i = 0; i < CCV.size(); i++){
        CCV[i]->SetUpSelectPointsForMinimization();
    }

    MC.SelectKPointsForMinimization(CCV, selectedk);

    out_trace << "Selected k " << selectedk << endl;

    int number_iters_local = 10;
    vector<int> equation_order;
    vector<int> equations_per_iter;
    int var_accumulator = 0;
    int num_equations = 0;
    int mod_solve = int(round(percentage_global_alg*double(MC.NumberVariables())));

    if (mod_solve == 0){mod_solve = 1;}

    CeresProblemClass CPC(Cali_Quaternion, MC, ceres_out);

    if (variable_order.size() > 0){
        num_equations = CPC.AddToProblemAlgebraicError(MC, variable_order, equation_order, out, variable_order.size());
        equations_per_iter.push_back(num_equations);
    }

    auto stage4_start = std::chrono::high_resolution_clock::now();
    for (int i = 0, vn = MC.NumberVariables(); i < vn && has_some_to_solve == true; i++){

        if (show_progress && (i % 10 == 0 || i == vn - 1 || i == 0)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - stage4_start).count();
            cout << "\rStage 4: variable " << (i+1) << "/" << vn
                 << " (" << (100*(i+1)/max(1, vn)) << "%), "
                 << elapsed << "s elapsed  " << flush;
        }
        cout << "counting " << i << endl;

        has_some_to_solve = MC.IterativelySolveForVariables2(out, false, variable_order,interleavedSolve,
                number_iters_local);

        if (has_some_to_solve){
            var_accumulator++;
        }

        if (!has_some_to_solve){
            bool can_solve = MC.CanSolveSystem();
            out_trace << "Can solve this system??? " << can_solve << endl;

            if (can_solve == false){
                MC.TestSolutionHasIdentityPstarTstar(MC.V_initial);
                bool used_closed_form = MC.SolveClique(out, variable_order, number_iters_local);
                var_accumulator+= 2;
                has_some_to_solve = true;
            }
        }

        if (options.checkpoint_enabled && i % 50 == 0 && i > 0) {
            MC.WriteCheckpoint(output_dir, 4, i);
        }

        if (i % mod_solve == 0){
            num_equations = CPC.AddToProblemAlgebraicError(MC, variable_order, equation_order, out, var_accumulator);
            equations_per_iter.push_back(num_equations);

            CPC.SolveWriteBackToMCAlgebraicError(MC, ceres_out, number_iters_local, true);

            var_accumulator = 0;
        }

    }

    if (show_progress) {
        cout << endl;
    }

    num_equations = CPC.AddToProblemAlgebraicError(MC, variable_order, equation_order, out, var_accumulator);
    equations_per_iter.push_back(num_equations);

    CPC.SolveWriteBackToMCAlgebraicError(MC, ceres_out, number_iters_local, true);

    cout << "break in between algebraic error and reprojection ... " << endl;

    ceres_out << "--------------- reprojection error ----------------------------" << endl;

    ///////////////////////////////////// Stage 5. ///////////////////////////////////////////
    ////////////////////////////// reprojection error. ////////////////////////////////////////////

    int number_equations = equation_order.size();
    int number_equations_before_solve = int(round(percentage_global_rp*float(number_equations)));
    int number_iters_rp = 10;

    CPC.SetUpXForReprojectionError(MC);

    int end_index = 0;

    cout << "Number of equations " << number_equations << endl;

    auto stage5_start = std::chrono::high_resolution_clock::now();
    int num_stage5_steps = max(1, (number_equations + number_equations_before_solve - 1) / number_equations_before_solve);
    int stage5_step = 0;

    for (int start_index = 0; start_index < number_equations; start_index+=number_equations_before_solve){

        if (show_progress) {
            stage5_step++;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - stage5_start).count();
            cout << "\rStage 5: step " << stage5_step << "/" << num_stage5_steps
                 << " (" << (100*stage5_step/num_stage5_steps) << "%), "
                 << elapsed << "s elapsed  " << flush;
        }

        end_index = min(start_index + number_equations_before_solve, number_equations);

        cout << "Adding equations  " << start_index << ", " << end_index << endl;

        CPC.AddEqsToProblemReprojectionError(MC, CCV, camera_params, start_index, end_index, equation_order);

        CPC.SolveWriteBackToMCRP(MC, ceres_out, number_iters_rp, variable_order, true);

    }

    if (show_progress) {
        cout << endl;
    }

    MC.V_progressive_solutions.push_back(MC.V_initial);

    auto end_stage_5 = std::chrono::high_resolution_clock::now();

    /////////////////////////////////// OUTPUT SECTION ////////////////////////////////////////////

    if (!options.no_visualization) {
        MC.ReconstructionAccuracyErrorAndWriteII(reconstructed_patterns_dir, 2, CCV, camera_params, rae_ceres_out);
    }

    rae_ceres_out.close();

    cout << "After iteratively solve " << endl;
    cout << "Number uninitialized " << MC.NumberUninitialized() << endl;
    cout << "Number to fill in " << MC.RemainingToFillIn() << endl;

    out_trace  << "After iteratively solve " << endl;
    out_trace << "Number uninitialized " << MC.NumberUninitialized() << endl;
    out_trace << "Number to fill in " << MC.RemainingToFillIn() << endl;

    out << "Times, seconds" << endl;
    out << "    Load, internal/external cali ... " << std::chrono::duration_cast<std::chrono::seconds>(end_find_id_calibrate - start_timer).count()
                                                                                                                            << " seconds "<< endl;
    out << " Set up cali structure           ... " << std::chrono::duration_cast<std::chrono::seconds>(start_stage_4 - end_find_id_calibrate).count()
                                                                                                                            << " seconds "<< endl;
    out << "Stage 4                           ... " << std::chrono::duration_cast<std::chrono::seconds>(end_stage_4 - start_stage_4).count()
                                                                                                                            << " seconds "<< endl;
    out << "Stage 5                           ... " << std::chrono::duration_cast<std::chrono::seconds>(end_stage_5 - end_stage_4).count()
                                                                                                                            << " seconds "<< endl;
    out << "Total, load and everything ... " <<   std::chrono::duration_cast<std::chrono::seconds>(end_stage_5 - start_timer).count()
                                                                                                                            << " seconds "<< endl;
    out << "MetaTotal, without load+internal/external cali, just network cali ... " <<   std::chrono::duration_cast<std::chrono::seconds>(end_stage_5 - end_find_id_calibrate).count()
                                                                                                                            << " seconds "<< endl;

    out << "Times,milliseconds" << endl;
    out << "    Load, internal/external cali ... ";
    out << std::chrono::duration_cast<std::chrono::milliseconds>(end_find_id_calibrate - start_timer).count() << " ms"<< endl;
    out << " Set up cali structure           ... ";
    out << std::chrono::duration_cast<std::chrono::milliseconds>(start_stage_4 - end_find_id_calibrate).count() << " ms "<< endl;
    out << "Stage 4                           ... ";
    out << std::chrono::duration_cast<std::chrono::milliseconds>(end_stage_4 - start_stage_4).count() << " ms "<< endl;
    out << "Stage 5                           ... ";
    out << std::chrono::duration_cast<std::chrono::milliseconds>(end_stage_5 - end_stage_4).count() << " ms "<< endl;
    out << "Total, load and everything ... ";
    out <<   std::chrono::duration_cast<std::chrono::milliseconds>(end_stage_5 - start_timer).count() << " ms "<< endl;
    out << "MetaTotal, without load+internal/external cali, just network cali ... ";
    out <<   std::chrono::duration_cast<std::chrono::milliseconds>(end_stage_5 - end_find_id_calibrate).count() << " milliseconds "<< endl;

    out << "Times, minutes" << endl;
    out << "Total, load and everything ... " <<   std::chrono::duration_cast<std::chrono::minutes>(end_stage_5 - start_timer).count()
                                                                                                                            << " minutes "<< endl;
    out << "MetaTotal, without load+internal/external cali, just network cali ... " <<   std::chrono::duration_cast<std::chrono::minutes>(end_stage_5 - end_find_id_calibrate).count()
                                                                                                                            << " minutes "<< endl;

    if (!options.no_visualization) {
        MC.WriteSolutionAssessErrorII(output_dir, camera_directories, CCV, 2, true, camera_size, track_size,
                options.camera_colors, options.pattern_colors, show_progress);
    }

    out.close();
    ceres_out.close();

    filename = output_dir + "total_results.txt";

     MC.OutputRunResults(filename);

    // Append per-camera calibration data to total_results.txt
    {
        ofstream total_out(filename.c_str(), std::ios::app);
        total_out << "\nPer-camera calibration:" << endl;
        for (int i = 0; i < number_cameras; i++) {
            total_out << "  " << camera_directories[i] << ": fx="
                      << CCV[i]->internal_parameters(0,0) << ", fy="
                      << CCV[i]->internal_parameters(1,1) << ", cx="
                      << CCV[i]->internal_parameters(0,2) << ", cy="
                      << CCV[i]->internal_parameters(1,2);
            if (CCV[i]->distortion.size() >= 5) {
                total_out << ", d=" << CCV[i]->distortion[0] << ","
                          << CCV[i]->distortion[1] << ","
                          << CCV[i]->distortion[2] << ","
                          << CCV[i]->distortion[3] << ","
                          << CCV[i]->distortion[4];
            }
            // --per-camera-mse: append mean reprojection error
            if (options.per_camera_mse && CCV[i]->reproj_error_per_board.size() > 0) {
                double sum_err = 0;
                int count = 0;
                for (size_t j = 0; j < CCV[i]->reproj_error_per_board.size(); j++) {
                    for (size_t k = 0; k < CCV[i]->reproj_error_per_board[j].size(); k++) {
                        if (CCV[i]->reproj_error_per_board[j][k] >= 0) {
                            sum_err += CCV[i]->reproj_error_per_board[j][k];
                            count++;
                        }
                    }
                }
                if (count > 0) {
                    double mse = sum_err / count;
                    total_out << ", mse=" << mse;
                }
            }
            total_out << endl;
        }
        total_out.close();
    }

    // --json output
    if (options.json_output) {
        string json_file = output_dir + "calibration.json";
        ofstream json_out(json_file.c_str());
        auto end_timer = std::chrono::high_resolution_clock::now();
        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(
            end_timer - start_timer).count();

        json_out << "{" << endl;
        json_out << "  \"tool\": \"calico-dec2023\"," << endl;
        json_out << "  \"num_cameras\": " << number_cameras << "," << endl;
        json_out << "  \"duration_sec\": " << total_sec << "," << endl;
        json_out << "  \"cameras\": [" << endl;
        for (int i = 0; i < number_cameras; i++) {
            json_out << "    {" << endl;
            json_out << "      \"name\": \"" << camera_directories[i] << "\"," << endl;
            json_out << "      \"fx\": " << CCV[i]->internal_parameters(0,0) << "," << endl;
            json_out << "      \"fy\": " << CCV[i]->internal_parameters(1,1) << "," << endl;
            json_out << "      \"cx\": " << CCV[i]->internal_parameters(0,2) << "," << endl;
            json_out << "      \"cy\": " << CCV[i]->internal_parameters(1,2);
            if (CCV[i]->distortion.size() >= 5) {
                json_out << "," << endl;
                json_out << "      \"k1\": " << CCV[i]->distortion[0] << "," << endl;
                json_out << "      \"k2\": " << CCV[i]->distortion[1] << "," << endl;
                json_out << "      \"p1\": " << CCV[i]->distortion[2] << "," << endl;
                json_out << "      \"p2\": " << CCV[i]->distortion[3] << "," << endl;
                json_out << "      \"k3\": " << CCV[i]->distortion[4] << endl;
            } else {
                json_out << endl;
            }
            json_out << "    }";
            if (i < number_cameras - 1) json_out << ",";
            json_out << endl;
        }
        json_out << "  ]" << endl;
        json_out << "}" << endl;
        json_out.close();
        cout << "JSON results written to " << json_file << endl;
    }

    // Restore stdout from quiet mode so summary is always visible
    if (quiet) {
        fflush(stdout);
        if (saved_stdout_fd >= 0) {
            dup2(saved_stdout_fd, STDOUT_FILENO);
            close(saved_stdout_fd);
        }
        if (null_fd >= 0) close(null_fd);
    }

    // --summary: print brief calibration summary
    if (options.summary) {
        auto end_timer = std::chrono::high_resolution_clock::now();
        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(
            end_timer - start_timer).count();
        cout << "\n=== CALIBRATION SUMMARY ===" << endl;
        cout << "Cameras: " << number_cameras << endl;
        cout << "Time: " << total_sec << "s" << endl;
        cout << "Output: " << output_dir << endl;
        cout << "===========================\n" << endl;
    }

    // dealloc
    for (int i = 0; i < number_cameras; i++){
        delete CCV[i];
    }

    CCV.clear();

    delete [] camera_params;

    delete P_Class;

}

void PrintDetectionSummary(const vector<CameraCali*>& CCV,
        const vector<string>& camera_directories,
        int number_patterns) {
    cout << "\n=== DETECTION SUMMARY ===" << endl;
    cout << "Camera" << "\t" << "Images" << "\t" << "Boards";
    for (int p = 0; p < number_patterns; p++) cout << "\tB" << p;
    cout << endl;

    for (size_t i = 0; i < CCV.size(); i++) {
        int n_images = CCV[i]->boards_detected.size();
        int board_count = 0;
        vector<bool> has_board(number_patterns, false);
        for (int j = 0; j < n_images; j++) {
            for (int p = 0; p < number_patterns && p < (int)CCV[i]->boards_detected[j].size(); p++) {
                if (CCV[i]->boards_detected[j][p]) {
                    has_board[p] = true;
                }
            }
        }
        for (int p = 0; p < number_patterns; p++) if (has_board[p]) board_count++;

        cout << camera_directories[i] << "\t" << n_images << "\t" << board_count;
        for (int p = 0; p < number_patterns; p++) {
            cout << "\t" << (has_board[p] ? "Y" : ".");
        }
        cout << endl;
    }
    cout << "=========================\n" << endl;
}

void DryRunValidation(const string& input_dir, const string& output_dir,
        const string& src_file, bool is_charuco) {

    cout << "\n=== DRY RUN ===" << endl;

    // Read camera directories
    vector<string> camera_directories;
    ReadDirectory(string(input_dir + "data"), camera_directories);

    cout << "Number of cameras found: " << camera_directories.size() << endl;

    // Read and validate pattern spec
    string spec_file = input_dir + "network_specification_file.yaml";
    patternParameters pp;

    bool spec_ok = false;
    if (is_charuco) {
        spec_ok = readCharucoSpecificationFile(spec_file, pp);
    } else {
        spec_ok = readAprilTagSpecificationFile(spec_file, pp);
    }

    if (spec_ok) {
        cout << "Pattern specification: " << endl;
        cout << "  Type: " << (is_charuco ? "charuco" : "april") << endl;
        cout << "  Boards: " << pp.numberBoards << endl;
        cout << "  Squares: " << pp.squaresX << " x " << pp.squaresY << endl;
        cout << "  Square length: " << pp.squareLength << " px" << endl;

        if (is_charuco) {
            cout << "  Marker length: " << pp.markerLength << " px" << endl;
            cout << "  Aruco dict code: " << pp.arc_code << endl;
        }

        bool valid = ValidatePatternParams(pp, spec_file);
        if (!valid) {
            cout << "  ** WARNING: Pattern parameters have issues (see above)." << endl;
        } else {
            cout << "  Pattern parameters valid." << endl;
        }
    } else {
        cout << "  ** WARNING: Could not read " << spec_file << endl;
    }

    // Check output directory
    cout << "Output directory: " << output_dir << endl;
    string results_file = output_dir + "total_results.txt";
    ifstream test(results_file.c_str());
    if (test.good()) {
        cout << "  ** WARNING: Output directory already contains results." << endl;
    } else {
        cout << "  Output directory is clean." << endl;
    }

    // Count images per camera
    cout << "\nPer-camera image counts:" << endl;
    for (size_t i = 0; i < camera_directories.size(); i++) {
        string cam_dir = input_dir + "data/" + camera_directories[i];
        string ext_dir = cam_dir + "/external/";

        vector<string> im_names;
        if (IsDirectory(ext_dir)) {
            ReadDirectory(ext_dir, im_names);
        } else {
            ReadDirectory(cam_dir, im_names);
        }

        int ext_count = 0;
        for (size_t j = 0; j < im_names.size(); j++) {
            string ext = im_names[j].substr(im_names[j].size() - 3);
            if (ext != "txt") ext_count++;
        }

        string int_dir = cam_dir + "/internal/";
        int int_count = 0;
        if (IsDirectory(int_dir)) {
            vector<string> int_names;
            ReadDirectory(int_dir, int_names);
            for (size_t j = 0; j < int_names.size(); j++) {
                string ext = int_names[j].substr(int_names[j].size() - 3);
                if (ext != "txt") int_count++;
            }
        }

        cout << "  " << camera_directories[i] << ": "
             << ext_count << " external, " << int_count << " internal" << endl;
    }

    cout << "\nDry-run complete. No calibration performed." << endl;
}
