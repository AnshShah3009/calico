#include "Includes.hpp"
#include "camera-visualization.hpp"

struct CalicoOptions {
    bool dry_run = false;
    bool show_progress = false;
    bool timestamp_output = false;
    bool no_overwrite = false;
    bool checkpoint_enabled = false;
    bool quiet = false;
    bool summary = false;
    bool force = false;
    bool no_progress = false;
    bool json_output = false;
    bool auto_rename = false;
    int num_threads = omp_get_max_threads();
    int max_images = -1;
    string resume_dir = "";
    string config_file = "";
    vector<vector<int> > camera_colors;
    vector<vector<int> > pattern_colors;
    vector<string> camera_names_override;
    vector<string> exclude_cameras;
};

void CreatePatterns(const string& input_dir, const string& output_dir,
        const string& src_file, bool is_charuco);

void MultipleCameraCalibration(const string& input_dir, const string& output_dir,
        const string& src_file, float camera_size, float track_size,  float initial_focal_px,
        int zero_tangent_dist, int zero_k3, int fix_principal_point,
        bool verbose, bool read_camera_calibration, bool only_camera_calibration,
        int max_internal_read, int max_internal_use, int max_external_positions, int selectedk,
        int number_points_needed_to_count_pattern, int synchronized_rotating_option, float percentage_global_alg,
        float percentage_global_rp,
        const bool is_charuco, const bool is_april, const bool interleavedSolve,
        const CalicoOptions& options = CalicoOptions());

void DryRunValidation(const string& input_dir, const string& output_dir,
        const string& src_file, bool is_charuco);
