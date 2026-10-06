/*
 * apriltag-file.cpp
 *
 *  Created on: Sep 1, 2020
 *      Author: atabb
 */


#include "apriltag-file.hpp"
#include "helper.hpp"
#include "cuda-detect.hpp"
#include <cctype>
#include <mutex>


AprilTagObject::AprilTagObject() :
// default settings, most can be modified through command line options (see below)
m_tagDetector(NULL),
m_tagCodes(AprilTags::tagCodes36h11),

m_draw(true),
m_timing(false),

m_width(640),
m_height(480),
m_tagSize(0.166),
m_fx(600),
m_fy(600),
m_px(m_width/2),
m_py(m_height/2),

m_deviceId(0),
m_apriltag_lib_tag_family(0)
{

    m_tag_strings =  vector<string>{"tag16h5", "tag25h7", "tag25h9", "tag36h9", "tag36h11" };
    m_vectorTagCodes = vector<AprilTags::TagCodes>{AprilTags::tagCodes16h5, AprilTags::tagCodes25h7, AprilTags::tagCodes25h9,
        AprilTags::tagCodes36h9, AprilTags::tagCodes36h11};

}

// changing the tag family
void AprilTagObject::setTagCodes(string s) {
    s = NormalizeAprilFamily(s);

    int tag_index = -1;
    uint number_types = uint(m_tag_strings.size());

    for (uint i = 0; i < number_types; i++){
        if (s.compare(m_tag_strings[i]) == 0){
            tag_index = i;
            m_tagCodes = m_vectorTagCodes[tag_index];
            i = number_types;
        }
    }

    cout << "Tag family " << s << endl;

    if (tag_index == -1){
        cout << "Invalid tag family specified. Supported: tag16h5, tag25h7, tag25h9, tag36h9, tag36h11" << endl;
        exit(1);
    }

    vector<string> tag_string{"tag36h11", "tag25h9", "tag16h5" };
    vector<std::function<apriltag_family_t *()> > tag_create_functions{tag36h11_create, tag25h9_create, tag16h5_create};

    tag_index = -1;
    number_types = uint(tag_string.size());

    for (uint i = 0; i < number_types; i++){
        if (s.compare(tag_string[i]) == 0){
            tag_index = i;
            m_apriltag_lib_tag_family = tag_create_functions[tag_index]();
            i = number_types;
        }
    }

    if (tag_index == -1){
        cout << "AprilRobotics image generator has no renderer for " << s
             << "; OpenCV generateImageMarker will be used if the dictionary exists." << endl;
    }

}

void AprilTagObject::setup() {

    m_tagDetector = new AprilTags::TagDetector(m_tagCodes);

}


Mat apriltag_to_image_local_black_border(apriltag_family_t *fam, int idx)
{

    uint64_t APRILTAG_U64_ONE = 1;

    assert(fam != NULL);
    assert(idx >= 0 && idx < int(fam->ncodes));

    uint64_t code = fam->codes[idx];

    int width = fam->total_width;

    Mat marker_mat = Mat::zeros(width, width, CV_8UC1);

    // make the border.

    int border_start = (fam->total_width - fam->width_at_border)/2;

    for (uint i = 0; i < fam->nbits; i++) {
        if (code & (APRILTAG_U64_ONE << (fam->nbits - i - 1))) {
            marker_mat.data[(fam->bit_y[i] + border_start)*width + fam->bit_x[i] + border_start] = 255;
        }
    }

    return marker_mat;
}

string NormalizeAprilFamily(const string& name) {
    string s = name;
    for (size_t i = 0; i < s.size(); i++) {
        s[i] = static_cast<char>(tolower(static_cast<unsigned char>(s[i])));
    }
    if (s.find("16h5") != string::npos) return "tag16h5";
    if (s.find("25h7") != string::npos) return "tag25h7";
    if (s.find("25h9") != string::npos) return "tag25h9";
    if (s.find("36h9") != string::npos) return "tag36h9";
    if (s.find("36h10") != string::npos) return "tag36h10";
    if (s.find("36h11") != string::npos) return "tag36h11";
    return s;
}

int AprilFamilyCapacity(const string& family) {
    const string s = NormalizeAprilFamily(family);
    if (s == "tag16h5") return 30;
    if (s == "tag25h7") return 242;
    if (s == "tag25h9") return 35;
    if (s == "tag36h9") return 532;
    if (s == "tag36h10") return 2320;
    if (s == "tag36h11") return 587;
    return -1;
}

int AprilFamilyOpenCVDict(const string& family) {
    const string s = NormalizeAprilFamily(family);
    if (s == "tag16h5") return 17;  // DICT_APRILTAG_16h5
    if (s == "tag25h9") return 18;  // DICT_APRILTAG_25h9
    if (s == "tag36h10") return 19; // DICT_APRILTAG_36h10
    if (s == "tag36h11") return 20; // DICT_APRILTAG_36h11
    return -1;
}

bool DetectAprilTagsOpenCV(const Mat& gray, const string& family,
        vector<AprilTags::TagDetection>& detections) {
#if CALICO_ARUCO_MODERN
    const int dict_code = AprilFamilyOpenCVDict(family);
    if (dict_code < 0) {
        return false;
    }
    Ptr<aruco::Dictionary> dict = CalicoGetDictionary(dict_code);
    Ptr<aruco::DetectorParameters> params = CalicoCreateDetectorParams();
    params->cornerRefinementMethod = (int)aruco::CORNER_REFINE_APRILTAG;
    vector<vector<Point2f> > corners, rejected;
    vector<int> ids;
    CalicoDetectMarkers(gray, dict, corners, ids, params, rejected);
    detections.clear();
    detections.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        AprilTags::TagDetection d;
        d.id = ids[i];
        d.good = true;
        const vector<Point2f>& c = corners[i];
        if (c.size() < 4) {
            continue;
        }
        d.p[0] = std::make_pair(c[3].x, c[3].y);
        d.p[1] = std::make_pair(c[2].x, c[2].y);
        d.p[2] = std::make_pair(c[1].x, c[1].y);
        d.p[3] = std::make_pair(c[0].x, c[0].y);
        detections.push_back(d);
    }
    return !detections.empty();
#else
    (void)gray;
    (void)family;
    (void)detections;
    return false;
#endif
}

int DetectAprilTagGrid(const Mat& gray, const string& family,
        AprilTags::TagDetector* kaess,
        vector<AprilTags::TagDetection>& detections) {
    detections.clear();
    if (gray.empty()) {
        return 0;
    }
    const string fam = NormalizeAprilFamily(family);
    static string logged_backend;
    static std::mutex detect_mutex;
    std::lock_guard<std::mutex> lock(detect_mutex);

    if (g_use_cuda && calico_cuda::DetectAprilTags(gray, fam, detections) && !detections.empty()) {
        if (logged_backend != "cuAprilTags") {
            cout << "AprilTag backend: cuAprilTags (GPU)" << endl;
            logged_backend = "cuAprilTags";
        }
        return int(detections.size());
    }

    if (kaess != 0) {
        detections = kaess->extractTags(gray);
        if (!detections.empty()) {
            if (logged_backend != "kaess") {
                cout << "AprilTag backend: Kaess/AprilTags CPU" << endl;
                logged_backend = "kaess";
            }
            return int(detections.size());
        }
    }

    if (DetectAprilTagsOpenCV(gray, fam, detections)) {
        if (logged_backend != "opencv") {
            cout << "AprilTag backend: OpenCV " << fam << endl;
            logged_backend = "opencv";
        }
        return int(detections.size());
    }

    return 0;
}

Mat RenderAprilTagMarker(const string& family, int idx, int side_px, apriltag_family_t* tf) {
    Mat marker;
    if (tf != 0) {
        Mat bits = apriltag_to_image_local_black_border(tf, idx);
        resize(bits, marker, Size(side_px, side_px), 0, 0, INTER_NEAREST);
        return marker;
    }
#if CALICO_ARUCO_MODERN
    const int dict_code = AprilFamilyOpenCVDict(family);
    if (dict_code >= 0) {
        Ptr<aruco::Dictionary> dict = CalicoGetDictionary(dict_code);
        dict->generateImageMarker(idx, side_px, marker, 1);
        if (marker.channels() == 3) {
            cvtColor(marker, marker, COLOR_BGR2GRAY);
        }
        return marker;
    }
#endif
    (void)family;
    (void)idx;
    (void)side_px;
    return marker;
}


