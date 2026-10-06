/*
 * apriltag-file.hpp
 *
 *  Created on: Sep 1, 2020
 *      Author: atabb
 */

#ifndef APRILTAG_FILE_HPP_
#define APRILTAG_FILE_HPP_

#include "Includes.hpp"

class AprilTagObject {

public:
    AprilTags::TagDetector* m_tagDetector;
    AprilTags::TagCodes m_tagCodes;

    bool m_draw; // draw image and April tag detections?
    bool m_timing; // print timing information for each tag extraction call

    int m_width; // image size in pixels
    int m_height;
    double m_tagSize; // April tag side length in meters of square black frame
    double m_fx; // camera focal length in pixels
    double m_fy;
    double m_px; // camera principal point
    double m_py;

    int m_deviceId; // camera id (in case of multiple cameras)

    // U Mich apriltag lib, alternate representation.
    apriltag_family_t *m_apriltag_lib_tag_family;

    vector<string> m_tag_strings;
    vector<AprilTags::TagCodes>  m_vectorTagCodes;
    // default constructor
    AprilTagObject();

    // changing the tag family
    void setTagCodes(string s);

    void setup() ;

};

std::string NormalizeAprilFamily(const std::string& name);
int AprilFamilyCapacity(const std::string& family);
int AprilFamilyOpenCVDict(const std::string& family);

// Fills Kaess-style detections (CCW from lower-left). Returns true if any tags found.
bool DetectAprilTagsOpenCV(const cv::Mat& gray, const std::string& family,
        std::vector<AprilTags::TagDetection>& detections);

// GPU cuAprilTags (tag36h11) if built, else Kaess, else OpenCV AprilTag dictionaries.
int DetectAprilTagGrid(const cv::Mat& gray, const std::string& family,
        AprilTags::TagDetector* kaess,
        std::vector<AprilTags::TagDetection>& detections);

cv::Mat apriltag_to_image_local_black_border(apriltag_family_t *fam, int idx);

cv::Mat RenderAprilTagMarker(const std::string& family, int idx, int side_px,
        apriltag_family_t* tf);

#endif /* APRILTAG_FILE_HPP_ */
