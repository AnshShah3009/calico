#ifndef OPENCV_COMPAT_HPP_
#define OPENCV_COMPAT_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#elif CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#else
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#endif

#define CALICO_ARUCO_MODERN ((CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7))

inline cv::Ptr<cv::aruco::DetectorParameters> CalicoCreateDetectorParams() {
#if CALICO_ARUCO_MODERN
    return cv::makePtr<cv::aruco::DetectorParameters>();
#else
    return cv::aruco::DetectorParameters::create();
#endif
}

inline cv::Ptr<cv::aruco::Dictionary> CalicoGetDictionary(int arc_code) {
#if CALICO_ARUCO_MODERN
    return cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(arc_code));
#else
    return cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(arc_code));
#endif
}

inline cv::Ptr<cv::aruco::CharucoBoard> CalicoCreateCharucoBoard(
        int squaresX, int squaresY, float squareLength, float markerLength,
        const cv::Ptr<cv::aruco::Dictionary>& dictionary,
        const std::vector<int>& ids = std::vector<int>()) {
#if CALICO_ARUCO_MODERN
    cv::Ptr<cv::aruco::CharucoBoard> board;
    if (ids.empty()) {
        board = cv::makePtr<cv::aruco::CharucoBoard>(
                cv::Size(squaresX, squaresY), squareLength, markerLength, *dictionary);
    } else {
        board = cv::makePtr<cv::aruco::CharucoBoard>(
                cv::Size(squaresX, squaresY), squareLength, markerLength, *dictionary, ids);
    }
    // Printed CALICO boards (and OpenCV < 4.6) use the pre-4.6 even-row layout.
    board->setLegacyPattern(true);
    return board;
#else
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
            squaresX, squaresY, squareLength, markerLength, dictionary);
    if (!ids.empty()) {
        for (size_t j = 0; j < ids.size() && j < board->ids.size(); j++) {
            board->ids[j] = ids[j];
        }
    }
    return board;
#endif
}

inline void CalicoGenerateBoardImage(const cv::Ptr<cv::aruco::CharucoBoard>& board,
        cv::Size imageSize, cv::Mat& boardImage, int margins, int borderBits) {
#if CALICO_ARUCO_MODERN
    board->generateImage(imageSize, boardImage, margins, borderBits);
#else
    board->draw(imageSize, boardImage, margins, borderBits);
#endif
}

inline void CalicoDetectMarkers(const cv::Mat& image,
        const cv::Ptr<cv::aruco::Dictionary>& dictionary,
        std::vector<std::vector<cv::Point2f> >& corners,
        std::vector<int>& ids,
        const cv::Ptr<cv::aruco::DetectorParameters>& params,
        std::vector<std::vector<cv::Point2f> >& rejected) {
#if CALICO_ARUCO_MODERN
    cv::aruco::ArucoDetector detector(*dictionary, *params);
    detector.detectMarkers(image, corners, ids, rejected);
#else
    cv::aruco::detectMarkers(image, dictionary, corners, ids, params, rejected);
#endif
}

inline std::vector<int> CharucoIds(const cv::Ptr<cv::aruco::CharucoBoard>& b) {
#if CALICO_ARUCO_MODERN
    return b->getIds();
#else
    return b->ids;
#endif
}

inline std::vector<std::vector<int> > CharucoNearestMarkerIdx(
        const cv::Ptr<cv::aruco::CharucoBoard>& b) {
#if CALICO_ARUCO_MODERN
    return b->getNearestMarkerIdx();
#else
    return b->nearestMarkerIdx;
#endif
}

inline std::vector<std::vector<int> > CharucoNearestMarkerCorners(
        const cv::Ptr<cv::aruco::CharucoBoard>& b) {
#if CALICO_ARUCO_MODERN
    return b->getNearestMarkerCorners();
#else
    return b->nearestMarkerCorners;
#endif
}

inline std::vector<cv::Point3f> CharucoChessboardCorners(
        const cv::Ptr<cv::aruco::CharucoBoard>& b) {
#if CALICO_ARUCO_MODERN
    return b->getChessboardCorners();
#else
    return b->chessboardCorners;
#endif
}

inline std::vector<std::vector<cv::Point3f> > CharucoObjPoints(
        const cv::Ptr<cv::aruco::CharucoBoard>& b) {
#if CALICO_ARUCO_MODERN
    return b->getObjPoints();
#else
    return b->objPoints;
#endif
}

inline cv::aruco::Dictionary CalicoBoardDictionary(const cv::Ptr<cv::aruco::CharucoBoard>& b) {
#if CALICO_ARUCO_MODERN
    return b->getDictionary();
#else
    return *b->dictionary;
#endif
}

inline void CalicoFilterMarkersForBoard(const cv::Ptr<cv::aruco::CharucoBoard>& board,
        const std::vector<std::vector<cv::Point2f> >& all_corners,
        const std::vector<int>& all_ids,
        std::vector<std::vector<cv::Point2f> >& corners,
        std::vector<int>& ids) {
    corners.clear();
    ids.clear();
    const std::vector<int> board_ids = CharucoIds(board);
    for (size_t k = 0; k < all_ids.size(); k++) {
        for (size_t b = 0; b < board_ids.size(); b++) {
            if (all_ids[k] == board_ids[b]) {
                corners.push_back(all_corners[k]);
                ids.push_back(all_ids[k]);
                break;
            }
        }
    }
}

#if CALICO_ARUCO_MODERN
inline void CalicoInterpolateCharuco(const cv::Mat& image,
        const cv::Ptr<cv::aruco::CharucoBoard>& board,
        std::vector<std::vector<cv::Point2f> >& corners,
        std::vector<int>& ids,
        std::vector<std::vector<cv::Point2f> >& rejected,
        std::vector<cv::Point2f>& charucoCorners,
        std::vector<int>& charucoIds,
        const cv::Ptr<cv::aruco::DetectorParameters>& params,
        const cv::Mat& cameraMatrix = cv::Mat(),
        const cv::Mat& distCoeffs = cv::Mat()) {
    cv::aruco::Dictionary dict = CalicoBoardDictionary(board);
    cv::aruco::ArucoDetector detector(dict, *params);
    if (!rejected.empty() && !corners.empty()) {
        detector.refineDetectedMarkers(image, *board, corners, ids, rejected);
    }
    cv::aruco::CharucoParameters cp;
    cp.minMarkers = 2;
    cp.tryRefineMarkers = false;
    if (!cameraMatrix.empty()) {
        cp.cameraMatrix = cameraMatrix;
        cp.distCoeffs = distCoeffs;
    }
    cv::aruco::CharucoDetector charuco_detector(*board, cp, *params);
    charuco_detector.detectBoard(image, charucoCorners, charucoIds, corners, ids);
}
#endif

#endif
