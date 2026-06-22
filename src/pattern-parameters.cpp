/*
 * pattern-parameters.cpp
 *
 *  Created on: Sep 1, 2020
 *      Author: atabb
 */


#include "pattern-parameters.hpp"

bool readAprilTagSpecificationFile(string filename, patternParameters &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["type"] >> params.type;
    fs["squaresX"] >> params.squaresX;
    fs["squaresY"] >> params.squaresY;
    fs["squareLength"] >> params.squareLength;
    fs["margins"] >> params.margins;
    fs["tagSpace"] >> params.tagSpace;
    fs["numberBoards"] >> params.numberBoards;
    fs["april_family"] >> params.april_family;
    return true;
}


bool readCharucoSpecificationFile(string filename, patternParameters &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
   // fs["type"] >> params.type;
    fs["squaresX"] >> params.squaresX;
    fs["squaresY"] >> params.squaresY;
    fs["squareLength"] >> params.squareLength;
    fs["markerLength"] >> params.markerLength;
    fs["numberBoards"] >> params.numberBoards;
    fs["arcCode"] >> params.arc_code;
    fs["margins"] >> params.margins;
    return true;
}

int GetArucoDictionarySize(int arc_code) {
    // from OpenCV's PREDEFINED_DICTIONARY_NAME enum
    switch (arc_code) {
        case 0: return 50;   // DICT_4X4_50
        case 1: return 100;  // DICT_4X4_100
        case 2: return 250;  // DICT_4X4_250
        case 3: return 1000; // DICT_4X4_1000
        case 4: return 50;   // DICT_5X5_50
        case 5: return 100;  // DICT_5X5_100
        case 6: return 250;  // DICT_5X5_250
        case 7: return 1000; // DICT_5X5_1000
        case 8: return 50;   // DICT_6X6_50
        case 9: return 100;  // DICT_6X6_100
        case 10: return 250; // DICT_6X6_250
        case 11: return 1000;// DICT_6X6_1000
        case 12: return 50;  // DICT_7X7_50
        case 13: return 100; // DICT_7X7_100
        case 14: return 250; // DICT_7X7_250
        case 15: return 1000;// DICT_7X7_1000
        case 16: return 1024;// DICT_ARUCO_ORIGINAL
        case 17: return 30;  // DICT_APRILTAG_16h5
        case 18: return 35;  // DICT_APRILTAG_25h9
        case 19: return 2320;// DICT_APRILTAG_36h10
        case 20: return 587; // DICT_APRILTAG_36h11
        default: return -1;
    }
}

bool ValidatePatternParams(const patternParameters& pp, const string& spec_file) {
    bool ok = true;

    if (pp.squaresX <= 0 || pp.squaresY <= 0) {
        cerr << "ERROR: squaresX and squaresY must be > 0 in " << spec_file << endl;
        ok = false;
    }

    if (pp.squareLength <= 0) {
        cerr << "ERROR: squareLength must be > 0 in " << spec_file << endl;
        ok = false;
    }

    if (pp.markerLength > pp.squareLength && pp.markerLength > 0) {
        cerr << "ERROR: markerLength (" << pp.markerLength
             << ") > squareLength (" << pp.squareLength
             << ") in " << spec_file << endl;
        ok = false;
    }

    if (pp.numberBoards <= 0) {
        cerr << "ERROR: numberBoards must be > 0 in " << spec_file << endl;
        ok = false;
    }

    if (pp.type == charuco) {
        int dict_size = GetArucoDictionarySize(pp.arc_code);
        int needed_markers = pp.squaresX * pp.squaresY * pp.numberBoards;
        if (dict_size > 0 && needed_markers > dict_size) {
            cerr << "ERROR: need " << needed_markers << " markers but aruco dictionary "
                 << pp.arc_code << " only has " << dict_size
                 << " markers in " << spec_file << endl;
            ok = false;
        }
    }

    return ok;
}


