// Calibration board registry: several static boards, each with its own layout. Three types:
//   aprilgrid    - Kalibr AprilGrid: tags with a 2-bit black border and black spacer squares at
//                  the tag corners (every tag corner is an X-junction); ids first_id + k.
//   charuco      - chessboard with ArUco / AprilTag markers in the white squares; the points are
//                  the inner chessboard corners; marker ids first_id + k.
//   checkerboard - plain chessboard; the points are the inner corners. Must be seen whole; use
//                  an odd x even square count so its orientation is unambiguous.
// Ids must not overlap between boards of the same marker family / dictionary.
#ifndef ROBOCAL_BOARDS_HPP_
#define ROBOCAL_BOARDS_HPP_

#include "se3.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace robocal {

enum class BoardType { AprilGrid, Charuco, Checkerboard };

inline BoardType BoardTypeFromName(const std::string& s) {
    if (s == "aprilgrid") return BoardType::AprilGrid;
    if (s == "charuco") return BoardType::Charuco;
    if (s == "checkerboard" || s == "chessboard") return BoardType::Checkerboard;
    throw std::runtime_error("unknown board type '" + s + "' (aprilgrid, charuco, checkerboard)");
}

inline std::string BoardTypeName(BoardType t) {
    switch (t) {
        case BoardType::AprilGrid: return "aprilgrid";
        case BoardType::Charuco: return "charuco";
        case BoardType::Checkerboard: return "checkerboard";
    }
    return "?";
}

// Board frame: x right, y up, z out of the print; origin at the bottom-left of the pattern.
//   aprilgrid: tag id = first_id + row * cols + col, row 0 at the bottom; tag_size is the black
//              square, tag_spacing the gap as a fraction of tag_size; rows / cols count tags.
//   charuco / checkerboard: rows / cols count squares; square_size; the top-left square is
//              black; inner corner id = j * (cols - 1) + i, i from the left, j from the top (OpenCV).
//              ChArUco markers (marker_size, dictionary) sit in the white squares, ids first_id + k
//              row-major from the top-left; legacy: true = the OpenCV < 4.6 layout (for an even
//              number of rows the top-left square is white).
struct BoardSpec {
    std::string name;
    std::string family = "tag36h11";  // aprilgrid tag family / charuco dictionary (e.g. DICT_5X5_1000)
    int rows = 0;
    int cols = 0;
    double tag_size = 0.0;
    double tag_spacing = 0.3;
    int first_id = 0;
    bool level = false;           // lies flat: its normal is held parallel to world z
    double level_sigma_deg = 0.3;
    BoardType type = BoardType::AprilGrid;  // (after the AprilGrid fields: aggregate initialisers)
    double square_size = 0.0;     // charuco / checkerboard
    double marker_size = 0.0;     // charuco
    bool legacy = false;          // charuco: OpenCV < 4.6 layout

    // Mounted on an arm link (e.g. a marker on the gripper) instead of static in the world.
    std::string mount_arm;              // arm name ("" = static board)
    std::string mount_link;             // the arm's tip link or a fixed child of it
    bool has_mount_init = false;
    Iso3 T_link_board_init = Iso3::Identity();
    bool mount_fixed = false;           // mount known (CAD / measured): makes the last joint's zero solvable

    // Board shape unknowns: bow along x / y and twist (m), x / y print scale (relative).
    bool fit_shape = false;
    bool fit_scale = false;
    double shape_sigma_m = 0.003;       // prior on bow / twist
    double scale_sigma = 0.01;          // prior on the print scale
    // Known print scale (relative, e.g. from a ruler check): the scale's start value, held fixed
    // unless fit_scale (then the centre of its prior).
    double print_scale[2] = {0, 0};
    bool has_shape_fixed = false;     // shape_fixed_mm: bow x / bow y / twist held (e.g. from an image-only fit)
    double shape_fixed_m[3] = {0, 0, 0};
    // Per-printed-point height above / below the board plane (m), each with a prior: paper sheets
    // are not flat at the 0.1-0.5 mm level and their bumps are too local for bow / twist.
    bool fit_points = false;
    double point_sigma_m = 0.0003;
    // In-plane print warp: X += a2 v + a3 u^2 + a4 u v + a5 v^2, Y += b1 u + b3 u^2 + b4 u v + b5 v^2
    // (m; u, v in [-1, 1] over the board), each with a prior: paper / printer feed distortion that
    // a uniform x / y scale cannot describe.
    bool fit_warp = false;
    double warp_sigma_m = 0.0003;

    // ---- AprilGrid tags (also: ChArUco markers through MarkerSquare)
    int NumTags() const {
        if (type == BoardType::AprilGrid) return rows * cols;
        if (type == BoardType::Charuco) return (rows * cols) / 2 + ((rows * cols) % 2 && !TopLeftBlack() ? 1 : 0);
        return 0;
    }
    bool HasId(int id) const { return id >= first_id && id < first_id + NumTags(); }
    double Pitch() const { return type == BoardType::AprilGrid ? tag_size * (1.0 + tag_spacing) : square_size; }
    double Width() const {
        return type == BoardType::AprilGrid ? cols * tag_size + (cols - 1) * tag_size * tag_spacing : cols * square_size;
    }
    double Height() const {
        return type == BoardType::AprilGrid ? rows * tag_size + (rows - 1) * tag_size * tag_spacing : rows * square_size;
    }

    // Corners of tag ``id`` in the board frame, ordered BL, BR, TR, TL (the order the
    // Kaess / Kalibr detector reports p[0..3]).
    std::array<Vec3, 4> TagCorners(int id) const {
        if (type != BoardType::AprilGrid || !HasId(id)) throw std::runtime_error("tag id not on board " + name);
        const int k = id - first_id;
        const double x0 = (k % cols) * Pitch();
        const double y0 = (k / cols) * Pitch();
        return {Vec3(x0, y0, 0), Vec3(x0 + tag_size, y0, 0), Vec3(x0 + tag_size, y0 + tag_size, 0),
                Vec3(x0, y0 + tag_size, 0)};
    }

    // ---- chessboard squares (charuco / checkerboard), r from the top, c from the left
    bool TopLeftBlack() const { return !(type == BoardType::Charuco && legacy && rows % 2 == 0); }
    bool SquareBlack(int r, int c) const { return ((r + c) % 2 == 0) == TopLeftBlack(); }
    // ChArUco marker id -> its square (r, c); false if the id is not on this board.
    bool MarkerSquare(int id, int& r, int& c) const {
        if (type != BoardType::Charuco || !HasId(id)) return false;
        int k = id - first_id;
        for (int rr = 0; rr < rows; ++rr) {
            for (int cc = 0; cc < cols; ++cc) {
                if (SquareBlack(rr, cc)) continue;
                if (k-- == 0) {
                    r = rr;
                    c = cc;
                    return true;
                }
            }
        }
        return false;
    }
    // ChArUco marker corners in the board frame, ordered TL, TR, BR, BL (cv::aruco order).
    std::array<Vec3, 4> MarkerCorners(int id) const {
        int r, c;
        if (!MarkerSquare(id, r, c)) throw std::runtime_error("marker id not on board " + name);
        const double m = 0.5 * (square_size - marker_size), H = Height();
        const double x0 = c * square_size + m, y0 = H - r * square_size - m;
        return {Vec3(x0, y0, 0), Vec3(x0 + marker_size, y0, 0), Vec3(x0 + marker_size, y0 - marker_size, 0),
                Vec3(x0, y0 - marker_size, 0)};
    }

    // ---- points: what the detector reports and the solver uses
    //   aprilgrid: 4 * k + corner (BL, BR, TR, TL of tag k); chessboards: inner corner id
    int NumPoints() const {
        return type == BoardType::AprilGrid ? 4 * rows * cols : (rows - 1) * (cols - 1);
    }
    Vec3 Point(int pid) const {
        if (pid < 0 || pid >= NumPoints()) throw std::runtime_error("point id not on board " + name);
        if (type == BoardType::AprilGrid) return TagCorners(first_id + pid / 4)[pid % 4];
        const int i = pid % (cols - 1), j = pid / (cols - 1);
        return Vec3((i + 1) * square_size, Height() - (j + 1) * square_size, 0);
    }
};

inline void ValidateBoards(const std::vector<BoardSpec>& boards) {
    for (size_t i = 0; i < boards.size(); ++i) {
        const BoardSpec& a = boards[i];
        if (a.rows <= 0 || a.cols <= 0) throw std::runtime_error("board " + a.name + ": rows and cols must be > 0");
        if (a.type == BoardType::AprilGrid && a.tag_size <= 0) {
            throw std::runtime_error("board " + a.name + ": tag_size must be > 0");
        }
        if (a.type != BoardType::AprilGrid && (a.square_size <= 0 || a.rows < 2 || a.cols < 2)) {
            throw std::runtime_error("board " + a.name + ": square_size > 0 and at least 2 x 2 squares");
        }
        if (a.type == BoardType::Charuco && !(a.marker_size > 0 && a.marker_size < a.square_size)) {
            throw std::runtime_error("board " + a.name + ": 0 < marker_size < square_size");
        }
        if (a.type == BoardType::Checkerboard && (a.rows + a.cols) % 2 == 0) {
            throw std::runtime_error("board " + a.name + ": use an odd x even number of squares (orientation)");
        }
        for (size_t j = i + 1; j < boards.size(); ++j) {
            const BoardSpec& b = boards[j];
            if (a.name == b.name) throw std::runtime_error("duplicate board name " + a.name);
            if (a.type == BoardType::Checkerboard || b.type == BoardType::Checkerboard) {
                if (a.type == b.type) {
                    throw std::runtime_error("boards " + a.name + " and " + b.name +
                                             ": two checkerboards cannot be told apart");
                }
                continue;
            }
            if (a.family != b.family) continue;
            const bool overlap = a.first_id < b.first_id + b.NumTags() && b.first_id < a.first_id + a.NumTags();
            if (overlap) {
                throw std::runtime_error("boards " + a.name + " and " + b.name + " share tag ids (" + a.family + ")");
            }
        }
    }
}

}  // namespace robocal

#endif  // ROBOCAL_BOARDS_HPP_
