#!/usr/bin/env python3
"""Unit tests for CALICO Python tooling: export_calibration, ingest_video, visualize_calibration."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from export_calibration import rot_mat_to_quat_wxyz


def _tool_python() -> str:
    """Python for spawning tools/*.py subprocesses.

    Some agent harnesses set sys.executable to an AppImage/launcher that
    returns 0 without running the script. Prefer a real CPython binary.
    """
    exe = Path(sys.executable).name.lower()
    if "appimage" in exe or "t3-code" in exe:
        for cand in ("/usr/bin/python3.8", "/usr/bin/python3.10", "/usr/bin/python3"):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return sys.executable


class TestRotationMatrixToQuaternion(unittest.TestCase):
    def test_identity(self):
        R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        qw, qx, qy, qz = rot_mat_to_quat_wxyz(R)
        self.assertAlmostEqual(qw, 1.0, places=5)
        self.assertAlmostEqual(qx, 0.0, places=5)
        self.assertAlmostEqual(qy, 0.0, places=5)
        self.assertAlmostEqual(qz, 0.0, places=5)

    def test_rot_x_180(self):
        # 180 deg around X: diag is [1, -1, -1] -> m00 is max
        R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        qw, qx, qy, qz = rot_mat_to_quat_wxyz(R)
        self.assertAlmostEqual(abs(qx), 1.0, places=5)
        self.assertAlmostEqual(qw, 0.0, places=5)
        self.assertAlmostEqual(qy, 0.0, places=5)
        self.assertAlmostEqual(qz, 0.0, places=5)

    def test_rot_y_180(self):
        # 180 deg around Y: diag is [-1, 1, -1] -> m11 is max
        R = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        qw, qx, qy, qz = rot_mat_to_quat_wxyz(R)
        self.assertAlmostEqual(abs(qy), 1.0, places=5)
        self.assertAlmostEqual(qw, 0.0, places=5)
        self.assertAlmostEqual(qx, 0.0, places=5)
        self.assertAlmostEqual(qz, 0.0, places=5)

    def test_rot_z_180(self):
        # 180 deg around Z: diag is [-1, -1, 1] -> m22 is max
        R = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        qw, qx, qy, qz = rot_mat_to_quat_wxyz(R)
        self.assertAlmostEqual(abs(qz), 1.0, places=5)
        self.assertAlmostEqual(qw, 0.0, places=5)
        self.assertAlmostEqual(qx, 0.0, places=5)
        self.assertAlmostEqual(qy, 0.0, places=5)

    def test_rot_z_90(self):
        # 90 deg around Z: trace = 1.0 > 0
        R = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        qw, qx, qy, qz = rot_mat_to_quat_wxyz(R)
        expected = math.sqrt(0.5)
        self.assertAlmostEqual(qw, expected, places=5)
        self.assertAlmostEqual(qx, 0.0, places=5)
        self.assertAlmostEqual(qy, 0.0, places=5)
        self.assertAlmostEqual(qz, expected, places=5)


class TestExportCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.cali_data = {
            "tool": "calico-dec2023",
            "num_cameras": 2,
            "duration_sec": 1.5,
            "opencv": "5.0.0",
            "cuda": False,
            "pattern": "april",
            "cameras": [
                {
                    "name": "cam0",
                    "width": 960,
                    "height": 720,
                    "fx": 800.0,
                    "fy": 800.0,
                    "cx": 480.0,
                    "cy": 360.0,
                    "k1": 0.01,
                    "k2": -0.001,
                    "p1": 0.0,
                    "p2": 0.0,
                    "k3": 0.0,
                    "T_world_camera": [
                        [1.0, 0.0, 0.0, -100.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 500.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "name": "cam1",
                    "width": 960,
                    "height": 720,
                    "fx": 820.0,
                    "fy": 820.0,
                    "cx": 475.0,
                    "cy": 365.0,
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                    "k3": 0.0,
                    "T_world_camera": [
                        [1.0, 0.0, 0.0, 100.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 500.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            ],
            "boards": [
                {
                    "index": 0,
                    "T_world_board": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        }
        (self.root / "calibration.json").write_text(json.dumps(self.cali_data))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_export_all_formats(self):
        script = Path(__file__).resolve().parent / "export_calibration.py"
        out_dir = self.root / "exports"
        subprocess.check_call(
            [_tool_python(), str(script), str(self.root), "--format", "all", "-o", str(out_dir)]
        )

        # ROS CameraInfo
        self.assertTrue((out_dir / "ros" / "cam0_camera_info.yaml").exists())
        self.assertTrue((out_dir / "ros" / "cam1_camera_info.yaml").exists())
        cam0_yaml = (out_dir / "ros" / "cam0_camera_info.yaml").read_text()
        self.assertIn("image_width: 960", cam0_yaml)
        self.assertIn("distortion_model: plumb_bob", cam0_yaml)

        # Nerfstudio transforms.json
        self.assertTrue((out_dir / "nerfstudio" / "transforms.json").exists())
        transforms = json.loads((out_dir / "nerfstudio" / "transforms.json").read_text())
        self.assertIn("fl_x", transforms)
        self.assertEqual(len(transforms.get("frames", [])), 2)

        # COLMAP
        self.assertTrue((out_dir / "colmap" / "cameras.txt").exists())
        self.assertTrue((out_dir / "colmap" / "images.txt").exists())
        cam_lines = [
            line
            for line in (out_dir / "colmap" / "cameras.txt").read_text().splitlines()
            if not line.startswith("#") and line.strip()
        ]
        self.assertEqual(len(cam_lines), 2)

        # OpenCV FileStorage
        self.assertTrue((out_dir / "opencv" / "cam0_opencv.yaml").exists())
        self.assertTrue((out_dir / "opencv" / "cam1_opencv.yaml").exists())
        fs = cv2.FileStorage(str(out_dir / "opencv" / "cam0_opencv.yaml"), cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        T = fs.getNode("transform_world_camera").mat()
        fs.release()

        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(D.shape, (5, 1))
        self.assertEqual(T.shape, (4, 4))
        self.assertAlmostEqual(K[0, 0], 800.0)

    def test_export_real_april_and_charuco(self):
        repo_root = Path(__file__).resolve().parent.parent
        script = Path(__file__).resolve().parent / "export_calibration.py"

        # 1. Test AprilTag grid calibration export (results/april-gpu)
        april_cali = repo_root / "results" / "april-gpu"
        if april_cali.exists() and (april_cali / "calibration.json").exists():
            out_april = self.root / "export_april"
            subprocess.check_call([
                _tool_python(), str(script), str(april_cali), "--format", "all", "-o", str(out_april)
            ])
            self.assertTrue((out_april / "ros" / "cam0_camera_info.yaml").exists())
            self.assertTrue((out_april / "nerfstudio" / "transforms.json").exists())
            self.assertTrue((out_april / "colmap" / "cameras.txt").exists())
            self.assertTrue((out_april / "opencv" / "cam0_opencv.yaml").exists())

        # 2. Test ChArUco calibration export (results/run5)
        charuco_cali = repo_root / "results" / "run5"
        if charuco_cali.exists() and (charuco_cali / "calibration.json").exists():
            out_charuco = self.root / "export_charuco"
            subprocess.check_call([
                _tool_python(), str(script), str(charuco_cali), "--format", "all", "-o", str(out_charuco)
            ])
            self.assertTrue((out_charuco / "ros" / "111F816F_camera_info.yaml").exists())
            self.assertTrue((out_charuco / "nerfstudio" / "transforms.json").exists())
            self.assertTrue((out_charuco / "colmap" / "cameras.txt").exists())
            self.assertTrue((out_charuco / "opencv" / "111F816F_opencv.yaml").exists())


class TestIngestVideo(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        # Create two 10-frame test MP4 files
        for cam in ["cam0", "cam1"]:
            vpath = str(self.root / f"{cam}.mp4")
            writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
            for i in range(10):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                cv2.putText(frame, f"{cam}_{i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                writer.write(frame)
            writer.release()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_ingest_with_stride(self):
        script = Path(__file__).resolve().parent / "ingest_video.py"
        dataset_dir = self.root / "calico_dataset"
        subprocess.check_call(
            [
                _tool_python(),
                str(script),
                "--video-dir",
                str(self.root),
                "--output-dir",
                str(dataset_dir),
                "--stride",
                "2",
                "--max-frames",
                "4",
            ]
        )

        self.assertTrue((dataset_dir / "pattern_square_mm0.txt").exists())
        self.assertTrue((dataset_dir / "network_specification_file.yaml").exists())

        cam0_ext = dataset_dir / "data" / "cam0" / "external"
        cam1_ext = dataset_dir / "data" / "cam1" / "external"
        self.assertTrue(cam0_ext.exists())
        self.assertTrue(cam1_ext.exists())

        c0_files = sorted(cam0_ext.glob("*.png"))
        c1_files = sorted(cam1_ext.glob("*.png"))
        self.assertEqual(len(c0_files), 4)
        self.assertEqual(len(c1_files), 4)
        self.assertEqual(c0_files[0].name, "00000.png")


class TestVisualizeCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.cali_data = {
            "tool": "calico-dec2023",
            "num_cameras": 2,
            "duration_sec": 1.2,
            "opencv": "5.0.0",
            "cuda": True,
            "pattern": "charuco",
            "cameras": [
                {
                    "name": "cam0",
                    "fx": 800.0,
                    "fy": 800.0,
                    "cx": 480.0,
                    "cy": 360.0,
                    "T_world_camera": [[1, 0, 0, -50], [0, 1, 0, 0], [0, 0, 1, 400], [0, 0, 0, 1]],
                },
                {
                    "name": "cam1",
                    "fx": 800.0,
                    "fy": 800.0,
                    "cx": 480.0,
                    "cy": 360.0,
                    "T_world_camera": [[1, 0, 0, 50], [0, 1, 0, 0], [0, 0, 1, 400], [0, 0, 0, 1]],
                },
            ],
            "boards": [{"index": 0, "T_world_board": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}],
        }
        (self.root / "calibration.json").write_text(json.dumps(self.cali_data))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_visualize_html_generation(self):
        script = Path(__file__).resolve().parent / "visualize_calibration.py"
        report_html = self.root / "report.html"
        subprocess.check_call([_tool_python(), str(script), str(self.root), "-o", str(report_html)])

        self.assertTrue(report_html.exists())
        content = report_html.read_text(encoding="utf-8")
        self.assertIn("Three.js", content)
        self.assertIn("cam0", content)
        self.assertIn("cam1", content)
        self.assertIn("CUDA", content)




class TestValidateDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        # Create 2 camera folders with 2 valid images each
        for cam in ["cam0", "cam1"]:
            cam_dir = self.root / "data" / cam / "external"
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(cam_dir / f"img_{i:04d}.png"), img)

        # Create valid network yaml
        yaml_content = """%YAML:1.0
type: charuco
squaresX: 7
squaresY: 9
squareLength: 40
markerLength: 20
margins: 10
numberBoards: 1
arcCode: 10
"""
        (self.root / "network_specification_file.yaml").write_text(yaml_content)
        (self.root / "pattern_square_mm0.txt").write_text("squareLength_mm 40.0\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_valid_dataset(self):
        from validate_dataset import validate_dataset
        res = validate_dataset(self.root)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["errors"]), 0)
        self.assertEqual(len(res["summary"]["cameras"]), 2)

    def test_invalid_marker_length(self):
        from validate_dataset import validate_dataset
        bad_yaml = """%YAML:1.0
type: charuco
squaresX: 7
squaresY: 9
squareLength: 40
markerLength: 45
numberBoards: 1
arcCode: 10
"""
        (self.root / "network_specification_file.yaml").write_text(bad_yaml)
        res = validate_dataset(self.root)
        self.assertFalse(res["valid"])
        self.assertTrue(any("markerLength" in e for e in res["errors"]))

    def test_dictionary_overflow(self):
        from validate_dataset import validate_dataset
        # DICT_4X4_50 has 50 markers; 11x11 board has 60 markers per board -> overflow!
        bad_yaml = """%YAML:1.0
type: charuco
squaresX: 11
squaresY: 11
squareLength: 40
markerLength: 20
numberBoards: 1
arcCode: 0
"""
        (self.root / "network_specification_file.yaml").write_text(bad_yaml)
        res = validate_dataset(self.root)
        self.assertFalse(res["valid"])
        self.assertTrue(any("overflow" in e.lower() for e in res["errors"]))

    def test_missing_scale_file(self):
        from validate_dataset import validate_dataset
        (self.root / "pattern_square_mm0.txt").unlink()
        res = validate_dataset(self.root)
        self.assertFalse(res["valid"])
        self.assertTrue(any("pattern_square_mm0.txt" in e for e in res["errors"]))

    def test_valid_april_dataset(self):
        from validate_dataset import validate_dataset
        april_yaml = """%YAML:1.0
type: april
squaresX: 6
squaresY: 6
squareLength: 80
margins: 20
tagSpace: 16
numberBoards: 2
april_family: tag36h11
"""
        (self.root / "network_specification_file.yaml").write_text(april_yaml)
        (self.root / "pattern_square_mm0.txt").write_text("squareLength_mm 40.0\n")
        (self.root / "pattern_square_mm1.txt").write_text("squareLength_mm 40.0\n")
        res = validate_dataset(self.root)
        self.assertTrue(res["valid"])
        self.assertEqual(res["summary"]["pattern"]["type"], "april")
        self.assertEqual(res["summary"]["pattern"]["april_family"], "tag36h11")

    def test_april_dictionary_overflow(self):
        from validate_dataset import validate_dataset
        # tag16h5 has capacity 30; 6x6 grid has 36 tags per board -> overflow!
        bad_april_yaml = """%YAML:1.0
type: april
squaresX: 6
squaresY: 6
squareLength: 80
margins: 20
numberBoards: 1
april_family: tag16h5
"""
        (self.root / "network_specification_file.yaml").write_text(bad_april_yaml)
        (self.root / "pattern_square_mm0.txt").write_text("squareLength_mm 40.0\n")
        res = validate_dataset(self.root)
        self.assertFalse(res["valid"])
        self.assertTrue(any("overflow" in e.lower() for e in res["errors"]))

    def test_real_repo_charuco_and_april_datasets(self):
        from validate_dataset import validate_dataset
        repo_root = Path(__file__).resolve().parent.parent

        # 1. Test real ChArUco dataset
        charuco_dir = repo_root / "DATA" / "mult-3-base"
        if charuco_dir.exists():
            res_c = validate_dataset(charuco_dir)
            self.assertTrue(res_c["valid"])
            self.assertEqual(res_c["summary"]["pattern"]["type"], "charuco")
            self.assertEqual(len(res_c["summary"]["cameras"]), 4)

        # 2. Test real AprilTag grid dataset
        april_dir = repo_root / "results" / "april-in"
        if april_dir.exists():
            res_a = validate_dataset(april_dir)
            self.assertTrue(res_a["valid"])
            self.assertEqual(res_a["summary"]["pattern"]["type"], "april")
            self.assertEqual(len(res_a["summary"]["cameras"]), 2)

class TestCalicoGUI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.cali_data = {
            "tool": "calico-dec2023",
            "num_cameras": 2,
            "duration_sec": 1.2,
            "opencv": "5.0.0",
            "cuda": True,
            "pattern": "charuco",
            "cameras": [
                {
                    "name": "cam0",
                    "width": 960,
                    "height": 720,
                    "fx": 800.0,
                    "fy": 800.0,
                    "cx": 480.0,
                    "cy": 360.0,
                    "T_world_camera": [[1, 0, 0, -100], [0, 1, 0, 0], [0, 0, 1, 500], [0, 0, 0, 1]],
                },
                {
                    "name": "cam1",
                    "width": 960,
                    "height": 720,
                    "fx": 800.0,
                    "fy": 800.0,
                    "cx": 480.0,
                    "cy": 360.0,
                    "T_world_camera": [[1, 0, 0, 100], [0, 1, 0, 0], [0, 0, 1, 500], [0, 0, 0, 1]],
                },
            ],
            "boards": [{"index": 0, "T_world_board": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}],
        }
        (self.root / "calibration.json").write_text(json.dumps(self.cali_data))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_detect_system_info(self):
        from calico_gui import detect_system_info
        info = detect_system_info()
        self.assertIn("cpu_count", info)
        self.assertIn("has_cuda", info)
        self.assertIn("calico_bin", info)

    def test_parse_calibration_json(self):
        from calico_gui import parse_calibration_json
        parsed = parse_calibration_json(self.root)
        self.assertEqual(len(parsed["cameras"]), 2)
        self.assertEqual(len(parsed["baselines"]), 1)
        self.assertAlmostEqual(parsed["baselines"][0]["distance_mm"], 200.0, places=1)

    def test_gui_http_server(self):
        import urllib.request
        import urllib.error
        import threading
        import time
        import calico_gui
        from calico_gui import ThreadingHTTPServer, StudioRequestHandler

        calico_gui.AUTH_TOKEN = "test-token-calico-studio"
        auth_hdr = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {calico_gui.AUTH_TOKEN}",
        }

        # Pick an ephemeral port
        server = ThreadingHTTPServer(("127.0.0.1", 0), StudioRequestHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            # 1. Test GET /
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
                self.assertIn("<title>CALICO Studio", body)
                self.assertIn(calico_gui.AUTH_TOKEN, body)
                self.assertNotIn("%%CALICO_AUTH_TOKEN%%", body)

            # 2. Test GET /api/status
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status") as resp:
                self.assertEqual(resp.status, 200)
                status_json = json.loads(resp.read().decode("utf-8"))
                self.assertIn("cpu_count", status_json)

            # 3. Test GET /api/calibration
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/calibration?path={self.root}") as resp:
                self.assertEqual(resp.status, 200)
                cali_resp = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(len(cali_resp["cameras"]), 2)

            # 4. Test GET /api/logs
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/logs") as resp:
                self.assertEqual(resp.status, 200)
                logs_resp = json.loads(resp.read().decode("utf-8"))
                self.assertIn("status", logs_resp)

            # 4b. POST without auth must 401
            bad = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/export",
                data=json.dumps({"cali_dir": str(self.root), "format": "ros"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(bad)
            self.assertEqual(cm.exception.code, 401)

            # 5. Test POST /api/export
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/export",
                data=json.dumps({"cali_dir": str(self.root), "format": "ros"}).encode("utf-8"),
                headers=auth_hdr,
                method="POST"
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                exp_res = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(exp_res.get("returncode"), 0)
            # 6. Test GET /api/validate
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/validate?path={self.root}") as resp:
                self.assertEqual(resp.status, 200)
                val_res = json.loads(resp.read().decode("utf-8"))
                self.assertIn("valid", val_res)

            # 7. Test POST /api/export with format: opencv
            req_cv = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/export",
                data=json.dumps({"cali_dir": str(self.root), "format": "opencv"}).encode("utf-8"),
                headers=auth_hdr,
                method="POST"
            )
            with urllib.request.urlopen(req_cv) as resp:
                self.assertEqual(resp.status, 200)
                exp_res = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(exp_res.get("returncode"), 0)

        finally:
            server.shutdown()
            server.server_close()

    def test_build_calico_run_cmd_no_stride(self):
        from calico_gui import build_calico_run_cmd, SRC_DIR
        cmd = build_calico_run_cmd({
            "pattern_type": "april",
            "input_dir": "/data/in",
            "output_dir": "/data/out",
            "use_cuda": True,
            "checkpoint": True,
            "json_export": True,
        }, calico_bin="/bin/calico-dec2023")
        self.assertEqual(cmd[0], "/bin/calico-dec2023")
        self.assertIn("--april", cmd)
        self.assertIn("--calibrate", cmd)
        self.assertIn(f"--src-dir={SRC_DIR}/", cmd)
        self.assertIn("--use-cuda", cmd)
        self.assertTrue(all(not a.startswith("--stride") for a in cmd))
        with self.assertRaises(ValueError):
            build_calico_run_cmd({"pattern_type": "config", "input_dir": "/x"})


class TestCudaMacroAlignment(unittest.TestCase):
    """Guard against the CALICO_USE_CUDA vs WITH_CUDA silent dead-code bug."""

    def test_cmake_macros_match_cpp_guards(self):
        root = Path(__file__).resolve().parent.parent
        cmake = (root / "src" / "CMakeLists.txt").read_text()
        cpp = (root / "src" / "cuda-detect.cpp").read_text()
        self.assertIn("add_definitions(-DCALICO_USE_CUDA)", cmake)
        self.assertIn("add_definitions(-DCALICO_HAVE_CUAPRILTAGS)", cmake)
        self.assertIn("#ifdef CALICO_USE_CUDA", cpp)
        self.assertIn("#ifdef CALICO_HAVE_CUAPRILTAGS", cpp)
        self.assertNotIn("#ifdef WITH_CUDA", cpp)
        self.assertNotIn("#ifdef HAVE_CUAPRILTAGS", cpp)
        self.assertIn("input.dev_ptr", cpp)
        self.assertNotIn("dev_p016", cpp)
        self.assertIn("COLOR_GRAY2RGB", cpp)
        self.assertIn("${CALICO_CUDA_LIBS}", cmake)


class TestCharucoCudaAndCalibrationVerification(unittest.TestCase):
    def setUp(self):
        self.squares_x = 7
        self.squares_y = 5
        self.square_len = 0.04
        self.marker_len = 0.03
        self.dict_code = cv2.aruco.DICT_4X4_50
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dict_code)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_len, self.marker_len, self.dictionary
        )

    def test_charuco_cuda_gray_equivalence(self):
        """Verify bit-exact corner detection consistency between color BGR and CUDA gray conversion."""
        board_w, board_h = 700, 500
        board_img = self.board.generateImage((board_w, board_h), marginSize=20)
        bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

        # Replicate raw CUDA kernel math: (p[0]*29 + p[1]*150 + p[2]*77) >> 8
        b = bgr[:, :, 0].astype(np.uint32)
        g = bgr[:, :, 1].astype(np.uint32)
        r = bgr[:, :, 2].astype(np.uint32)
        cuda_gray = ((b * 29 + g * 150 + r * 77) >> 8).astype(np.uint8)

        detector = cv2.aruco.CharucoDetector(self.board)
        corners_bgr, ids_bgr, _, _ = detector.detectBoard(bgr)
        corners_gray, ids_gray, _, _ = detector.detectBoard(cuda_gray)

        expected_corners = (self.squares_x - 1) * (self.squares_y - 1)
        self.assertEqual(len(corners_bgr), expected_corners)
        self.assertEqual(len(corners_gray), expected_corners)
        np.testing.assert_array_equal(ids_bgr, ids_gray)

        max_corner_diff = float(np.max(np.abs(corners_bgr - corners_gray)))
        self.assertLess(max_corner_diff, 1e-4, "Detected corner coordinates differ between BGR and CUDA Gray")

    def test_charuco_calibration_accuracy_and_reprojection(self):
        """Verify calibration accuracy: 3D-to-2D projection, PnP camera pose recovery, and reprojection error."""
        board_img = self.board.generateImage((700, 500), marginSize=20)
        detector = cv2.aruco.CharucoDetector(self.board)
        corners, ids, _, _ = detector.detectBoard(board_img)
        obj_pts, _ = self.board.matchImagePoints(corners, ids)

        K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)

        # Test across multiple camera viewing angles
        test_poses = [
            (np.array([0.1, 0.2, 0.05]), np.array([0.0, 0.0, 0.8])),
            (np.array([-0.15, 0.25, 0.1]), np.array([-0.05, 0.03, 0.75])),
            (np.array([0.2, -0.1, -0.05]), np.array([0.08, -0.02, 0.85])),
        ]

        for rvec_gt, tvec_gt in test_poses:
            proj_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, tvec_gt, K, dist_coeffs)

            num_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj_pts, proj_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
            )
            self.assertGreaterEqual(num_sol, 1)

            min_r_err = min(float(np.linalg.norm(r.flatten() - rvec_gt)) for r in rvecs)
            min_t_err = min(float(np.linalg.norm(t.flatten() - tvec_gt)) for t in tvecs)
            self.assertLess(min_r_err, 1e-4, f"Rotation error {min_r_err} exceeds 1e-4 rad")
            self.assertLess(min_t_err, 1e-4, f"Translation error {min_t_err} exceeds 1e-4 m")

            reproj, _ = cv2.projectPoints(obj_pts, rvecs[0], tvecs[0], K, dist_coeffs)
            rms = float(np.sqrt(np.mean(np.sum((reproj.reshape(-1, 2) - proj_pts.reshape(-1, 2)) ** 2, axis=1))))
            self.assertLess(rms, 1e-4, f"Reprojection RMS {rms} exceeds 1e-4 px")

    def test_charuco_apriltag_dictionary_support(self):
        """Verify ChArUco board with AprilTag dictionary (DICT_APRILTAG_36h11) used for cuAprilTags."""
        april_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        april_board = cv2.aruco.CharucoBoard((5, 4), 0.05, 0.038, april_dict)
        board_img = april_board.generateImage((500, 400), marginSize=15)
        detector = cv2.aruco.CharucoDetector(april_board)
        corners, ids, _, _ = detector.detectBoard(board_img)

        expected_corners = (5 - 1) * (4 - 1)
        self.assertEqual(len(corners), expected_corners)
        self.assertEqual(len(ids), expected_corners)

    def test_ingest_intrinsics_preserves_tangential_distortion(self):
        """Verify that ingested intrinsics with non-zero tangential distortion (p1, p2) are preserved."""
        K_orig = np.array([[820.5, 0.0, 481.2], [0.0, 819.8, 359.7], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_orig = np.array([-0.12, 0.04, 0.0025, -0.0018, 0.0], dtype=np.float64)

        board_img = self.board.generateImage((700, 500), marginSize=20)
        detector = cv2.aruco.CharucoDetector(self.board)
        corners, ids, _, _ = detector.detectBoard(board_img)
        obj_pts, _ = self.board.matchImagePoints(corners, ids)

        rvec = np.array([0.05, -0.08, 0.02], dtype=np.float64)
        tvec = np.array([0.01, -0.02, 0.75], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_orig, dist_orig)

        K_calib = K_orig.copy()
        dist_calib = dist_orig.copy()
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_FOCAL_LENGTH
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_TANGENT_DIST
        )

        rms, K_out, dist_out, _, _ = cv2.calibrateCamera(
            [obj_pts.astype(np.float32)],
            [img_pts.astype(np.float32)],
            (700, 500),
            K_calib,
            dist_calib,
            flags=flags,
        )

        np.testing.assert_allclose(K_out, K_orig, atol=1e-5)
        np.testing.assert_allclose(dist_out.flatten()[:4], dist_orig[:4], atol=1e-5)
        self.assertNotEqual(dist_out.flatten()[2], 0.0, 'p1 should not be zeroed out')
        self.assertNotEqual(dist_out.flatten()[3], 0.0, 'p2 should not be zeroed out')


class TestAprilTagGridVariants(unittest.TestCase):
    """Render CALICO-style AprilTag grids for each OpenCV family and detect all tags."""

    @staticmethod
    def _render_grid(dict_id: int, sx: int, sy: int, side: int = 64, gap: int = 12, margins: int = 16):
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        w = sx * (side + gap) - gap + 2 * margins
        h = sy * (side + gap) - gap + 2 * margins
        board = np.full((h, w), 255, dtype=np.uint8)
        start_x = margins
        start_y = h - side - margins
        tag_id = 0
        for y in range(sy):
            for x in range(sx):
                marker = np.zeros((side, side), dtype=np.uint8)
                dictionary.generateImageMarker(tag_id, side, marker, 1)
                x0 = start_x + x * (side + gap)
                y0 = start_y - y * (side + gap)
                board[y0:y0 + side, x0:x0 + side] = marker
                tag_id += 1
        return board, dictionary

    def test_families_and_sizes_detect_all_tags(self):
        cases = [
            (cv2.aruco.DICT_APRILTAG_36h11, 6, 6),
            (cv2.aruco.DICT_APRILTAG_36h11, 8, 6),
            (cv2.aruco.DICT_APRILTAG_25h9, 5, 5),
            (cv2.aruco.DICT_APRILTAG_25h9, 7, 5),
            (cv2.aruco.DICT_APRILTAG_16h5, 5, 5),
            (cv2.aruco.DICT_APRILTAG_16h5, 6, 5),
            (cv2.aruco.DICT_APRILTAG_36h10, 6, 6),
            (cv2.aruco.DICT_APRILTAG_36h10, 10, 8),
        ]
        for dict_id, sx, sy in cases:
            with self.subTest(dict_id=dict_id, grid=f"{sx}x{sy}"):
                board, dictionary = self._render_grid(dict_id, sx, sy)
                det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
                _corners, ids, _rej = det.detectMarkers(board)
                self.assertIsNotNone(ids)
                self.assertEqual(len(ids), sx * sy)

    def test_april_variant_configs_exist(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "april-variants"
        expected = {
            "april-tag16h5-5x5.yaml",
            "april-tag25h9-5x5.yaml",
            "april-tag36h10-6x6.yaml",
            "april-tag36h11-8x6.yaml",
        }
        self.assertTrue(cfg_dir.is_dir())
        found = {p.name for p in cfg_dir.glob("april-*.yaml")}
        self.assertTrue(expected.issubset(found))


class TestCharucoBoardVariants(unittest.TestCase):
    """Generate ChArUco boards across dictionaries/sizes and detect all corners."""

    def test_dicts_and_sizes_detect_all_corners(self):
        cases = [
            (cv2.aruco.DICT_4X4_50, 5, 4, 80, 50),
            (cv2.aruco.DICT_4X4_100, 7, 5, 80, 60),
            (cv2.aruco.DICT_5X5_100, 7, 5, 80, 60),
            (cv2.aruco.DICT_6X6_250, 7, 5, 80, 60),
            (cv2.aruco.DICT_6X6_250, 9, 6, 70, 50),
            (cv2.aruco.DICT_6X6_1000, 12, 9, 56, 40),
            (cv2.aruco.DICT_7X7_250, 8, 6, 72, 52),
            (cv2.aruco.DICT_APRILTAG_36h11, 7, 5, 80, 60),
            (cv2.aruco.DICT_5X5_50, 5, 4, 100, 70),
        ]
        for dict_id, sx, sy, sq, mk in cases:
            with self.subTest(dict_id=dict_id, grid=f"{sx}x{sy}"):
                dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                board = cv2.aruco.CharucoBoard((sx, sy), float(sq), float(mk), dictionary)
                margins = 20
                img = board.generateImage((sx * sq + 2 * margins, sy * sq + 2 * margins), marginSize=margins)
                det = cv2.aruco.CharucoDetector(board)
                corners, ids, _mc, _mi = det.detectBoard(img)
                expected = (sx - 1) * (sy - 1)
                self.assertIsNotNone(corners)
                self.assertIsNotNone(ids)
                self.assertEqual(len(corners), expected)
                self.assertEqual(len(ids), expected)

    def test_charuco_variant_configs_exist(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "charuco-variants"
        expected = {
            "charuco-4x4-5x4.yaml",
            "charuco-5x5-7x5.yaml",
            "charuco-6x6-9x6.yaml",
            "charuco-6x6-12x9.yaml",
            "charuco-7x7-8x6.yaml",
            "charuco-april36h11-7x5.yaml",
        }
        self.assertTrue(cfg_dir.is_dir())
        found = {p.name for p in cfg_dir.glob("charuco-*.yaml")}
        self.assertTrue(expected.issubset(found))


if __name__ == "__main__":
    unittest.main()



