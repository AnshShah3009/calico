#!/usr/bin/env python3
"""Unit tests for CALICO Python tooling: export_calibration, ingest_video, visualize_calibration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


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
            [sys.executable, str(script), str(self.root), "--format", "all", "-o", str(out_dir)]
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
                sys.executable,
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
        subprocess.check_call([sys.executable, str(script), str(self.root), "-o", str(report_html)])

        self.assertTrue(report_html.exists())
        content = report_html.read_text(encoding="utf-8")
        self.assertIn("Three.js", content)
        self.assertIn("cam0", content)
        self.assertIn("cam1", content)
        self.assertIn("CUDA", content)


if __name__ == "__main__":
    unittest.main()
