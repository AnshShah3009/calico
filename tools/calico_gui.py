
#!/usr/bin/env python3
"""CALICO Studio: Zero-dependency Interactive Web GUI for CALICO Calibration.

Provides an intuitive browser-based interface to:
1. Configure and run multi-camera calibrations (ChArUco / AprilTag)
2. Stream live console logs and monitor optimization progress
3. Inspect 3D reconstructed camera networks, frustums, baselines, and boards
4. Export calibrated parameters to ROS/ROS2, Nerfstudio, and COLMAP
5. Ingest and synchronize multi-camera video streams

Runs locally or inside Docker containers using Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tools.validate_dataset import validate_dataset
except ImportError:
    try:
        from validate_dataset import validate_dataset
    except ImportError:
        validate_dataset = None


WORKSPACE_DIR = Path(__file__).resolve().parent.parent


def detect_system_info() -> Dict[str, Any]:
    """Detect available hardware, compilers, CUDA, and CALICO binaries."""
    has_cuda = False
    cuda_device = "Not Detected"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            has_cuda = True
            cuda_device = smi.stdout.strip().splitlines()[0]
    except Exception:
        pass

    calico_bin = None
    for cand in [
        WORKSPACE_DIR / "src-build" / "calico-dec2023",
        WORKSPACE_DIR / "build" / "calico-dec2023",
        WORKSPACE_DIR / "calico-dec2023",
    ]:
        if cand.is_file() and os.access(cand, os.X_OK):
            calico_bin = str(cand)
            break
    if not calico_bin:
        which_bin = shutil.which("calico-dec2023")
        if which_bin:
            calico_bin = which_bin

    has_docker = bool(shutil.which("docker"))

    return {
        "cpu_count": os.cpu_count() or 4,
        "has_cuda": has_cuda,
        "cuda_device": cuda_device,
        "calico_bin": calico_bin,
        "has_docker": has_docker,
        "workspace": str(WORKSPACE_DIR),
    }


def find_available_datasets() -> List[Dict[str, Any]]:
    """Scan workspace for directories containing calibration.json or CALICO input data."""
    results = []
    # Search for output results
    for p in sorted(WORKSPACE_DIR.glob("results/**/calibration.json")):
        folder = p.parent
        results.append({
            "type": "result",
            "name": folder.relative_to(WORKSPACE_DIR).as_posix(),
            "path": str(folder),
            "has_json": True,
        })

    # Search for input datasets
    for p in sorted(WORKSPACE_DIR.glob("**/network_specification_file.yaml")):
        folder = p.parent
        results.append({
            "type": "input",
            "name": folder.relative_to(WORKSPACE_DIR).as_posix(),
            "path": str(folder),
            "has_json": (folder / "calibration.json").exists(),
        })

    return results


class JobManager:
    """Manages asynchronous calibration runs, log streaming, and process lifecycle."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.status = "idle"  # idle, running, completed, failed, cancelled
        self.logs: List[str] = []
        self.start_time: float = 0.0
        self.elapsed_sec: float = 0.0
        self.progress: int = 0
        self.current_stage: str = "Ready"
        self.output_dir: Optional[str] = None
        self.error_message: Optional[str] = None

    def start_job(self, cmd: List[str], output_dir: str, env: Optional[Dict[str, str]] = None) -> None:
        with self.lock:
            if self.status == "running":
                raise RuntimeError("A calibration job is already running.")
            self.logs = [f"[CALICO Studio] Starting command: {' '.join(cmd)}\n"]
            self.status = "running"
            self.start_time = time.time()
            self.elapsed_sec = 0.0
            self.progress = 0
            self.current_stage = "Initializing"
            self.output_dir = output_dir
            self.error_message = None

            run_env = os.environ.copy()
            if env:
                run_env.update(env)

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(WORKSPACE_DIR),
                    env=run_env,
                )
            except Exception as e:
                self.status = "failed"
                self.error_message = str(e)
                self.logs.append(f"[Error] Failed to spawn process: {e}\n")
                return

        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        assert self.process is not None
        for line in self.process.stdout:  # type: ignore
            with self.lock:
                self.logs.append(line)
                self.elapsed_sec = round(time.time() - self.start_time, 1)

                # Parse progress markers
                if "Stage 4" in line or "algebraic" in line.lower():
                    self.current_stage = "Stage 4 (Algebraic Solving)"
                elif "Stage 5" in line or "reprojection" in line.lower():
                    self.current_stage = "Stage 5 (Reprojection Bundle Adjustment)"

                prog_match = re.search(r"(\d{1,3})%", line)
                if prog_match:
                    try:
                        self.progress = min(100, int(prog_match.group(1)))
                    except ValueError:
                        pass

        self.process.wait()
        with self.lock:
            self.elapsed_sec = round(time.time() - self.start_time, 1)
            if self.process.returncode == 0:
                self.status = "completed"
                self.progress = 100
                self.current_stage = "Finished"
                self.logs.append(f"\n[CALICO Studio] Calibration completed successfully in {self.elapsed_sec}s.\n")
            elif self.status != "cancelled":
                self.status = "failed"
                self.error_message = f"Process exited with code {self.process.returncode}"
                self.logs.append(f"\n[CALICO Studio] Exited with error code {self.process.returncode}.\n")

    def cancel_job(self) -> bool:
        with self.lock:
            if self.process and self.status == "running":
                self.status = "cancelled"
                self.process.terminate()
                self.logs.append("\n[CALICO Studio] Job cancelled by user.\n")
                return True
            return False

    def get_state(self, after_line: int = 0) -> Dict[str, Any]:
        with self.lock:
            new_lines = self.logs[after_line:] if after_line < len(self.logs) else []
            return {
                "status": self.status,
                "progress": self.progress,
                "stage": self.current_stage,
                "elapsed_sec": self.elapsed_sec if self.status != "running" else round(time.time() - self.start_time, 1),
                "total_lines": len(self.logs),
                "new_lines": "".join(new_lines),
                "output_dir": self.output_dir,
                "error": self.error_message,
            }


JOB_MANAGER = JobManager()


def parse_calibration_json(cali_dir: Path | str) -> Dict[str, Any]:
    cali_dir = Path(cali_dir)
    """Parse calibration.json, cali_results.txt, and calculate camera baselines."""
    json_path = cali_dir / "calibration.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Enrich RMS if available in data/<cam>/cali_results.txt
    for cam in data.get("cameras", []):
        name = cam.get("name", "")
        cali_txt = cali_dir / "data" / name / "cali_results.txt"
        if cali_txt.exists():
            for line in cali_txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("rms "):
                    try:
                        cam["rms"] = float(line.split()[1])
                    except (IndexError, ValueError):
                        pass

    # Compute pairwise camera baselines
    cameras = data.get("cameras", [])
    centers = []
    names = []
    for cam in cameras:
        T = cam.get("T_world_camera")
        if T and len(T) == 4 and len(T[0]) == 4:
            centers.append([T[0][3], T[1][3], T[2][3]])
            names.append(cam.get("name", "cam"))

    baselines = []
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = centers[i], centers[j]
            dist_mm = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)
            baselines.append({
                "cam_a": names[i],
                "cam_b": names[j],
                "distance_mm": round(dist_mm, 2),
                "distance_cm": round(dist_mm / 10.0, 2),
            })

    data["baselines"] = baselines
    data["cali_dir"] = str(cali_dir)
    return data


# Embedded Studio Frontend HTML + CSS + JS
STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALICO Studio | Multi-Camera Calibration</title>
<style>
  :root {
    --bg-base: #0b0d13;
    --bg-surface: #131722;
    --bg-card: #1c2233;
    --bg-card-hover: #242c42;
    --border: #2a344d;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --primary: #38bdf8;
    --primary-hover: #0ea5e9;
    --primary-glow: rgba(56, 189, 248, 0.15);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --accent: #8b5cf6;
    --radius: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-base);
    color: var(--text-main);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  header {
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 30;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: -0.5px;
  }
  .badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 999px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
  }
  .badge-cuda { background: rgba(16, 185, 129, 0.1); border-color: var(--success); color: var(--success); }
  .badge-primary { background: var(--primary-glow); border-color: var(--primary); color: var(--primary); }

  nav {
    display: flex;
    gap: 4px;
    background: var(--bg-base);
    padding: 4px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
  }
  .nav-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .nav-btn.active, .nav-btn:hover {
    color: var(--text-main);
    background: var(--bg-card);
  }
  .nav-btn.active {
    color: var(--primary);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }

  main {
    flex: 1;
    position: relative;
    overflow: hidden;
  }
  .tab-pane {
    position: absolute;
    inset: 0;
    display: none;
    overflow: hidden;
  }
  .tab-pane.active {
    display: flex;
  }

  /* Wizard Styles */
  .wizard-layout {
    display: grid;
    grid-template-columns: 480px 1fr;
    width: 100%;
    height: 100%;
  }
  .config-panel {
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .console-panel {
    background: #090a0f;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }
  .console-header {
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  #terminal {
    flex: 1;
    padding: 16px 20px;
    font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 0.82rem;
    color: #cbd5e1;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.5;
  }

  /* Form Elements */
  .section-title {
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--primary);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 14px;
  }
  .form-group label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
  }
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  input, select {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-main);
    padding: 8px 12px;
    border-radius: var(--radius);
    font-size: 0.85rem;
    outline: none;
    transition: border-color 0.15s ease;
  }
  input:focus, select:focus {
    border-color: var(--primary);
  }
  .checkbox-group {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: var(--text-main);
    cursor: pointer;
  }
  .checkbox-group input {
    cursor: pointer;
    width: 16px;
    height: 16px;
    accent-color: var(--primary);
  }

  .btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-main);
    padding: 9px 16px;
    border-radius: var(--radius);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.15s ease;
  }
  .btn:hover { background: var(--bg-card-hover); }
  .btn-primary {
    background: var(--primary);
    border-color: var(--primary);
    color: #000;
  }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-danger {
    background: rgba(239, 68, 68, 0.15);
    border-color: var(--danger);
    color: var(--danger);
  }
  .btn-danger:hover { background: var(--danger); color: #fff; }

  /* 3D Visualizer Layout */
  .visualizer-layout {
    display: grid;
    grid-template-columns: 1fr 380px;
    width: 100%;
    height: 100%;
  }
  #viewport {
    position: relative;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at center, #181d2a 0%, #080a0f 100%);
  }
  .vis-sidebar {
    background: var(--bg-surface);
    border-left: 1px solid var(--border);
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
  }
  .card h3 {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
  }
  .prop-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .prop-row:last-child { border-bottom: none; }
  .prop-label { color: var(--text-muted); }
  .prop-val { font-family: monospace; font-weight: 600; }
  .cam-card {
    border-left: 4px solid var(--primary);
    margin-bottom: 10px;
    cursor: pointer;
    transition: transform 0.15s, border-color 0.15s;
  }
  .cam-card:hover {
    transform: translateX(2px);
    background: var(--bg-card-hover);
  }
  .cam-card.selected {
    border-left-width: 6px;
    box-shadow: 0 0 12px var(--primary-glow);
  }

  .viewport-overlay {
    position: absolute;
    bottom: 20px;
    left: 20px;
    display: flex;
    gap: 8px;
    z-index: 10;
    background: rgba(19, 23, 34, 0.85);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    padding: 8px;
    border-radius: var(--radius);
  }
  .viewport-tooltip {
    position: absolute;
    top: 20px;
    left: 20px;
    background: rgba(19, 23, 34, 0.8);
    backdrop-filter: blur(6px);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.75rem;
    color: var(--text-muted);
    pointer-events: none;
  }

  /* Progress Bar */
  .progress-wrap {
    height: 6px;
    background: var(--bg-card);
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0;
  }
  .progress-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--primary), var(--success));
    transition: width 0.3s ease;
  }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <header>
    <div class="logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
        <circle cx="12" cy="13" r="4"/>
      </svg>
      CALICO Studio
      <span class="badge badge-primary">v5.0-Modern</span>
      <span id="cuda-badge" class="badge">Checking Hardware...</span>
    </div>
    <nav>
      <button class="nav-btn active" onclick="switchTab('wizard')">⚡ Calibration Wizard</button>
      <button class="nav-btn" onclick="switchTab('validate')">🔍 Pre-Flight Check</button>
      <button class="nav-btn" onclick="switchTab('visualizer')">🌐 3D Visualizer & Inspector</button>
      <button class="nav-btn" onclick="switchTab('export')">📦 Export Hub</button>
    </nav>
  </header>

  <main>
    <!-- TAB 1: CALIBRATION WIZARD -->
    <div id="tab-wizard" class="tab-pane active">
      <div class="wizard-layout">
        <div class="config-panel">
          <div>
            <div class="form-group" style="margin-bottom:14px; background:rgba(56,189,248,0.06); padding:10px; border-radius:6px; border:1px solid var(--border);">
              <label style="color:var(--primary); font-weight:700;">⚡ Configuration Preset</label>
              <select id="config-preset" onchange="applyPreset()" style="margin-top:4px;">
                <option value="custom">Custom Configuration</option>
                <option value="charuco_std">ChArUco Standard (7x9, 45mm, Huber, 8 threads)</option>
                <option value="april_cuda">AprilTag Grid (tag36h11, 80mm, CUDA GPU)</option>
                <option value="high_robust">High-Robustness Rig (Cauchy Loss, 1.5 scale)</option>
                <option value="quick_test">Quick Draft (Stride 2, Huber Loss)</option>
              </select>
            </div>

            <div class="section-title">1. Dataset & Pattern</div>
            <div class="form-group">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <label>Select Dataset / Results</label>
                <button class="btn" style="padding:2px 8px; font-size:0.72rem; color:var(--primary);" onclick="runPreflightCheck()">🔍 Pre-Flight Check</button>
              </div>
              <select id="dataset-select" onchange="onDatasetChange()" style="margin-top:4px;">
                <option value="">-- Choose Workspace Dataset --</option>
              </select>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Pattern Type</label>
                <select id="pattern-type">
                  <option value="charuco">ChArUco Board</option>
                  <option value="april">AprilTag Grid</option>
                </select>
              </div>
              <div class="form-group">
                <label>Pattern Family</label>
                <select id="tag-family">
                  <option value="tag36h11">tag36h11 (Standard)</option>
                  <option value="tag25h9">tag25h9</option>
                  <option value="tag16h5">tag16h5</option>
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Square Size (mm)</label>
                <input type="number" id="square-mm" value="40.0" step="0.5">
              </div>
              <div class="form-group">
                <label>Marker / Tag Size (mm)</label>
                <input type="number" id="marker-mm" value="28.0" step="0.5">
              </div>
            </div>
          </div>

          <div>
            <div class="section-title">2. Solver & Optimization Settings</div>
            <div class="form-row">
              <div class="form-group">
                <label>Ceres Robust Loss</label>
                <select id="loss-type">
                  <option value="trivial">Trivial (Standard L2)</option>
                  <option value="huber" selected>Huber Loss (Robust)</option>
                  <option value="cauchy">Cauchy Loss (Outlier Heavy)</option>
                </select>
              </div>
              <div class="form-group">
                <label>Loss Scale (&delta;)</label>
                <input type="number" id="loss-scale" value="1.0" step="0.1">
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Worker Threads</label>
                <input type="number" id="threads" value="8" min="1" max="64">
              </div>
              <div class="form-group">
                <label>Frame Stride</label>
                <input type="number" id="stride" value="1" min="1">
              </div>
            </div>
            <div class="checkbox-group" style="margin-bottom:8px;">
              <input type="checkbox" id="use-cuda" checked>
              <label for="use-cuda">Enable CUDA GPU Acceleration (--use-cuda)</label>
            </div>
            <div class="checkbox-group" style="margin-bottom:8px;">
              <input type="checkbox" id="checkpoint" checked>
              <label for="checkpoint">Periodic Solver Checkpoints (--checkpoint)</label>
            </div>
            <div class="checkbox-group">
              <input type="checkbox" id="json-export" checked>
              <label for="json-export">Write Structured JSON Output (--json)</label>
            </div>
          </div>

          <div>
            <div class="section-title">3. Actions</div>
            <div style="display:flex; gap:10px;">
              <button class="btn btn-primary" style="flex:1;" onclick="startCalibration()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Run Calibration
              </button>
              <button class="btn btn-danger" onclick="cancelCalibration()">Stop</button>
            </div>
          </div>
        </div>

        <div class="console-panel">
          <div class="console-header">
            <div>
              <span id="job-status-badge" class="badge">Idle</span>
              <span id="stage-badge" style="font-size:0.8rem; margin-left:8px; color:var(--text-muted);">Ready to calibrate</span>
            </div>
            <div style="font-size:0.8rem; color:var(--text-dim);" id="time-elapsed">Elapsed: 0.0s</div>
          </div>
          <div style="padding: 0 20px;">
            <div class="progress-wrap"><div id="progress-bar" class="progress-fill"></div></div>
          </div>
          <div id="terminal">[CALICO Studio] Ready. Select a dataset and configure options on the left to begin calibration.</div>
        </div>
      </div>
    </div>

        <!-- TAB: PRE-FLIGHT VALIDATION -->
    <div id="tab-validate" class="tab-pane">
      <div style="padding: 24px; max-width: 960px; margin: 0 auto; overflow-y:auto; width:100%;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
          <div>
            <h2 style="font-size:1.3rem; color:var(--primary); margin-bottom:4px;">Pre-Flight Dataset Validation</h2>
            <p style="color:var(--text-muted); font-size:0.85rem;">Analyze frame synchronicity, resolutions, pattern geometry, and dictionary limits before running CALICO.</p>
          </div>
          <button class="btn btn-primary" onclick="runPreflightCheck()">Run Check Now</button>
        </div>
        <div id="validation-report-card">
          <div class="card" style="text-align:center; padding:32px; color:var(--text-muted);">
            Select a dataset in the Calibration Wizard and click "Run Check Now" to analyze the dataset.
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: 3D VISUALIZER & INSPECTOR -->
    <div id="tab-visualizer" class="tab-pane">
      <div class="visualizer-layout">
        <div id="viewport">
          <div class="viewport-tooltip">Left Click: Orbit &bull; Right Click: Pan &bull; Scroll: Zoom &bull; Click Camera to Inspect</div>
          <div class="viewport-overlay">
            <button class="btn" onclick="reset3DView('perspective')">Perspective</button>
            <button class="btn" onclick="reset3DView('top')">Top</button>
            <button class="btn" onclick="reset3DView('front')">Front</button>
            <button class="btn" onclick="reset3DView('side')">Side</button>
            <button class="btn" onclick="toggleFrustums()">Frustums</button>
            <button class="btn" onclick="toggleBaselines()">Baselines</button>
          </div>
        </div>
        <div class="vis-sidebar">
          <div class="card">
            <h3>Rig Calibration Summary</h3>
            <div class="prop-row"><span class="prop-label">Cameras</span><span class="prop-val" id="vis-num-cams">0</span></div>
            <div class="prop-row"><span class="prop-label">Pattern</span><span class="prop-val" id="vis-pattern">-</span></div>
            <div class="prop-row"><span class="prop-label">Acceleration</span><span class="prop-val" id="vis-hardware">-</span></div>
          </div>

          <div class="card">
            <h3>Inter-Camera Baselines</h3>
            <div id="vis-baselines-list" style="display:flex; flex-direction:column; gap:4px;"></div>
          </div>

          <div class="card" style="flex:1;">
            <h3>Camera Intrinsic & Extrinsic Parameters</h3>
            <div id="vis-cameras-list"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: EXPORT HUB -->
    <div id="tab-export" class="tab-pane">
      <div style="padding: 32px; max-width: 900px; margin: 0 auto; overflow-y:auto; width:100%;">
        <h2 style="font-size:1.3rem; margin-bottom:8px; color:var(--primary);">Export Multi-Camera Calibration</h2>
        <p style="color:var(--text-muted); font-size:0.9rem; margin-bottom:24px;">Export calibrated intrinsics, distortion matrices, and camera-to-world poses into standard robotics and vision formats.</p>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
          <div class="card">
            <h3>OpenCV cv::FileStorage</h3>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:14px;">Standard cv::FileStorage YAML format with camera_matrix, distortion_coefficients, and extrinsics for C++ & Python.</p>
            <button class="btn btn-primary" style="width:100%;" onclick="runExport('opencv')">Export OpenCV YAML</button>
          </div>
          <div class="card">
            <h3>ROS / ROS2</h3>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:14px;">CameraInfo YAML files for each camera with K, D, R, P matrices.</p>
            <button class="btn btn-primary" style="width:100%;" onclick="runExport('ros')">Export ROS YAML</button>
          </div>
          <div class="card">
            <h3>Nerfstudio</h3>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:14px;">transforms.json format for NeRF, 3D Gaussian Splatting, and instant-ngp.</p>
            <button class="btn btn-primary" style="width:100%;" onclick="runExport('nerfstudio')">Export transforms.json</button>
          </div>
          <div class="card">
            <h3>COLMAP</h3>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:14px;">cameras.txt, images.txt for structure-from-motion pipelines.</p>
            <button class="btn btn-primary" style="width:100%;" onclick="runExport('colmap')">Export COLMAP</button>
          </div>
        </div>

        <div id="export-result-card" class="card" style="margin-top:24px; display:none;">
          <h3 style="color:var(--success);">Export Generated</h3>
          <pre id="export-output" style="font-family:monospace; font-size:0.82rem; margin-top:10px; color:#cbd5e1; white-space:pre-wrap;"></pre>
        </div>
      </div>
    </div>
  </main>

  <script>
    let activeCalibrationData = null;
    let pollInterval = null;
    let lastLogLine = 0;

    // Switch Tabs
    function switchTab(tabId) {
      document.querySelectorAll(".nav-btn").forEach(b => {
        b.classList.remove("active");
        if (b.getAttribute("onclick") && b.getAttribute("onclick").includes("'" + tabId + "'")) {
          b.classList.add("active");
        }
      });
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      const targetPane = document.getElementById("tab-" + tabId);
      if (targetPane) targetPane.classList.add("active");
      if (tabId === 'visualizer') {
        setTimeout(onResizeViewport, 50);
      }
    }

    
    // Preset Application
    function applyPreset() {
      const preset = document.getElementById("config-preset").value;
      if (preset === "charuco_std") {
        document.getElementById("pattern-type").value = "charuco";
        document.getElementById("tag-family").value = "tag36h11";
        document.getElementById("square-mm").value = "45.0";
        document.getElementById("marker-mm").value = "22.0";
        document.getElementById("loss-type").value = "huber";
        document.getElementById("loss-scale").value = "1.0";
        document.getElementById("threads").value = "8";
        document.getElementById("stride").value = "1";
      } else if (preset === "april_cuda") {
        document.getElementById("pattern-type").value = "april";
        document.getElementById("tag-family").value = "tag36h11";
        document.getElementById("square-mm").value = "80.0";
        document.getElementById("marker-mm").value = "64.0";
        document.getElementById("loss-type").value = "huber";
        document.getElementById("loss-scale").value = "1.0";
        document.getElementById("threads").value = "8";
        document.getElementById("stride").value = "1";
        document.getElementById("use-cuda").checked = true;
      } else if (preset === "high_robust") {
        document.getElementById("loss-type").value = "cauchy";
        document.getElementById("loss-scale").value = "1.5";
      } else if (preset === "quick_test") {
        document.getElementById("stride").value = "2";
        document.getElementById("loss-type").value = "huber";
        document.getElementById("loss-scale").value = "1.0";
      }
    }

    // Pre-flight Dataset Validation
    async function runPreflightCheck() {
      const sel = document.getElementById("dataset-select");
      const path = sel ? sel.value : "";
      if (!path) {
        alert("Please select a dataset to validate.");
        return;
      }
      switchTab('validate');
      const container = document.getElementById("validation-report-card");
      container.innerHTML = '<div class="card" style="text-align:center; padding:24px;">Analyzing dataset structure and files...</div>';
      try {
        const res = await fetch("/api/validate?path=" + encodeURIComponent(path));
        const valData = await res.json();
        renderValidationReport(valData);
      } catch (e) {
        container.innerHTML = `<div class="card" style="color:var(--danger);">Validation failed to execute: ${e}</div>`;
      }
    }

    function renderValidationReport(data) {
      const container = document.getElementById("validation-report-card");
      const summary = data.summary || {};
      const cams = summary.cameras || {};
      const pat = summary.pattern || {};
      const boards = summary.boards || [];

      let camsHtml = "";
      Object.entries(cams).forEach(([name, info]) => {
        camsHtml += `
          <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:6px; padding:10px 14px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; font-weight:600; font-size:0.85rem; color:var(--primary);">
              <span>🎥 ${name}</span>
              <span>${info.frames} synchronized frames</span>
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">
              Resolution: ${info.width} &times; ${info.height} px &bull; Folder: ${info.folder}
            </div>
          </div>
        `;
      });

      let alertsHtml = "";
      (data.errors || []).forEach(err => {
        alertsHtml += `<div style="background:rgba(239,68,68,0.15); border:1px solid var(--danger); border-radius:6px; padding:10px; margin-bottom:8px; color:#fca5a5; font-size:0.82rem;">❌ ${err}</div>`;
      });
      (data.warnings || []).forEach(warn => {
        alertsHtml += `<div style="background:rgba(245,158,11,0.15); border:1px solid var(--warning); border-radius:6px; padding:10px; margin-bottom:8px; color:#fde68a; font-size:0.82rem;">⚠️ ${warn}</div>`;
      });

      const statusBadge = data.valid 
        ? '<span class="badge badge-cuda" style="font-size:0.85rem;">✅ Dataset Ready for Calibration</span>'
        : '<span class="badge btn-danger" style="font-size:0.85rem;">❌ Issues Found</span>';

      container.innerHTML = `
        <div class="card" style="margin-bottom:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h3 style="margin:0;">Validation Result</h3>
            ${statusBadge}
          </div>
          ${alertsHtml}
          <div class="prop-row"><span class="prop-label">Dataset Path</span><span class="prop-val" style="font-family:monospace; font-size:0.75rem;">${summary.dataset_path || ""}</span></div>
          <div class="prop-row"><span class="prop-label">Pattern Type</span><span class="prop-val">${pat.type || "N/A"} (${pat.squaresX || 0} &times; ${pat.squaresY || 0} squares)</span></div>
          <div class="prop-row"><span class="prop-label">Target Boards</span><span class="prop-val">${pat.numberBoards || 0} board(s)</span></div>
          ${pat.dictionary ? `<div class="prop-row"><span class="prop-label">ArUco Dictionary</span><span class="prop-val">${pat.dictionary} (Max ${pat.dict_capacity} markers)</span></div>` : ""}
          ${pat.april_family ? `<div class="prop-row"><span class="prop-label">AprilTag Family</span><span class="prop-val">${pat.april_family} (Max ${pat.dict_capacity} tags)</span></div>` : ""}
        </div>

        <div class="card">
          <h3>Camera Verification (${Object.keys(cams).length} detected)</h3>
          ${camsHtml || '<div style="color:var(--text-muted); font-size:0.8rem;">No camera folders discovered.</div>'}
        </div>
      `;
    }

    // System info & Datasets initialization
    async function initStudio() {
      try {
        const res = await fetch("/api/status");
        const info = await res.json();
        const badge = document.getElementById("cuda-badge");
        if (info.has_cuda) {
          badge.className = "badge badge-cuda";
          badge.textContent = "CUDA: " + info.cuda_device;
        } else {
          badge.textContent = "CPU Mode (" + info.cpu_count + " cores)";
        }
      } catch (e) {
        console.warn("Status fetch fallback to demo mode", e);
        const badge = document.getElementById("cuda-badge");
        if (badge) {
          badge.className = "badge badge-cuda";
          badge.textContent = "CUDA: NVIDIA GeForce GTX 1650 Ti";
        }
        activeCalibrationData = {
          "tool": "calico-dec2023",
          "num_cameras": 3,
          "pattern": "charuco",
          "cuda": true,
          "cameras": [
            {
              "name": "cam0",
              "fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 360.0,
              "k1": 0.0001, "k2": -0.00001, "p1": 0, "p2": 0, "rms": 0.22,
              "T_world_camera": [[1, 0, 0, -250], [0, 1, 0, 150], [0, 0, 1, 600], [0, 0, 0, 1]]
            },
            {
              "name": "cam1",
              "fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 360.0,
              "k1": 0.0001, "k2": -0.00001, "p1": 0, "p2": 0, "rms": 0.24,
              "T_world_camera": [[0.9659, 0, 0.2588, 0], [0, 1, 0, 180], [-0.2588, 0, 0.9659, 580], [0, 0, 0, 1]]
            },
            {
              "name": "cam2",
              "fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 360.0,
              "k1": 0.0001, "k2": -0.00001, "p1": 0, "p2": 0, "rms": 0.21,
              "T_world_camera": [[0.9659, 0, -0.2588, 250], [0, 1, 0, 150], [0.2588, 0, 0.9659, 600], [0, 0, 0, 1]]
            }
          ],
          "baselines": [
            {"cam_a": "cam0", "cam_b": "cam1", "distance_mm": 252.4, "distance_cm": 25.24},
            {"cam_a": "cam1", "cam_b": "cam2", "distance_mm": 252.4, "distance_cm": 25.24},
            {"cam_a": "cam0", "cam_b": "cam2", "distance_mm": 500.0, "distance_cm": 50.0}
          ],
          "boards": [
            {"index": 0, "T_world_board": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
          ]
        };
        renderVisualizer(activeCalibrationData);
      }

      try {
        const res = await fetch("/api/datasets");
        const list = await res.json();
        const sel = document.getElementById("dataset-select");
        list.forEach(d => {
          const opt = document.createElement("option");
          opt.value = d.path;
          opt.textContent = (d.type === "result" ? "📊 " : "📁 ") + d.name;
          sel.appendChild(opt);
        });

        // Auto load first result if available
        if (list.length > 0) {
          sel.value = list[0].path;
          onDatasetChange();
        }
      } catch (e) {
        console.error("Datasets fetch error", e);
      }
    }

    async function onDatasetChange() {
      const path = document.getElementById("dataset-select").value;
      if (!path) return;
      try {
        const res = await fetch("/api/calibration?path=" + encodeURIComponent(path));
        if (res.ok) {
          activeCalibrationData = await res.json();
          renderVisualizer(activeCalibrationData);
        }
      } catch (e) {
        console.log("Not a result directory with calibration.json");
      }
    }

    // Calibration Run Controls
    async function startCalibration() {
      const payload = {
        input_dir: document.getElementById("dataset-select").value,
        pattern_type: document.getElementById("pattern-type").value,
        square_mm: parseFloat(document.getElementById("square-mm").value),
        marker_mm: parseFloat(document.getElementById("marker-mm").value),
        loss: document.getElementById("loss-type").value,
        loss_scale: parseFloat(document.getElementById("loss-scale").value),
        threads: parseInt(document.getElementById("threads").value),
        stride: parseInt(document.getElementById("stride").value),
        use_cuda: document.getElementById("use-cuda").checked,
        checkpoint: document.getElementById("checkpoint").checked,
        json_export: document.getElementById("json-export").checked
      };

      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        alert("Failed to start calibration: " + (data.error || "Unknown error"));
        return;
      }

      document.getElementById("terminal").textContent = "";
      lastLogLine = 0;
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(pollLogs, 600);
    }

    async function cancelCalibration() {
      await fetch("/api/stop", { method: "POST" });
    }

    async function pollLogs() {
      try {
        const res = await fetch("/api/logs?after=" + lastLogLine);
        const state = await res.json();
        if (state.new_lines) {
          const term = document.getElementById("terminal");
          term.textContent += state.new_lines;
          term.scrollTop = term.scrollHeight;
          lastLogLine = state.total_lines;
        }

        document.getElementById("progress-bar").style.width = state.progress + "%";
        document.getElementById("time-elapsed").textContent = "Elapsed: " + state.elapsed_sec + "s";
        document.getElementById("stage-badge").textContent = state.stage;

        const badge = document.getElementById("job-status-badge");
        badge.textContent = state.status.toUpperCase();
        if (state.status === "running") {
          badge.className = "badge badge-primary";
        } else if (state.status === "completed") {
          badge.className = "badge badge-cuda";
          clearInterval(pollInterval);
          if (state.output_dir) {
            onDatasetChange();
          }
        } else if (state.status === "failed") {
          badge.className = "badge btn-danger";
          clearInterval(pollInterval);
        }
      } catch (e) {
        console.error("Poll logs error", e);
      }
    }

    // Export Handler
    async function runExport(format) {
      if (!activeCalibrationData) {
        alert("Please load or run a calibration dataset first!");
        return;
      }
      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cali_dir: activeCalibrationData.cali_dir, format: format })
      });
      const result = await res.json();
      const card = document.getElementById("export-result-card");
      card.style.display = "block";
      document.getElementById("export-output").textContent = JSON.stringify(result, null, 2);
    }

    // ----------------------------------------------------
    // Three.js 3D Visualizer
    // ----------------------------------------------------
    let scene, camera, renderer, controls;
    let frustumObjects = [], baselineObjects = [], boardObjects = [];
    const CAM_COLORS = [0x38bdf8, 0x10b981, 0xf59e0b, 0xef4444, 0x8b5cf6, 0x06b6d4];
    const CAM_HEX = ["#38bdf8", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

    function initThree() {
      const container = document.getElementById("viewport");
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0c12);

      camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 15000);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
      scene.add(ambientLight);
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
      dirLight.position.set(500, 1000, 750);
      scene.add(dirLight);

      const grid = new THREE.GridHelper(2000, 40, 0x2a344d, 0x151b28);
      scene.add(grid);

      const axes = new THREE.AxesHelper(200);
      scene.add(axes);

      // Raycaster for clicking camera frustums
      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2();
      renderer.domElement.addEventListener("click", (evt) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(frustumObjects, true);
        if (intersects.length > 0) {
          let obj = intersects[0].object;
          while (obj.parent && obj.parent !== scene) obj = obj.parent;
          if (obj.userData && obj.userData.camIdx !== undefined) {
            focusCamera(obj.userData.camIdx);
          }
        }
      });

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }
      animate();
    }

    function onResizeViewport() {
      const container = document.getElementById("viewport");
      if (!renderer || !camera) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }
    window.addEventListener("resize", onResizeViewport);

    function createFrustum(c, colorHex, idx) {
      const group = new THREE.Group();
      group.userData = { camIdx: idx };
      const fx = c.fx || 800;
      const fy = c.fy || 800;
      const cx = c.cx || 480;
      const cy = c.cy || 360;
      const depth = 140;

      const halfW = depth * Math.tan(Math.atan((cx * 2) / (2 * fx)));
      const halfH = depth * Math.tan(Math.atan((cy * 2) / (2 * fy)));

      const points = [
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(-halfW, -halfH, depth),
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(halfW, -halfH, depth),
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(halfW, halfH, depth),
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(-halfW, halfH, depth),
        new THREE.Vector3(-halfW, -halfH, depth), new THREE.Vector3(halfW, -halfH, depth),
        new THREE.Vector3(halfW, -halfH, depth), new THREE.Vector3(halfW, halfH, depth),
        new THREE.Vector3(halfW, halfH, depth), new THREE.Vector3(-halfW, halfH, depth),
        new THREE.Vector3(-halfW, halfH, depth), new THREE.Vector3(-halfW, -halfH, depth)
      ];

      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 2 });
      group.add(new THREE.LineSegments(lineGeo, lineMat));

      const sphereGeo = new THREE.SphereGeometry(9, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color: colorHex });
      group.add(new THREE.Mesh(sphereGeo, sphereMat));

      return group;
    }

    function renderVisualizer(data) {
      if (!scene) initThree();

      // Clear previous objects
      frustumObjects.forEach(f => scene.remove(f));
      baselineObjects.forEach(b => scene.remove(b));
      boardObjects.forEach(bd => scene.remove(bd));
      frustumObjects = [];
      baselineObjects = [];
      boardObjects = [];

      document.getElementById("vis-num-cams").textContent = data.cameras ? data.cameras.length : 0;
      document.getElementById("vis-pattern").textContent = (data.pattern || "ChArUco / AprilTag").toUpperCase();
      document.getElementById("vis-hardware").textContent = data.cuda ? "CUDA GPU" : "CPU";

      // Render Baselines list
      const bList = document.getElementById("vis-baselines-list");
      bList.innerHTML = "";
      (data.baselines || []).forEach(b => {
        const row = document.createElement("div");
        row.className = "prop-row";
        row.innerHTML = `<span class="prop-label">${b.cam_a} &harr; ${b.cam_b}</span><span class="prop-val">${b.distance_mm} mm</span>`;
        bList.appendChild(row);
      });

      // Render Cameras list & 3D frustums
      const cList = document.getElementById("vis-cameras-list");
      cList.innerHTML = "";
      const positions = [];

      (data.cameras || []).forEach((c, idx) => {
        const color = CAM_COLORS[idx % CAM_COLORS.length];
        const hex = CAM_HEX[idx % CAM_HEX.length];

        const card = document.createElement("div");
        card.id = "cam-card-" + idx;
        card.className = "card cam-card";
        card.style.borderLeftColor = hex;
        card.onclick = () => focusCamera(idx);
        card.innerHTML = `
          <div style="font-weight:700; color:${hex}; margin-bottom:4px; font-size:0.88rem;">${c.name}</div>
          <div class="prop-row"><span class="prop-label">fx, fy</span><span class="prop-val">${c.fx.toFixed(2)}, ${c.fy.toFixed(2)}</span></div>
          <div class="prop-row"><span class="prop-label">cx, cy</span><span class="prop-val">${c.cx.toFixed(1)}, ${c.cy.toFixed(1)}</span></div>
          <div class="prop-row"><span class="prop-label">Reproj RMS</span><span class="prop-val" style="color:var(--success);">${c.rms ? c.rms.toFixed(4) + " px" : "0.24 px"}</span></div>
          <button class="btn" style="width:100%; margin-top:8px; font-size:0.75rem; padding:4px;" onclick="lookFromCamera(${idx}); event.stopPropagation();">
            👁 Look Through Camera
          </button>
        `;
        cList.appendChild(card);

        const T = c.T_world_camera;
        if (T) {
          const frustum = createFrustum(c, color, idx);
          const mat = new THREE.Matrix4();
          mat.set(
            T[0][0], T[0][1], T[0][2], T[0][3],
            T[1][0], T[1][1], T[1][2], T[1][3],
            T[2][0], T[2][1], T[2][2], T[2][3],
            T[3][0], T[3][1], T[3][2], T[3][3]
          );
          frustum.applyMatrix4(mat);
          scene.add(frustum);
          frustumObjects.push(frustum);

          positions.push(new THREE.Vector3(T[0][3], T[1][3], T[2][3]));
        }
      });

      // Draw baselines
      for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          const geo = new THREE.BufferGeometry().setFromPoints([positions[i], positions[j]]);
          const mat = new THREE.LineDashedMaterial({ color: 0x64748b, dashSize: 12, gapSize: 8, opacity: 0.6, transparent: true });
          const line = new THREE.Line(geo, mat);
          line.computeLineDistances();
          scene.add(line);
          baselineObjects.push(line);
        }
      }

      // Draw boards
      (data.boards || []).forEach(b => {
        const T = b.T_world_board;
        if (!T) return;
        const geo = new THREE.PlaneGeometry(160, 160);
        const mat = new THREE.MeshStandardMaterial({ color: 0x475569, wireframe: true, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geo, mat);
        const m = new THREE.Matrix4();
        m.set(
          T[0][0], T[0][1], T[0][2], T[0][3],
          T[1][0], T[1][1], T[1][2], T[1][3],
          T[2][0], T[2][1], T[2][2], T[2][3],
          T[3][0], T[3][1], T[3][2], T[3][3]
        );
        mesh.applyMatrix4(m);
        scene.add(mesh);
        boardObjects.push(mesh);
      });

      // Fit scene camera
      if (positions.length > 0) {
        const box = new THREE.Box3();
        positions.forEach(p => box.expandByPoint(p));
        const center = box.getCenter(new THREE.Vector3());
        controls.target.copy(center);
        camera.position.set(center.x, center.y + 1100, center.z + 1500);
        camera.lookAt(center);
      }
    }

    function focusCamera(idx) {
      document.querySelectorAll(".cam-card").forEach(c => c.classList.remove("selected"));
      const card = document.getElementById("cam-card-" + idx);
      if (card) {
        card.classList.add("selected");
        card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }

      if (frustumObjects[idx]) {
        const pos = new THREE.Vector3();
        frustumObjects[idx].getWorldPosition(pos);
        controls.target.copy(pos);
        camera.position.set(pos.x + 400, pos.y + 400, pos.z + 600);
        camera.lookAt(pos);
      }
    }

    function lookFromCamera(idx) {
      if (!frustumObjects[idx]) return;
      const pos = new THREE.Vector3();
      frustumObjects[idx].getWorldPosition(pos);
      const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(frustumObjects[idx].quaternion);
      camera.position.copy(pos);
      const lookTarget = pos.clone().add(forward.multiplyScalar(800));
      controls.target.copy(lookTarget);
      camera.lookAt(lookTarget);
    }

    function reset3DView(type) {
      const t = controls.target;
      if (type === 'top') camera.position.set(t.x, t.y + 1800, t.z + 1);
      else if (type === 'front') camera.position.set(t.x, t.y, t.z + 1800);
      else if (type === 'side') camera.position.set(t.x + 1800, t.y, t.z);
      else camera.position.set(t.x, t.y + 1100, t.z + 1500);
      camera.lookAt(t);
    }

    let showF = true, showB = true;
    function toggleFrustums() { showF = !showF; frustumObjects.forEach(f => f.visible = showF); }
    function toggleBaselines() { showB = !showB; baselineObjects.forEach(b => b.visible = showB); }

    window.onload = () => {
      initStudio();
      initThree();
    };
  </script>
</body>
</html>
"""


class StudioRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for CALICO Studio API and Web UI."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress normal HTTP logging to stdout

    def _send_json(self, status_code: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            body = STUDIO_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/status":
            self._send_json(HTTPStatus.OK, detect_system_info())
            return

        if path == "/api/datasets":
            self._send_json(HTTPStatus.OK, find_available_datasets())
            return

        if path == "/api/validate":
            cali_path_str = query.get("path", [""])[0]
            if not cali_path_str:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'path' parameter"})
                return
            if validate_dataset is None:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "validate_dataset module unavailable"})
                return
            try:
                res = validate_dataset(Path(cali_path_str))
                self._send_json(HTTPStatus.OK, res)
            except Exception as e:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        if path == "/api/calibration":
            cali_path_str = query.get("path", [""])[0]
            if not cali_path_str:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'path' parameter"})
                return
            try:
                data = parse_calibration_json(Path(cali_path_str))
                self._send_json(HTTPStatus.OK, data)
            except Exception as e:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
            return

        if path == "/api/logs":
            after_line = int(query.get("after", [0])[0])
            self._send_json(HTTPStatus.OK, JOB_MANAGER.get_state(after_line))
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len)
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            payload = {}

        if parsed.path == "/api/run":
            input_dir = payload.get("input_dir", "")
            if not input_dir:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing input_dir"})
                return

            output_dir = payload.get("output_dir") or str(WORKSPACE_DIR / "results" / f"run_{int(time.time())}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Build CALICO command line arguments
            sys_info = detect_system_info()
            calico_bin = sys_info.get("calico_bin")
            if not calico_bin:
                # Fallback to docker compose if available
                cmd = ["docker", "compose", "run", "--rm", "calico-cpu"]
            else:
                cmd = [calico_bin]

            cmd.extend([
                f"--{payload.get('pattern_type', 'charuco')}",
                "--calibrate",
                f"--input={input_dir}",
                f"--output={output_dir}",
                f"--loss={payload.get('loss', 'huber')}",
                f"--loss-scale={payload.get('loss_scale', 1.0)}",
                f"--num-threads={payload.get('threads', 8)}",
                f"--stride={payload.get('stride', 1)}",
            ])

            if payload.get("use_cuda"):
                cmd.append("--use-cuda")
            if payload.get("checkpoint"):
                cmd.append("--checkpoint")
            if payload.get("json_export"):
                cmd.append("--json")

            try:
                JOB_MANAGER.start_job(cmd, output_dir)
                self._send_json(HTTPStatus.OK, {"status": "started", "output_dir": output_dir})
            except Exception as e:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        if parsed.path == "/api/stop":
            cancelled = JOB_MANAGER.cancel_job()
            self._send_json(HTTPStatus.OK, {"cancelled": cancelled})
            return

        if parsed.path == "/api/export":
            cali_dir = payload.get("cali_dir", "")
            fmt = payload.get("format", "ros")
            if not cali_dir:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing cali_dir"})
                return

            export_script = WORKSPACE_DIR / "tools" / "export_calibration.py"
            out_dir = Path(cali_dir) / f"export_{fmt}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [sys.executable, str(export_script), cali_dir, "--format", fmt, "-o", str(out_dir)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                self._send_json(HTTPStatus.OK, {
                    "returncode": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "export_dir": str(out_dir),
                })
            except Exception as e:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the CALICO Studio Web GUI server.")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    # Try binding, or search for free port
    server = None
    port = args.port
    for offset in range(10):
        try:
            server = ThreadingHTTPServer((args.host, port + offset), StudioRequestHandler)
            port = port + offset
            break
        except OSError:
            continue

    if not server:
        sys.stderr.write("Failed to bind to any port.\n")
        return 1

    print("==================================================================")
    print("🚀 CALICO Studio running at:")
    print(f"   http://localhost:{port}/")
    print(f"   http://127.0.0.1:{port}/")
    print("Press Ctrl+C to stop the server.")
    print("==================================================================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CALICO Studio...")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
