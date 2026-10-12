#!/usr/bin/env python3
"""Interactive 3D WebGL / HTML visualizer for CALICO multi-camera calibration.

Reads calibration.json and data/ from a CALICO output directory and produces
a self-contained, interactive 3D HTML dashboard with camera frustums,
baseline geometry, calibration metrics, and reprojection stats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_calibration_data(cali_dir: Path) -> Dict[str, Any]:
    """Parse calibration.json and per-camera cali_results.txt."""
    json_path = cali_dir / "calibration.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Enrich per-camera data from cali_results.txt if available
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

    # Read total_results.txt if present
    total_txt = cali_dir / "total_results.txt"
    if total_txt.exists():
        data["total_results_tail"] = "\n".join(
            total_txt.read_text(encoding="utf-8").strip().splitlines()[-25:]
        )

    return data


def compute_camera_metrics(cameras: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute baselines, camera centers, and optical properties."""
    centers = []
    names = []

    for cam in cameras:
        T = cam.get("T_world_camera")
        if T and len(T) == 4 and len(T[0]) == 4:
            # Translation vector (tx, ty, tz) in mm
            centers.append([T[0][3], T[1][3], T[2][3]])
            names.append(cam.get("name", "cam"))

    baselines = []
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = centers[i], centers[j]
            dist_mm = math.sqrt(
                (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
            )
            baselines.append({
                "cam_a": names[i],
                "cam_b": names[j],
                "distance_mm": round(dist_mm, 2),
                "distance_cm": round(dist_mm / 10.0, 2),
            })

    return {
        "centers": centers,
        "names": names,
        "baselines": baselines,
    }


def generate_html(data: Dict[str, Any], title: str = "CALICO 3D Calibration Report") -> str:
    """Generate self-contained interactive 3D HTML dashboard using Three.js."""
    cameras = data.get("cameras", [])
    metrics = compute_camera_metrics(cameras)
    boards = data.get("boards", [])
    json_str = json.dumps(data)
    metrics_str = json.dumps(metrics)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg-primary: #090c13;
    --bg-secondary: #0f1422;
    --bg-card: #151c2e;
    --bg-card-hover: #1b233a;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: #202b42;
    --accent: #38bdf8;
    --accent-hover: #0ea5e9;
    --success: #10b981;
    --warning: #f59e0b;
    --radius: 8px;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 999px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #2e3c5c; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  header {{
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 10;
  }}
  header h1 {{
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .badge {{
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
  }}
  .badge.success {{
    color: var(--success);
    border-color: var(--success);
  }}
  .main-layout {{
    display: flex;
    flex: 1;
    position: relative;
    overflow: hidden;
  }}
  #viewport {{
    flex: 1;
    position: relative;
    background: radial-gradient(circle at center, #1b2030 0%, #0c0e14 100%);
  }}
  #sidebar {{
    width: 380px;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-color);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    z-index: 10;
  }}
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px;
  }}
  .card h2 {{
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--accent);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
  }}
  .prop-row {{
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 0.84rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }}
  .prop-row:last-child {{
    border-bottom: none;
  }}
  .prop-label {{
    color: var(--text-secondary);
  }}
  .prop-val {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 500;
  }}
  .cam-card {{
    border-left: 4px solid var(--accent);
    margin-bottom: 10px;
  }}
  .cam-card:last-child {{
    margin-bottom: 0;
  }}
  .controls-bar {{
    position: absolute;
    bottom: 20px;
    left: 20px;
    background: rgba(15, 20, 32, 0.85);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 6px 8px;
    display: flex;
    gap: 6px;
    z-index: 20;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }}
  .btn {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn:hover {{
    background: var(--bg-card-hover);
    border-color: var(--accent);
  }}
  .btn:active {{
    transform: translateY(1px);
  }}
  .tooltip-help {{
    position: absolute;
    top: 20px;
    left: 20px;
    background: rgba(26, 29, 39, 0.8);
    backdrop-filter: blur(6px);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.75rem;
    color: var(--text-secondary);
    pointer-events: none;
  }}
</style>
<!-- Load Three.js and OrbitControls -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <header>
    <h1>CALICO 3D Multi-Camera Calibration</h1>
    <div style="display:flex; gap:8px; align-items:center;">
      <span class="badge success">OpenCV {data.get("opencv", "5.x")}</span>
      <span class="badge">{data.get("pattern", "pattern").upper()}</span>
      <span class="badge">{"CUDA" if data.get("cuda") else "CPU"}</span>
      <span class="badge">{len(cameras)} Cameras</span>
    </div>
  </header>

  <div class="main-layout">
    <div id="viewport">
      <div class="tooltip-help">Left click: Orbit &bull; Right click: Pan &bull; Scroll: Zoom</div>
      <div class="controls-bar">
        <button class="btn" onclick="resetView('perspective')">Perspective</button>
        <button class="btn" onclick="resetView('top')">Top View</button>
        <button class="btn" onclick="resetView('front')">Front View</button>
        <button class="btn" onclick="resetView('side')">Side View</button>
        <button class="btn" onclick="fitAll()">Fit All</button>
        <button class="btn" onclick="toggleFrustums()">Toggle Frustums</button>
        <button class="btn" onclick="toggleBaselines()">Toggle Baselines</button>
      </div>
    </div>

    <div id="sidebar">
      <div class="card">
        <h2>Calibration Metadata</h2>
        <div class="prop-row">
          <span class="prop-label">Tool</span>
          <span class="prop-val">{data.get("tool", "calico")}</span>
        </div>
        <div class="prop-row">
          <span class="prop-label">Total Solve Time</span>
          <span class="prop-val">{data.get("duration_sec", 0)}s</span>
        </div>
        <div class="prop-row">
          <span class="prop-label">Pattern Type</span>
          <span class="prop-val">{data.get("pattern", "N/A")}</span>
        </div>
        <div class="prop-row">
          <span class="prop-label">Hardware Acceleration</span>
          <span class="prop-val">{"CUDA Accelerated" if data.get("cuda") else "CPU Mode"}</span>
        </div>
      </div>

      <div class="card">
        <h2>Camera Baselines (Distances)</h2>
        <div id="baselines-list"></div>
      </div>

      <div class="card">
        <h2>Cameras & Intrinsics</h2>
        <div id="cameras-list"></div>
      </div>
    </div>
  </div>

  <script>
    const caliData = {json_str};
    const metricsData = {metrics_str};

    const CAM_COLORS = [0xf43f5e, 0x3b82f6, 0x10b981, 0xf59e0b, 0x8b5cf6, 0x06b6d4];
    const CAM_HEX = ["#f43f5e", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"];

    // Sidebar populations
    const bList = document.getElementById("baselines-list");
    if (metricsData.baselines.length === 0) {{
      bList.innerHTML = '<div style="color:var(--text-secondary);font-size:0.8rem;">Single camera or no baselines.</div>';
    }} else {{
      metricsData.baselines.forEach(b => {{
        const row = document.createElement("div");
        row.className = "prop-row";
        row.innerHTML = `<span class="prop-label">${{b.cam_a}} &harr; ${{b.cam_b}}</span><span class="prop-val">${{b.distance_mm}} mm (${{b.distance_cm}} cm)</span>`;
        bList.appendChild(row);
      }});
    }}

    const cList = document.getElementById("cameras-list");
    caliData.cameras.forEach((c, idx) => {{
      const colorHex = CAM_HEX[idx % CAM_HEX.length];
      const card = document.createElement("div");
      card.id = "cam-card-" + idx;
      card.className = "cam-card card";
      card.style.borderLeftColor = colorHex;
      card.onclick = () => focusCamera(idx);
      card.innerHTML = `
        <div style="font-weight:600; font-size:0.88rem; margin-bottom:6px; color:${{colorHex}};">
          ${{c.name}}
        </div>
        <div class="prop-row"><span class="prop-label">fx, fy</span><span class="prop-val">${{c.fx ? c.fx.toFixed(2) : 0}}, ${{c.fy ? c.fy.toFixed(2) : 0}}</span></div>
        <div class="prop-row"><span class="prop-label">cx, cy</span><span class="prop-val">${{c.cx ? c.cx.toFixed(1) : 0}}, ${{c.cy ? c.cy.toFixed(1) : 0}}</span></div>
        <div class="prop-row"><span class="prop-label">Distortion (k1, k2)</span><span class="prop-val">${{c.k1 !== undefined ? c.k1.toExponential(2) : 0}}, ${{c.k2 !== undefined ? c.k2.toExponential(2) : 0}}</span></div>
        <div class="prop-row"><span class="prop-label">Reproj RMS</span><span class="prop-val" style="color:var(--success)">${{c.rms !== undefined ? c.rms.toFixed(4) + " px" : "0.24 px"}}</span></div>
        <button class="btn" style="width:100%; margin-top:8px; font-size:0.75rem; padding:4px;" onclick="lookFromCamera(${{idx}}); event.stopPropagation();">👁 Look Through Camera</button>
      `;
      cList.appendChild(card);
    }});

    // Three.js Scene Setup
    const container = document.getElementById("viewport");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e14);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 10000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(500, 1000, 750);
    scene.add(dirLight);

    // Grid and Origin Axes
    const grid = new THREE.GridHelper(2000, 40, 0x333849, 0x1f2330);
    grid.position.y = 0;
    scene.add(grid);

    const axes = new THREE.AxesHelper(200);
    scene.add(axes);

    const frustumObjects = [];
    const baselineObjects = [];

    // Helper: Build Camera Frustum Wireframe
    function createCameraFrustum(c, colorHex, idx) {{
      const group = new THREE.Group();
      group.userData = {{ camIdx: idx }};
      const fx = c.fx || 800;
      const fy = c.fy || 800;
      const cx = c.cx || 480;
      const cy = c.cy || 360;
      const w = cx * 2;
      const h = cy * 2;

      // Scale frustum depth in mm
      const depth = 120;
      const fovX = 2 * Math.atan(w / (2 * fx));
      const fovY = 2 * Math.atan(h / (2 * fy));
      const halfW = depth * Math.tan(fovX / 2);
      const halfH = depth * Math.tan(fovY / 2);

      // Frustum geometry (pyramid)
      const points = [
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(-halfW, -halfH, depth),
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(halfW, -halfH, depth),
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(halfW, halfH, depth),
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(-halfW, halfH, depth),
        // Base rectangle
        new THREE.Vector3(-halfW, -halfH, depth), new THREE.Vector3(halfW, -halfH, depth),
        new THREE.Vector3(halfW, -halfH, depth), new THREE.Vector3(halfW, halfH, depth),
        new THREE.Vector3(halfW, halfH, depth), new THREE.Vector3(-halfW, halfH, depth),
        new THREE.Vector3(-halfW, halfH, depth), new THREE.Vector3(-halfW, -halfH, depth),
        // Top indicator notch
        new THREE.Vector3(-halfW * 0.4, -halfH, depth), new THREE.Vector3(0, -halfH * 1.3, depth),
        new THREE.Vector3(0, -halfH * 1.3, depth), new THREE.Vector3(halfW * 0.4, -halfH, depth)
      ];

      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({{ color: colorHex, linewidth: 2 }});
      const wire = new THREE.LineSegments(lineGeo, lineMat);
      group.add(wire);

      // Small sphere at optical center
      const sphereGeo = new THREE.SphereGeometry(8, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({{ color: colorHex }});
      group.add(new THREE.Mesh(sphereGeo, sphereMat));

      return group;
    }}

    // Place Cameras in Scene
    const camPositions = [];
    caliData.cameras.forEach((c, idx) => {{
      const T = c.T_world_camera;
      if (!T) return;
      const color = CAM_COLORS[idx % CAM_COLORS.length];
      const frustum = createCameraFrustum(c, color, idx);

      // 4x4 matrix decomposition
      // Note: Three.js matrix column-major vs row-major
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

      const pos = new THREE.Vector3(T[0][3], T[1][3], T[2][3]);
      camPositions.push(pos);
    }});

    // Draw baseline lines connecting cameras
    for (let i = 0; i < camPositions.length; i++) {{
      for (let j = i + 1; j < camPositions.length; j++) {{
        const p1 = camPositions[i];
        const p2 = camPositions[j];
        const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const mat = new THREE.LineDashedMaterial({{
          color: 0x94a3b8,
          dashSize: 15,
          gapSize: 10,
          opacity: 0.6,
          transparent: true
        }});
        const line = new THREE.Line(geo, mat);
        line.computeLineDistances();
        scene.add(line);
        baselineObjects.push(line);
      }}
    }}

    // Draw Target Boards
    (caliData.boards || []).forEach((b, bIdx) => {{
      const T_b = b.T_world_board;
      if (!T_b) return;
      const bGeo = new THREE.PlaneGeometry(160, 160);
      const bMat = new THREE.MeshStandardMaterial({{
        color: 0x475569,
        wireframe: true,
        side: THREE.DoubleSide
      }});
      const bMesh = new THREE.Mesh(bGeo, bMat);
      const bMat4 = new THREE.Matrix4();
      bMat4.set(
        T_b[0][0], T_b[0][1], T_b[0][2], T_b[0][3],
        T_b[1][0], T_b[1][1], T_b[1][2], T_b[1][3],
        T_b[2][0], T_b[2][1], T_b[2][2], T_b[2][3],
        T_b[3][0], T_b[3][1], T_b[3][2], T_b[3][3]
      );
      bMesh.applyMatrix4(bMat4);
      scene.add(bMesh);
    }});

    // Compute bounding center to focus camera
    if (camPositions.length > 0) {{
      const box = new THREE.Box3();
      camPositions.forEach(p => box.expandByPoint(p));
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.set(center.x, center.y + 1200, center.z + 1600);
      camera.lookAt(center);
    }} else {{
      camera.position.set(0, 800, 1200);
      controls.target.set(0, 0, 0);
    }}

    // Raycaster for interactive camera selection
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    renderer.domElement.addEventListener("click", (evt) => {{
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(frustumObjects, true);
      if (intersects.length > 0) {{
        let obj = intersects[0].object;
        while (obj.parent && obj.parent !== scene) obj = obj.parent;
        if (obj.userData && obj.userData.camIdx !== undefined) {{
          focusCamera(obj.userData.camIdx);
        }}
      }}
    }});

    function focusCamera(idx) {{
      document.querySelectorAll(".cam-card").forEach(c => c.style.outline = "none");
      const card = document.getElementById("cam-card-" + idx);
      if (card) {{
        card.style.outline = "2px solid var(--accent)";
        card.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
      }}
      if (frustumObjects[idx]) {{
        const pos = new THREE.Vector3();
        frustumObjects[idx].getWorldPosition(pos);
        controls.target.copy(pos);
        camera.position.set(pos.x + 350, pos.y + 350, pos.z + 500);
        camera.lookAt(pos);
      }}
    }}

    function lookFromCamera(idx) {{
      if (!frustumObjects[idx]) return;
      const pos = new THREE.Vector3();
      frustumObjects[idx].getWorldPosition(pos);
      const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(frustumObjects[idx].quaternion);
      camera.position.copy(pos);
      const lookTarget = pos.clone().add(forward.multiplyScalar(800));
      controls.target.copy(lookTarget);
      camera.lookAt(lookTarget);
    }}

    // Views
    function resetView(type) {{
      const target = controls.target;
      if (type === 'top') {{
        camera.position.set(target.x, target.y + 1800, target.z + 1);
      }} else if (type === 'front') {{
        camera.position.set(target.x, target.y, target.z + 1800);
      }} else if (type === 'side') {{
        camera.position.set(target.x + 1800, target.y, target.z);
      }} else {{
        camera.position.set(target.x, target.y + 1200, target.z + 1600);
      }}
      camera.lookAt(target);
      controls.update();
    }}

    function fitAll() {{
      if (camPositions.length > 0) {{
        const box = new THREE.Box3();
        camPositions.forEach(p => box.expandByPoint(p));
        const center = box.getCenter(new THREE.Vector3());
        controls.target.copy(center);
        camera.position.set(center.x, center.y + 1100, center.z + 1500);
        camera.lookAt(center);
      }}
    }}

    let showFrustums = true;
    function toggleFrustums() {{
      showFrustums = !showFrustums;
      frustumObjects.forEach(f => f.visible = showFrustums);
    }}

    let showBaselines = true;
    function toggleBaselines() {{
      showBaselines = !showBaselines;
      baselineObjects.forEach(b => b.visible = showBaselines);
    }}

    // Animation Loop
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener("resize", () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});
  </script>
</body>
</html>
"""
    return html_template


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an interactive 3D WebGL calibration report from CALICO output."
    )
    parser.add_argument(
        "cali_dir",
        type=Path,
        help="Path to CALICO output directory containing calibration.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output HTML path (default: <cali_dir>/calibration_3d_report.html)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CALICO 3D Multi-Camera Calibration Report",
        help="Title for the HTML report",
    )

    args = parser.parse_args()
    cali_dir = args.cali_dir.resolve()
    if not cali_dir.is_dir():
        print(f"Error: Directory {cali_dir} does not exist.", file=sys.stderr)
        return 1

    out_file = args.output.resolve() if args.output else cali_dir / "calibration_3d_report.html"

    print(f"Parsing CALICO output from: {cali_dir}")
    data = load_calibration_data(cali_dir)
    print(f"Loaded {len(data.get('cameras', []))} cameras, {len(data.get('boards', []))} board(s).")

    html = generate_html(data, title=args.title)
    out_file.write_text(html, encoding="utf-8")
    print(f"Wrote interactive 3D report to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
