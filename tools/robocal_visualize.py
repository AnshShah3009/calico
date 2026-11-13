#!/usr/bin/env python3
"""Qualitative views of a calico-robocal intrinsics (or robot) solve.

  tools/robocal_visualize.py --out-dir=<calico out dir> --images-root=<dir the frame paths resolve in>
                             [--camera=left] [--examples=12] [--overlays=25] [--review=review.json]

Reads <out>/residuals.csv (written by calico-robocal) and <out>/<camera>.camera_info.yaml, writes
<out>/viz/:
  coverage.png        corner density over the image (log), rings at 30/60/90 deg off axis
  residual_map.png    mean residual magnitude per image cell
  residual_angle.png  residual vs off-axis angle (radial / tangential), all and hold-out
  sharpness.png       per-view RMS residual vs the recorder's sharpness score (if --review)
  examples/*.jpg      frames with detected corners (green), the model's projection (red ring) and
                      the residual x20 (yellow); held-out views in cyan
  overlays/*.jpg      the frames picked for a visual review (worst, best, edge, random)
"""
import argparse
import collections
import csv
import json
import math
import os
import random

import cv2
import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_camera_info(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    d = yaml.safe_load("".join(lines))
    k = d["k"]
    return d["distortion_model"], [k[0], k[4], k[2], k[5]] + list(d["d"]), d["width"], d["height"]


def project(model, p, X):
    """Wide-angle models of calico (equidistant, kb_rt, double_sphere)."""
    x, y, z = X
    fx, fy, cx, cy = p[:4]
    d = p[4:]
    if model == "double_sphere":
        xi, al = d[0], d[1]
        d1 = math.sqrt(x * x + y * y + z * z)
        zz = xi * d1 + z
        d2 = math.sqrt(x * x + y * y + zz * zz)
        den = al * d2 + (1 - al) * zz
        return fx * x / den + cx, fy * y / den + cy
    rho = math.hypot(x, y)
    th = math.atan2(rho, z)
    t2 = th * th
    thd = th * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
    s = thd / rho if rho > 1e-12 else 1.0 / z
    xd, yd = s * x, s * y
    if model == "kb_rt":
        r2 = xd * xd + yd * yd
        xd, yd = xd + 2 * d[4] * xd * yd + d[5] * (r2 + 2 * xd * xd), yd + d[4] * (r2 + 2 * yd * yd) + 2 * d[5] * xd * yd
    return fx * xd + cx, fy * yd + cy


def ring(model, p, theta_deg, n=360):
    pts = []
    for k in range(n + 1):
        ph = 2 * math.pi * k / n
        t = math.radians(theta_deg)
        pts.append(project(model, p, (math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph), math.cos(t))))
    return np.array(pts, np.int32)


def draw_rings(img, model, p, scale=1.0, color=(200, 170, 110)):
    for th in (30, 60, 90):
        r = (ring(model, p, th) * scale).astype(np.int32)
        cv2.polylines(img, [r], False, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"{th} deg", tuple(r[45]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale + 0.2, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--images-root", default="")
    ap.add_argument("--camera", default="left")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--overlays", type=int, default=25)
    ap.add_argument("--review", default="")
    ap.add_argument("--viz-dir", default="", help="output directory (default <out-dir>/viz)")
    a = ap.parse_args()
    out = a.viz_dir or os.path.join(a.out_dir, "viz")
    os.makedirs(os.path.join(out, "examples"), exist_ok=True)
    os.makedirs(os.path.join(out, "overlays"), exist_ok=True)
    model, p, W, H = load_camera_info(os.path.join(a.out_dir, f"{a.camera}.camera_info.yaml"))

    rows = []
    with open(os.path.join(a.out_dir, "residuals.csv")) as f:
        for r in csv.DictReader(f):
            if r["camera"] != a.camera:
                continue
            rows.append((r["image"], r["board"], float(r["u"]), float(r["v"]), float(r["du"]), float(r["dv"]),
                         int(r["holdout"]), float(r["theta_deg"])))
    u = np.array([r[2] for r in rows])
    v = np.array([r[3] for r in rows])
    du = np.array([r[4] for r in rows])
    dv = np.array([r[5] for r in rows])
    hold = np.array([r[6] for r in rows])
    th = np.array([r[7] for r in rows])
    e = np.hypot(du, dv)
    print(f"{len(rows)} corners, rms {math.sqrt((e ** 2).mean()):.3f} px, hold-out rms "
          f"{math.sqrt((e[hold == 1] ** 2).mean()) if (hold == 1).any() else float('nan'):.3f} px")

    # Coverage heatmap (log) and residual map, 32 px cells.
    cs = 32
    gx, gy = W // cs + 1, H // cs + 1
    cnt = np.zeros((gy, gx))
    esum = np.zeros((gy, gx))
    for x, y, ee in zip(u, v, e):
        i, j = int(y // cs), int(x // cs)
        if 0 <= i < gy and 0 <= j < gx:
            cnt[i, j] += 1
            esum[i, j] += ee
    for name, data, label, cmap in (
        ("coverage.png", np.log10(cnt + 1), "log10(corners per 32 px cell)", "viridis"),
        ("residual_map.png", np.where(cnt >= 5, esum / np.maximum(cnt, 1), np.nan), "mean |residual| (px), cells with >= 5 corners", "magma"),
    ):
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(data, extent=(0, gx * cs, gy * cs, 0), cmap=cmap, interpolation="nearest")
        for t in (30, 60, 90):
            r = ring(model, p, t)
            ax.plot(r[:, 0], r[:, 1], color="w", lw=0.8, alpha=0.8)
            ax.text(r[45, 0], r[45, 1], f"{t}°", color="w", fontsize=9)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f"{a.camera}: {label} — {len(rows)} corners, {len(set(r[0] for r in rows))} frames")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(out, name), dpi=110, bbox_inches="tight")
        plt.close(fig)

    # Residual vs angle: radial / tangential components about the principal point.
    rx, ry = u - p[2], v - p[3]
    rn = np.maximum(np.hypot(rx, ry), 1e-9)
    rad = (du * rx + dv * ry) / rn
    tan = (-du * ry + dv * rx) / rn
    bins = np.arange(0, 100, 5)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, (lab, comp) in enumerate((("radial", rad), ("tangential", tan))):
        means, rmss, mids = [], [], []
        for b0 in bins:
            m = (th >= b0) & (th < b0 + 5)
            if m.sum() < 20:
                continue
            mids.append(b0 + 2.5)
            means.append(comp[m].mean())
            rmss.append(math.sqrt((comp[m] ** 2).mean()))
        ax[k].plot(mids, rmss, "o-", label="rms")
        ax[k].plot(mids, means, "s-", label="mean (bias)")
        ax[k].axhline(0, color="k", lw=0.5)
        ax[k].set_xlabel("off-axis angle (deg)")
        ax[k].set_ylabel("px")
        ax[k].set_title(f"{lab} residual by angle")
        ax[k].legend()
    fig.savefig(os.path.join(out, "residual_angle.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Per view (image, board) rms.
    views = collections.defaultdict(list)
    for r in rows:
        views[(r[0], r[1])].append(r)
    view_rms = {k: math.sqrt(np.mean([x[4] ** 2 + x[5] ** 2 for x in vs])) for k, vs in views.items()}
    if a.review and os.path.exists(a.review):
        rev = json.load(open(a.review))
        sharp = {}
        for fr in rev.get("frames", []):
            sharp[(fr["raw"], fr["board"])] = fr.get("sharpness", float("nan"))
        board_map = {f"k{k}": n for k, n in enumerate(["kalibr_297x210_6x8_24", "K1", "K2", "K3", "K4", "K5", "K6"])}
        xs, ys = [], []
        for (img, b), rr in view_rms.items():
            key = (os.path.basename(img), board_map.get(b, b))
            if key in sharp and sharp[key] == sharp[key]:
                xs.append(sharp[key])
                ys.append(rr)
        if xs:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(xs, ys, s=6, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("recorder sharpness score (log)")
            ax.set_ylabel("view rms residual (px)")
            c = np.corrcoef(np.log(np.array(xs)), np.array(ys))[0, 1]
            ax.set_title(f"per-view residual vs sharpness (corr {c:+.2f}, {len(xs)} views)")
            fig.savefig(os.path.join(out, "sharpness.png"), dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"sharpness vs view rms: corr(log sharpness, rms) = {c:+.3f} over {len(xs)} views")

    # Frames: worst / best / edge / random.
    by_img = collections.defaultdict(list)
    for r in rows:
        by_img[r[0]].append(r)
    img_rms = {k: math.sqrt(np.mean([x[4] ** 2 + x[5] ** 2 for x in vs])) for k, vs in by_img.items()}
    img_edge = {k: max(x[7] for x in vs) for k, vs in by_img.items()}
    ordered = sorted(img_rms, key=img_rms.get)
    random.seed(1)
    pick = []
    for lst in (ordered[-8:], ordered[:4], sorted(img_edge, key=img_edge.get)[-8:], random.sample(ordered, min(10, len(ordered)))):
        for k in lst:
            if k not in pick:
                pick.append(k)
    pick = pick[: max(a.overlays, a.examples)]

    def overlay(img_path, scale_err=20.0):
        # The frame path as written (may be a container path), else by name under --images-root.
        full = img_path
        for sub in ("", "images", "raw_images"):
            if os.path.exists(full):
                break
            full = os.path.join(a.images_root, sub, os.path.basename(img_path))
        im = cv2.imread(full)
        if im is None:
            return None
        draw_rings(im, model, p)
        for r in by_img[img_path]:
            x, y, ddu, ddv, h = r[2], r[3], r[4], r[5], r[6]
            col = (255, 255, 0) if h else (60, 220, 60)
            cv2.circle(im, (int(round(x)), int(round(y))), 2, col, -1, cv2.LINE_AA)
            cv2.circle(im, (int(round(x + ddu)), int(round(y + ddv))), 5, (60, 60, 255), 1, cv2.LINE_AA)
            cv2.line(im, (int(round(x)), int(round(y))), (int(round(x + scale_err * ddu)), int(round(y + scale_err * ddv))),
                     (0, 220, 255), 1, cv2.LINE_AA)
        txt = f"{os.path.basename(img_path)}  rms {img_rms[img_path]:.2f} px  max angle {img_edge[img_path]:.0f} deg  {len(by_img[img_path])} corners"
        cv2.rectangle(im, (0, 0), (im.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(im, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, "green: detected  red ring: model projection  yellow: residual x20  cyan: held-out",
                    (10, im.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return im

    for n, k in enumerate(pick):
        im = overlay(k)
        if im is None:
            continue
        cv2.imwrite(os.path.join(out, "overlays", f"{n:02d}_{os.path.basename(k)[:-4]}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if n < a.examples:
            cv2.imwrite(os.path.join(out, "examples", f"{n:02d}_{os.path.basename(k)[:-4]}.jpg"), im,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(os.path.join(out, "picked.txt"), "w") as f:
        for k in pick:
            f.write(f"{k} rms={img_rms[k]:.3f} max_angle={img_edge[k]:.1f} corners={len(by_img[k])}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
