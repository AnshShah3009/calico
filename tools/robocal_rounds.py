#!/usr/bin/env python3
"""Corner density per guided round (calico-robocal `detection: {dump_rounds: <dir>}`).

Reads <dir>/<camera>_round<k>.csv (+ _lens.txt) written after detection (k = 0) and after each
guided round, attributes every corner to the round that first found it, and writes:

  density_rounds.png   corner density per image cell: detector, added per round, final (log scale)
  theta_rounds.png     corners per 5 deg off-axis angle, stacked by round of origin
  board_rounds.png     corners per board, stacked by round of origin
  qualitative/*.jpg    frames with the most added corners and the widest-angle corners:
                       green = detector, orange = guided round 1 (incl. rectified views),
                       magenta = later rounds; zoom insets of the region with most additions
  summary.txt          counts per round

usage: robocal_rounds.py <dump_dir> [--camera left] [--path-map /sim=$HOME/robocal_work/sim]
                         [--out <dump_dir>/viz] [--frames 6]
"""
import argparse
import collections
import csv
import glob
import math
import os
import re

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

COL_BGR = [(80, 200, 60), (0, 150, 255), (220, 60, 220), (255, 200, 0)]
COL_MPL = ["#3cb450", "#ff9600", "#dc3cdc", "#00c8ff"]


def project(model, p, X):
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
    if model == "kb_rt" and len(d) >= 6:
        r2 = xd * xd + yd * yd
        xd, yd = (xd + 2 * d[4] * xd * yd + d[5] * (r2 + 2 * xd * xd),
                  yd + d[4] * (r2 + 2 * yd * yd) + 2 * d[5] * xd * yd)
    return fx * xd + cx, fy * yd + cy


class ThetaLookup:
    """Pixel -> off-axis angle, by tabulating the projection per azimuth."""

    def __init__(self, model, p, n_az=180, th_max=125.0):
        self.model, self.p = model, p
        self.az = np.linspace(-math.pi, math.pi, n_az + 1)
        self.th = np.radians(np.arange(0.0, th_max, 0.25))
        cx, cy = p[2], p[3]
        self.r = np.zeros((len(self.az), len(self.th)))
        for i, a in enumerate(self.az):
            for j, t in enumerate(self.th):
                u, v = project(model, p, (math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)))
                self.r[i, j] = math.hypot((u - cx) / p[0], (v - cy) / p[1])
            self.r[i] = np.maximum.accumulate(self.r[i])  # monotone for interp (past the fold: clamp)

    def __call__(self, u, v):
        x = (np.asarray(u) - self.p[2]) / self.p[0]
        y = (np.asarray(v) - self.p[3]) / self.p[1]
        a = np.arctan2(y, x)
        r = np.hypot(x, y)
        idx = np.clip(np.round((a + math.pi) / (2 * math.pi) * (len(self.az) - 1)).astype(int), 0, len(self.az) - 1)
        out = np.empty_like(r)
        for i in np.unique(idx):
            m = idx == i
            out[m] = np.interp(r[m], self.r[i], self.th)
        return np.degrees(out)


def ring(model, p, theta_deg, n=360):
    t = math.radians(theta_deg)
    return np.array([project(model, p, (math.sin(t) * math.cos(2 * math.pi * k / n),
                                        math.sin(t) * math.sin(2 * math.pi * k / n), math.cos(t)))
                     for k in range(n + 1)])


def load_round(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["image"], r["board"], int(r["pid"]), float(r["u"]), float(r["v"]), int(r["holdout"])))
    return rows


def load_lens(path):
    t = open(path).read().split()
    return t[0], [float(x) for x in t[1:]]


def map_path(p, maps):
    for a, b in maps:
        if p.startswith(a):
            return b + p[len(a):]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("--camera", default="left")
    ap.add_argument("--path-map", action="append", default=[],
                    help="prefix rewrite for image paths, e.g. /sim=/home/me/robocal_work/sim")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dump_dir, "viz")
    os.makedirs(os.path.join(out, "qualitative"), exist_ok=True)
    maps = [tuple(m.split("=", 1)) for m in a.path_map]

    files = glob.glob(os.path.join(a.dump_dir, f"{a.camera}_round*.csv"))
    rounds = sorted(int(re.search(r"_round(\d+)\.csv$", f).group(1)) for f in files)
    if not rounds:
        raise SystemExit(f"no {a.camera}_round*.csv in {a.dump_dir}")
    first = {}  # (image, board, pid) -> (round, u, v)
    per_round_total = []
    for k in rounds:
        rows = load_round(os.path.join(a.dump_dir, f"{a.camera}_round{k}.csv"))
        per_round_total.append(len(rows))
        for im, b, pid, u, v, ho in rows:
            key = (im, b, pid)
            if key not in first:
                first[key] = (k, u, v)
    final = {}
    for im, b, pid, u, v, ho in load_round(os.path.join(a.dump_dir, f"{a.camera}_round{rounds[-1]}.csv")):
        final[(im, b, pid)] = (u, v)
    model, p = load_lens(os.path.join(a.dump_dir, f"{a.camera}_round{rounds[-1]}_lens.txt"))
    theta_of = ThetaLookup(model, p)

    keys = list(first.keys())
    rnd = np.array([first[k][0] for k in keys])
    U = np.array([first[k][1] for k in keys])
    V = np.array([first[k][2] for k in keys])
    TH = theta_of(U, V)
    rlabel = {0: "detector"}
    for k in rounds[1:]:
        rlabel[k] = f"guided round {k}" + (" (+ rectified)" if k == 1 else "")

    # --- density maps ---
    cell = 40
    bx = np.arange(0, a.width + cell, cell)
    by = np.arange(0, a.height + cell, cell)
    panels = [("detector (round 0)", rnd == 0)] + \
             [(f"added in round {k}", rnd == k) for k in rounds[1:]] + [("final (all rounds)", np.ones_like(rnd, bool))]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), constrained_layout=True)
    axs = np.atleast_1d(axs)
    vmax = None
    for ax, (title, m) in zip(axs, panels):
        H, _, _ = np.histogram2d(U[m], V[m], bins=[bx, by])
        H = H.T
        vmax = vmax or max(1, H.max())
        im = ax.imshow(np.where(H > 0, H, np.nan), extent=[0, a.width, a.height, 0], cmap="viridis",
                       norm=LogNorm(vmin=1, vmax=max(vmax, H.max())), interpolation="nearest")
        for th, ls in ((30, ":"), (60, ":"), (90, "-"), (100, "--")):
            r = ring(model, p, th)
            ax.plot(r[:, 0], r[:, 1], ls, color="w", lw=0.8)
        ax.set_xlim(0, a.width)
        ax.set_ylim(a.height, 0)
        ax.set_title(f"{title}: {int(m.sum())} corners", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label=f"corners per {cell}x{cell} px cell")
    fig.suptitle(f"{a.camera}: corner density per guided round (rings 30/60/90/100 deg, final lens)", fontsize=11)
    fig.savefig(os.path.join(out, "density_rounds.png"), dpi=110)
    plt.close(fig)

    # --- off-axis angle histogram ---
    edges = np.arange(0, 120, 5)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    bottom = np.zeros(len(edges) - 1)
    for k in rounds:
        h, _ = np.histogram(TH[rnd == k], bins=edges)
        ax.bar(edges[:-1], h, width=5, align="edge", bottom=bottom, color=COL_MPL[min(k, 3)], label=rlabel[k])
        bottom += h
    ax.set_yscale("log")
    ax.set_xlabel("off-axis angle (deg)")
    ax.set_ylabel("corners")
    ax.legend(fontsize=8)
    ax.set_title("corners per 5 deg, by round of origin")
    h0, _ = np.histogram(TH[rnd == 0], bins=edges)
    ha, _ = np.histogram(TH, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        ax2.plot(edges[:-1] + 2.5, 100 * (1 - h0 / ha), "o-", color=COL_MPL[1])
    ax2.set_xlabel("off-axis angle (deg)")
    ax2.set_ylabel("% of corners added by guided rounds")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    ax2.set_title("share recovered by guided / rectified passes")
    fig.savefig(os.path.join(out, "theta_rounds.png"), dpi=110)
    plt.close(fig)

    # --- per board ---
    boards = sorted({k[1] for k in keys})
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(boards), 3.6), constrained_layout=True)
    bottom = np.zeros(len(boards))
    bidx = {b: i for i, b in enumerate(boards)}
    for k in rounds:
        c = np.zeros(len(boards))
        for key, r_ in zip(keys, rnd):
            if r_ == k:
                c[bidx[key[1]]] += 1
        ax.bar(boards, c, bottom=bottom, color=COL_MPL[min(k, 3)], label=rlabel[k])
        bottom += c
    ax.set_ylabel("corners")
    ax.legend(fontsize=8)
    ax.set_title("corners per board, by round of origin")
    fig.savefig(os.path.join(out, "board_rounds.png"), dpi=110)
    plt.close(fig)

    # --- qualitative frames ---
    by_image = collections.defaultdict(list)
    for key, r_, u, v, th in zip(keys, rnd, U, V, TH):
        by_image[key[0]].append((r_, u, v, th, key[1]))
    added = sorted(by_image, key=lambda im: -sum(1 for c in by_image[im] if c[0] > 0))
    widest = sorted(by_image, key=lambda im: -max(c[3] for c in by_image[im]))
    pick = []
    for im in added[: max(1, a.frames // 2)] + widest[: a.frames]:
        if im not in pick:
            pick.append(im)
        if len(pick) >= a.frames:
            break
    rings = {th: ring(model, p, th).astype(np.int32) for th in (30, 60, 90, 100)}
    tiles = []
    for n, im in enumerate(pick):
        img = cv2.imread(map_path(im, maps), cv2.IMREAD_COLOR)
        if img is None:
            print("missing image", im)
            continue
        if img.shape[0] != a.height:  # full-sensor frame of a cropped stream: centre crop
            y0 = (img.shape[0] - a.height) // 2
            img = img[y0:y0 + a.height]
        vis = (img * 0.75).astype(np.uint8)
        for th, r in rings.items():
            cv2.polylines(vis, [r], False, (200, 170, 110), 1, cv2.LINE_AA)
            cv2.putText(vis, f"{th}", tuple(r[45]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 170, 110), 1, cv2.LINE_AA)
        cs = by_image[im]
        for r_, u, v, th, b in sorted(cs, key=lambda c: c[0]):
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, COL_BGR[min(r_, 3)], 1, cv2.LINE_AA)
        add = [(u, v) for r_, u, v, th, b in cs if r_ > 0]
        cnt = collections.Counter(c[0] for c in cs)
        txt = "  ".join(f"{rlabel[k].split(' (')[0]}: {cnt.get(k, 0)}" for k in rounds)
        cv2.putText(vis, os.path.basename(im) + f"   max {max(c[3] for c in cs):.1f} deg", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if add:  # zoom inset (3x) around the densest cluster of added corners
            A = np.array(add)
            half = 70
            # densest 140 px window of added corners (cell histogram, 3x3 cell sum)
            H, xe, ye = np.histogram2d(A[:, 0], A[:, 1], bins=[np.arange(0, a.width + 47, 47),
                                                                np.arange(0, a.height + 47, 47)])
            Hs = cv2.boxFilter(H.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
            i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
            c0 = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
            x0 = int(np.clip(c0[0] - half, 0, a.width - 2 * half))
            y0 = int(np.clip(c0[1] - half, 0, a.height - 2 * half))
            crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
            z = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            for r_, u, v, th, b in cs:
                if x0 <= u < x0 + 2 * half and y0 <= v < y0 + 2 * half:
                    cv2.circle(z, (int((u - x0) * 3 + 1), int((v - y0) * 3 + 1)), 5, COL_BGR[min(r_, 3)], 2, cv2.LINE_AA)
            cv2.rectangle(vis, (x0, y0), (x0 + 2 * half, y0 + 2 * half), (255, 255, 255), 1)
            zx = a.width - z.shape[1] - 10 if c0[0] < a.width / 2 else 10
            zy = a.height - z.shape[0] - 10
            vis[zy:zy + z.shape[0], zx:zx + z.shape[1]] = z
            cv2.rectangle(vis, (zx, zy), (zx + z.shape[1], zy + z.shape[0]), (255, 255, 255), 2)
        path = os.path.join(out, "qualitative", f"{n:02d}_{os.path.splitext(os.path.basename(im))[0]}.jpg")
        cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles.append(cv2.resize(vis, (960, 540), interpolation=cv2.INTER_AREA))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(os.path.join(out, "qualitative_montage.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])

    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(f"camera {a.camera}, lens {model}\n")
        for k, tot in zip(rounds, per_round_total):
            m = rnd == k
            f.write(f"round {k} ({rlabel[k]}): {tot} corners in use, {int(m.sum())} first found here, "
                    f"{int((TH[m] > 75).sum())} beyond 75 deg, {int((TH[m] > 90).sum())} beyond 90 deg\n")
        lost = len(first) - len(final)
        f.write(f"found in some round but not in use at the end (dropped views / replaced): {lost}\n")
    print(open(os.path.join(out, "summary.txt")).read(), end="")
    print("wrote", out)


if __name__ == "__main__":
    main()
