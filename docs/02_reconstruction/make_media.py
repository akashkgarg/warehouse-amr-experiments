"""Figures and videos for the reconstruction report. Run with ~/.venvs/recon/bin/python from this directory.

    make_media.py [sensors|video|lidar|coverage|views|error|est|fly|icp|all]
"""

import glob
import json
import math
import os
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "32")
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.rendering as rendering

sys.path.insert(0, os.path.expanduser("~/isaacsim/tools/warehouse_amr"))
from recon_views import views  # noqa: E402

R = os.path.expanduser("~/warehouse_amr/recon")
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
RUNS = ["data_s7", "data_s11", "data_s13", "data_s17"]
os.makedirs(M, exist_ok=True)


def recs(run):
    return [json.loads(l) for l in open(f"{R}/{run}/captures.jsonl")]


def jpg(path, img, q=85):
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


def depth_vis(d):
    v = cv2.applyColorMap((np.clip(d / 12.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    v[d == 0] = 0
    return v


def frame_panels(stem):
    rgb = cv2.imread(stem + ".jpg")
    d = cv2.imread(stem + "_d.png", -1).astype(np.float32) / 1000
    m = cv2.imread(stem + "_m.png", 0) > 0
    ov = rgb.copy()
    ov[m] = (0.4 * ov[m] + 0.6 * np.array([40, 40, 230])).astype(np.uint8)
    return rgb, depth_vis(d), ov


def label(img, text):
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sensors():
    best = None
    for r in recs("data_s7"):
        if r["type"] != "carter":
            continue
        frac = sum((cv2.imread(f"{R}/data_s7/frames/{c['stem']}_m.png", 0) > 0).mean() for c in r["cams"])
        if 0.02 < frac < 0.12:
            best = r
            break
    rows = []
    for c in best["cams"]:
        rgb, dv, ov = frame_panels(f"{R}/data_s7/frames/{c['stem']}")
        rows.append(np.hstack([label(rgb, f"{c['name']} hawk RGB"), label(dv, "depth (0-12 m)"),
                               label(ov, "masked: people / robots")]))
    jpg(f"{M}/sensor_frames.jpg", cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    json.dump({"capture": best["capture"], "robot": best["robot"]}, open(f"{M}/.sensor_pick.json", "w"))


def video():
    rs = [r for r in recs("data_s7") if r["robot"] == "carter_1"][:300]
    out = f"{M}/robot_view.mp4"
    w, h = 1280, 400
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", "8", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
                          "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for r in rs:
        rgbs, deps = [], []
        for c in r["cams"]:
            rgb, dv, ov = frame_panels(f"{R}/data_s7/frames/{c['stem']}")
            rgbs.append(label(cv2.resize(ov, (320, 200)), c["name"]))
            deps.append(cv2.resize(dv, (320, 200)))
        grid = lambda a: np.vstack([np.hstack(a[:2]), np.hstack(a[2:])])
        frame = np.hstack([grid(rgbs), grid(deps)])
        cv2.putText(frame, f"carter_1  t={r['t']:.0f}s", (650, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


def gt_slice():
    g = np.load(f"{R}/data_s7/gt_mesh.npz")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                     o3d.utility.Vector3iVector(g["faces"]))
    pts = np.asarray(mesh.sample_points_uniformly(1_500_000).points)
    return mesh, pts[(pts[:, 2] > 0.1) & (pts[:, 2] < 2.0)]


def lidar():
    pick = json.load(open(f"{M}/.sensor_pick.json"))
    rs = recs("data_s7")
    car = rs[pick["capture"]]
    fork = next(r for r in rs if r["type"] == "forklift" and r["capture"] > 300)
    _, sl = gt_slice()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, r, title in ((axes[0], car, "Carter XT-32 sweep (32 channels)"),
                         (axes[1], fork, "Forklift planar lidar (3 layers)")):
        z = np.load(f"{R}/data_s7/lidar/{r['capture']:06d}_{r['robot']}.npz")
        p, dyn = z["points"], z["dynamic"]
        x, y = r["gt_pose"][:2]
        near = np.hypot(sl[:, 0] - x, sl[:, 1] - y) < 22
        ax.scatter(sl[near, 0], sl[near, 1], s=0.05, c="0.8")
        keep = np.hypot(p[:, 0] - x, p[:, 1] - y) < 22
        sc = ax.scatter(p[keep & ~dyn, 0], p[keep & ~dyn, 1], s=0.3, c=p[keep & ~dyn, 2], cmap="viridis", vmin=0,
                        vmax=4)
        ax.scatter(p[keep & dyn, 0], p[keep & dyn, 1], s=4, c="red", label="dynamic hits (dropped)")
        ax.plot(x, y, "k^", ms=10, label="robot")
        ax.set_aspect("equal")
        ax.set_xlim(x - 22, x + 22)
        ax.set_ylim(y - 22, y + 22)
        ax.set_title(f"{title}: {len(p)} returns")
        ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=axes, label="hit height (m)", shrink=0.8)
    fig.savefig(f"{M}/lidar_sweeps.jpg", dpi=90, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote lidar_sweeps.jpg")


def coverage():
    _, sl = gt_slice()
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.scatter(sl[:, 0], sl[:, 1], s=0.02, c="0.75")
    colors = plt.cm.tab10(np.arange(4))
    for run, col in zip(RUNS, colors):
        rs = recs(run)
        for robot in sorted({r["robot"] for r in rs}):
            xy = np.array([r["gt_pose"][:2] for r in rs if r["robot"] == robot])
            # Respawns are jumps; break the line there.
            jump = np.r_[False, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 3.0]
            for seg in np.split(xy, np.nonzero(jump)[0]):
                ax.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=0.6, alpha=0.7)
        ax.plot([], [], color=col, label=f"run {run[6:]} ({len(rs)} captures)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Where the fleet drove (4 runs x 10 min, 5 robots each)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.savefig(f"{M}/coverage.jpg", dpi=80, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote coverage.jpg")


def side_by_side():
    for v in ("overview", "aisle", "top"):
        a = cv2.imread(f"{R}/out_gt/view_{v}_recon.png")
        b = cv2.imread(f"{R}/out_gt/view_{v}_gt.png")
        img = np.hstack([label(a, "reconstruction"), label(b, "true geometry")])
        jpg(f"{M}/recon_vs_gt_{v}.jpg", cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    jpg(f"{M}/recon_aisle.jpg", cv2.imread(f"{R}/out_gt/view_aisle_recon.png"), 88)


def renderer(w=1600, h=1000):
    r = rendering.OffscreenRenderer(w, h)
    r.scene.set_background([1, 1, 1, 1])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    return r, mat


def load_mesh(path):
    m = o3d.io.read_triangle_mesh(path)
    m.compute_vertex_normals()
    return m


def crop_box(mesh):
    b = mesh.get_axis_aligned_bounding_box()
    lo, hi = b.get_min_bound(), b.get_max_bound()
    return o3d.geometry.AxisAlignedBoundingBox(lo - [0.5, 0.5, 0.5], [hi[0] + 0.5, hi[1] + 0.5, min(hi[2], 6.0)])


def gt_scene():
    g = np.load(f"{R}/data_s7/gt_mesh.npz")
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3c.Tensor(g["vertices"]), o3c.Tensor(g["faces"].astype(np.uint32)))
    return s


def error():
    mesh = load_mesh(f"{R}/out_gt/clean_mesh.ply")
    scene = gt_scene()
    d = scene.compute_distance(o3c.Tensor(np.asarray(mesh.vertices).astype(np.float32))).numpy()
    cmap = plt.cm.turbo(np.clip(d / 0.05, 0, 1))[:, :3]
    em = o3d.geometry.TriangleMesh(mesh)
    em.vertex_colors = o3d.utility.Vector3dVector(cmap)
    box = crop_box(mesh)
    r, mat = renderer()
    imgs = []
    for v in ("overview", "aisle"):
        eye, target, fov = views(box)[v]
        r.scene.clear_geometry()
        r.scene.add_geometry("m", em.crop(box), mat)
        r.setup_camera(fov, target, eye, [0, 0, 1])
        imgs.append(np.asarray(r.render_to_image())[..., ::-1])
    img = np.hstack(imgs)
    bar = (plt.cm.turbo(np.linspace(0, 1, img.shape[1]))[:, :3] * 255).astype(np.uint8)[None, :, ::-1]
    bar = np.repeat(bar, 40, 0).copy()
    for k, t in enumerate(("0 cm", "1", "2", "3", "4", "5+ cm")):
        cv2.putText(bar, t, (int(k / 5 * (img.shape[1] - 80)) + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2, cv2.LINE_AA)
    jpg(f"{M}/error_map.jpg", cv2.resize(np.vstack([img, bar]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    # Accuracy histograms for both pose sources (sampled surface points).
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, label_, col in (("out_gt", "true poses", "C0"), ("out_est", "robots' own pose estimates", "C3")):
        m = load_mesh(f"{R}/{tag}/clean_mesh.ply")
        p = np.asarray(m.sample_points_uniformly(1_000_000).points).astype(np.float32)
        e = scene.compute_distance(o3c.Tensor(p)).numpy()
        ax.hist(np.clip(e * 100, 0, 30), bins=120, histtype="step", lw=1.6, density=True, color=col,
                label=f"{label_}: median {np.median(e) * 100:.1f} cm")
    ax.set_yscale("log")
    ax.set_xlabel("distance from reconstructed surface to true surface (cm, clipped at 30)")
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(f"{M}/accuracy_hist.png", dpi=90, bbox_inches="tight")
    print("wrote accuracy_hist.png")


def est():
    gt_m = load_mesh(f"{R}/out_gt/clean_mesh.ply")
    est_m = load_mesh(f"{R}/out_est/clean_mesh.ply")
    box = crop_box(gt_m)
    eye, target, fov = views(box)["aisle"]
    # Closer in, so the doubled rack uprights are visible.
    eye = target + 0.45 * (eye - target)
    r, mat = renderer(1200, 800)
    imgs = []
    for m, t in ((gt_m, "true poses"), (est_m, "robots' own pose estimates (~12 cm error)")):
        r.scene.clear_geometry()
        r.scene.add_geometry("m", m.crop(box), mat)
        r.setup_camera(fov, target, eye, [0, 0, 1])
        imgs.append(label(np.asarray(r.render_to_image())[..., ::-1].copy(), t))
    jpg(f"{M}/est_vs_gt.jpg", cv2.resize(np.hstack(imgs), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))


def fly():
    mesh = load_mesh(f"{R}/out_gt/clean_mesh.ply")
    box = crop_box(mesh)
    mesh = mesh.crop(box)
    lo, hi = box.get_min_bound(), box.get_max_bound()
    c = (lo + hi) / 2
    w, h = 1280, 720
    r, mat = renderer(w, h)
    r.scene.add_geometry("m", mesh, mat)
    out = f"{M}/flythrough.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
                          "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    radius = 0.75 * max(hi[0] - lo[0], hi[1] - lo[1])
    shots = []
    for k in range(300):  # orbit
        a = -math.pi / 2 + 2 * math.pi * k / 300
        shots.append((c + [radius * math.cos(a), radius * math.sin(a), 22.0], c, 50))
    # Low pass along the long axis between the racks and the open floor.
    y0, y1 = lo[1] + 4, hi[1] - 4
    for k in range(240):
        s = k / 239
        y = y0 + s * (y1 - y0)
        eye = np.array([c[0] - 3.0, y, 1.6])
        shots.append((eye, eye + [3.0 * math.cos(0.5 * math.sin(2 * math.pi * s)), 4.0, -0.3], 70))
    for eye, target, fov in shots:
        r.setup_camera(fov, np.asarray(target), np.asarray(eye), [0, 0, 1])
        p.stdin.write(np.asarray(r.render_to_image()).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


def icp():
    small = [0.124, 0.107, 0.066, 0.039, 0.031]
    small_p90 = [0.212, 0.240, 0.183, 0.137, 0.101]
    full = [0.119, 0.118, 0.116]
    full_p90 = [0.215, 0.233, 0.237]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(small)), small, "o-", color="C0", label="600 captures, median")
    ax.plot(range(len(small)), small_p90, "o--", color="C0", alpha=0.5, label="600 captures, p90")
    ax.plot(range(len(full)), full, "s-", color="C3", label="all 11,815 captures, median")
    ax.plot(range(len(full)), full_p90, "s--", color="C3", alpha=0.5, label="all 11,815 captures, p90")
    ax.set_xticks(range(5), ["start", "pass 1", "pass 2", "pass 3", "pass 4"])
    ax.set_ylabel("robot position error (m)")
    ax.set_ylim(0, 0.3)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Lidar ICP pose refinement")
    fig.savefig(f"{M}/icp_refinement.png", dpi=90, bbox_inches="tight")
    print("wrote icp_refinement.png")


if __name__ == "__main__":
    steps = {"sensors": sensors, "video": video, "lidar": lidar, "coverage": coverage, "views": side_by_side,
             "error": error, "est": est, "fly": fly, "icp": icp}
    for name in (sys.argv[1:] or ["all"]):
        for k, f in steps.items():
            if name in (k, "all"):
                f()
