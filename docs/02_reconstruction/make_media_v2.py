"""Figures and videos for the v2 part of the reconstruction report (SLAM, stereo, aisle coverage, lidar in the mesh).
Run with ~/.venvs/recon/bin/python from this directory.

    make_media_v2.py [stereo|forkcam|slam|coverage|views|lidarfill|error|ablation|fly|all]
"""

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

sys.path.insert(0, os.path.expanduser("~/isaacsim/tools/warehouse_amr"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slam  # noqa: E402
from make_media import crop_box, jpg, label, load_mesh, renderer  # noqa: E402
from recon_views import views  # noqa: E402

R = os.path.expanduser("~/warehouse_amr/recon")
V2 = f"{R}/v2"
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
RUNS2 = [f"data_s{s}" for s in range(21, 28)]
RUNS1 = ["data_s7", "data_s11", "data_s13", "data_s17"]
BEST = f"{V2}/out_final_slam"
UPPER = f"{V2}/out_gt_gt_lidar"
LIDAR_A, LIDAR_B = f"{V2}/out_slam_stereo", f"{V2}/out_slam_stereo_lidar"


def recs(path):
    return [json.loads(line) for line in open(f"{path}/captures.jsonl")]


def depth_vis(d, vmax=12.0):
    v = cv2.applyColorMap((np.clip(d / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    v[d == 0] = 0
    return v


def gt_scene():
    g = np.load(f"{V2}/data_s21/gt_mesh.npz")
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3c.Tensor(g["vertices"]), o3c.Tensor(g["faces"].astype(np.uint32)))
    return s, g


def gt_slice():
    g = np.load(f"{V2}/data_s21/gt_mesh.npz")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                     o3d.utility.Vector3iVector(g["faces"]))
    pts = np.asarray(mesh.sample_points_uniformly(1_500_000).points)
    return pts[(pts[:, 2] > 0.1) & (pts[:, 2] < 2.0)]


# ----------------------------------------------------------------------------------------------------------- stereo
def stereo():
    rs = recs(f"{V2}/data_s22")
    pick = next(r for r in rs if r["type"] == "carter" and r["capture"] > 400 and
                (cv2.imread(f"{V2}/data_s22/frames/{r['cams'][0]['stem']}_m.png", 0) > 0).mean() > 0.01)
    stem = f"{V2}/data_s22/frames/{pick['cams'][0]['stem']}"
    left, right = cv2.imread(stem + ".jpg"), cv2.imread(stem + "_r.jpg")
    gt = cv2.imread(stem + "_d.png", -1) / 1000.0
    st = cv2.imread(stem + "_s.png", -1) / 1000.0
    err = np.where((st > 0) & (gt > 0), np.abs(st - gt), 0)
    ev = cv2.applyColorMap((np.clip(err / 0.10, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ev[(st == 0) | (gt == 0)] = 0
    top = np.hstack([label(left.copy(), "left hawk"), label(right.copy(), "right hawk (15 cm baseline)")])
    bot = np.hstack([label(depth_vis(gt), "rendered depth (truth)"), label(depth_vis(st), "FoundationStereo depth")])
    row3 = np.hstack([label(ev, "|stereo - truth| (0-10 cm)"),
                      label(cv2.addWeighted(left, 0.5, depth_vis(st), 0.5, 0), "stereo over left image")])
    jpg(f"{M}/stereo_panel.jpg", cv2.resize(np.vstack([top, bot, row3]), None, fx=0.5, fy=0.5,
                                            interpolation=cv2.INTER_AREA))
    # Error vs range over a sample of frames from every run.
    bins = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
    errs = [[] for _ in bins[:-1]]
    valid = np.zeros((len(bins) - 1, 2))
    rng = np.random.default_rng(0)
    for run in RUNS2:
        rr = recs(f"{V2}/{run}")
        for r in rng.choice(rr, 40, replace=False):
            for c in r["cams"]:
                s = f"{V2}/{run}/frames/{c['stem']}"
                g = cv2.imread(s + "_d.png", -1) / 1000.0
                d = cv2.imread(s + "_s.png", -1)
                if d is None:
                    continue
                d = d / 1000.0
                m = (cv2.imread(s + "_m.png", 0) == 0) & (g > 0)
                for k in range(len(bins) - 1):
                    b = m & (g >= bins[k]) & (g < bins[k + 1])
                    v = b & (d > 0)
                    valid[k] += (b.sum(), v.sum())
                    if v.any():
                        e = np.abs(d[v] - g[v])
                        errs[k].append(e[rng.random(len(e)) < 0.05])
    mids = 0.5 * (bins[1:] + bins[:-1])
    med = np.array([np.median(np.concatenate(e)) if e else np.nan for e in errs])
    p90 = np.array([np.percentile(np.concatenate(e), 90) if e else np.nan for e in errs])
    fx, B = pick["cams"][0]["K"][0], pick["cams"][0]["baseline"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mids, med * 100, "o-", label="median error")
    ax.plot(mids, p90 * 100, "s--", label="90th percentile")
    ax.plot(mids, mids ** 2 / (fx * B) * 0.25 * 100, ":", color="0.4", label="0.25 px disparity error")
    ax.axvline(6.0, color="C3", alpha=0.5)
    ax.text(6.1, 1, "fusion cap\n(6 m)", color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("true depth (m)")
    ax.set_ylabel("depth error (cm)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title(f"Stereo depth error vs range ({len(RUNS2) * 40} captures; valid pixels "
                 f"{valid[:, 1].sum() / valid[:, 0].sum():.0%})")
    fig.savefig(f"{M}/stereo_error.png", dpi=90, bbox_inches="tight")
    json.dump({"bins": bins.tolist(), "median_m": med.tolist(), "p90_m": p90.tolist(),
               "valid": (valid[:, 1] / np.maximum(valid[:, 0], 1)).tolist()}, open(f"{M}/.stereo_stats.json", "w"))
    print("wrote stereo_error.png")


def forkcam():
    rs = recs(f"{V2}/data_s23")
    # The forklift is too wide for the rack aisles; it sees them from the cross aisle, facing in.
    r = [r for r in rs if r["type"] == "forklift" and 5 < r["gt_pose"][1] < 9 and -20 < r["gt_pose"][0] < 0
         and abs(slam.wrap(r["gt_pose"][3] - math.pi / 2)) < 0.4][3]
    imgs = []
    for c in r["cams"]:
        s = f"{V2}/data_s23/frames/{c['stem']}"
        imgs.append(label(cv2.imread(s + ".jpg"), f"forklift mast cam: {c['name']}"))
    jpg(f"{M}/forklift_cams.jpg", cv2.resize(np.hstack(imgs), None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))


# ------------------------------------------------------------------------------------------------------------- slam
def slam_fig():
    rs = []
    for run in RUNS2:
        for r in recs(f"{V2}/{run}"):
            r["key"] = f"{run}/{r['robot']}"
            rs.append(r)
    x = np.load(f"{V2}/slam_poses.npy")
    stages = json.load(open(f"{V2}/slam_poses_stages.json"))
    closures = np.load(f"{V2}/slam_poses_closures.npy")
    gt = np.array([r["gt_pose"] for r in rs])[:, [0, 1, 3]]
    est = np.array([r["est_pose"] for r in rs])
    odom = np.array([r["odom_pose"] for r in rs])
    # Dead reckoning per robot segment, as SLAM initializes.
    dr = np.zeros_like(x)
    order = sorted(range(len(rs)), key=lambda k: (rs[k]["key"], rs[k]["capture"]))
    prev = None
    for k in order:
        tag = (rs[k]["key"], rs[k]["segment"])
        if tag != prev:
            dr[k] = est[k]
        else:
            dr[k] = slam.compose(dr[kp][None], slam.between(odom[kp][None], odom[k][None]))[0]
        prev, kp = tag, k
    sl = gt_slice()
    fig, axes = plt.subplots(1, 2, figsize=(15, 11))
    for ax in axes:
        ax.scatter(sl[:, 0], sl[:, 1], s=0.02, c="0.8")
        ax.set_aspect("equal")
    sel = [k for k in order if rs[k]["key"] == "data_s24/carter_2"]
    seg = np.array([rs[k]["segment"] for k in sel])
    longest = np.bincount(seg).argmax()
    sel = [k for k, s in zip(sel, seg) if s == longest]
    ax = axes[0]
    for arr, col, lab in ((gt, "k", "truth"), (dr, "C3", "wheel odometry (dead reckoning)"),
                          (est, "C1", "robot's own localization"), (x, "C0", "pose-graph SLAM")):
        ax.plot(arr[sel, 0], arr[sel, 1], "-", color=col, lw=1.2 if col != "k" else 2.5, alpha=0.9, label=lab)
    ax.legend(loc="upper right")
    e_dr = np.linalg.norm(dr[sel, :2] - gt[sel, :2], axis=1)
    e_sl = np.linalg.norm(x[sel, :2] - gt[sel, :2], axis=1)
    ax.set_title(f"one Carter, {len(sel)} s: odometry drifts to {e_dr.max():.1f} m, SLAM max {e_sl.max() * 100:.1f} cm")
    ax = axes[1]
    a, b = closures[:, 0], closures[:, 1]
    cross = np.array([rs[i]["key"] != rs[j]["key"] for i, j in closures])
    rng = np.random.default_rng(0)
    pick = rng.random(len(closures)) < min(1.0, 6000 / len(closures))
    for m, col, lab in ((pick & ~cross, "C2", "same robot"), (pick & cross, "C4", "different robot or run")):
        segs = np.stack([x[a[m], :2], x[b[m], :2]], 1)
        from matplotlib.collections import LineCollection

        ax.add_collection(LineCollection(segs, colors=col, linewidths=0.5, alpha=0.6, label=lab))
    ax.plot(x[:, 0], x[:, 1], ".", ms=0.5, color="0.3")
    ax.legend(loc="upper right")
    ax.set_title(f"{len(closures):,} accepted loop closures ({cross.mean():.0%} between robots/runs; 6k drawn)")
    fig.savefig(f"{M}/slam_traj.jpg", dpi=75, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote slam_traj.jpg")
    # Error per stage.
    e_est = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    names = ["robot's own\nlocalization", "wheel odometry\n(dead reckoning)", "+ sequential scan\nmatching + priors",
             "+ loop closures\n(round 1)", "+ loop closures\n(final)"]
    keys = list(stages)
    med = [np.median(e_est), stages["odometry"]["median_m"], stages["odometry+seq"]["median_m"],
           stages[keys[2]]["median_m"], stages[keys[-1]]["median_m"]]
    p90 = [np.percentile(e_est, 90), stages["odometry"]["p90_m"], stages["odometry+seq"]["p90_m"],
           stages[keys[2]]["p90_m"], stages[keys[-1]]["p90_m"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, np.array(med) * 100, 0.4, label="median")
    ax.bar(xs + 0.2, np.array(p90) * 100, 0.4, label="90th percentile")
    for xi, m_, p_ in zip(xs, med, p90):
        ax.text(xi - 0.2, m_ * 100 * 1.1, f"{m_ * 100:.1f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, p_ * 100 * 1.1, f"{p_ * 100:.1f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_ylabel("position error (cm)")
    ax.legend()
    ax.set_title(f"Robot pose error, {len(rs):,} captures from 7 runs (no alignment)")
    fig.savefig(f"{M}/slam_stages.png", dpi=90, bbox_inches="tight")
    print("wrote slam_stages.png")


# --------------------------------------------------------------------------------------------------------- coverage
def coverage():
    sl = gt_slice()
    fig, axes = plt.subplots(1, 2, figsize=(14, 11))
    for ax, root, runs, title in ((axes[0], R, RUNS1, "v1: goals from the task generator"),
                                  (axes[1], V2, RUNS2, "v2: least-visited exploration goals")):
        xy = np.concatenate([np.array([r["gt_pose"][:2] for r in recs(f"{root}/{run}")]) for run in runs])
        lo, hi = sl[:, :2].min(0), sl[:, :2].max(0)
        H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[np.arange(lo[0], hi[0] + 1, 1.0),
                                                              np.arange(lo[1], hi[1] + 1, 1.0)])
        ax.scatter(sl[:, 0], sl[:, 1], s=0.02, c="0.6")
        im = ax.imshow(np.where(H.T > 0, H.T, np.nan), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap="magma_r", norm=matplotlib.colors.LogNorm(1, 300), alpha=0.85)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{(H > 0).sum()} of the 1 m cells visited, {len(xy):,} captures")
    fig.colorbar(im, ax=axes, shrink=0.6, label="captures per 1 m cell")
    fig.savefig(f"{M}/coverage_v2.jpg", dpi=75, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote coverage_v2.jpg")


# ----------------------------------------------------------------------------------------------------------- renders
def side_by_side():
    for v in ("overview", "aisle", "top"):
        a = cv2.imread(f"{BEST}/view_{v}_recon.png")
        b = cv2.imread(f"{BEST}/view_{v}_gt.png")
        img = np.hstack([label(a, "reconstruction (SLAM poses, stereo depth, lidar)"), label(b, "true geometry")])
        jpg(f"{M}/v2_recon_vs_gt_{v}.jpg", cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    jpg(f"{M}/v2_recon_aisle.jpg", cv2.imread(f"{BEST}/view_aisle_recon.png"), 88)


def lidarfill():
    """Share of the true surface reconstructed (within 10 cm), by height band."""
    _, g = gt_scene()
    gm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                   o3d.utility.Vector3iVector(g["faces"]))
    gp = np.asarray(gm.sample_points_uniformly(3_000_000).points)
    edges = [-1, 0.05, 3, 6, 9, 11]
    names = ["floor", "0.05-3 m", "3-6 m", "6-9 m", "roof (9.3 m)"]
    band = np.digitize(gp[:, 2], edges) - 1
    runs = (("out_gt_gt", "true poses, rendered depth to 8 m", "0.45"),
            ("out_slam_stereo", "SLAM, stereo to 6 m", "C3"),
            ("out_slam_stereo_lidar", "SLAM, stereo to 6 m + lidar fill", "C1"),
            ("out_final_slam", "SLAM, final stereo recipe + lidar fill", "C2"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for k, (d, lab, col) in enumerate(runs):
        m = o3d.io.read_triangle_mesh(f"{V2}/{d}/clean_mesh.ply")
        m.transform(np.array(json.load(open(f"{V2}/{d}/metrics.json")).get("align_T", np.eye(4).tolist())))
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        e = sc.compute_distance(o3c.Tensor(gp.astype(np.float32))).numpy()
        frac = [np.mean(e[band == i] < 0.10) * 100 for i in range(len(names))]
        xs = np.arange(len(names)) + (k - 1.5) * w
        ax.bar(xs, frac, w, color=col, label=lab)
        for x_, f_ in zip(xs, frac):
            ax.text(x_, f_ + 1, f"{f_:.0f}", ha="center", fontsize=7)
    share = [np.mean(band == i) * 100 for i in range(len(names))]
    ax.set_xticks(range(len(names)), [f"{n}\n({s_:.0f}% of the area)" for n, s_ in zip(names, share)], fontsize=8)
    ax.set_ylabel("true surface reconstructed within 10 cm (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(f"{M}/coverage_by_height.png", dpi=90, bbox_inches="tight")
    print("wrote coverage_by_height.png")


def error():
    scene, _ = gt_scene()
    mesh = load_mesh(f"{BEST}/clean_mesh.ply")
    T = np.array(json.load(open(f"{BEST}/metrics.json")).get("align_T", np.eye(4).tolist()))
    mesh.transform(T)
    d = scene.compute_distance(o3c.Tensor(np.asarray(mesh.vertices).astype(np.float32))).numpy()
    em = o3d.geometry.TriangleMesh(mesh)
    em.vertex_colors = o3d.utility.Vector3dVector(plt.cm.turbo(np.clip(d / 0.05, 0, 1))[:, :3])
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
    bar = np.repeat((plt.cm.turbo(np.linspace(0, 1, img.shape[1]))[:, :3] * 255).astype(np.uint8)[None, :, ::-1],
                    40, 0).copy()
    for k, t in enumerate(("0 cm", "1", "2", "3", "4", "5+ cm")):
        cv2.putText(bar, t, (int(k / 5 * (img.shape[1] - 80)) + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    jpg(f"{M}/v2_error_map.jpg", cv2.resize(np.vstack([img, bar]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


RUNS_TABLE = [
    ("out_gt_gt", "true poses, rendered depth 8 m"),
    ("out_gt_gt_lidar", "true poses, rendered depth 8 m, lidar"),
    ("out_est_stereo_lidar", "own localization, stereo 6 m, lidar"),
    ("out_slamnp_stereo_lidar", "SLAM without priors, stereo 6 m, lidar"),
    ("out_slam_stereo", "SLAM, stereo 6 m"),
    ("out_slam_stereo_lidar", "SLAM, stereo 6 m, lidar"),
    ("out_gt_stereo_lidar", "true poses, stereo 6 m, lidar"),
    ("out_final_gt", "true poses, final stereo recipe, lidar"),
    ("out_final_slam", "SLAM, final stereo recipe, lidar (best)"),
]


def ablation():
    rows = [(lab, json.load(open(f"{V2}/{d}/metrics.json"))) for d, lab in RUNS_TABLE
            if os.path.exists(f"{V2}/{d}/metrics.json")]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.5))
    y = np.arange(len(rows))
    keys = (("accuracy_within_5cm", "surface within 5 cm of the truth"),
            ("completeness_all_within_5cm", "building covered within 5 cm"),
            ("completeness_high_within_10cm", "above 3 m covered within 10 cm"))
    h = 0.27
    for k, (key, lab) in enumerate(keys):
        vals = [m[key] * 100 for _, m in rows]
        ax.barh(y + (k - 1) * h, vals, h, label=lab, color=f"C{k}")
        for yi, v in zip(y, vals):
            ax.text(v + 0.5, yi + (k - 1) * h, f"{v:.0f}", va="center", fontsize=7)
    ax.set_yticks(y, [lab for lab, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(f"{M}/ablation.png", dpi=90, bbox_inches="tight")
    print("wrote ablation.png")
    scene, _ = gt_scene()
    fig, ax = plt.subplots(figsize=(8, 4))
    for d, lab, col in (("out_gt_gt_lidar", "true poses, rendered depth", "C0"),
                        ("out_slam_stereo_lidar", "SLAM, stereo 6 m", "C3"),
                        ("out_final_slam", "SLAM, final stereo recipe", "C2")):
        if not os.path.exists(f"{V2}/{d}/clean_mesh.ply"):
            continue
        m = load_mesh(f"{V2}/{d}/clean_mesh.ply")
        m.transform(np.array(json.load(open(f"{V2}/{d}/metrics.json")).get("align_T", np.eye(4).tolist())))
        p = np.asarray(m.sample_points_uniformly(1_000_000).points).astype(np.float32)
        e = scene.compute_distance(o3c.Tensor(p)).numpy()
        ax.hist(np.clip(e * 100, 0, 30), bins=120, histtype="step", lw=1.6, density=True, color=col,
                label=f"{lab}: median {np.median(e) * 100:.2f} cm")
    ax.set_yscale("log")
    ax.set_xlabel("distance from reconstructed surface to true surface (cm, clipped at 30)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(f"{M}/v2_accuracy_hist.png", dpi=90, bbox_inches="tight")
    print("wrote v2_accuracy_hist.png")


def floaters():
    """Free-space filter on the 3k-capture tuning subset: stereo mesh before / after, colored by error."""
    scene, _ = gt_scene()
    imgs = []
    r, mat = renderer(1200, 800)
    for path, t in (("/tmp/sweep/s6/clean_mesh.ply", "stereo 6 m: floaters"),
                    ("/tmp/sweep/s6/fs_mesh.ply", "after the free-space filter")):
        m = load_mesh(path)
        e = scene.compute_distance(o3c.Tensor(np.asarray(m.vertices).astype(np.float32))).numpy()
        m.vertex_colors = o3d.utility.Vector3dVector(plt.cm.turbo(np.clip(e / 0.10, 0, 1))[:, :3])
        box = crop_box(m)
        eye, target, fov = views(box)["aisle"]
        eye = target + 0.5 * (eye - target)
        r.scene.clear_geometry()
        r.scene.add_geometry("m", m.crop(box), mat)
        r.setup_camera(fov, target, eye, [0, 0, 1])
        imgs.append(label(np.asarray(r.render_to_image())[..., ::-1].copy(), t + " (error 0-10 cm)"))
    jpg(f"{M}/floaters.jpg", cv2.resize(np.hstack(imgs), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def fly():
    mesh = load_mesh(f"{BEST}/clean_mesh.ply")
    box = crop_box(mesh)
    mesh = mesh.crop(box)
    lo, hi = box.get_min_bound(), box.get_max_bound()
    c = (lo + hi) / 2
    w, h = 1280, 720
    r, mat = renderer(w, h)
    r.scene.add_geometry("m", mesh, mat)
    out = f"{M}/v2_flythrough.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
                          "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    radius = 0.75 * max(hi[0] - lo[0], hi[1] - lo[1])
    shots = []
    for k in range(300):
        a = -math.pi / 2 + 2 * math.pi * k / 300
        shots.append((c + [radius * math.cos(a), radius * math.sin(a), 22.0], c, 50))
    y0, y1 = lo[1] + 4, hi[1] - 4
    for k in range(300):
        s = k / 299
        eye = np.array([c[0] - 3.0, y0 + s * (y1 - y0), 1.6])
        shots.append((eye, eye + [3.0 * math.cos(0.5 * math.sin(2 * math.pi * s)), 4.0, 0.2], 75))
    for eye, target, fov in shots:
        r.setup_camera(fov, np.asarray(target), np.asarray(eye), [0, 0, 1])
        p.stdin.write(np.asarray(r.render_to_image()).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    steps = {"stereo": stereo, "forkcam": forkcam, "slam": slam_fig, "coverage": coverage, "views": side_by_side,
             "lidarfill": lidarfill, "error": error, "ablation": ablation, "floaters": floaters, "fly": fly}
    for name in (sys.argv[1:] or ["all"]):
        for k, f in steps.items():
            if name in (k, "all"):
                f()
