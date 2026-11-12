"""Figures and videos for the v3 reconstruction report (place recognition, kidnapped robots, left-right stereo,
forklift mast lidar, pole landmarks). Run with ~/.venvs/recon/bin/python from this directory.

    make_media_v3.py [mast|sc|slam|kidnap|stereo|poles|views|height|error|ablation|compare|fly|all]
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
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.expanduser("~/isaacsim/tools/warehouse_amr"))
sys.path.insert(0, os.path.expanduser("~/warehouse_amr/recon/report"))
import recon  # noqa: E402
import slam  # noqa: E402
from make_media import crop_box, jpg, label, load_mesh, renderer  # noqa: E402
from recon_views import views  # noqa: E402

R = os.path.expanduser("~/warehouse_amr/recon")
V2, V3 = f"{R}/v2", f"{R}/v3"
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(M, exist_ok=True)
RUNS3 = [f"{V3}/data_s{s}" for s in range(31, 38)]
RUNS2 = [f"{V2}/data_s{s}" for s in range(21, 28)]
BEST = f"{V3}/out_final"


def gt_scene():
    g = np.load(f"{V3}/data_s31/gt_mesh.npz")
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3c.Tensor(g["vertices"]), o3c.Tensor(g["faces"].astype(np.uint32)))
    return s, g


def gt_slice():
    g = np.load(f"{V3}/data_s31/gt_mesh.npz")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                     o3d.utility.Vector3iVector(g["faces"]))
    pts = np.asarray(mesh.sample_points_uniformly(1_500_000).points)
    return pts[(pts[:, 2] > 0.1) & (pts[:, 2] < 2.0)]


def floor(ax, sl):
    ax.scatter(sl[:, 0], sl[:, 1], s=0.02, c="0.8")
    ax.set_aspect("equal")


def load_recs():
    return recon.load(RUNS3)


def chain_ids(recs):
    order = sorted(range(len(recs)), key=lambda k: (recs[k]["key"], recs[k]["capture"]))
    chain = np.zeros(len(recs), int)
    prev, cid = None, -1
    for k in order:
        tag = (recs[k]["key"], recs[k]["segment"])
        if tag != prev:
            cid, prev = cid + 1, tag
        chain[k] = cid
    return chain, order


def align2d(x, gt, trim: float = 0.3):
    """Best rigid 2D transform of x onto gt (gauge-free error), refit on the captures within ``trim`` so a few
    misplaced segments do not tilt the whole map."""
    inl = np.ones(len(x), bool)
    for _ in range(5):
        a, b = x[inl, :2], gt[inl, :2]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1
            Rm = (U @ Vt).T
        out = x.copy()
        out[:, :2] = (x[:, :2] - ma) @ Rm.T + mb
        out[:, 2] = slam.wrap(x[:, 2] + math.atan2(Rm[1, 0], Rm[0, 0]))
        inl = np.linalg.norm(out[:, :2] - gt[:, :2], axis=1) < trim
    return out


# ------------------------------------------------------------------------------------------------------------- mast
def mast():
    recs = recon.load([RUNS3[2]])
    fk = [r for r in recs if r["type"] == "forklift" and 4 < r["gt_pose"][1] < 10][40]
    ct = min((r for r in recs if r["type"] == "carter"),
             key=lambda r: np.hypot(r["gt_pose"][0] - fk["gt_pose"][0], r["gt_pose"][1] - fk["gt_pose"][1]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, r, src, title in ((axes[0], ct, "auto", "Carter XT-32 (0.5 m up, -16..+15 deg)"),
                              (axes[1], fk, "mast", "forklift mast lidar (2.3 m up, +-45 deg)")):
        p, _ = slam.noisy_world_sweep(r["dir"], r, source=src)
        o = np.array(r["mast_lidar_origin"] if src == "mast" else r["lidar_origin"])
        d = np.hypot(p[:, 0] - o[0], p[:, 1] - o[1])
        ax.scatter(d, p[:, 2], s=0.3, c=p[:, 2], cmap="viridis", vmin=0, vmax=9.5)
        ax.axhline(9.3, color="C3", lw=0.8)
        ax.text(0.5, 9.45, "roof", color="C3", fontsize=8)
        ax.set_xlim(0, 30)
        ax.set_ylim(-0.3, 10.5)
        ax.set_xlabel("horizontal distance from the sensor (m)")
        ax.set_ylabel("height (m)")
        hi = np.mean(p[:, 2] > 6) * 100
        ax.set_title(f"{title}\n{len(p):,} returns, {hi:.0f}% above 6 m")
        ax.grid(alpha=0.3)
    fig.savefig(f"{M}/mast_lidar.png", dpi=80, bbox_inches="tight")
    print("wrote mast_lidar.png")


# ------------------------------------------------------------------------------------- scan context (descriptor demo)
def sc_fig():
    recs = load_recs()
    st = np.load(f"{V3}/slam_pr_poses_closures.npy")
    gt = np.array([r["gt_pose"] for r in recs])[:, [0, 1, 3]]
    rng = np.random.default_rng(3)
    # A cross-robot place-recognition match between two different runs with a big heading difference.
    best = None
    for a, b in st[rng.permutation(len(st))[:4000]]:
        ra, rb = recs[a], recs[b]
        if ra["key"].split("/")[0] == rb["key"].split("/")[0] or ra["type"] != "carter" or rb["type"] != "carter":
            continue
        dyaw = abs(slam.wrap(gt[a, 2] - gt[b, 2]))
        if dyaw > 1.5 and np.linalg.norm(gt[a, :2] - gt[b, :2]) < 1.0:
            best = (a, b)
            break
    a, b = best
    far = int(np.argmax(np.linalg.norm(gt[:, :2] - gt[a, :2], axis=1) > 15) + rng.integers(0, 50))
    while recs[far]["type"] != "carter":
        far += 1
    scs = []
    for k in (a, b, far):
        p, _ = slam.noisy_world_sweep(recs[k]["dir"], recs[k])
        scs.append(slam.scan_context(slam.to_local(p, recs[k]["gt_pose"])))

    def dist(A, B):
        An = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-9)
        Bn = B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-9)
        ok_a, ok_b = A.any(0), B.any(0)
        out = []
        for s in range(slam.SC_SECTORS):
            Bs = np.roll(Bn, s, 1)
            v = ok_a & np.roll(ok_b, s)
            out.append(1 - (An * Bs).sum(0)[v].mean() if v.any() else 1)
        return np.array(out)

    d_match, d_far = dist(scs[0], scs[1]), dist(scs[0], scs[2])
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), gridspec_kw={"height_ratios": [1, 1]})
    for ax, scd, t in ((axes[0, 0], scs[0], f"{recs[a]['key']} #{recs[a]['capture']}"),
                       (axes[0, 1], scs[1], f"{recs[b]['key']} #{recs[b]['capture']} (same spot, "
                                            f"{math.degrees(abs(slam.wrap(gt[a, 2] - gt[b, 2]))):.0f} deg turned)"),
                       (axes[1, 0], scs[2], f"{recs[far]['key']} #{recs[far]['capture']} (15+ m away)")):
        ax.imshow(scd, aspect="auto", cmap="magma", vmin=0, vmax=slam.SC_ZMAX, origin="lower",
                  extent=[-180, 180, 0, slam.SC_RANGE])
        ax.set_xlabel("bearing (deg)")
        ax.set_ylabel("range (m)")
        ax.set_title(t, fontsize=9)
    ax = axes[1, 1]
    sh = np.arange(slam.SC_SECTORS) * 360 / slam.SC_SECTORS
    ax.plot(sh, d_match, label=f"same place: best {d_match.min():.2f}")
    ax.plot(sh, d_far, label=f"other place: best {d_far.min():.2f}")
    ax.axhline(0.25, color="0.4", ls=":", label="candidate threshold")
    ax.set_xlabel("column shift = heading difference (deg)")
    ax.set_ylabel("descriptor distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{M}/scan_context.png", dpi=80, bbox_inches="tight")
    print("wrote scan_context.png")


# ------------------------------------------------------------------------------------------------------------- slam
def mode_stats(mode, recs, gt, typ, stage=None):
    path = f"{V3}/{mode}_poses_stage_poses.npz"
    st = np.load(path)
    loc = np.load(f"{V3}/{mode}_poses_localized.npy")
    out = {}
    for name in st.files:
        x = st[name]
        xa = x.copy()
        xa[loc] = align2d(x[loc], gt[loc])
        e = np.linalg.norm(xa[:, :2] - gt[:, :2], axis=1)
        row = {"localized": float(loc.mean())}
        for t in ("all", "carter", "forklift"):
            m = loc & ((typ == t) if t != "all" else True)
            row[t] = {"median": float(np.median(e[m])), "p90": float(np.percentile(e[m], 90)),
                      "max": float(e[m].max()), "over10cm": float(np.mean(e[m] > 0.10)),
                      "over50cm": float(np.mean(e[m] > 0.5))}
        out[name] = row
    return out


def slam_fig():
    recs = load_recs()
    gt = np.array([r["gt_pose"] for r in recs])[:, [0, 1, 3]]
    typ = np.array([r["type"] for r in recs])
    stats = {m: mode_stats(m, recs, gt, typ) for m in ("slam", "slam_pr", "slam_kidnap")
             if os.path.exists(f"{V3}/{m}_poses_stage_poses.npz")}
    lm = {}
    for m in ("slam_pr_lm", "slam_kidnap_lm"):
        p = f"{V3}/slam_try1/{m}_poses_stage_poses.npz"
        if os.path.exists(p):
            st = np.load(p)
            loc = np.load(f"{V3}/slam_try1/{m}_poses_localized.npy")
            for name in ("closures 2", "landmarks"):
                xa = st[name].copy()
                xa[loc] = align2d(st[name][loc], gt[loc])
                e = np.linalg.norm(xa[:, :2] - gt[:, :2], axis=1)[loc & (typ == "carter")]
                lm[f"{m} {name}"] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                                     "max": float(e.max())}
    json.dump({"modes": stats, "landmarks": lm}, open(f"{M}/.slam_stats.json", "w"), indent=1)
    # Error after each stage, per prior mode (all robots, rigidly aligned).
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = {"slam": "priors on every pose", "slam_pr": "segment starts only + place recognition",
             "slam_kidnap": "one anchor per robot + place recognition"}
    axis = ["odometry", "odometry+seq", "place recognition", "closures 0", "closures 1", "closures 2"]
    ticks = ["wheel odometry", "+ sequential\nscan matching", "+ place\nrecognition", "+ loop closures\n(round 1)",
             "(round 2)", "(round 3)"]
    for k, (m, st) in enumerate(stats.items()):
        stg = [s_ for s_ in axis if s_ in st]
        xs = [axis.index(s_) for s_ in stg]
        ax.plot(xs, [st[s_]["all"]["median"] * 100 for s_ in stg], "o-", color=f"C{k}", label=f"{names[m]}")
        ax.plot(xs, [st[s_]["all"]["p90"] * 100 for s_ in stg], "s--", color=f"C{k}", alpha=0.5)
    ax.plot([], [], "k-o", label="median")
    ax.plot([], [], "k--s", alpha=0.5, label="90th percentile")
    ax.set_yscale("log")
    ax.set_xticks(range(len(axis)), ticks, fontsize=8)
    ax.set_ylabel("position error (cm, rigidly aligned)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Robot pose error per stage, {len(recs):,} captures from 7 runs")
    fig.savefig(f"{M}/slam_modes.png", dpi=90, bbox_inches="tight")
    print("wrote slam_modes.png")
    # Trajectories without priors: before / after place recognition.
    sl = gt_slice()
    st = np.load(f"{V3}/slam_pr_poses_stage_poses.npz")
    clo = np.load(f"{V3}/slam_pr_poses_closures.npy")
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    for ax, name, t in ((axes[0], "odometry+seq", "no priors, odometry + scan matching only"),
                        (axes[1], st.files[-1], "after place recognition and loop closures")):
        floor(ax, sl)
        x = align2d(st[name], gt) if name != "odometry+seq" else st[name]
        e = np.linalg.norm(x[:, :2] - gt[:, :2], axis=1)
        sc_ = ax.scatter(x[:, 0], x[:, 1], s=0.4, c=np.clip(e, 0, 2), cmap="turbo", vmin=0, vmax=2)
        ax.set_title(f"{t}\nmedian error {np.median(e) * 100:.1f} cm, {np.mean(e > 0.5):.1%} more than 0.5 m off")
    fig.colorbar(sc_, ax=axes, shrink=0.5, label="position error (m, clipped at 2)")
    fig.savefig(f"{M}/pr_before_after.jpg", dpi=70, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote pr_before_after.jpg")
    # The place-recognition closures themselves.
    fig, ax = plt.subplots(figsize=(8, 10))
    floor(ax, sl)
    x = st[st.files[-1]]
    n_pr = int(json.load(open(f"{M}/.slam_counts.json"))["slam_pr_place"]) if os.path.exists(
        f"{M}/.slam_counts.json") else min(len(clo), 12000)
    pr = clo[:n_pr]
    cross = np.array([recs[i]["key"] != recs[j]["key"] for i, j in pr])
    for msk, col, lab in ((~cross, "C2", "same robot"), (cross, "C4", "different robot or run")):
        ax.add_collection(LineCollection(np.stack([x[pr[msk, 0], :2], x[pr[msk, 1], :2]], 1), colors=col,
                                         linewidths=0.4, alpha=0.5, label=lab))
    ax.plot(x[:, 0], x[:, 1], ".", ms=0.4, color="0.3")
    ax.legend(loc="upper right")
    ax.set_title(f"{len(pr):,} place-recognition closures ({cross.mean():.0%} across robots/runs)")
    fig.savefig(f"{M}/pr_closures.jpg", dpi=70, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote pr_closures.jpg")


def kidnap():
    recs = load_recs()
    gt = np.array([r["gt_pose"] for r in recs])[:, [0, 1, 3]]
    chain, order = chain_ids(recs)
    st = np.load(f"{V3}/slam_kidnap_poses_stage_poses.npz")
    loc = np.load(f"{V3}/slam_kidnap_poses_localized.npy")
    x = st[st.files[-1]].copy()
    x[loc] = align2d(x[loc], gt[loc])
    first_key = {}
    for k in order:
        first_key.setdefault(recs[k]["key"], chain[k])
    anchored = np.isin(chain, list(first_key.values()))
    e = np.linalg.norm(x[:, :2] - gt[:, :2], axis=1)
    sl = gt_slice()
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    ax = axes[0]
    floor(ax, sl)
    ax.plot(gt[anchored, 0], gt[anchored, 1], ".", ms=0.6, color="C0", label="anchored (first segment per robot)")
    ax.plot(gt[~anchored & loc, 0], gt[~anchored & loc, 1], ".", ms=0.6, color="C2",
            label="respawned, found by place recognition")
    ax.plot(gt[~loc, 0], gt[~loc, 1], ".", ms=2, color="C3", label="respawned, never placed")
    ax.legend(loc="upper right", markerscale=8)
    n_ch = chain.max() + 1
    ax.set_title(f"{n_ch} robot segments: {len(first_key)} anchored, "
                 f"{len(set(chain[loc & ~anchored]))} relocalized, {len(set(chain[~loc]))} not placed (true poses)")
    ax = axes[1]
    floor(ax, sl)
    sc_ = ax.scatter(x[loc, 0], x[loc, 1], s=0.4, c=np.clip(e[loc] * 100, 0, 5), cmap="turbo", vmin=0, vmax=5)
    fig.colorbar(sc_, ax=ax, shrink=0.5, label="position error (cm, clipped at 5)")
    ax.set_title(f"SLAM result (rigidly aligned): median {np.median(e[loc]) * 1000:.1f} mm, "
                 f"{np.mean(e[loc] > 0.5):.2%} more than 0.5 m off")
    fig.savefig(f"{M}/kidnap.jpg", dpi=70, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote kidnap.jpg")


# ----------------------------------------------------------------------------------------------------------- stereo
def depth_vis(d, vmax=12.0):
    v = cv2.applyColorMap((np.clip(d / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    v[d == 0] = 0
    return v


def stereo():
    rs = recon.load([RUNS3[3]])
    best, score = None, -1
    for r in rs[200:1400:25]:
        for c in r["cams"]:
            s = f"{RUNS3[3]}/frames/{c['stem']}"
            g = cv2.imread(s + "_d.png", -1)
            a, b = cv2.imread(s + "_s.png", -1), cv2.imread(s + "_sl.png", -1)
            if g is None or a is None or b is None:
                continue
            g, a, b = g / 1000.0, a / 1000.0, b / 1000.0
            bad_a = (a > 0) & (g > 0) & (g < 6) & (np.abs(a - g) > 0.05)
            fixed = bad_a & (b == 0)
            if fixed.sum() > score:
                best, score = s, fixed.sum()
    s = best
    left = cv2.imread(s + ".jpg")
    g, a, b = (cv2.imread(s + suf, -1) / 1000.0 for suf in ("_d.png", "_s.png", "_sl.png"))

    def err(d):
        e = np.where((d > 0) & (g > 0), np.abs(d - g), 0)
        v = cv2.applyColorMap((np.clip(e / 0.10, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        v[(d == 0) | (g == 0)] = 0
        return v

    rej = left.copy()
    m = (a > 0) & (b == 0)
    rej[m] = (0.4 * rej[m] + 0.6 * np.array([60, 60, 255])).astype(np.uint8)
    top = np.hstack([label(left.copy(), "left image"), label(rej, "dropped by the left-right check (red)")])
    bot = np.hstack([label(err(a), "|plain stereo - truth| (0-10 cm)"),
                     label(err(b), "|left-right checked - truth| (0-10 cm)")])
    jpg(f"{M}/stereo_lr.jpg", cv2.resize(np.vstack([top, bot]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    print("wrote stereo_lr.jpg")
    # Bad pixels and density by range, plain vs checked, over a sample of every run.
    bins = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    acc = {k: np.zeros((len(bins) - 1, 3)) for k in ("_s", "_sl")}
    rng = np.random.default_rng(0)
    for run in RUNS3:
        rr = recon.load([run])
        for r in rng.choice(rr, 30, replace=False):
            for c in r["cams"]:
                p = f"{run}/frames/{c['stem']}"
                gd = cv2.imread(p + "_d.png", -1)
                mk = cv2.imread(p + "_m.png", 0)
                if gd is None:
                    continue
                gd = gd / 1000.0
                valid = (gd > 0) & (mk == 0)
                for suf in acc:
                    d = cv2.imread(p + suf + ".png", -1)
                    if d is None:
                        continue
                    d = d / 1000.0
                    for k in range(len(bins) - 1):
                        bm = valid & (gd >= bins[k]) & (gd < bins[k + 1])
                        kept = bm & (d > 0)
                        acc[suf][k] += (bm.sum(), kept.sum(), (kept & (np.abs(d - gd) > 0.05)).sum())
    mids = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for suf, lab, col in (("_s", "plain", "C1"), ("_sl", "left-right checked", "C0")):
        a_ = acc[suf]
        axes[0].plot(mids, a_[:, 2] / np.maximum(a_[:, 1], 1) * 100, "o-", color=col, label=lab)
        axes[1].plot(mids, a_[:, 1] / np.maximum(a_[:, 0], 1) * 100, "o-", color=col, label=lab)
    axes[0].set_ylabel("pixels off by more than 5 cm (%)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("pixels with a depth (%)")
    for ax in axes:
        ax.set_xlabel("true depth (m)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        ax.axvline(4, color="0.5", ls=":")
    fig.savefig(f"{M}/stereo_lr_range.png", dpi=90, bbox_inches="tight")
    near = slice(0, 6)
    stats = {suf: {"bad_le6m": float(a_[near, 2].sum() / a_[near, 1].sum()),
                   "density_le6m": float(a_[near, 1].sum() / a_[near, 0].sum()),
                   "bad_le4m": float(a_[:4, 2].sum() / a_[:4, 1].sum()),
                   "density_le4m": float(a_[:4, 1].sum() / a_[:4, 0].sum())} for suf, a_ in acc.items()}
    json.dump(stats, open(f"{M}/.stereo_lr_stats.json", "w"), indent=1)
    print("wrote stereo_lr_range.png", stats)


# ------------------------------------------------------------------------------------------------------------ poles
def poles():
    recs = [r for r in recon.load([RUNS3[1]]) if r["type"] == "carter"][::4]
    W = []
    for r in recs:
        p, _ = slam.noisy_world_sweep(r["dir"], r)
        d = slam.detect_poles(slam.to_local(p, r["gt_pose"]))
        x, y, _, yaw = r["gt_pose"]
        c, s = math.cos(yaw), math.sin(yaw)
        W.append(np.c_[x + c * d[:, 0] - s * d[:, 1], y + s * d[:, 0] + c * d[:, 1]])
    W = np.concatenate(W)
    sl = gt_slice()
    fig, ax = plt.subplots(figsize=(8, 10))
    floor(ax, sl)
    ax.plot(W[:, 0], W[:, 1], "+", ms=3, color="C3", alpha=0.4)
    ax.set_title(f"pole detections from {len(recs)} Carter sweeps, placed with the true poses\n"
                 f"({len(W) / len(recs):.1f} per sweep)")
    fig.savefig(f"{M}/poles.jpg", dpi=70, bbox_inches="tight", pil_kwargs={"quality": 85})
    print("wrote poles.jpg")


# ----------------------------------------------------------------------------------------------------------- renders
def side_by_side():
    for v in ("overview", "aisle", "top"):
        a = cv2.imread(f"{BEST}/view_{v}_recon.png")
        b = cv2.imread(f"{BEST}/view_{v}_gt.png")
        img = np.hstack([label(a, "v3 reconstruction (no priors, stereo, lidar)"), label(b, "true geometry")])
        jpg(f"{M}/recon_vs_gt_{v}.jpg", cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    jpg(f"{M}/recon_aisle.jpg", cv2.imread(f"{BEST}/view_aisle_recon.png"), 88)
    print("wrote recon views")


def compare():
    a = cv2.imread(f"{V2}/out_final_slam/view_aisle_recon.png")
    b = cv2.imread(f"{BEST}/view_aisle_recon.png")
    img = np.hstack([label(a, "v2: SLAM with priors, stereo"), label(b, "v3: no priors, checked stereo, mast lidar")])
    jpg(f"{M}/v2_vs_v3.jpg", cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    a = cv2.imread(f"{V2}/out_final_slam/view_overview_recon.png")
    b = cv2.imread(f"{BEST}/view_overview_recon.png")
    img = np.hstack([label(a, "v2"), label(b, "v3")])
    jpg(f"{M}/v2_vs_v3_overview.jpg", cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    print("wrote v2_vs_v3")


HEIGHT_RUNS = ((f"{V3}/out_gt_gt_lidar", "v3 upper bound (true poses, rendered depth)", "0.45"),
               (f"{V2}/out_final_slam", "v2 final", "C1"),
               (f"{V3}/out_nomast", "v3 without the mast lidar", "C3"),
               (f"{V3}/out_final", "v3 final", "C2"))


def height():
    _, g = gt_scene()
    gm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                   o3d.utility.Vector3iVector(g["faces"]))
    gp = np.asarray(gm.sample_points_uniformly(3_000_000).points)
    edges = [-1, 0.05, 3, 6, 9, 11]
    names = ["floor", "0.05-3 m", "3-6 m", "6-9 m", "roof (9.3 m)"]
    band = np.digitize(gp[:, 2], edges) - 1
    runs = [r for r in HEIGHT_RUNS if os.path.exists(f"{r[0]}/clean_mesh.ply")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(runs)
    out = {}
    for k, (d, lab, col) in enumerate(runs):
        m = o3d.io.read_triangle_mesh(f"{d}/clean_mesh.ply")
        m.transform(np.array(json.load(open(f"{d}/metrics.json")).get("align_T", np.eye(4).tolist())))
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        e = sc.compute_distance(o3c.Tensor(gp.astype(np.float32))).numpy()
        frac = [np.mean(e[band == i] < 0.10) * 100 for i in range(len(names))]
        out[lab] = frac
        xs = np.arange(len(names)) + (k - (len(runs) - 1) / 2) * w
        ax.bar(xs, frac, w, color=col, label=lab)
        for x_, f_ in zip(xs, frac):
            ax.text(x_, f_ + 1, f"{f_:.0f}", ha="center", fontsize=7)
    share = [np.mean(band == i) * 100 for i in range(len(names))]
    ax.set_xticks(range(len(names)), [f"{n}\n({s_:.0f}% of the area)" for n, s_ in zip(names, share)], fontsize=8)
    ax.set_ylabel("true surface reconstructed within 10 cm (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(f"{M}/coverage_by_height.png", dpi=90, bbox_inches="tight")
    json.dump(out, open(f"{M}/.height_stats.json", "w"), indent=1)
    print("wrote coverage_by_height.png", out)


def error():
    scene, _ = gt_scene()
    mesh = load_mesh(f"{BEST}/clean_mesh.ply")
    mesh.transform(np.array(json.load(open(f"{BEST}/metrics.json")).get("align_T", np.eye(4).tolist())))
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
    jpg(f"{M}/error_map.jpg", cv2.resize(np.vstack([img, bar]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    print("wrote error_map.jpg")


RUNS_TABLE = [
    (f"{V3}/out_gt_gt_lidar", "upper bound: true poses, rendered depth"),
    (f"{V2}/out_final_slam", "v2 final (priors, plain stereo, no mast lidar)"),
    (f"{V3}/out_final", "v3 final: no priors, place recognition"),
    (f"{V3}/out_final_unlinked", "v3 final, keeping 2 segments with no closures"),
    (f"{V3}/out_kidnap", "v3, kidnapped (one anchor per robot)"),
    (f"{V3}/out_final_gt", "v3 recipe with true poses"),
    (f"{V3}/out_plain", "v3 final, plain stereo (no left-right check)"),
    (f"{V3}/out_nomast", "v3 final, without the mast lidar"),
]


def ablation():
    rows = [(lab, json.load(open(f"{d}/metrics.json"))) for d, lab in RUNS_TABLE if os.path.exists(f"{d}/metrics.json")]
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
            ax.text(v + 0.5, yi + (k - 1) * h, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(y, [lab for lab, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("%")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(f"{M}/ablation.png", dpi=90, bbox_inches="tight")
    print("wrote ablation.png")
    scene, _ = gt_scene()
    fig, ax = plt.subplots(figsize=(8, 4))
    for d, lab, col in ((f"{V3}/out_gt_gt_lidar", "upper bound", "C0"), (f"{V2}/out_final_slam", "v2 final", "C1"),
                        (f"{V3}/out_final", "v3 final", "C2")):
        if not os.path.exists(f"{d}/clean_mesh.ply"):
            continue
        m = load_mesh(f"{d}/clean_mesh.ply")
        m.transform(np.array(json.load(open(f"{d}/metrics.json")).get("align_T", np.eye(4).tolist())))
        p = np.asarray(m.sample_points_uniformly(1_000_000).points).astype(np.float32)
        e = scene.compute_distance(o3c.Tensor(p)).numpy()
        ax.hist(np.clip(e * 100, 0, 30), bins=120, histtype="step", lw=1.6, density=True, color=col,
                label=f"{lab}: median {np.median(e) * 100:.2f} cm")
    ax.set_yscale("log")
    ax.set_xlabel("distance from reconstructed surface to true surface (cm, clipped at 30)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(f"{M}/accuracy_hist.png", dpi=90, bbox_inches="tight")
    print("wrote accuracy_hist.png")


def fly():
    mesh = load_mesh(f"{BEST}/clean_mesh.ply")
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
    for k in range(300):
        a = -math.pi / 2 + 2 * math.pi * k / 300
        shots.append((c + [radius * math.cos(a), radius * math.sin(a), 22.0], c, 50))
    y0, y1 = lo[1] + 4, hi[1] - 4
    for k in range(300):
        s = k / 299
        eye = np.array([c[0] - 3.0, y0 + s * (y1 - y0), 1.6])
        shots.append((eye, eye + [3.0 * math.cos(0.5 * math.sin(2 * math.pi * s)), 4.0, 0.2], 75))
    # Look up at the high racks and the roof from the middle of the floor.
    for k in range(150):
        a = 2 * math.pi * k / 150
        eye = c + [0, 0, -c[2] + 1.6]
        shots.append((eye, eye + [4 * math.cos(a), 4 * math.sin(a), 3.0], 80))
    for eye, target, fov in shots:
        r.setup_camera(fov, np.asarray(target), np.asarray(eye), [0, 0, 1])
        p.stdin.write(np.asarray(r.render_to_image()).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    steps = {"mast": mast, "sc": sc_fig, "slam": slam_fig, "kidnap": kidnap, "stereo": stereo, "poles": poles,
             "views": side_by_side, "height": height, "error": error, "ablation": ablation, "compare": compare,
             "fly": fly}
    for name in (sys.argv[1:] or ["all"]):
        for k, f in steps.items():
            if name in (k, "all"):
                f()
