"""Reconstruct the warehouse from the fleet's logged sensing (``collect_recon.py``) and export a GLB.

Pipeline:

1. Sensor poses. ``--poses gt`` uses the logged ground-truth poses. ``--poses est`` uses each robot's own
   localization estimate (x, y, yaw; ~0.1 m / 2 deg error). ``--poses est_icp`` refines those estimates by
   registering every lidar sweep (Carter XT-32 and forklift planar layers) against a map built only from other
   robots' sweeps (robots split into folds), alternating map building and point-to-plane ICP (x, y, yaw only).
2. Colored TSDF fusion (Open3D ``ScalableTSDFVolume``) of every Hawk RGB-D frame, with workers, robots and pallets masked
   out, then marching cubes, small-fragment removal and quadric decimation.
3. Lidar points (dynamic hits dropped), colored from the simultaneous Hawk images where they are visible.
4. Evaluation against the warehouse's true triangles: accuracy (reconstruction to truth) and completeness (truth to
   reconstruction, inside the region the cameras could see), after a rigid ICP alignment for estimated poses.
5. GLB export (Y-up, meters): ``reconstruction`` mesh node plus a ``lidar_points`` node, and a decimated ``_lite``
   copy.

Usage (Open3D + trimesh venv)::

    ~/.venvs/recon/bin/python tools/warehouse_amr/recon.py ~/warehouse_amr/recon/data_s7 [more dirs] \\
        --poses gt --out ~/warehouse_amr/recon/out_gt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

# One OpenMP thread per core of a big shared host makes TSDF integration crawl (spin-waiting).
os.environ.setdefault("OMP_NUM_THREADS", "32")

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

DEPTH_SCALE = 1000.0
CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def load(dirs: list[str]) -> list[dict]:
    recs = []
    for d in dirs:
        for line in open(os.path.join(d, "captures.jsonl")):
            r = json.loads(line)
            r["dir"] = d
            r["key"] = f"{os.path.basename(d.rstrip('/'))}/{r['robot']}"
            recs.append(r)
    return recs


def planar(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = (x, y, z)
    return T


def lidar_points(r: dict) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(os.path.join(r["dir"], "lidar", f"{r['capture']:06d}_{r['robot']}.npz"))
    return z["points"].astype(np.float64), z["dynamic"]


def to_pcd(p: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))


def calibrate(recs: list[dict]) -> dict:
    """Fixed sensor extrinsics per robot type, as a calibration would give: base height, each camera's pose in the
    base frame and the lidar origin (rotation averages projected back onto SO(3)). The per-capture body tilt and
    bounce are then unmodeled, as on a real robot."""
    calib: dict = {}
    for r in recs:
        x, y, z, yaw = r["gt_pose"]
        T_inv = np.linalg.inv(planar(x, y, z, yaw))
        e = calib.setdefault(r["type"], {"z": [], "lidar": [], "mast": [], "cams": {}})
        e["z"].append(z)
        e["lidar"].append(T_inv[:3, :3] @ np.array(r["lidar_origin"]) + T_inv[:3, 3])
        if "mast_lidar_origin" in r:
            e["mast"].append(T_inv[:3, :3] @ np.array(r["mast_lidar_origin"]) + T_inv[:3, 3])
        for c in r.get("cams", []):
            e["cams"].setdefault(c["name"], []).append(T_inv @ np.array(c["T_world_cam"]))
    out = {}
    for t, e in calib.items():
        cams = {}
        for name, Ts in e["cams"].items():
            Ts = np.array(Ts)
            u, _, vt = np.linalg.svd(Ts[:, :3, :3].mean(0))
            T = np.eye(4)
            T[:3, :3] = u @ vt
            T[:3, 3] = np.median(Ts[:, :3, 3], 0)
            cams[name] = T
        out[t] = {"z": float(np.median(e["z"])), "lidar": np.median(np.array(e["lidar"]), 0), "cams": cams}
        if e["mast"]:
            out[t]["mast"] = np.median(np.array(e["mast"]), 0)
    return out


# Which stereo depth images to read (stereo_depth.py --suffix); an env var so spawned workers see it too.
STEREO_SUFFIX = os.environ.get("RECON_STEREO_SUFFIX", "_s")

# --poses slam*: (prior mode, place recognition, landmarks); see slam.run.
SLAM_MODES = {
    "slam": ("all", False, False),
    "slam_noprior": ("first", False, False),
    "slam_pr": ("first", True, False),
    "slam_pr_all": ("all", True, False),
    "slam_kidnap": ("kidnapped", True, False),
    "slam_lm": ("all", True, True),
    "slam_pr_lm": ("first", True, True),
    "slam_kidnap_lm": ("kidnapped", True, True),
}


def estimate_poses(recs: list[dict], mode: str, iters: int = 5, cache: str = "") -> None:
    """Sets ``r['T_robot']`` (robot-to-world used for reconstruction) and ``r['T_gt']`` on every capture."""
    calib = calibrate(recs)
    for r in recs:
        x, y, z, yaw = r["gt_pose"]
        r["T_gt"] = planar(x, y, z, yaw)
        r["calib"] = calib[r["type"]]
        if mode == "gt":
            r["T_robot"] = r["T_gt"]
        else:
            ex, ey, eyaw = r["est_pose"]
            r["T_robot"] = planar(ex, ey, r["calib"]["z"], eyaw)
    if mode.startswith("slam"):
        import slam

        if cache and os.path.exists(cache) and len(np.load(cache)) == len(recs):
            x = np.load(cache)
            print(f"[recon] slam poses from {cache}", flush=True)
        else:
            x = slam.run(recs, priors=SLAM_MODES[mode][0], place=SLAM_MODES[mode][1],
                         landmarks=SLAM_MODES[mode][2], log=lambda s: print(s, flush=True))
            np.save(cache[:-4] + "_localized.npy", slam.run.localized) if cache else None
            gt = np.array([r["gt_pose"] for r in recs])[:, [0, 1, 3]]
            stages = {}
            for name, xs in slam.run.stages:
                e = np.linalg.norm(xs[:, :2] - gt[:, :2], axis=1)
                ey = np.degrees(np.abs(slam.wrap(xs[:, 2] - gt[:, 2])))
                stages[name] = {"median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)),
                                "max_m": float(e.max()), "yaw_median_deg": float(np.median(ey))}
                print(f"[recon] slam stage {name}: {stages[name]}", flush=True)
            if cache:
                np.savez(cache[:-4] + "_stage_poses.npz", **{name: xs for name, xs in slam.run.stages})
                np.save(cache, x)
                np.save(cache[:-4] + "_closures.npy", slam.run.closures)
                json.dump(stages, open(cache[:-4] + "_stages.json", "w"), indent=1)
        loc = cache[:-4] + "_localized.npy" if cache else ""
        localized = np.load(loc) if loc and os.path.exists(loc) else np.ones(len(recs), bool)
        for r, (px, py, pyaw), ok in zip(recs, x, localized):
            r["T_robot"] = planar(px, py, r["calib"]["z"], pyaw)
            r["localized"] = bool(ok)
        return
    if mode != "est_icp":
        return
    import shutil
    from multiprocessing import get_context

    keys = sorted({r["key"] for r in recs})
    key_idx = np.array([keys.index(r["key"]) for r in recs])
    n_folds = min(4, len(keys))
    procs = min(64, os.cpu_count() or 1)
    shared = f"/dev/shm/recon_icp_{os.getpid()}"
    os.makedirs(shared, exist_ok=True)
    # Open3D's OpenMP runtime deadlocks in forked children; spawned ones read OMP_NUM_THREADS at import. Workers
    # memory-map the arrays in ``shared`` so they hold one copy between them.
    os.environ["OMP_NUM_THREADS"] = "2"
    ctx = get_context("spawn")
    with ctx.Pool(procs) as pool:
        local = pool.map(_sensor_frame_sweep, [(r["dir"], r["capture"], r["robot"], r["T_gt"]) for r in recs],
                         chunksize=32)
    offsets = np.cumsum([0] + [len(p) for p in local])
    np.save(f"{shared}/local.npy", np.concatenate(local))
    np.save(f"{shared}/offsets.npy", offsets)
    np.save(f"{shared}/key_idx.npy", key_idx)
    try:
        for it in range(iters):
            t0 = time.time()
            # A sweep registered against a map containing itself (or its own robot's sweeps, whose localization
            # errors are correlated in time) stays where it is: robots are split into folds and each fold is
            # aligned to the map of all the other folds.
            for f in range(n_folds):
                pts = np.concatenate([pl @ r["T_robot"][:3, :3].T + r["T_robot"][:3, 3]
                                      for r, k, pl in zip(recs, key_idx, local) if k % n_folds != f])
                pcd = to_pcd(pts).voxel_down_sample(0.08)
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=20))
                p, n = np.asarray(pcd.points), np.asarray(pcd.normals)
                order = np.argsort(p[:, 0])
                np.save(f"{shared}/p{f}.npy", p[order])
                np.save(f"{shared}/n{f}.npy", n[order])
                np.save(f"{shared}/x{f}.npy", np.ascontiguousarray(p[order, 0]))
            radius = (0.6, 0.3, 0.15, 0.1, 0.08)[min(it, 4)]
            np.save(f"{shared}/init.npy", np.array([r["T_robot"] for r in recs]))
            with ctx.Pool(procs, initializer=_init_icp_worker, initargs=(shared, n_folds, radius, ICP_CROP)) as pool:
                results = pool.map(_register, range(len(recs)), chunksize=16)
            moved, rejected = [], 0
            for r, (T, fitness) in zip(recs, results):
                # 2D localization: keep x, y, yaw from ICP; height, roll and pitch stay at the floor plane.
                yaw = math.atan2(T[1, 0], T[0, 0])
                cand = planar(T[0, 3], T[1, 3], r["T_robot"][2, 3], yaw)
                ex, ey, eyaw = r["est_pose"]
                if fitness < 0.3 or math.hypot(cand[0, 3] - ex, cand[1, 3] - ey) > 0.6:
                    rejected += 1
                    continue
                moved.append(np.linalg.norm(cand[:2, 3] - r["T_robot"][:2, 3]))
                r["T_robot"] = cand
            err = pose_errors(recs, aligned=True)
            print(f"[recon] icp pass {it}: radius {radius}, median update {np.median(moved):.3f} m, rejected "
                  f"{rejected}, aligned pose error median {np.median(err):.3f} p90 {np.percentile(err, 90):.3f} m "
                  f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        shutil.rmtree(shared, ignore_errors=True)


_ICP: dict = {}
ICP_CROP = 25.0


def _sensor_frame_sweep(job) -> np.ndarray:
    """The logged world points were placed with the true pose; undo that to get the sensor-frame measurement."""
    d, capture, robot, T_gt = job
    z = np.load(os.path.join(d, "lidar", f"{capture:06d}_{robot}.npz"))
    p = z["points"][~z["dynamic"]].astype(np.float64)
    return np.asarray(to_pcd((p - T_gt[:3, 3]) @ T_gt[:3, :3]).voxel_down_sample(0.08).points)


def _init_icp_worker(path: str, n_folds: int, radius: float, crop: float) -> None:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    for name in ("local", "offsets", "key_idx", "init"):
        _ICP[name] = np.load(f"{path}/{name}.npy", mmap_mode="r")
    _ICP["maps"] = [tuple(np.load(f"{path}/{a}{j}.npy", mmap_mode="r") for a in "xpn") for j in range(n_folds)]
    _ICP["radius"] = radius
    _ICP["crop"] = crop


def _register(i: int):
    x, p, n = _ICP["maps"][int(_ICP["key_idx"][i]) % len(_ICP["maps"])]
    T0 = np.array(_ICP["init"][i])
    cx, cy = T0[0, 3], T0[1, 3]
    # Map points are sorted by x: slice the x window, then filter y.
    crop = _ICP["crop"]
    lo, hi = np.searchsorted(x, [cx - crop, cx + crop])
    px, nx = np.asarray(p[lo:hi]), np.asarray(n[lo:hi])
    keep = np.abs(px[:, 1] - cy) < crop
    target = to_pcd(px[keep])
    target.normals = o3d.utility.Vector3dVector(nx[keep])
    o = _ICP["offsets"]
    res = o3d.pipelines.registration.registration_icp(
        to_pcd(np.array(_ICP["local"][o[i]:o[i + 1]])), target, _ICP["radius"], T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    return np.asarray(res.transformation), res.fitness


def pose_errors(recs: list[dict], aligned: bool) -> np.ndarray:
    """Position error of the reconstruction poses, optionally after the best rigid 2D fit of all of them to the
    truth (a map built only from robot data has its own frame)."""
    est = np.array([r["T_robot"][:2, 3] for r in recs])
    gt = np.array([r["T_gt"][:2, 3] for r in recs])
    if aligned:
        me, mg = est.mean(0), gt.mean(0)
        u, _, vt = np.linalg.svd((est - me).T @ (gt - mg))
        R = (u @ vt).T
        if np.linalg.det(R) < 0:
            R = (u @ np.diag([1, -1]) @ vt).T
        est = (est - me) @ R.T + mg
    return np.linalg.norm(est - gt, axis=1)


def cam_pose(r: dict, c: dict) -> np.ndarray:
    """Camera-to-world (OpenCV axes): the true logged pose with ground-truth poses, otherwise the calibrated
    extrinsic on the estimated base pose."""
    if r["T_robot"] is r["T_gt"]:
        return np.array(c["T_world_cam"])
    return r["T_robot"] @ r["calib"]["cams"][c["name"]]


def read_frame(r: dict, c: dict, depth_max: float, source: str = "gt"):
    stem = os.path.join(r["dir"], "frames", c["stem"])
    rgb = cv2.imread(stem + ".jpg")[..., ::-1].copy()
    depth = cv2.imread(stem + ("_d.png" if source == "gt" else STEREO_SUFFIX + ".png"), -1)
    mask = cv2.imread(stem + "_m.png", 0) > 0
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    depth[mask | (depth > depth_max * DEPTH_SCALE)] = 0
    return rgb, depth


# Lidar sweeps enter the TSDF as four virtual 90-degree pinhole depth images around the sensor, interpolated from
# its range image: the Carter's XT-32 (32 x 720, 1 deg x 0.5 deg) or the forklift's mast lidar (64 x 512 over +-45).
LIDAR_SPECS = {"xt32": (32, 720, -16.0, 1.0), "mast": (64, 512, -45.0, 90.0 / 63)}  # rows, cols, el0, el step
VL_W, VL_F = 360, 180.0


def _virtual_lidar_lookup(rows, cols, el0, el_step):
    half = max(abs(el0), abs(el0 + el_step * (rows - 1)))
    vl_h = min(VL_W, 2 * int(math.ceil(VL_F * math.tan(math.radians(half)))))
    u, v = np.meshgrid(np.arange(VL_W) + 0.5, np.arange(vl_h) + 0.5)
    d = np.stack([(u - VL_W / 2) / VL_F, (v - vl_h / 2) / VL_F, np.ones_like(u)], -1)
    out = []
    for k in range(4):
        psi = k * math.pi / 2
        fwd, right, down = np.array([math.cos(psi), math.sin(psi), 0]), np.array([math.sin(psi), -math.cos(psi), 0]), \
            np.array([0, 0, -1.0])
        R = np.stack([right, down, fwd], 1)
        dl = d @ R.T
        az = np.arctan2(dl[..., 1], dl[..., 0])
        el = np.degrees(np.arctan2(dl[..., 2], np.hypot(dl[..., 0], dl[..., 1])))
        col = (az + math.pi) / (2 * math.pi) * cols
        row = (el - el0) / el_step
        out.append((R, col, row, np.linalg.norm(d, axis=-1)))
    return out, vl_h


def lidar_depth_images(points_local: np.ndarray, origin_local: np.ndarray, depth_max: float, kind: str = "xt32"):
    """Robot-frame returns -> [(R_lidar_cam, depth m)] for the four virtual cameras of lidar ``kind``."""
    rows, cols, el0, el_step = LIDAR_SPECS[kind]
    q = points_local - origin_local
    rng_m = np.linalg.norm(q, axis=1)
    az = np.arctan2(q[:, 1], q[:, 0])
    el = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
    ci = np.round((az + math.pi) / (2 * math.pi) * cols - 0.5).astype(int) % cols
    ri = np.round((el - el0) / el_step).astype(int)
    ok = (ri >= 0) & (ri < rows)
    R = np.full((rows, cols), np.inf)
    np.minimum.at(R, (ri[ok], ci[ok]), rng_m[ok])
    R[~np.isfinite(R)] = 0
    out = []
    for Rc, col, row, norm in _VL[kind][0]:
        c0 = np.floor(col - 0.5).astype(int)
        r0 = np.floor(row).astype(int)
        fc, fr = col - 0.5 - c0, row - r0
        valid = (r0 >= 0) & (r0 + 1 < rows)
        r0c = np.clip(r0, 0, rows - 2)
        c0m, c1m = c0 % cols, (c0 + 1) % cols
        q00, q01, q10, q11 = R[r0c, c0m], R[r0c, c1m], R[r0c + 1, c0m], R[r0c + 1, c1m]
        stack = np.stack([q00, q01, q10, q11])
        lo, hi = stack.min(0), stack.max(0)
        valid &= (lo > 0) & (hi - lo < 0.03 * lo + 0.05)
        rr = (q00 * (1 - fc) + q01 * fc) * (1 - fr) + (q10 * (1 - fc) + q11 * fc) * fr
        depth = np.where(valid, rr / norm, 0.0)
        depth[depth > depth_max] = 0
        out.append((Rc, depth))
    return out


_VL = {k: _virtual_lidar_lookup(*v) for k, v in LIDAR_SPECS.items()}


def _tile_job(job):
    """Fuses one tile: a geometry volume (camera depth up to ``geo_max`` plus lidar) and a color volume (camera
    depth up to ``col_max``); returns the tile's mesh, cropped to the tile, colored from the color volume."""
    (idx, lo, hi, cam_frames, lidar_frames, voxel, geo_max, col_max, lidar_max, source, out_dir, lidar_mode,
     lidar_gap, stereo_edge, near_max, far_voxel) = job
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    integ = o3d.pipelines.integration

    def tsdf():
        return integ.ScalableTSDFVolume(voxel_length=voxel, sdf_trunc=4 * voxel,
                                        color_type=integ.TSDFVolumeColorType.NoColor)

    geo = tsdf()
    # Layers by confidence: camera depth up to near_max, then (filling only gaps) up to geo_max, then lidar.
    far_geo = None
    if 0 < near_max < geo_max:
        # Far stereo is noisier but unbiased: a coarser grid with a wider band lets more views average.
        fv = far_voxel or voxel
        far_geo = integ.ScalableTSDFVolume(voxel_length=fv, sdf_trunc=(6 if far_voxel else 4) * fv,
                                           color_type=integ.TSDFVolumeColorType.NoColor)
    col = integ.ScalableTSDFVolume(voxel_length=1.5 * voxel, sdf_trunc=6 * voxel,
                                   color_type=integ.TSDFVolumeColorType.RGB8)
    margin = 0.3
    lo_m, hi_m = np.asarray(lo) - margin, np.asarray(hi) + margin
    grids = {}

    def in_tile(depth, K, T):
        fx, fy, cx, cy = K
        h, w = depth.shape
        if (h, w, fx) not in grids:
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            grids[(h, w, fx)] = ((u - cx) / fx, (v - cy) / fy)
        gu, gv = grids[(h, w, fx)]
        x = T[0, 0] * gu + T[0, 1] * gv + T[0, 2]
        y = T[1, 0] * gu + T[1, 1] * gv + T[1, 2]
        wx, wy = T[0, 3] + x * depth, T[1, 3] + y * depth
        return (wx > lo_m[0]) & (wx < hi_m[0]) & (wy > lo_m[1]) & (wy < hi_m[1])

    def integrate(vol, rgb, depth_m, K, T, trunc):
        fx, fy, cx, cy = K
        h, w = depth_m.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image((depth_m * DEPTH_SCALE).astype(np.uint16)),
            depth_scale=DEPTH_SCALE, depth_trunc=trunc, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy), np.linalg.inv(T))

    for stem, K, T in cam_frames:
        depth = cv2.imread(stem + ("_d.png" if source == "gt" else STEREO_SUFFIX + ".png"), -1)
        if depth is None:
            continue
        mask = cv2.dilate((cv2.imread(stem + "_m.png", 0) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        depth = depth.astype(np.float64) / DEPTH_SCALE
        depth[mask] = 0
        if stereo_edge and source == "stereo":
            # Stereo smears depth across discontinuities: drop a band of pixels around every jump.
            z = np.where(depth > 0, depth, np.nan)
            jump = np.zeros(depth.shape, bool)
            for ax in (0, 1):
                dz = np.abs(np.diff(z, axis=ax))
                j = ~(dz < 0.03 * np.fmin(z.take(range(1, z.shape[ax]), ax), z.take(range(z.shape[ax] - 1), ax)))
                if ax == 0:
                    jump[1:] |= j
                    jump[:-1] |= j
                else:
                    jump[:, 1:] |= j
                    jump[:, :-1] |= j
            k = 2 * stereo_edge + 1
            depth[cv2.dilate(jump.astype(np.uint8), np.ones((k, k), np.uint8)) > 0] = 0
        keep = in_tile(depth, K, T) & (depth > 0)
        if not keep.any():
            continue
        rgb = cv2.imread(stem + ".jpg")[..., ::-1].copy()
        d_geo = np.where(keep & (depth <= geo_max), depth, 0)
        if far_geo is not None:
            if (d_geo > 0).any():
                integrate(far_geo, rgb, d_geo, K, T, geo_max)
            d_near = np.where(d_geo <= near_max, d_geo, 0)
            if (d_near > 0).any():
                integrate(geo, rgb, d_near, K, T, near_max)
        elif (d_geo > 0).any():
            integrate(geo, rgb, d_geo, K, T, geo_max)
        integrate(col, rgb, np.where(keep & (depth <= col_max), depth, 0), K, T, col_max)
    import slam

    # Lidar fills what the cameras did not reconstruct (its own volume, merged below) instead of being averaged
    # into the more precise camera surfaces.
    cam_geo = geo
    if lidar_frames and lidar_mode == "union":
        geo = integ.ScalableTSDFVolume(voxel_length=voxel, sdf_trunc=4 * voxel,
                                       color_type=integ.TSDFVolumeColorType.NoColor)
    for d, rec, T_robot, origin_local, kind in lidar_frames:
        p, _ = slam.noisy_world_sweep(d, rec)
        p_local = slam.to_local(p, rec["gt_pose"])
        vl_h = _VL[kind][1]
        dummy = np.zeros((vl_h, VL_W, 3), np.uint8)
        K_vl = (VL_F, VL_F, VL_W / 2, vl_h / 2)
        for Rc, depth in lidar_depth_images(p_local, origin_local, lidar_max, kind):
            T = np.eye(4)
            T[:3, :3] = Rc
            T[:3, 3] = origin_local
            T = T_robot @ T
            keep = in_tile(depth, K_vl, T) & (depth > 0)
            if keep.any():
                integrate(geo, dummy, np.where(keep, depth, 0), K_vl, T, lidar_max)
    def fill(mesh, extra):
        """``mesh`` plus the triangles of ``extra`` farther than ``lidar_gap`` from it."""
        if not len(extra.triangles):
            return mesh
        ev, ef = np.asarray(extra.vertices), np.asarray(extra.triangles)
        if len(mesh.triangles):
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            ef = ef[scene.compute_distance(o3c.Tensor(ev[ef].mean(1).astype(np.float32))).numpy() > lidar_gap]
        v0 = len(mesh.vertices)
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.concatenate([np.asarray(mesh.vertices).reshape(-1, 3), ev])),
            o3d.utility.Vector3iVector(np.concatenate([np.asarray(mesh.triangles).reshape(-1, 3), ef + v0])))

    mesh = cam_geo.extract_triangle_mesh()
    if far_geo is not None:
        mesh = fill(mesh, far_geo.extract_triangle_mesh())
    if geo is not cam_geo:
        mesh = fill(mesh, geo.extract_triangle_mesh())
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    path = os.path.join(out_dir, f"tile_{idx:03d}.npz")
    if len(f):
        cen = v[f].mean(1)
        f = f[(cen[:, 0] >= lo[0]) & (cen[:, 0] < hi[0]) & (cen[:, 1] >= lo[1]) & (cen[:, 1] < hi[1])]
    colors = np.full((len(v), 3), 0.55)
    pc = col.extract_point_cloud()
    if len(pc.points) and len(v):
        from scipy.spatial import cKDTree

        dist, nb = cKDTree(np.asarray(pc.points)).query(v, distance_upper_bound=4 * voxel)
        hit = np.isfinite(dist)
        colors[hit] = np.asarray(pc.colors)[nb[hit]]
    np.savez(path, vertices=v.astype(np.float32), faces=f.astype(np.int32), colors=colors.astype(np.float32))
    return path, len(cam_frames), len(lidar_frames), len(f)


def fuse(recs: list[dict], voxel: float, depth_max: float, source: str = "gt", lidar: bool = False,
         col_max: float = 0.0, lidar_max: float = 20.0, tile: float = 9.0, procs: int = 40,
         work: str = "/tmp", lidar_mode: str = "union", lidar_gap: float = 0.08,
         stereo_edge: int = 0, near_max: float = 0.0, far_voxel: float = 0.0) -> o3d.geometry.TriangleMesh:
    """Tiled TSDF fusion (one process per tile, legacy ``ScalableTSDFVolume``; the tensor VoxelBlockGrid segfaults
    extracting warehouse-sized CPU grids)."""
    from multiprocessing import get_context

    col_max = col_max or depth_max
    cams = [(os.path.join(r["dir"], "frames", c["stem"]), tuple(c["K"]), cam_pose(r, c))
            for r in recs for c in r.get("cams", [])]
    lidars = []
    if lidar:
        for r in recs:
            rec = {k: r[k] for k in ("capture", "robot", "type", "gt_pose", "lidar_origin", "mast_lidar_origin")
                   if k in r}
            if r["type"] == "carter":
                lidars.append((r["dir"], rec, r["T_robot"], r["calib"]["lidar"], "xt32"))
            elif "mast_lidar_origin" in r and "mast" in r["calib"] and not os.environ.get("RECON_NO_MAST"):
                lidars.append((r["dir"], rec, r["T_robot"], r["calib"]["mast"], "mast"))
    cam_xy = np.array([T[:2, 3] for _, _, T in cams])
    lid_xy = np.array([L[2][:2, 3] for L in lidars]) if lidars else np.zeros((0, 2))
    all_xy = np.concatenate([cam_xy, lid_xy])
    lo, hi = all_xy.min(0) - 1, all_xy.max(0) + 1
    nx, ny = int(math.ceil((hi[0] - lo[0]) / tile)), int(math.ceil((hi[1] - lo[1]) / tile))
    reach_c, reach_l = max(depth_max, col_max), lidar_max
    jobs = []
    tmp = os.path.join(work, f"tiles_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    for ix in range(nx):
        for iy in range(ny):
            # Outer tiles extend to infinity so nothing the sensors saw past the robots' reach is cropped away.
            tlo = [lo[0] + ix * tile if ix else -1e4, lo[1] + iy * tile if iy else -1e4]
            thi = [lo[0] + (ix + 1) * tile if ix < nx - 1 else 1e4, lo[1] + (iy + 1) * tile if iy < ny - 1 else 1e4]

            def near(xy, reach):
                dx = np.maximum(np.maximum(tlo[0] - xy[:, 0], xy[:, 0] - thi[0]), 0)
                dy = np.maximum(np.maximum(tlo[1] - xy[:, 1], xy[:, 1] - thi[1]), 0)
                return np.hypot(dx, dy) < reach

            ci = np.nonzero(near(cam_xy, reach_c))[0]
            li = np.nonzero(near(lid_xy, reach_l))[0] if len(lidars) else []
            jobs.append((len(jobs), tlo, thi, [cams[k] for k in ci], [lidars[k] for k in li], voxel, depth_max,
                         col_max, lidar_max, source, tmp, lidar_mode, lidar_gap, stereo_edge, near_max, far_voxel))
    # Largest tiles first so the pool's tail is short.
    jobs.sort(key=lambda j: -(len(j[3]) + 0.5 * len(j[4])))
    print(f"[recon] fusing {len(cams)} camera frames + {len(lidars)} lidar sweeps in {len(jobs)} tiles", flush=True)
    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = "4"
    verts, faces, cols, off = [], [], [], 0
    with get_context("spawn").Pool(min(procs, len(jobs))) as pool:
        for path, nc, nl, nf in pool.imap_unordered(_tile_job, jobs):
            z = np.load(path)
            verts.append(z["vertices"])
            faces.append(z["faces"] + off)
            cols.append(z["colors"])
            off += len(z["vertices"])
            os.remove(path)
            print(f"[recon] tile done: {nc} cam / {nl} lidar frames, {nf} triangles ({time.time() - t0:.0f} s)",
                  flush=True)
    os.environ["OMP_NUM_THREADS"] = "32"
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(verts).astype(np.float64)),
                                     o3d.utility.Vector3iVector(np.concatenate(faces)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(cols).astype(np.float64))
    mesh.remove_unreferenced_vertices()
    mesh.merge_close_vertices(1e-4)
    print(f"[recon] raw mesh: {len(mesh.triangles)} triangles ({time.time() - t0:.0f} s)", flush=True)
    return mesh


_FS: dict = {}


def _fs_init(path: str) -> None:
    _FS["v"] = np.load(f"{path}/v.npy", mmap_mode="r")
    from scipy.spatial import cKDTree

    _FS["tree"] = cKDTree(np.asarray(_FS["v"])[:, :2])


def _fs_job(job):
    """Per-vertex (seen, seen-through) counts over a chunk of camera frames."""
    frames, source, reach, weighted = job
    v = _FS["v"]
    seen = np.zeros(len(v), np.float64)
    through = np.zeros(len(v), np.float64)
    for stem, K, T in frames:
        depth = cv2.imread(stem + ("_d.png" if source == "gt" else STEREO_SUFFIX + ".png"), -1)
        if depth is None:
            continue
        mask = cv2.dilate((cv2.imread(stem + "_m.png", 0) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        depth = depth.astype(np.float32) / DEPTH_SCALE
        depth[mask] = 0
        idx = np.asarray(_FS["tree"].query_ball_point(T[:2, 3], reach), np.int64)
        if not len(idx):
            continue
        Tcw = np.linalg.inv(T)
        p = np.asarray(v[idx]) @ Tcw[:3, :3].T + Tcw[:3, 3]
        fx, fy, cx, cy = K
        z = p[:, 2]
        ok = z > 0.2
        idx, p, z = idx[ok], p[ok], z[ok]
        u = np.floor(p[:, 0] / z * fx + cx).astype(np.int64)
        w_ = np.floor(p[:, 1] / z * fy + cy).astype(np.int64)
        h, w = depth.shape
        ok = (u >= 0) & (u < w) & (w_ >= 0) & (w_ < h)
        idx, z, u, w_ = idx[ok], z[ok], u[ok], w_[ok]
        dm = depth[w_, u]
        ok = dm > 0
        idx, z, dm = idx[ok], z[ok], dm[ok]
        tol = 0.05 + 0.02 * z
        # A close view is far more reliable than a distant one (stereo error grows with z^2).
        wt = 1.0 / np.maximum(z, 1.0) ** 2 if weighted else np.ones_like(z)
        s_ = np.abs(dm - z) < tol
        t_ = dm > z + 2 * tol
        np.add.at(seen, idx[s_], wt[s_])
        np.add.at(through, idx[t_], wt[t_])
    return seen, through


def _color_vertex_job(job):
    """Per-vertex color sums from the frames that see each vertex (depth agrees), weighted 1 / z^2."""
    frames, source, reach = job
    v = _FS["v"]
    acc = np.zeros((len(v), 3), np.float64)
    wsum = np.zeros(len(v), np.float64)
    for stem, K, T in frames:
        depth = cv2.imread(stem + ("_d.png" if source == "gt" else STEREO_SUFFIX + ".png"), -1)
        if depth is None:
            continue
        mask = cv2.dilate((cv2.imread(stem + "_m.png", 0) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        depth = depth.astype(np.float32) / DEPTH_SCALE
        depth[mask] = 0
        idx = np.asarray(_FS["tree"].query_ball_point(T[:2, 3], reach), np.int64)
        if not len(idx):
            continue
        Tcw = np.linalg.inv(T)
        p = np.asarray(v[idx]) @ Tcw[:3, :3].T + Tcw[:3, 3]
        fx, fy, cx, cy = K
        z = p[:, 2]
        ok = (z > 0.2) & (z < reach)
        idx, p, z = idx[ok], p[ok], z[ok]
        u = np.floor(p[:, 0] / z * fx + cx).astype(np.int64)
        w_ = np.floor(p[:, 1] / z * fy + cy).astype(np.int64)
        h, w = depth.shape
        ok = (u >= 0) & (u < w) & (w_ >= 0) & (w_ < h)
        idx, z, u, w_ = idx[ok], z[ok], u[ok], w_[ok]
        dm = depth[w_, u]
        ok = (dm > 0) & (np.abs(dm - z) < 0.04 + 0.015 * z)
        if not ok.any():
            continue
        idx, z, u, w_ = idx[ok], z[ok], u[ok], w_[ok]
        rgb = cv2.imread(stem + ".jpg")[..., ::-1].astype(np.float64) / 255.0
        wt = 1.0 / z ** 2
        np.add.at(acc, idx, rgb[w_, u] * wt[:, None])
        np.add.at(wsum, idx, wt)
    return acc, wsum


def recolor(mesh, recs, source: str, reach: float = 8.0, procs: int = 40, fallback_reach: float = 25.0):
    """Vertex colors straight from the camera images (nearest views weighted most). Vertices no camera saw within
    ``reach`` get a second pass out to ``fallback_reach``; the rest keep their fused color."""
    import shutil
    from multiprocessing import get_context

    v = np.asarray(mesh.vertices).astype(np.float32)
    frames = [(os.path.join(r["dir"], "frames", c["stem"]), tuple(c["K"]), cam_pose(r, c))
              for r in recs for c in r.get("cams", [])]
    shared = f"/dev/shm/recon_col_{os.getpid()}"
    os.makedirs(shared, exist_ok=True)
    np.save(f"{shared}/v.npy", v)
    chunks = [frames[k::procs * 4] for k in range(procs * 4)]
    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with get_context("spawn").Pool(procs, initializer=_fs_init, initargs=(shared,)) as pool:
            acc = np.zeros((len(v), 3))
            wsum = np.zeros(len(v))
            for a_, w_ in pool.imap_unordered(_color_vertex_job, [(c, source, reach) for c in chunks]):
                acc += a_
                wsum += w_
            hit1 = wsum > 0
            if fallback_reach > reach and not hit1.all():
                for a_, w_ in pool.imap_unordered(_color_vertex_job, [(c, source, fallback_reach) for c in chunks]):
                    acc[~hit1] += a_[~hit1]
                    wsum[~hit1] += w_[~hit1]
    finally:
        shutil.rmtree(shared, ignore_errors=True)
        os.environ["OMP_NUM_THREADS"] = "32"
    cols = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else np.full((len(v), 3), 0.55)
    hit = wsum > 0
    cols[hit] = acc[hit] / wsum[hit, None]
    out = o3d.geometry.TriangleMesh(mesh)
    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    print(f"[recon] recolored {hit.mean():.1%} of vertices from the images ({time.time() - t0:.0f} s)", flush=True)
    return out


def free_space_filter(mesh, recs, source: str, reach: float = 12.0, procs: int = 40, weighted: bool = False):
    """Drops surfaces that camera rays see straight through more often than they see them (floaters from a few bad
    depth frames, which the TSDF's narrow truncation band never carves away)."""
    import shutil
    from multiprocessing import get_context

    v = np.asarray(mesh.vertices).astype(np.float32)
    frames = [(os.path.join(r["dir"], "frames", c["stem"]), tuple(c["K"]), cam_pose(r, c))
              for r in recs for c in r.get("cams", [])]
    shared = f"/dev/shm/recon_fs_{os.getpid()}"
    os.makedirs(shared, exist_ok=True)
    np.save(f"{shared}/v.npy", v)
    chunks = [frames[k::procs * 4] for k in range(procs * 4)]
    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with get_context("spawn").Pool(procs, initializer=_fs_init, initargs=(shared,)) as pool:
            seen = np.zeros(len(v))
            through = np.zeros(len(v))
            for s_, t_ in pool.imap_unordered(_fs_job, [(c, source, reach, weighted) for c in chunks]):
                seen += s_
                through += t_
    finally:
        shutil.rmtree(shared, ignore_errors=True)
        os.environ["OMP_NUM_THREADS"] = "32"
    bad = (through > seen) & (through > (2.0 / 16 if weighted else 2))
    tri = np.asarray(mesh.triangles)
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(bad[tri].any(1))
    out.remove_unreferenced_vertices()
    print(f"[recon] free-space filter: removed {bad.mean():.1%} of vertices ({time.time() - t0:.0f} s)", flush=True)
    return out


def clean(mesh: o3d.geometry.TriangleMesh, min_tris: int, target: int) -> o3d.geometry.TriangleMesh:
    ids, counts, _ = mesh.cluster_connected_triangles()
    ids, counts = np.asarray(ids), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[ids] < min_tris)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _color_job(job):
    """One capture's static lidar returns under the reconstruction pose, colored from its cameras."""
    import slam

    r, cams, source, voxel = job
    p, _ = slam.noisy_world_sweep(r["dir"], r)
    if not len(p):
        return np.zeros((0, 3)), np.zeros((0, 3))
    p = slam.to_local(p, r["gt_pose"]) @ r["T_robot"][:3, :3].T + r["T_robot"][:3, 3]
    col = np.full((len(p), 3), 0.55)
    done = np.zeros(len(p), bool)
    for stem, K, T_wc in cams:
        Tcw = np.linalg.inv(T_wc)
        pc = p @ Tcw[:3, :3].T + Tcw[:3, 3]
        fx, fy, cx, cy = K
        z = pc[:, 2]
        ok = (z > 0.1) & ~done
        u = np.full(len(p), -1)
        v = np.full(len(p), -1)
        u[ok] = np.floor(pc[ok, 0] / z[ok] * fx + cx).astype(int)
        v[ok] = np.floor(pc[ok, 1] / z[ok] * fy + cy).astype(int)
        depth = cv2.imread(stem + ("_d.png" if source == "gt" else STEREO_SUFFIX + ".png"), -1)
        if depth is None:
            continue
        depth = depth.astype(np.float64) / DEPTH_SCALE
        h, w = depth.shape
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.nonzero(ok)[0]
        agree = np.abs(depth[v[idx], u[idx]] - z[idx]) < 0.08 + 0.02 * z[idx]
        idx = idx[agree]
        if len(idx):
            rgb = cv2.imread(stem + ".jpg")[..., ::-1]
            col[idx] = rgb[v[idx], u[idx]] / 255.0
            done[idx] = True
    pcd = to_pcd(p)
    pcd.colors = o3d.utility.Vector3dVector(col)
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def color_lidar(recs: list[dict], voxel: float, source: str = "gt", procs: int = 40) -> o3d.geometry.PointCloud:
    """Static lidar returns under the reconstruction poses; points take the color of the camera pixel they
    project to when the camera sees the same surface there (depth agrees), others stay gray."""
    from multiprocessing import get_context

    keys = ("dir", "capture", "robot", "type", "gt_pose", "lidar_origin", "mast_lidar_origin", "T_robot")
    jobs = [({k: r[k] for k in keys if k in r}, [(os.path.join(r["dir"], "frames", c["stem"]), c["K"], cam_pose(r, c))
                                       for c in r.get("cams", [])], source, voxel) for r in recs]
    os.environ["OMP_NUM_THREADS"] = "1"
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(_color_job, jobs, chunksize=16)
    os.environ["OMP_NUM_THREADS"] = "32"
    pcd = to_pcd(np.concatenate([p for p, _ in res]))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([c for _, c in res]))
    return pcd.voxel_down_sample(voxel)


def gt_scene(dirs: list[str]):
    g = np.load(os.path.join(dirs[0], "gt_mesh.npz"))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g["vertices"].astype(np.float64)),
                                     o3d.utility.Vector3iVector(g["faces"]))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return mesh, scene


def evaluate(mesh, recs, dirs, depth_max: float, align: bool) -> dict:
    gt_mesh, gt = gt_scene(dirs)
    rec_pts = mesh.sample_points_uniformly(2_000_000)
    T_align = np.eye(4)
    if align:
        gt_pcd = gt_mesh.sample_points_uniformly(3_000_000)
        gt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        # A single 30 cm pass settles ~2 cm off (it is pulled by the far, noisy surface), so tighten it.
        for thr in (0.3, 0.1, 0.05, 0.03):
            T_align = o3d.pipelines.registration.registration_icp(
                rec_pts, gt_pcd, thr, T_align,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
        rec_pts.transform(T_align)
    p = np.asarray(rec_pts.points).astype(np.float32)
    acc = gt.compute_distance(o3c.Tensor(p)).numpy()
    rec_scene = o3d.t.geometry.RaycastingScene()
    m = o3d.geometry.TriangleMesh(mesh).transform(T_align)
    rec_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    gt_pts = np.asarray(gt_mesh.sample_points_uniformly(4_000_000).points)
    cams = np.array([cam_pose(r, c)[:3, 3] for r in recs for c in r.get("cams", [])])
    from scipy.spatial import cKDTree

    dist_cam, _ = cKDTree(cams[:, :2]).query(gt_pts[:, :2], k=1)
    seen = (dist_cam < depth_max * 0.7) & (gt_pts[:, 2] < 3.0)
    high = gt_pts[:, 2] >= 3.0
    comp = rec_scene.compute_distance(o3c.Tensor(gt_pts.astype(np.float32))).numpy()
    out = {
        "accuracy_median_m": float(np.median(acc)), "accuracy_mean_m": float(acc.mean()),
        "accuracy_p90_m": float(np.percentile(acc, 90)), "accuracy_within_2cm": float((acc < 0.02).mean()),
        "accuracy_within_5cm": float((acc < 0.05).mean()),
        "completeness_seen_within_5cm": float((comp[seen] < 0.05).mean()),
        "completeness_all_within_5cm": float((comp < 0.05).mean()),
        "completeness_all_within_10cm": float((comp < 0.10).mean()),
        "completeness_high_within_10cm": float((comp[high] < 0.10).mean()), "seen_fraction_of_gt": float(seen.mean()),
        "align_T": np.asarray(T_align).tolist(),
        "align_translation_m": float(np.linalg.norm(T_align[:3, 3])),
        "align_rotation_deg": float(math.degrees(math.acos(np.clip((np.trace(T_align[:3, :3]) - 1) / 2, -1, 1)))),
    }
    return out


def export_glb(path: str, mesh, pcd) -> None:
    import trimesh

    # glTF is Y-up: (x, y, z)_sim -> (x, z, -y).
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
    v = np.asarray(mesh.vertices) @ R.T
    colors = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    tm = trimesh.Trimesh(v, np.asarray(mesh.triangles), vertex_colors=colors, process=False)
    scene = trimesh.Scene()
    scene.add_geometry(tm, node_name="reconstruction", geom_name="reconstruction")
    if pcd is not None and len(pcd.points):
        pc = trimesh.PointCloud(np.asarray(pcd.points) @ R.T,
                                colors=(np.clip(np.asarray(pcd.colors), 0, 1) * 255).astype(np.uint8))
        scene.add_geometry(pc, node_name="lidar_points", geom_name="lidar_points")
    scene.export(path)
    print(f"[recon] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--poses", choices=("gt", "est", "est_icp", *SLAM_MODES), default="gt")
    ap.add_argument("--depth", choices=("gt", "stereo"), default="gt", help="rendered or stereo (_s.png) depth")
    ap.add_argument("--lidar-fuse", action="store_true", help="also integrate Carter lidar sweeps into the TSDF")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--depth-max", type=float, default=8.0, help="camera depth cap for geometry")
    ap.add_argument("--color-max", type=float, default=0.0, help="camera depth cap for color (default depth-max)")
    ap.add_argument("--lidar-max", type=float, default=20.0)
    ap.add_argument("--lidar-mode", choices=("union", "joint"), default="union",
                    help="union: lidar volume fills only where the camera mesh is absent; joint: one shared volume")
    ap.add_argument("--lidar-gap", type=float, default=0.08)
    ap.add_argument("--stereo-edge", type=int, default=0, help="drop stereo pixels within N px of a depth jump")
    ap.add_argument("--depth-near", type=float, default=0.0,
                    help="camera depth up to this builds the mesh; depth up to --depth-max only fills its gaps")
    ap.add_argument("--far-voxel", type=float, default=0.0, help="voxel of the --depth-near..--depth-max layer")
    ap.add_argument("--free-space", action="store_true", help="drop surfaces camera rays see through (floaters)")
    ap.add_argument("--no-recolor", action="store_true", help="keep the TSDF-fused vertex colors")
    ap.add_argument("--stereo-suffix", default="_s", help="stereo depth images to read (<stem><suffix>.png)")
    ap.add_argument("--no-mast", action="store_true", help="leave the forklift's mast lidar out (SLAM and fusion)")
    ap.add_argument("--eval-range", type=float, default=8.0, help="'seen' region: within 0.7x this of a camera")
    ap.add_argument("--slam-cache", default="", help=".npy of SLAM poses to reuse (written if missing)")
    ap.add_argument("--tile", type=float, default=9.0)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--target-tris", type=int, default=2_000_000)
    ap.add_argument("--lite-tris", type=int, default=300_000)
    ap.add_argument("--max-captures", type=int, default=0, help="debug: use only the first N captures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global STEREO_SUFFIX
    STEREO_SUFFIX = os.environ["RECON_STEREO_SUFFIX"] = args.stereo_suffix
    if args.no_mast:
        os.environ["RECON_NO_MAST"] = "1"

    os.makedirs(args.out, exist_ok=True)
    recs = load(args.dirs)
    if args.max_captures:
        recs = recs[:args.max_captures]
    print(f"[recon] {len(recs)} captures, {sum(len(r.get('cams', [])) for r in recs)} camera frames", flush=True)
    estimate_poses(recs, args.poses, cache=args.slam_cache)
    n_all = len(recs)
    recs = [r for r in recs if r.get("localized", True)]
    if len(recs) < n_all:
        print(f"[recon] fusing {len(recs)}/{n_all} captures (the rest were never tied to the map)", flush=True)
    np.save(os.path.join(args.out, "poses.npy"), np.array([r["T_robot"] for r in recs]))
    raw = fuse(recs, args.voxel, args.depth_max, source=args.depth, lidar=args.lidar_fuse, col_max=args.color_max,
               lidar_max=args.lidar_max, tile=args.tile, procs=args.procs, lidar_mode=args.lidar_mode,
               lidar_gap=args.lidar_gap, stereo_edge=args.stereo_edge, near_max=args.depth_near,
               far_voxel=args.far_voxel)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "raw_mesh.ply"), raw)
    mesh = clean(raw, min_tris=200, target=args.target_tris)
    if args.free_space:
        mesh = clean(free_space_filter(mesh, recs, args.depth, procs=args.procs), min_tris=200, target=args.target_tris)
    print(f"[recon] cleaned mesh: {len(mesh.triangles)} triangles", flush=True)
    if not args.no_recolor:
        mesh = recolor(mesh, recs, args.depth, procs=args.procs)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "clean_mesh.ply"), mesh)
    pcd = color_lidar(recs, 0.04, args.depth, args.procs)
    print(f"[recon] lidar points: {len(pcd.points)}", flush=True)
    metrics = evaluate(mesh, recs, args.dirs, args.eval_range, align=args.poses != "gt")
    pose_err, pose_err_al = pose_errors(recs, aligned=False), pose_errors(recs, aligned=True)
    metrics.update({"poses": args.poses, "depth": args.depth, "lidar_fuse": args.lidar_fuse,
                    "depth_near": args.depth_near, "far_voxel": args.far_voxel, "free_space": args.free_space, "stereo_edge": args.stereo_edge,
                    "depth_max": args.depth_max, "voxel_m": args.voxel, "captures": len(recs),
                    "camera_frames": sum(len(r.get("cams", [])) for r in recs), "triangles": len(mesh.triangles),
                    "lidar_points": len(pcd.points), "pose_error_median_m": float(np.median(pose_err)),
                    "pose_error_p90_m": float(np.percentile(pose_err, 90)),
                    "pose_error_aligned_median_m": float(np.median(pose_err_al)),
                    "pose_error_aligned_p90_m": float(np.percentile(pose_err_al, 90))})
    print(json.dumps(metrics, indent=1), flush=True)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    export_glb(os.path.join(args.out, "warehouse_recon.glb"), mesh, pcd)
    lite = o3d.geometry.TriangleMesh(mesh).simplify_quadric_decimation(args.lite_tris)
    lite.compute_vertex_normals()
    export_glb(os.path.join(args.out, "warehouse_recon_lite.glb"), lite, pcd.voxel_down_sample(0.12))


if __name__ == "__main__":
    main()
