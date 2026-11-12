"""Log the fleet's onboard sensing while it drives the warehouse, for offline 3D reconstruction (``recon.py``).

Robots drive expert episodes (random goals, auto-reset) among walking workers. Every ``--every`` control steps the
simulation pauses for a synchronized capture per active robot:

* Nova Carter: RGB, depth (distance to image plane) and a dynamic-object mask (workers, robots, pallets) from the
  left imager of each of its four Hawk stereo cameras, plus a sweep of its Hesai XT-32 (32 channels, -16..+15 deg,
  emulated with PhysX ray casts at the sensor's mount pose).
* Forklift: its planar lidar layers, three Hawk-style stereo pairs on a roof rig, and a wide-field mast lidar
  (Ouster OS0-style: ``--mast-channels`` beams over +-45 deg) just above the rig.

Poses are logged both as ground truth and as the robot's own estimate (localization noise model) and wheel odometry.

Output in ``--out``: ``frames/`` (jpg / 16-bit depth png in mm / mask png), ``lidar/`` (npz per capture),
``captures.jsonl`` (intrinsics, extrinsics, poses), and ``gt_mesh.npz`` (warehouse triangles, for evaluation only).

Usage::

    ./python.sh tools/warehouse_amr/collect_recon.py --gpu 5 --steps 9000 --out ~/warehouse_amr/recon/data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warehouse_env as we

HAWKS = ("front", "left", "right", "back")
XT32_CHANNELS = np.radians(np.arange(-16, 16, 1.0))
FORKLIFT_LAYERS = (0.12, 0.6, 1.8)
DYNAMIC_ROOTS = ("/World/Characters", "/World/Robots", "/World/Pallets")
STEREO_BASELINE = 0.15  # Hawk
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
CV_TO_USD = USD_TO_CV
# Forklift roof rig (no cameras on the stock ForkliftC): Hawk-style stereo pairs, (name, yaw deg, pitch deg).
FORKLIFT_CAMS = ("front", "left", "right")
FORKLIFT_RIG = ((0.0, 10.0), (90.0, 30.0), (-90.0, 30.0))
MAST_LIDAR_FOV = 45.0  # deg, symmetric
MAST_LIDAR_ABOVE_RIG = 0.10


def world_matrix(prim) -> np.ndarray:
    from pxr import Usd, UsdGeom

    # Gf matrices are row-vector; return the column-vector 4x4.
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def export_gt_mesh(stage, path: str) -> None:
    """Triangles of every visible static mesh of the warehouse, in world coordinates."""
    from pxr import Usd, UsdGeom

    verts, faces, n = [], [], 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Root"), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts, idx, cnt = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        if not pts or not idx or not cnt:
            continue
        m = world_matrix(prim)
        p = np.asarray(pts, np.float64)
        p = p @ m[:3, :3].T + m[:3, 3]
        idx, cnt = np.asarray(idx), np.asarray(cnt)
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        tris = [np.stack([idx[s] * np.ones(c - 2, int), idx[s + 1:s + c - 1], idx[s + 2:s + c]], 1)
                for s, c in zip(starts, cnt) if c >= 3]
        if not tris:
            continue
        verts.append(p.astype(np.float32))
        faces.append(np.concatenate(tris) + n)
        n += len(p)
    np.savez_compressed(path, vertices=np.concatenate(verts), faces=np.concatenate(faces).astype(np.int32))
    print(f"[collect] gt mesh: {n} vertices, {sum(len(f) for f in faces)} triangles", flush=True)


def look(yaw: float, pitch: float) -> np.ndarray:
    """Rotation (columns = OpenCV camera axes in the parent frame) looking along yaw / pitch (radians, pitch up)."""
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    x = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    return np.stack([x, np.cross(f, x), f], 1)


def root_matrix(r) -> np.ndarray:
    pos, quat = (a.numpy()[0].astype(np.float64) for a in r.art.get_world_poses())
    from pxr import Gf

    q = Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:])))
    T = np.eye(4)
    T[:3, :3] = np.array(Gf.Matrix3d(q.GetNormalized())).T
    T[:3, 3] = pos
    return T


def forklift_rig(env, r) -> list[np.ndarray]:
    """Left-camera poses of the roof rig in the robot root frame, centered over the cab just above the roof."""
    from pxr import Usd, UsdGeom

    T_root = root_matrix(r)
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
        env.stage.GetPrimAtPath(r.prim_path)).ComputeAlignedRange()
    top = rng.GetMax()[2] - T_root[2, 3] + 0.05
    out = []
    for yaw, pitch in FORKLIFT_RIG:
        R = look(math.radians(yaw), math.radians(pitch))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [r.center_offset[0] + 0.25 * math.cos(math.radians(yaw)) - 0.5 * STEREO_BASELINE * R[0, 0],
                    r.center_offset[1] + 0.25 * math.sin(math.radians(yaw)) - 0.5 * STEREO_BASELINE * R[1, 0], top]
        out.append(T)
    print(f"[collect] {r.name} rig at {top:.2f} m above the root", flush=True)
    return out


class Lidar:
    """PhysX ray-cast lidar; returns world hit points and whether each hit belongs to a dynamic object."""

    def __init__(self, own_prefix: str):
        from omni.physx import get_physx_scene_query_interface

        self.sq = get_physx_scene_query_interface()
        self.own = own_prefix + "/"

    def cast(self, origin, dirs, max_range: float, min_range: float = 0.05):
        import carb

        pts, dyn = [], []
        for d in dirs:
            start = min_range
            for _ in range(4):
                o = origin + d * start
                hit = self.sq.raycast_closest(carb.Float3(*o), carb.Float3(*d), max_range - start)
                if not hit["hit"]:
                    break
                body = hit["rigidBody"] or hit["collision"]
                if body.startswith(self.own):
                    start += hit["distance"] + 0.02
                    continue
                pts.append(o + d * hit["distance"])
                dyn.append(any(body.startswith(r) for r in DYNAMIC_ROOTS))
                break
        return np.array(pts, np.float32).reshape(-1, 3), np.array(dyn, bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--robots", default="forklift,carter,carter,carter,carter")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=9000, help="control steps (0.1 s each)")
    ap.add_argument("--every", type=int, default=10, help="capture period in control steps")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--explore", type=float, default=0.8, help="probability a new goal is the least-visited pick")
    ap.add_argument("--azimuths", type=int, default=720, help="XT-32 columns per sweep")
    ap.add_argument("--mast-channels", type=int, default=64, help="forklift mast lidar beams (0 disables it)")
    ap.add_argument("--mast-azimuths", type=int, default=512, help="forklift mast lidar columns per sweep")
    ap.add_argument("--out", default=os.path.expanduser("~/warehouse_amr/recon/data"))
    args = ap.parse_args()

    app = we.launch_app(args.gpu)
    import cv2
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    env = we.WarehouseEnv(app, robots=args.robots.split(","), workers=args.workers, seed=args.seed)
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "lidar"), exist_ok=True)
    export_gt_mesh(env.stage, os.path.join(args.out, "gt_mesh.npz"))
    env.set_render(True)

    carters = [r for r in env.robots if r.type_name == "carter"]
    forklifts = [r for r in env.robots if r.type_name == "forklift"]
    cam_paths = {r.name: [f"{r.prim_path}/chassis_link/sensors/{h}_hawk/left/camera_left" for h in HAWKS] for r in carters}
    w, h = args.width, args.height
    # The Hawk prims use a deprecated fisheye model, and retargeting render products between cameras crashes the
    # annotator readback. Instead pinhole proxies with the Hawk's field of view (90 deg) are moved onto each robot's
    # cameras in turn: a rectified left / right pair per Hawk.
    hawk = UsdGeom.Camera(env.stage.GetPrimAtPath(cam_paths[carters[0].name][0]))
    focal, hap = hawk.GetFocalLengthAttr().Get(), hawk.GetHorizontalApertureAttr().Get()
    fx = focal / hap * w
    K = [fx, fx, w / 2.0, h / 2.0]
    n_pairs = len(HAWKS)
    rps, annos, proxies = [], [], []
    for k in range(2 * n_pairs):
        cam = UsdGeom.Camera.Define(env.stage, f"/World/ReconCam_{k}")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(hap)
        cam.CreateVerticalApertureAttr(hap * h / w)
        cam.CreateClippingRangeAttr((0.05, 200.0))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        proxies.append(xf.MakeMatrixXform())
        rp = rep.create.render_product(str(cam.GetPath()), (w, h))
        names = ("rgb", "distance_to_image_plane", "instance_id_segmentation_fast") if k % 2 == 0 else ("rgb",)
        a = {n: rep.AnnotatorRegistry.get_annotator(n, init_params=({"colorize": False} if "seg" in n else None))
             for n in names}
        for an in a.values():
            an.attach([rp])
        rps.append(rp)
        annos.append(a)
    xt32 = {r.name: env.stage.GetPrimAtPath(f"{r.prim_path}/chassis_link/sensors/XT_32/PandarXT_32_10hz") for r in carters}
    lidars = {r.name: Lidar(r.prim_path) for r in env.robots}
    az = np.linspace(-math.pi, math.pi, args.azimuths, endpoint=False)
    el = XT32_CHANNELS
    xt_dirs_local = np.stack([np.cos(el)[:, None] * np.cos(az)[None], np.cos(el)[:, None] * np.sin(az)[None],
                              np.sin(el)[:, None] * np.ones_like(az)[None]], -1).reshape(-1, 3)
    fl_az = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    rig = {r.name: forklift_rig(env, r) for r in forklifts}
    mast_az = np.linspace(-math.pi, math.pi, args.mast_azimuths, endpoint=False)
    mast_el = np.radians(np.linspace(-MAST_LIDAR_FOV, MAST_LIDAR_FOV, max(args.mast_channels, 1)))
    mast_dirs_local = np.stack([np.cos(mast_el)[:, None] * np.cos(mast_az)[None],
                                np.cos(mast_el)[:, None] * np.sin(mast_az)[None],
                                np.sin(mast_el)[:, None] * np.ones_like(mast_az)[None]], -1).reshape(-1, 3)

    def render_pairs(lefts):
        """Render rectified stereo pairs whose left cameras have the given camera-to-world poses (OpenCV axes)."""
        for k, T in enumerate(lefts):
            right = T.copy()
            right[:3, 3] += STEREO_BASELINE * T[:3, 0]
            for j, Tc in enumerate((T, right)):
                proxies[2 * k + j].Set(Gf.Matrix4d(*(Tc @ CV_TO_USD).T.flatten().tolist()))
        rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=False)

    # Least-visited goal selection (1 m cells) pushes the fleet into aisles it has neglected.
    m = env.map
    visits = np.zeros((int(m.h * m.res) + 1, int(m.w * m.res) + 1))
    segment = {r.name: 0 for r in env.robots}
    goal_rng = np.random.default_rng(args.seed + 1000)

    def visit_cell(xy):
        c = ((np.asarray(xy[:2]) - m.origin)).astype(int)
        return int(np.clip(c[1], 0, visits.shape[0] - 1)), int(np.clip(c[0], 0, visits.shape[1] - 1))

    def explore_goal(r):
        cells = m.nav(r.spec.inflation)["cells"]
        cand = [m.to_world(c) for c in cells[goal_rng.integers(len(cells), size=400)]]
        cand = [g for g in cand if np.linalg.norm(g - r.pose[0][:2]) > 5.0] or cand
        wgt = np.array([(1.0 + visits[visit_cell(g)]) ** -2.0 for g in cand])
        return cand[int(goal_rng.choice(len(cand), p=wgt / wgt.sum()))]

    meta = open(os.path.join(args.out, "captures.jsonl"), "w")
    env.reset(seed=args.seed)
    t0, n_cap = time.time(), 0
    for step in range(args.steps):
        env.step(env.expert_actions())
        for r in env.robots:
            if r.teleported or not r.active:
                segment[r.name] += 1
            if not r.active:
                continue
            visits[visit_cell(r.pose[0])] += 1
            if r.steps == 0 and goal_rng.random() < args.explore:
                geo = env._set_goal_keep(r, explore_goal(r))
                r.prev_geo = r.best_geo = r.metrics["geodesic"] = geo
        if step % args.every:
            continue
        for ri, r in enumerate(env.robots):
            if not r.active or r.steps < 3:
                continue
            r.read_state()
            center, yaw, _ = r.pose
            rec = {"capture": n_cap, "step": step, "t": step * we.CONTROL_DT, "robot": r.name, "type": r.type_name,
                   "segment": segment[r.name], "episode": r.episode_id,
                   "gt_pose": [float(center[0]), float(center[1]), float(center[2]), float(yaw)],
                   "est_pose": [float(v) for v in r.est_pose], "odom_pose": [float(v) for v in r.odom_pose]}
            if r.type_name == "carter":
                lefts = [world_matrix(env.stage.GetPrimAtPath(p)) @ USD_TO_CV for p in cam_paths[r.name]]
                names = HAWKS
            else:
                T_root = root_matrix(r)
                lefts = [T_root @ T for T in rig[r.name]]
                names = FORKLIFT_CAMS
            render_pairs(lefts)
            rec["cams"] = []
            for k, T in enumerate(lefts):
                a, a_right = annos[2 * k], annos[2 * k + 1]
                rgb = np.asarray(a["rgb"].get_data())[..., :3]
                rgb_r = np.asarray(a_right["rgb"].get_data())[..., :3]
                depth = np.asarray(a["distance_to_image_plane"].get_data(), np.float32)
                seg = a["instance_id_segmentation_fast"].get_data()
                ids, labels = np.asarray(seg["data"]), seg["info"]["idToLabels"]
                dyn_ids = [int(i) for i, lab in labels.items() if any(str(lab).startswith(d) for d in DYNAMIC_ROOTS)]
                mask = np.isin(ids, dyn_ids).astype(np.uint8) * 255
                stem = f"{n_cap:06d}_{r.name}_{names[k]}"
                depth_mm = np.where(np.isfinite(depth) & (depth > 0) & (depth < 65.0), depth * 1000.0, 0)
                frames = os.path.join(args.out, "frames")
                cv2.imwrite(os.path.join(frames, stem + ".jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(frames, stem + "_r.jpg"), rgb_r[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(frames, stem + "_d.png"), depth_mm.astype(np.uint16))
                cv2.imwrite(os.path.join(frames, stem + "_m.png"), mask)
                rec["cams"].append({"name": names[k], "stem": stem, "K": K, "baseline": STEREO_BASELINE,
                                    "T_world_cam": T.tolist()})
            if r.type_name == "carter":
                Tl = world_matrix(xt32[r.name])
                origin = Tl[:3, 3]
                pts, dyn = lidars[r.name].cast(origin, xt_dirs_local @ Tl[:3, :3].T, 60.0)
                rec["lidar_origin"] = origin.tolist()
            else:
                pts_l, dyn_l = [], []
                for z in FORKLIFT_LAYERS:
                    dirs = np.stack([np.cos(fl_az + yaw), np.sin(fl_az + yaw), np.zeros_like(fl_az)], 1)
                    origin = np.array([center[0], center[1], z])
                    p, d = lidars[r.name].cast(origin, dirs, we.SCAN_RANGE, min_range=float(r.half[0]) + 0.05)
                    pts_l.append(p)
                    dyn_l.append(d)
                pts, dyn = np.concatenate(pts_l), np.concatenate(dyn_l)
                rec["lidar_origin"] = [float(center[0]), float(center[1]), 1.0]
            extra = {}
            if r.type_name == "forklift" and args.mast_channels:
                T_root = root_matrix(r)
                o_local = np.array([r.center_offset[0], r.center_offset[1], rig[r.name][0][2, 3] + MAST_LIDAR_ABOVE_RIG])
                origin = T_root[:3, :3] @ o_local + T_root[:3, 3]
                mp, md = lidars[r.name].cast(origin, mast_dirs_local @ T_root[:3, :3].T, 60.0)
                extra = {"mast_points": mp, "mast_dynamic": md}
                rec["mast_lidar_origin"] = origin.tolist()
                rec["mast_lidar_R"] = T_root[:3, :3].tolist()
            np.savez_compressed(os.path.join(args.out, "lidar", f"{n_cap:06d}_{r.name}.npz"), points=pts, dynamic=dyn,
                                **extra)
            meta.write(json.dumps(rec) + "\n")
            n_cap += 1
        meta.flush()
        if step % (args.every * 30) == 0:
            rate = (step + 1) * we.CONTROL_DT / (time.time() - t0)
            print(f"[collect] step {step}/{args.steps} captures {n_cap} ({rate:.2f}x real time) "
                  f"events {[r.events for r in env.robots]}", flush=True)
    meta.close()
    print(f"[collect] done: {n_cap} captures in {time.time() - t0:.0f} s", flush=True)
    app.close()


if __name__ == "__main__":
    main()
