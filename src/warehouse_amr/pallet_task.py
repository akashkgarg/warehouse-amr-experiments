"""Forklift pallet pick-and-place on top of warehouse navigation (``WarehouseEnv(task="pallet")``).

Every forklift owns one dynamic pallet (block pallet with a carton load). An episode is: drive to the pallet,
insert the forks through the pockets of the requested face, lift, carry it to a drop pose, lower it within
``PLACE_TOL`` and back the forks out. Phases: 0 pick, 1 carry (pallet on the raised forks), 2 retreat.

The forks enter along the pallet's y axis (the 1.0 m faces; the 9-block pockets on the other faces are too narrow
for the ForkliftC tines). A *target pose* is the forklift center pose at full insertion; the staging pose is
``STAGE`` metres in front of it along the approach line. A Hybrid-A* path (``hybrid_astar.py``, forward and reverse
arcs) leads to the staging pose; from there the forklift docks straight along the fork axis.

Sensors for the student, on top of the navigation observation: a simulated pallet-pocket detector (noisy relative
target pose when the face is in view of the fork carriage, dead-reckoned on wheel odometry otherwise, a noisy
warehouse-system prior before the first detection), lift height and a load sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PALLET_USD = "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd"
PALLET_HALF = (0.605, 0.5)
PALLET_ZSCALE = 1.3  # stock pockets (0.03-0.136 m) are lower than the ForkliftC blades (0.104-0.141 m)
PALLET_TOP = 0.211 * PALLET_ZSCALE
LOAD_SIZE = (1.10, 0.90, 0.80)
PALLET_MASS, LOAD_MASS = 25.0, 120.0
INSERT_OFFSET = 1.35  # forklift center -> pallet center at full insertion (pallet face at the fork heel)
FRONT_AXLE = 0.26  # fixed (front) axle ahead of the forklift center; the steered axle is at the rear
FORK_TIP = 1.867
STAGE = 4.0
RETREAT = 1.3  # forklift backs out this far after placing (fork tips clear the pallet by ~0.3 m)
LIFT_MAX = 0.30
LIFT_RATE = 0.2  # m/s
CARRY_SPEED = 1.0  # m/s cap with raised forks
CARRY_ACCEL = 1.5  # m/s^2 cap with raised forks
PLACE_TOL = (0.25, math.radians(8.0))
PUSH_TOL = (0.3, math.radians(15.0))
DET_RANGE = (0.3, 6.0)
DET_FOV = math.radians(40.0)
DET_NOISE = (0.02, 0.02, math.radians(1.0))
DET_DROPOUT = 0.1
PRIOR_STD = (0.3, 0.3, math.radians(5.0))
TASK_OBS = 14
TASK_PRIV = 15
PARK_Z = 0.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)])


def rel_pose(base, p) -> np.ndarray:
    """Pose ``p`` (x, y, yaw) expressed in the frame of ``base``."""
    c, s = math.cos(base[2]), math.sin(base[2])
    dx, dy = p[0] - base[0], p[1] - base[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(p[2] - base[2])])


def compose(base, rel) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    return np.array([base[0] + c * rel[0] - s * rel[1], base[1] + s * rel[0] + c * rel[1], wrap(base[2] + rel[2])])


def invert(p) -> np.ndarray:
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([-c * p[0] - s * p[1], s * p[0] - c * p[1], -p[2]])


def drop_target(t, carry) -> np.ndarray:
    """Forklift pose that puts a pallet carried at ``carry`` (forklift frame) onto the drop pose (either yaw)."""
    nominal = target_from_pallet(t.drop, t.drop_side)
    if carry is None:
        return nominal
    cands = [compose(np.array([t.drop[0], t.drop[1], t.drop[2] + f]), invert(carry)) for f in (0.0, math.pi)]
    return min(cands, key=lambda c: abs(wrap(c[2] - nominal[2])))


def target_from_pallet(pallet, side: int) -> np.ndarray:
    """Forklift center pose with the forks fully inserted through the ``side`` (+-1 along pallet y) face."""
    yaw = wrap(pallet[2] + side * math.pi / 2.0)
    d = heading(yaw)
    return np.array([pallet[0] - INSERT_OFFSET * d[0], pallet[1] - INSERT_OFFSET * d[1], yaw])


def dock_errors(center, yaw: float, T) -> tuple[float, float, float]:
    """(distance still to go along the approach line, lateral offset, heading error) to target pose T."""
    d = heading(T[2])
    rel = np.asarray(center[:2]) - T[:2]
    return float(-(rel @ d)), float(rel[0] * -d[1] + rel[1] * d[0]), wrap(yaw - T[2])


def yaw_err_mod_pi(a: float, b: float) -> float:
    e = (a - b) % math.pi
    return min(e, math.pi - e)


@dataclass
class Task:
    pallet0: np.ndarray  # pallet pose (x, y, yaw) at the start of the episode
    side: int
    drop: np.ndarray  # pallet pose to place at
    drop_side: int
    prior: np.ndarray = None  # warehouse-system pallet pose estimate (map frame)
    phase: int = 0
    ref: np.ndarray = None  # pallet rest pose that pushes are measured from
    det_odom: np.ndarray | None = None  # last detected target pose, in the odometry frame
    det_age: float = 1e9
    retreat_odom: np.ndarray | None = None
    retreat: np.ndarray | None = None  # true retreat target (map frame)
    best_phi: float = math.inf
    prev_phi: float | None = None
    t_pick: float | None = None
    repicks: int = 0
    place_err: tuple | None = None
    mode: str = "nav"
    mode_steps: int = 0
    plan: np.ndarray | None = None  # Hybrid-A* path (fixed-axle x, y, yaw, direction)
    plan_idx: int = 0
    plan_goal: np.ndarray | None = None
    plan_attempt: int = 20
    plan_blocked: int = 0
    plans: int = 0
    plan_fail: int = 0
    h_field: np.ndarray | None = None
    h_goal: np.ndarray | None = None
    pallet: np.ndarray = field(default_factory=lambda: np.zeros(3))  # true pallet pose this step
    pallet_z: float = 0.0
    pallet_up: float = 1.0
    loaded: bool = False
    lifted: bool = False
    miss_err: list = field(default_factory=list)  # placement errors of missed drops
    block_run: int = 0
    unstick_dir: int = 0
    carry: np.ndarray | None = None  # pallet pose in the forklift frame while carried (true / own estimate)
    carry_est: np.ndarray | None = None

    def to_json(self) -> dict:
        return {"pallet": self.pallet0.tolist(), "side": self.side, "drop": self.drop.tolist(),
                "drop_side": self.drop_side}


def task_from_json(g: dict) -> Task:
    return Task(np.asarray(g["pallet"], float), int(g["side"]), np.asarray(g["drop"], float), int(g["drop_side"]))


# ---------------------------------------------------------------------------------------------------- scene


def create_pallet(stage, path: str, assets_root: str) -> None:
    """Dynamic block pallet + carton load: box colliders (clean fork pockets) under the stock pallet visual."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    root = UsdGeom.Xform.Define(stage, path)
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(PALLET_MASS + LOAD_MASS)
    vis = stage.DefinePrim(path + "/visual", "Xform")
    vis.GetReferences().AddReference(assets_root + PALLET_USD)
    UsdGeom.Xformable(vis).AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, PALLET_ZSCALE))
    mesh = stage.OverridePrim(path + "/visual/SM_PaletteA_01")
    mesh.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    mesh.RemoveAPI(UsdPhysics.CollisionAPI)

    mat = UsdShade.Material.Define(stage, path + "/friction")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.9)
    api.CreateDynamicFrictionAttr(0.8)
    api.CreateRestitutionAttr(0.0)

    z = PALLET_ZSCALE
    boxes = []
    for y in (-0.405, 0.0, 0.405):  # bottom boards along x
        boxes.append(((0.0, y, 0.015 * z), (1.21, 0.19, 0.03 * z)))
    for x in (-0.52, 0.0, 0.52):  # 9 blocks
        for y in (-0.405, 0.0, 0.405):
            boxes.append(((x, y, 0.083 * z), (0.16, 0.19, 0.106 * z)))
    boxes.append(((0.0, 0.0, 0.1735 * z), (1.21, 1.0, 0.075 * z)))  # top deck
    for k, (c, size) in enumerate(boxes):
        cube = UsdGeom.Cube.Define(stage, f"{path}/col_{k}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*c))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    load = UsdGeom.Cube.Define(stage, path + "/load")
    load.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(load)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, PALLET_TOP + LOAD_SIZE[2] / 2.0))
    xf.AddScaleOp().Set(Gf.Vec3f(*LOAD_SIZE))
    load.CreateDisplayColorAttr([(0.62, 0.47, 0.30)])
    UsdPhysics.CollisionAPI.Apply(load.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(load.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def fit_fork_colliders(stage, robot_path: str) -> None:
    """Replace the fork mesh colliders with boxes (blade + shank) on the lift link, at the identity robot pose."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    lift = stage.GetPrimAtPath(robot_path + "/lift")
    for name in ("SM_Forklift_C01_Fork01_01", "SM_Forklift_C01_Fork02_01"):
        p = stage.GetPrimAtPath(f"{robot_path}/lift/{name}")
        if p.IsInstance():
            p.SetInstanceable(False)
        for q in Usd.PrimRange(p):
            if q.HasAPI(UsdPhysics.CollisionAPI):
                q.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    to_lift = UsdGeom.Xformable(lift).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    mat = UsdShade.Material.Define(stage, robot_path + "/fork_friction")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.9)
    api.CreateDynamicFrictionAttr(0.8)
    k = 0
    for y0, y1 in ((0.242, 0.344), (-0.344, -0.242)):
        for (x0, x1), (z0, z1) in (((0.80, FORK_TIP), (0.104, 0.141)), ((0.749, 0.80), (0.104, 0.609))):
            c = to_lift.Transform(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
            # The lift link is authored in cm (and possibly rotated): express the world box size in its frame.
            size = sum((np.abs(np.array(to_lift.TransformDir(Gf.Vec3d(*v)))) for v in
                        ((x1 - x0, 0, 0), (0, y1 - y0, 0), (0, 0, z1 - z0))), np.zeros(3))
            cube = UsdGeom.Cube.Define(stage, f"{robot_path}/lift/fork_col_{k}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(c)
            xf.AddScaleOp().Set(Gf.Vec3f(*size.tolist()))
            cube.MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            k += 1


class Pallet:
    def __init__(self, path: str):
        self.path = path
        self.view = None

    def initialize(self) -> None:
        from isaacsim.core.experimental.prims import RigidPrim

        self.view = RigidPrim(self.path)

    def place(self, xy, yaw: float, z: float = 0.005) -> None:
        import warp as wp

        self.view.set_world_poses(
            positions=wp.array([[float(xy[0]), float(xy[1]), z]], dtype=wp.float32),
            orientations=wp.array([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], dtype=wp.float32),
        )
        self.view.set_velocities(
            linear_velocities=wp.zeros((1, 3), dtype=wp.float32), angular_velocities=wp.zeros((1, 3), dtype=wp.float32)
        )

    def read(self) -> tuple[np.ndarray, float, float]:
        """(x, y, yaw), z, up-axis z component."""
        pos, q = (a.numpy()[0].astype(np.float64) for a in self.view.get_world_poses())
        w, x, y, z = q
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([pos[0], pos[1], yaw]), float(pos[2]), 1.0 - 2.0 * (x * x + y * y)


# ---------------------------------------------------------------------------------------------------- episodes


def sample_spot(env, robot, rng, near=None, dist_range=(6.0, 30.0), avoid=(), tries: int = 600):
    """A pallet pose + entry side whose footprint and straight approach corridor are free, staging on the nav graph."""
    nav = env.map.nav(robot.spec.inflation)
    for _ in range(tries):
        c = env.map.to_world(nav["cells"][rng.integers(len(nav["cells"]))])
        if near is not None and not (dist_range[0] <= np.linalg.norm(c - near) <= dist_range[1]):
            continue
        if any(np.linalg.norm(c - np.asarray(a)[:2]) < 5.0 for a in avoid):
            continue
        psi = float(rng.uniform(-math.pi, math.pi))
        side = int(rng.choice([-1, 1]))
        if not env.map.footprint_clear(c, psi, PALLET_HALF, margin=0.3):
            continue
        T = target_from_pallet((c[0], c[1], psi), side)
        d = heading(T[2])
        if not all(env.map.footprint_clear(T[:2] - d * a, T[2], robot.half, margin=0.1)
                   for a in np.arange(0.0, STAGE + 0.81, 0.4)):
            continue
        lane = [env.map.to_cell(T[:2] - d * a) for a in np.arange(0.0, STAGE + 0.61, 0.2)]
        if not all(nav["mask"][rc] for rc in lane):
            continue
        return np.array([c[0], c[1], psi]), side
    raise RuntimeError(f"no pallet spot for {robot.name}")


def sample_task(env, robot, start, rng, avoid=()) -> Task:
    pallet, side = sample_spot(env, robot, rng, near=np.asarray(start), dist_range=(6.0, 25.0), avoid=avoid)
    drop, drop_side = sample_spot(env, robot, rng, near=pallet[:2], dist_range=(8.0, 30.0),
                                  avoid=list(avoid) + [pallet[:2]])
    return Task(pallet, side, drop, drop_side)


# ---------------------------------------------------------------------------------------------------- per step


def begin(env, r, task: Task) -> None:
    """Start ``task`` for forklift ``r`` (pallet already placed at ``task.pallet0``)."""
    task.prior = task.pallet0 + env.rng.normal(0.0, PRIOR_STD)
    task.ref = task.pallet0.copy()
    task.pallet = task.pallet0.copy()
    r.task = task
    env._set_goal(r, staging(target_from_pallet(task.prior, task.side)))
    task.h_field = r.field


def staging(T) -> np.ndarray:
    return T[:2] - STAGE * heading(T[2])


def true_target(r) -> np.ndarray:
    t = r.task
    if t.phase == 0:
        side = t.side if t.repicks == 0 else _side_facing(r, t.pallet)
        return target_from_pallet(t.pallet, side)
    if t.phase == 1:
        return drop_target(t, t.carry)
    return t.retreat


def _side_facing(r, pallet) -> int:
    yaw_axis = heading(pallet[2] + math.pi / 2.0)
    return 1 if heading(r.pose[1]) @ yaw_axis >= 0 else -1


def pose3(r) -> np.ndarray:
    return np.array([r.pose[0][0], r.pose[0][1], r.pose[1]])


def detect(env, r) -> None:
    """Pallet-pocket detector on the fork carriage (phase 0 only): noisy relative target pose, else dead reckoning."""
    import carb
    from omni.physx import get_physx_scene_query_interface

    t = r.task
    t.det_age += 0.1
    if t.phase != 0:
        return
    T = true_target(r)
    me = pose3(r)
    d = heading(T[2])
    face = T[:2] + d * (INSERT_OFFSET - PALLET_HALF[1])
    sensor = me[:2] + heading(me[2]) * 0.9
    v = face - sensor
    dist = float(np.linalg.norm(v))
    if not (DET_RANGE[0] <= dist <= DET_RANGE[1]):
        return
    if abs(wrap(math.atan2(v[1], v[0]) - me[2])) > DET_FOV or (v / dist) @ d < math.cos(math.radians(60)):
        return
    if env.rng.random() < DET_DROPOUT:
        return
    hit = get_physx_scene_query_interface().raycast_closest(
        carb.Float3(sensor[0], sensor[1], 0.11), carb.Float3(v[0] / dist, v[1] / dist, 0.0), dist + 0.3)
    body = hit["rigidBody"] or hit["collision"] if hit["hit"] else ""
    if body != r.pallet.path and not body.startswith(r.pallet.path + "/"):
        return
    rel = rel_pose(me, T) + env.rng.normal(0.0, DET_NOISE)
    t.det_odom = compose(r.odom_pose, rel)
    t.det_age = 0.0


def observed_target(env, r) -> tuple[np.ndarray, float]:
    """Target pose in the robot frame as the robot's own sensors see it, plus detection freshness."""
    t = r.task
    if t.phase == 0:
        if t.det_odom is not None:
            return rel_pose(r.odom_pose, t.det_odom), math.exp(-t.det_age / 2.0)
        return rel_pose(r.est_pose, target_from_pallet(t.prior, t.side)), 0.0
    if t.phase == 1:
        return rel_pose(r.est_pose, drop_target(t, t.carry_est)), 0.0
    return rel_pose(r.odom_pose, t.retreat_odom), 0.0


def _axle(pose) -> np.ndarray:
    return np.array([pose[0] + FRONT_AXLE * math.cos(pose[2]), pose[1] + FRONT_AXLE * math.sin(pose[2]), pose[2]])


def in_lane(along: float, lat: float, yerr: float) -> bool:
    return -0.3 < along < STAGE + 1.0 and abs(lat) < 0.5 and abs(yerr) < 0.5


def ensure_plan(env, r) -> None:
    """Keep a Hybrid-A* path (fixed-axle poses) from the forklift to the staging pose of the current target."""
    t = r.task
    t.plan_attempt += 1
    if t.phase == 2:
        t.plan = None
        return
    T = true_target(r)
    me = pose3(r)
    if in_lane(*dock_errors(me[:2], me[2], T)):
        t.plan = None
        return
    S = staging(T)
    goal = _axle((S[0], S[1], T[2]))
    ax = _axle(me)
    need = t.plan is None or np.linalg.norm(t.plan_goal[:2] - goal[:2]) > 0.3 or t.plan_blocked > 30
    if t.plan is not None and not need:
        seg_end = _segment_end(t.plan, t.plan_idx)
        lo, hi = max(t.plan_idx - 2, 0), min(t.plan_idx + 40, seg_end + 1)
        d = np.linalg.norm(t.plan[lo:hi, :2] - ax[:2], axis=1)
        k = int(np.argmin(d))
        t.plan_idx = lo + k
        if t.plan_idx >= seg_end - 2 and seg_end + 1 < len(t.plan):
            t.plan_idx = seg_end + 1  # at the cusp: continue with the next direction
        # Off the path, or at its end without having reached the approach lane: plan again.
        need = d[k] > 1.0 or t.plan_idx >= len(t.plan) - 2
    if not need or t.plan_attempt < 20:
        return
    t.plan_attempt = 0
    if t.h_goal is None or np.linalg.norm(t.h_goal - S) > 0.3:
        t.h_field, t.h_goal = env.map.distance_field(r.spec.inflation, S), S.copy()
        r.goal, r.field = S, t.h_field
    planner = env.planner(r)
    inf = r.spec.inflation

    def h(x, y):
        g = env.map.geodesic(inf, t.h_field, (x, y))
        return g if math.isfinite(g) else 1e3

    obstacles = [pallet_outline(o.task.pallet) for o in env.robots
                 if o.task is not None and o.active and not (o is r and t.phase == 1)]
    if t.plan_blocked > 30:
        hits = r.scan - r.scan_body < 8.0
        ang = r.scan_angles[hits] + me[2]
        rng_ = r.scan[hits]
        obstacles.append(np.stack([me[0] + rng_ * np.cos(ang), me[1] + rng_ * np.sin(ang)], 1))
    obstacles = np.concatenate(obstacles) if obstacles else None
    path = planner.plan(ax, goal, h, obstacles=obstacles, lane=(0.0, 2.5))
    t.plan_blocked = 0
    t.plans += 1
    if path is None:
        t.plan_fail += 1
        return
    t.plan, t.plan_idx, t.plan_goal = path, 0, goal
    t.prev_phi = None


def pallet_outline(pallet, spacing: float = 0.15) -> np.ndarray:
    """Points along the pallet footprint boundary (planner obstacles; pallets are not in the static map)."""
    hx, hy = PALLET_HALF
    xs = np.arange(-hx, hx + 1e-6, spacing)
    ys = np.arange(-hy, hy + 1e-6, spacing)
    local = np.concatenate([np.stack([xs, np.full_like(xs, -hy)], 1), np.stack([xs, np.full_like(xs, hy)], 1),
                            np.stack([np.full_like(ys, -hx), ys], 1), np.stack([np.full_like(ys, hx), ys], 1)])
    c, s = math.cos(pallet[2]), math.sin(pallet[2])
    return np.stack([pallet[0] + c * local[:, 0] - s * local[:, 1], pallet[1] + s * local[:, 0] + c * local[:, 1]], 1)


def _segment_end(p, i: int) -> int:
    j = i
    while j + 1 < len(p) and p[j + 1, 3] == p[i, 3]:
        j += 1
    return j


def _plan_state(t, ax, look: float = 1.5):
    """(lookahead axle point, direction, distance to the end of the current direction segment, remaining length)."""
    p = t.plan
    i = t.plan_idx
    d = p[i, 3]
    seg_end = _segment_end(p, i)
    steps = np.linalg.norm(np.diff(p[i:, :2], axis=0), axis=1)
    to_end = float(steps[: seg_end - i].sum()) if seg_end > i else 0.0
    acc, j = 0.0, i
    while j < seg_end and acc < look:
        acc += float(np.linalg.norm(p[j + 1, :2] - p[j, :2]))
        j += 1
    target = p[j, :2]
    if acc < look and j == len(p) - 1:
        target = p[j, :2] + d * (look - acc) * heading(p[j, 2])
    return target, int(d), to_end, float(steps.sum())


def _path_features(env, r, frame) -> np.ndarray:
    """Lookahead point on the path (robot frame of ``frame``), direction, distance to the next cusp / end."""
    t = r.task
    if t.plan is not None:
        target, d, to_end, _ = _plan_state(t, _axle(pose3(r)))
        rel = rel_pose(frame, (target[0], target[1], 0.0))[:2]
        return np.array([*np.clip(rel / 3.0, -2, 2), d, min(to_end / 5.0, 2.0)])
    T = true_target(r)
    along = dock_errors(r.pose[0], r.pose[1], T)[0]
    d = -1 if t.phase == 2 else 1
    ahead = T[:2] - heading(T[2]) * (along - d * 1.5)
    rel = rel_pose(frame, (ahead[0], ahead[1], 0.0))[:2]
    return np.array([*np.clip(rel / 3.0, -2, 2), d, min(max(along, 0.0) / 5.0, 2.0)])


def _task_vec(rel, fresh: float, lift: float, loaded: bool, phase: int, path) -> np.ndarray:
    return np.concatenate([
        np.clip(rel[:2] / 5.0, -3, 3), [math.cos(rel[2]), math.sin(rel[2]), fresh, lift / LIFT_MAX, float(loaded)],
        np.eye(3)[phase], path,
    ])


def observe(env, r) -> np.ndarray:
    if r.task is None:
        return np.zeros(TASK_OBS)
    rel, fresh = observed_target(env, r)
    return _task_vec(rel, fresh, r.lift, r.task.loaded, r.task.phase, _path_features(env, r, r.est_pose))


def observe_privileged(env, r) -> np.ndarray:
    if r.task is None:
        return np.zeros(TASK_PRIV)
    t = r.task
    me = pose3(r)
    rel = rel_pose(me, true_target(r))
    push = np.linalg.norm(t.pallet[:2] - t.ref[:2]) / PUSH_TOL[0] if t.phase == 0 else 0.0
    return np.concatenate([_task_vec(rel, 1.0, r.lift, t.loaded, t.phase, _path_features(env, r, me)), [push]])


def potential(env, r, T) -> float:
    """Remaining plan length + staging distance; inside the approach lane the docking error."""
    t = r.task
    center, yaw, _ = r.pose
    along, lat, yerr = dock_errors(center, yaw, T)
    if t.phase == 2:
        return abs(along) + abs(lat)
    if in_lane(along, lat, yerr) or t.plan is None and along < STAGE + 1.0 and abs(lat) < 1.0:
        return along + 1.5 * abs(lat) + abs(yerr)
    if t.plan is not None:
        return _plan_state(t, _axle(pose3(r)))[3] + STAGE
    geo = env.map.geodesic(r.spec.inflation, r.field, center[:2])
    return (geo if math.isfinite(geo) else 50.0) + STAGE


def _new_phase(env, r, phase: int) -> None:
    t = r.task
    t.phase = phase
    t.prev_phi = None
    t.best_phi = math.inf
    t.mode, t.mode_steps = "nav", 0
    t.plan, t.plan_attempt = None, 20
    r.since_progress = 0
    if phase in (0, 1):
        T = target_from_pallet(t.pallet, _side_facing(r, t.pallet)) if phase == 0 else \
            drop_target(t, t.carry)
        env._set_goal_keep(r, staging(T))
        t.h_field, t.h_goal = r.field, staging(T)
    else:
        me = pose3(r)
        t.retreat = compose(me, (-RETREAT, 0.0, 0.0))
        t.retreat_odom = compose(r.odom_pose, (-RETREAT, 0.0, 0.0))
        env._set_goal_keep(r, t.retreat[:2])


def update(env, r) -> tuple[float, str | None]:
    """Read the pallet, run phase transitions and failure checks; returns (task reward, task event)."""
    t = r.task
    t.pallet, t.pallet_z, t.pallet_up = r.pallet.read()
    detect(env, r)
    me = pose3(r)
    if t.pallet_up < 0.9 or t.pallet_z < -0.3:
        return -10.0, "pallet_drop"
    rel = rel_pose(me, t.pallet)
    t.lifted = t.pallet_z > 0.03
    on_forks = abs(rel[0] - INSERT_OFFSET) < 0.35 and abs(rel[1]) < 0.15
    t.loaded = t.lifted and on_forks and r.lift > 0.04
    bonus = 0.0
    if t.phase == 0:
        if t.loaded:
            t.t_pick = r.steps * 0.1
            t.carry = rel
            if t.det_odom is not None:
                # Own estimate: the last detected pallet, carried along by odometry.
                t.carry_est = rel_pose(r.odom_pose, compose(t.det_odom, (INSERT_OFFSET, 0.0, -t.side * math.pi / 2)))
            else:
                t.carry_est = np.array([INSERT_OFFSET, 0.0, rel[2]])
            _new_phase(env, r, 1)
            bonus = 5.0
        elif not t.lifted:
            moved = np.linalg.norm(t.pallet[:2] - t.ref[:2]) > PUSH_TOL[0] or \
                yaw_err_mod_pi(t.pallet[2], t.ref[2]) > PUSH_TOL[1]
            if moved:
                if np.linalg.norm(me[:2] - t.pallet[:2]) < 4.0:
                    return -10.0, "pallet_push"
                t.ref = t.pallet.copy()  # nudged by a worker
                t.det_odom = None
    elif t.phase == 1:
        if not t.lifted:
            err = (float(np.linalg.norm(t.pallet[:2] - t.drop[:2])), yaw_err_mod_pi(t.pallet[2], t.drop[2]))
            t.ref = t.pallet.copy()
            if err[0] < PLACE_TOL[0] and err[1] < PLACE_TOL[1]:
                t.place_err = err
                _new_phase(env, r, 2)
                bonus = 5.0
            else:
                t.repicks += 1
                t.miss_err.append((round(err[0], 3), round(math.degrees(err[1]), 1)))
                t.det_odom = None
                _new_phase(env, r, 0)
                bonus = -2.0
    else:
        if t.lifted:
            _new_phase(env, r, 1)
            bonus = -2.0
        else:
            along = dock_errors(me[:2], me[2], t.retreat)[0]
            if along < 0.1:
                err = (float(np.linalg.norm(t.pallet[:2] - t.drop[:2])), yaw_err_mod_pi(t.pallet[2], t.drop[2]))
                if err[0] < PLACE_TOL[0] and err[1] < PLACE_TOL[1]:
                    t.place_err = err
                    return 10.0, "success"
                t.repicks += 1
                _new_phase(env, r, 0)
                bonus = -2.0
    replans = t.plans
    ensure_plan(env, r)
    phi = potential(env, r, true_target(r))
    if t.plans != replans:
        t.prev_phi = None
    prog = 0.0 if t.prev_phi is None else float(np.clip(t.prev_phi - phi, -0.5, 0.5))
    t.prev_phi = phi
    if phi < t.best_phi - 0.2:
        t.best_phi, r.since_progress = phi, 0
    elif t.lifted and t.phase == 0 or r.lift_moving:
        r.since_progress = 0
    else:
        r.since_progress += 1
        if t.plans != replans:
            t.best_phi = phi
    if r.since_progress > int(20.0 / 0.1):
        return prog + bonus - 2.0, "stuck"
    return prog + bonus, None


# ---------------------------------------------------------------------------------------------------- expert


def steer_to(r, point, direction: int) -> float:
    """Pure-pursuit steering (normalized) of the fixed front axle toward ``point``, driving forward or reverse."""
    center, yaw, _ = r.pose
    fa = center[:2] + FRONT_AXLE * heading(yaw)
    v = np.asarray(point) - fa
    look = max(float(np.linalg.norm(v)), 0.5)
    alpha = wrap(math.atan2(v[1], v[0]) - yaw)
    wb = r.spec.controller["wheel_base"]
    if direction < 0:
        steer = math.atan2(2.0 * wb * math.sin(wrap(alpha - math.pi)), look)
        return float(np.clip(-steer / r.spec.max_turn, -1, 1))
    steer = math.atan2(2.0 * wb * math.sin(alpha), look)
    return float(np.clip(steer / r.spec.max_turn, -1, 1))


def pursuit(r, T, reverse: bool, look: float = 1.2) -> float:
    """Steering onto the approach line of T (carrot ``look`` m ahead of the axle's projection)."""
    center, yaw, _ = r.pose
    fa = center[:2] + FRONT_AXLE * heading(yaw)
    d = heading(T[2])
    tf = T[:2] + FRONT_AXLE * d
    a = float((fa - tf) @ d)
    return steer_to(r, tf + d * (a - look if reverse else a + look), -1 if reverse else 1)


def expert(env, r, humans) -> np.ndarray:
    """Privileged teacher: follow the Hybrid-A* path to the staging pose, then pure-pursuit docking along the fork
    axis (backing off and retrying when the fork tips would miss the pockets); lift / lower in position."""
    t = r.task
    center, yaw, _ = r.pose
    T = true_target(r)
    along, lat, yerr = dock_errors(center, yaw, T)
    lift = 0.0 if t.phase == 1 else -1.0
    front = np.abs(np.array([wrap(a) for a in r.beam_angles])) < math.radians(35)
    rear = np.abs(np.array([wrap(a - math.pi) for a in r.beam_angles])) < math.radians(35)
    clear_f, clear_r = float(r.clearance[front].min()), float(r.clearance[rear].min())
    t.mode_steps += 1
    dbg = {}

    def out(speed, turn, lift_=lift):
        if speed:
            hs = env._human_scale(r, speed)
            speed *= hs
            dbg.update(hs=round(hs, 2), cf=round(clear_f, 2), cr=round(clear_r, 2))
            if speed > 0 and clear_f < 0.3 or speed < 0 and clear_r < 0.3:
                t.plan_blocked += 1
                t.block_run += 1
                if t.block_run > 40 and (clear_r if speed > 0 else clear_f) > 0.8:
                    # Blocked for 4 s: drive ~1 m the other way, then plan again.
                    t.unstick_dir, t.block_run = (-1 if speed > 0 else 1), 0
                    switch("unstick")
                speed = 0.0
            else:
                t.block_run = 0
        r.expert_dbg = {"mode": t.mode, "phase": t.phase, "along": round(along, 2), "lat": round(lat, 3),
                        "yerr": round(float(yerr), 3), "plans": t.plans, "fails": t.plan_fail, **dbg}
        return np.array([speed, turn, lift_], np.float32)

    def switch(mode):
        t.mode, t.mode_steps = mode, 0

    if t.phase == 2:
        return out(-0.2, pursuit(r, T, reverse=True), -1.0)
    if t.mode == "unstick":
        if t.mode_steps < 20:
            return out(0.25 * t.unstick_dir, 0.0)
        switch("nav")
        t.plan, t.plan_attempt = None, 20
    if t.mode == "backoff":
        if along < STAGE - 0.5 and t.mode_steps < 150:
            return out(-0.25, pursuit(r, T, reverse=True))
        switch("dock")
    if t.mode != "dock" and in_lane(along, lat, yerr):
        switch("dock")
    if t.mode == "dock":
        if not (-0.5 < along < STAGE + 2.5 and abs(lat) < 0.9 and abs(yerr) < 0.8):
            switch("nav")
        else:
            tip_lat = lat + FORK_TIP * math.sin(yerr)
            if t.phase == 0 and 0.2 < along < 1.25 and (abs(tip_lat) > 0.06 or abs(yerr) > 0.06) or \
                    t.phase == 1 and along < 0.6 and (abs(lat) > 0.12 or abs(yerr) > 0.08):
                switch("backoff")
                return out(-0.25, pursuit(r, T, reverse=True))
            if along <= 0.03:
                # In position: lift (pick) or lower (drop) with the truck stopped.
                return out(0.0, 0.0, 0.0 if t.phase == 0 else -1.0)
            speed = 0.3 if along > 2.0 else 0.12 if along > 0.4 else 0.07
            return out(speed, pursuit(r, T, reverse=False))
    if t.plan is not None:
        target, d, to_end, _ = _plan_state(t, _axle(pose3(r)))
        turn = steer_to(r, target, d)
        v = (0.6 if abs(turn) < 0.3 else 0.35) if d > 0 else 0.4
        v *= float(np.clip(to_end / 1.5, 0.4, 1.0))
        return out(d * v, turn)
    if np.linalg.norm(t.pallet[:2] - center[:2]) < 3.0 and not t.loaded:
        # No plan next to the pallet (the forks overlap it after a missed placement): back straight out first.
        return out(-0.25, 0.0)
    # No plan (planning failed or pending): plain geodesic driving.
    nav = env._expert_nav(r, humans)
    r.expert_dbg = {"mode": "geo", "phase": t.phase, "plans": t.plans, "fails": t.plan_fail}
    return np.array([nav[0], nav[1], lift], np.float32)
