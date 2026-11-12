"""Multi-robot warehouse environment for testing and training mobile-robot policies.

Scene: ``full_warehouse.usd`` with walking workers (Isaac Replicator Agent wander routines on the warehouse
navmesh) and physics-driven robots: the articulated ``ForkliftC`` (Ackermann, rear-wheel steering) and Nova
Carters (differential drive). Every robot is an independent agent that auto-resets: on success it receives a
new goal, on failure (collision, tip-over, stuck) it respawns at a free location.

Action per robot, normalized to ``[-1, 1]^2``: ``[speed, turn]`` where turn is the steering angle for the
forklift and the yaw rate for the Carter. With ``task="pallet"`` forklifts run pallet pick-and-place episodes
(``pallet_task.py``) and the action gains a third entry, the fork lift height command (ignored by Carters); the
observations grow by the previous lift command and the task block.

Two observations per robot:

* sensor (``OBS_DIM``, what a deployed policy sees): goal and global-path direction from the *estimated* pose
  (localization noise model, or an external estimate such as AMCL passed to ``step``), geodesic distance on the
  static map, wheel odometry, previous action, the current and a 0.3 s old binned lidar scan, robot type.
  Workers are visible only through the lidar.
* privileged (``PRIV_DIM``, for the scripted teacher / critic): the same quantities from ground truth plus the two
  nearest workers' true positions.

Each robot carries a planar ``SCAN_BEAMS`` lidar (flattened over the robot's mount heights, like a 3D lidar
projected to 2D) that is also published to ROS by ``nav2_bridge.py``.

Must be imported after ``launch_app()`` has created the SimulationApp.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pallet_task as pt

EXPERIENCE = "isaacsim.exp.action_and_event_data_generation.base.kit"
WAREHOUSE_USD = "Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
CHARACTERS_ROOT = "Isaac/People/Characters/"

PHYSICS_DT = 1.0 / 60.0
CONTROL_DT = 0.1
SCAN_BEAMS = 180
SCAN_RANGE = 20.0
SCAN_START = 0.02  # rays start this far outside the footprint so they never hit the robot itself
LIDAR_BEAMS = 36  # clearance sectors (min over SCAN_BEAMS // LIDAR_BEAMS beams) used by the teacher and rewards
LIDAR_RANGE = 10.0
SCAN_BINS = 90
SCAN_DELAY = 3  # control steps between the two scans in the sensor observation
MAP_RES = 0.1
MAP_SLICES = (0.08, 0.35, 0.6, 1.0, 1.5, 2.1)
HUMAN_RADIUS = 0.3
PRIV_DIM = 2 + 1 + 2 + 2 + 2 + LIDAR_BEAMS + 6 + 2
OBS_DIM = 2 + 1 + 2 + 2 + 2 + 2 * SCAN_BINS + 2
ACT_DIM = 2

# Sensor noise: lidar range noise / dropout, wheel odometry scale bias + noise, and a localization error that
# wanders like an AMCL estimate (Ornstein-Uhlenbeck, stationary std LOC_STD, correlation time LOC_TAU).
SCAN_NOISE = 0.02
SCAN_DROPOUT = 0.005
ODOM_SCALE_STD = 0.02
ODOM_NOISE = (0.02, 0.02)
LOC_STD = (0.10, 0.10, math.radians(2.0))
LOC_TAU = 5.0
TYPE_NAMES = ("forklift", "carter")


def launch_app(
    gpu: int = 0,
    headless: bool = True,
    width: int = 1280,
    height: int = 720,
    texture_budget_mb: int = 3072,
):
    """Create the SimulationApp with the Action and Event Data Generation experience (needed for IRA workers).

    ``gpu`` is the nvidia-smi index.
    """
    from isaacsim import SimulationApp

    # Two NVIDIA Vulkan ICD manifests list every GPU twice, which doubles Kit's device indices.
    icd = "/etc/vulkan/icd.d/nvidia_icd.json"
    if os.path.exists(icd) and os.path.exists("/etc/vulkan/icd.d/nvidia_icd.x86_64.json"):
        os.environ.setdefault("VK_DRIVER_FILES", icd)
        os.environ.setdefault("VK_ICD_FILENAMES", icd)

    app = SimulationApp(
        {
            "headless": headless,
            "width": width,
            "height": height,
            "multi_gpu": False,
            "active_gpu": gpu,
            "physics_gpu": gpu,
            # The default texture budget is a fraction of total VRAM, which overcommits GPUs shared with other jobs.
            "extra_args": [
                "--/rtx-transient/resourcemanager/enableTextureStreaming=true",
                f"--/rtx-transient/resourcemanager/texturestreaming/streamingBudgetMB={texture_budget_mb}",
                "--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.03",
                # Allow PhysX to substep a whole control period inside one app update.
                "--/persistent/simulation/minFrameRate=5",
            ],
        },
        experience=os.path.join(os.environ["EXP_PATH"], EXPERIENCE),
    )
    import carb
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    settings = carb.settings.get_settings()
    settings.set("/app/scripting/ignoreWarningDialog", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)
    settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
    return app


@dataclass(frozen=True)
class RobotSpec:
    kind: str
    usd: str
    max_speed: float
    max_turn: float
    goal_tolerance: float
    inflation: float
    lookahead: float
    stop_distance: float
    lidar_layers: tuple[float, ...]
    controller: dict = field(default_factory=dict)
    max_lateral_accel: float = 0.0
    yaw_feedback: float = 0.0


ROBOT_SPECS = {
    "forklift": RobotSpec(
        kind="ackermann",
        usd="/Isaac/Robots_Multiphysics/IsaacSim/ForkliftC/forklift_c/forklift_c.usda",
        max_speed=2.0,
        max_turn=0.7,
        goal_tolerance=1.5,
        inflation=0.95,
        lookahead=3.0,
        stop_distance=1.2,
        lidar_layers=(0.12, 0.6, 1.8),
        controller=dict(
            left_steerable_wheel_joint="left_back_wheel_joint",
            right_steerable_wheel_joint="right_back_wheel_joint",
            left_steering_joint="left_rotator_joint",
            right_steering_joint="right_rotator_joint",
            left_non_steerable_wheel_joint="left_front_wheel_joint",
            right_non_steerable_wheel_joint="right_front_wheel_joint",
            non_steerable_wheel_radius=0.325,
            non_steerable_track_width=1.044,
            steerable_wheel_radius=0.255,
            wheel_base=1.652,
            track_width=1.140,
            steerable_wheels_at_rear=True,
        ),
    ),
    "carter": RobotSpec(
        kind="differential",
        usd="/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        max_speed=1.5,
        max_turn=1.5,
        goal_tolerance=0.6,
        inflation=0.6,
        lookahead=1.5,
        stop_distance=0.4,
        lidar_layers=(0.12, 0.4),
        controller=dict(
            left_wheel_joint="joint_wheel_left",
            right_wheel_joint="joint_wheel_right",
            wheel_radius=0.14,
            wheel_base=0.4132,
        ),
        max_lateral_accel=0.7,
        yaw_feedback=0.8,
    ),
}


def yaw_from_quat(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def up_z_from_quat(q) -> float:
    w, x, y, z = q
    return 1.0 - 2.0 * (x * x + y * y)


def quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GridMap:
    """2D occupancy grid of the warehouse floor with per-robot inflated navigation graphs."""

    def __init__(self, free: np.ndarray, occupied: np.ndarray, origin_xy: tuple[float, float], res: float):
        self.free = free
        self.occupied = occupied
        self.origin = np.asarray(origin_xy, dtype=np.float64)
        self.res = res
        self.h, self.w = free.shape
        self._nav: dict[float, dict] = {}

    def to_cell(self, xy) -> tuple[int, int]:
        c = ((np.asarray(xy[:2]) - self.origin) / self.res).astype(int)
        return int(np.clip(c[1], 0, self.h - 1)), int(np.clip(c[0], 0, self.w - 1))

    def to_world(self, rc) -> np.ndarray:
        return self.origin + (np.array([rc[1], rc[0]], dtype=np.float64) + 0.5) * self.res

    def nav(self, inflation: float) -> dict:
        """Eroded free mask, nearest-valid-cell lookup, and an 8-connected sparse graph for that inflation."""
        if inflation in self._nav:
            return self._nav[inflation]
        import cv2
        import scipy.ndimage as ndi
        import scipy.sparse as sp

        r = int(math.ceil(inflation / self.res))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        blocked = cv2.dilate((~self.free).astype(np.uint8), kernel) > 0
        mask = ~blocked
        labels, n = ndi.label(mask, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndi.sum(mask, labels, range(1, n + 1))
            mask = labels == (int(np.argmax(sizes)) + 1)
        _, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
        index = -np.ones(mask.shape, dtype=np.int64)
        cells = np.argwhere(mask)
        index[mask] = np.arange(len(cells))
        rows, cols, weights = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            rr, cc = cells[:, 0] + dr, cells[:, 1] + dc
            ok = (rr < self.h) & (cc >= 0) & (cc < self.w)
            ok[ok] = mask[rr[ok], cc[ok]]
            rows.append(index[cells[ok, 0], cells[ok, 1]])
            cols.append(index[rr[ok], cc[ok]])
            weights.append(np.full(int(ok.sum()), math.hypot(dr, dc) * self.res))
        rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
        graph = sp.coo_matrix((weights, (rows, cols)), shape=(len(cells), len(cells))).tocsr()
        graph = graph + graph.T
        self._nav[inflation] = dict(mask=mask, nearest=nearest, index=index, cells=cells, graph=graph)
        return self._nav[inflation]

    def distance_field(self, inflation: float, goal_xy) -> np.ndarray:
        """Geodesic distance (m) from every cell to the goal over the inflated free space (inf where blocked)."""
        from scipy.sparse.csgraph import dijkstra

        nav = self.nav(inflation)
        gr, gc = self.snap(inflation, goal_xy)
        dist = dijkstra(nav["graph"], indices=int(nav["index"][gr, gc]))
        field_ = np.full(nav["mask"].shape, np.inf)
        field_[nav["mask"]] = dist
        return field_

    def snap(self, inflation: float, xy) -> tuple[int, int]:
        nav = self.nav(inflation)
        r, c = self.to_cell(xy)
        return int(nav["nearest"][0, r, c]), int(nav["nearest"][1, r, c])

    def geodesic(self, inflation: float, field_: np.ndarray, xy) -> float:
        r, c = self.to_cell(xy)
        sr, sc = self.snap(inflation, xy)
        return float(field_[sr, sc] + math.hypot(sr - r, sc - c) * self.res)

    def lookahead(self, inflation: float, field_: np.ndarray, xy, distance: float) -> np.ndarray:
        """Follow steepest descent of the distance field from xy for ``distance`` metres."""
        r, c = self.snap(inflation, xy)
        travelled = 0.0
        while travelled < distance and field_[r, c] > 0:
            best, step = field_[r, c], None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < self.h and 0 <= cc < self.w and field_[rr, cc] < best:
                        best, step = field_[rr, cc], (rr, cc)
            if step is None:
                break
            travelled += math.hypot(step[0] - r, step[1] - c) * self.res
            r, c = step
        return self.to_world((r, c))

    def footprint_clear(self, center_xy, yaw: float, half: tuple[float, float], margin: float = 0.1) -> bool:
        import cv2

        r, c = self.to_cell(center_xy)
        rect = ((c, r), (2 * (half[0] + margin) / self.res, 2 * (half[1] + margin) / self.res), math.degrees(yaw))
        pts = cv2.boxPoints(rect).astype(np.int32)
        lo = np.maximum(pts.min(axis=0) - 1, 0)
        hi = np.minimum(pts.max(axis=0) + 2, (self.w, self.h))
        if np.any(pts.min(axis=0) < 0) or np.any(pts.max(axis=0) >= (self.w, self.h)):
            return False
        mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
        cv2.fillPoly(mask, [pts - lo], 1)
        return not np.any((mask > 0) & ~self.free[lo[1]:hi[1], lo[0]:hi[0]])


class Robot:
    """One physics-driven robot plus its controller and per-episode bookkeeping."""

    def __init__(self, name: str, type_name: str, prim_path: str, assets_root: str):
        import isaacsim.core.experimental.utils.stage as stage_utils

        self.name = name
        self.type_name = type_name
        self.spec = ROBOT_SPECS[type_name]
        self.prim_path = prim_path
        stage_utils.add_reference_to_stage(usd_path=assets_root + self.spec.usd, path=prim_path)
        self.art = None
        self.controller = None
        self.goal = np.zeros(2)
        self.goal_yaw = 0.0
        self.field = None
        self.prev_geo = 0.0
        self.prev_action = np.zeros(ACT_DIM, np.float32)
        self.steps = 0
        self.ep_return = 0.0
        self.best_geo = 0.0
        self.since_progress = 0
        self.pose = (np.zeros(3), 0.0, 1.0)
        self.vel = np.zeros(2)
        self.clearance = np.full(LIDAR_BEAMS, LIDAR_RANGE, np.float32)
        self.scan = np.full(SCAN_BEAMS, SCAN_RANGE, np.float32)
        self.scan_noisy = self.scan.copy()
        self.scan_filtered = self.scan.copy()
        self.scan_hist: deque = deque(maxlen=SCAN_DELAY + 1)
        self.closest_hit = ""
        self.blocked_steps = 0
        self.recovery_steps = 0
        self.events: dict[str, int] = {}
        self.odom_scale = np.ones(2)
        self.odom_vel = np.zeros(2)
        self.odom_pose = np.zeros(3)
        self.loc_err = np.zeros(3)
        self.est_override: np.ndarray | None = None
        self.episode_id = 0
        self.teleported = True
        self.metrics: dict = {}
        self.active = True
        self.pending = False
        self.pending_steps = 0
        self.task = None
        self.pallet = None
        self.has_lift = False
        self.lift = 0.0
        self.lift_target = 0.0
        self.lift_moving = False
        self.v_cmd = 0.0

    def measure_footprint(self, stage) -> None:
        """Footprint rectangle in the body frame, from the authored collision geometry at the identity pose."""
        from pxr import Usd, UsdGeom

        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
            stage.GetPrimAtPath(self.prim_path)
        ).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        self.center_offset = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0])
        self.half = ((mx[0] - mn[0]) / 2.0, (mx[1] - mn[1]) / 2.0)
        self.z_offset = -mn[2] + 0.02
        self.beam_angles = np.linspace(-math.pi, math.pi, LIDAR_BEAMS, endpoint=False)
        self.scan_angles = np.linspace(-math.pi, math.pi, SCAN_BEAMS, endpoint=False)
        with np.errstate(divide="ignore"):
            self.scan_body = np.minimum(
                self.half[0] / np.abs(np.cos(self.scan_angles)), self.half[1] / np.abs(np.sin(self.scan_angles))
            ).astype(np.float32)
        self.scan_dirs = np.stack([np.cos(self.scan_angles), np.sin(self.scan_angles)], axis=1)

    def measure_scan_body(self) -> None:
        """Per-beam extent of the robot's own collision shapes (which can exceed the render bounds), probed from
        outside while the robot is parked alone; scan rays start just beyond it."""
        import carb
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
        self.read_state()
        center, yaw, _ = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        prefix = self.prim_path + "/"
        far = 8.0
        for k, (bx, by) in enumerate(self.scan_dirs):
            dx, dy = c * bx - s * by, s * bx + c * by
            nearest = [far]

            def report(hit, nearest=nearest):
                if (hit.rigid_body or hit.collision).startswith(prefix):
                    nearest[0] = min(nearest[0], hit.distance)
                return True

            for z in self.spec.lidar_layers:
                origin = carb.Float3(center[0] + dx * far, center[1] + dy * far, center[2] + z)
                sq.raycast_all(origin, carb.Float3(-dx, -dy, 0.0), far, report)
            self.scan_body[k] = max(self.scan_body[k], far - nearest[0])

    @property
    def est_pose(self) -> np.ndarray:
        """Estimated (x, y, yaw): an external estimate if one was supplied this step, else truth + noise model."""
        if self.est_override is not None:
            return self.est_override
        center, yaw, _ = self.pose
        return np.array([center[0] + self.loc_err[0], center[1] + self.loc_err[1], wrap(yaw + self.loc_err[2])])

    def initialize(self) -> None:
        import isaacsim.robot_motion.controllers as ctrl
        import isaacsim.robot_motion.experimental.motion_generation as mg
        from isaacsim.core.experimental.prims import Articulation

        self.art = Articulation(self.prim_path)
        if self.spec.kind == "ackermann":
            self.controller = ctrl.AckermannController(
                robot_joint_space=self.art.dof_names,
                max_linear_speed=self.spec.max_speed,
                max_turning_angle=self.spec.max_turn,
                direct_command=True,
                **self.spec.controller,
            )
        else:
            self.controller = ctrl.DifferentialDriveController(
                robot_joint_space=self.art.dof_names,
                max_linear_speed=self.spec.max_speed,
                max_angular_speed=2.0 * self.spec.max_turn,
                **self.spec.controller,
            )
        self.controller.reset(mg.RobotState(), mg.RobotState(), 0.0)
        if self.has_lift:
            self.lift_idx = self.art.get_dof_indices(["lift_joint"])

    def apply(self, action: np.ndarray) -> None:
        import isaacsim.robot_motion.experimental.motion_generation as mg
        import warp as wp

        a = np.clip(np.nan_to_num(action), -1.0, 1.0)
        v, turn = float(a[0]) * self.spec.max_speed, float(a[1]) * self.spec.max_turn
        if self.has_lift:
            cmd = pt.LIFT_MAX * (float(a[2]) + 1.0) / 2.0 if len(a) > 2 else 0.0
            step = pt.LIFT_RATE * CONTROL_DT
            new = float(np.clip(cmd, self.lift_target - step, self.lift_target + step))
            self.lift_moving = abs(new - self.lift_target) > 1e-4
            self.lift_target = new
            self.art.set_dof_position_targets(wp.array([[new]], dtype=wp.float32), dof_indices=self.lift_idx)
            if self.lift > 0.05:
                # Raised forks: speed and acceleration caps keep the load on the forks.
                v = float(np.clip(v, -pt.CARRY_SPEED, pt.CARRY_SPEED))
                dv = pt.CARRY_ACCEL * CONTROL_DT
                v = float(np.clip(v, self.v_cmd - dv, self.v_cmd + dv))
        self.v_cmd = v
        if self.spec.max_lateral_accel and abs(v * turn) > self.spec.max_lateral_accel:
            # The drive wheels lose traction and the chassis spins out above this lateral acceleration.
            v = math.copysign(self.spec.max_lateral_accel / abs(turn), v)
        if self.spec.kind == "ackermann":
            desired = self.controller.forward(mg.RobotState(), None, 0.0, linear_speed=v, turning_angle=turn)
        else:
            # Gyro feedback like a real base controller: the chassis slips and over-rotates relative to the wheel
            # kinematics, so correct the wheel yaw command by the measured yaw-rate error.
            turn = float(np.clip(turn + self.spec.yaw_feedback * (turn - float(self.vel[1])), -2.0 * self.spec.max_turn,
                                 2.0 * self.spec.max_turn))
            setpoint = mg.RobotState(
                sites=mg.SpatialState.from_name(
                    spatial_space=["control_point"],
                    linear_velocities=(["control_point"], wp.array([[v, 0.0, 0.0]], dtype=wp.float32)),
                    angular_velocities=(["control_point"], wp.array([[0.0, 0.0, turn]], dtype=wp.float32)),
                )
            )
            desired = self.controller.forward(mg.RobotState(), setpoint, 0.0)
        if desired is None or desired.joints is None:
            return
        if desired.joints.velocities is not None:
            self.art.set_dof_velocity_targets(
                desired.joints.velocities, dof_indices=self.art.get_dof_indices(desired.joints.velocity_names)
            )
        if desired.joints.positions is not None:
            self.art.set_dof_position_targets(
                desired.joints.positions, dof_indices=self.art.get_dof_indices(desired.joints.position_names)
            )

    def teleport(self, xy, yaw: float) -> None:
        import warp as wp

        n = len(self.art.dof_names)
        self.art.set_dof_velocity_targets(wp.zeros((1, n), dtype=wp.float32))
        self.art.set_dof_velocities(wp.zeros((1, n), dtype=wp.float32))
        self.art.set_dof_positions(wp.zeros((1, n), dtype=wp.float32))
        self.art.set_dof_position_targets(wp.zeros((1, n), dtype=wp.float32))
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = self.center_offset
        root = [xy[0] - (c * ox - s * oy), xy[1] - (s * ox + c * oy), self.z_offset]
        self.art.set_world_poses(
            positions=wp.array([root], dtype=wp.float32), orientations=wp.array([quat_from_yaw(yaw)], dtype=wp.float32)
        )
        self.art.set_velocities(
            linear_velocities=wp.zeros((1, 3), dtype=wp.float32), angular_velocities=wp.zeros((1, 3), dtype=wp.float32)
        )
        self.lift = self.lift_target = self.v_cmd = 0.0

    def read_state(self) -> None:
        pos, quat = (a.numpy()[0].astype(np.float64) for a in self.art.get_world_poses())
        lin, ang = (a.numpy()[0].astype(np.float64) for a in self.art.get_velocities())
        yaw = yaw_from_quat(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = self.center_offset
        center = np.array([pos[0] + c * ox - s * oy, pos[1] + s * ox + c * oy, pos[2]])
        self.pose = (center, yaw, up_z_from_quat(quat))
        self.vel = np.array([c * lin[0] + s * lin[1], ang[2]])
        if self.has_lift:
            self.lift = float(self.art.get_dof_positions(dof_indices=self.lift_idx).numpy()[0, 0])


class WarehouseEnv:
    """Full warehouse, N walking workers, and a heterogeneous fleet of robots exposed as a multi-agent env."""

    def __init__(
        self,
        app,
        robots: list[str] = ("forklift", "carter", "carter"),
        workers: int = 8,
        seed: int = 0,
        max_episode_steps: int = 1500,
        goal_range: tuple[float, float] = (6.0, 30.0),
        frames_per_step: int = 1,
        render: bool = False,
        task: str = "nav",
    ):
        """``frames_per_step`` app updates (render + worker animation) run per control step; PhysX substeps at
        ``PHYSICS_DT`` inside each. 1 is fastest; 3 gives smooth 30 fps recordings. ``task`` is ``"nav"`` (every
        robot drives to goals) or ``"pallet"`` (forklifts pick and place pallets, Carters navigate)."""
        if task not in ("nav", "pallet"):
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        pallets = task == "pallet"
        self.act_dim = 3 if pallets else ACT_DIM
        self.obs_dim = OBS_DIM + (1 + pt.TASK_OBS if pallets else 0)
        self.priv_dim = PRIV_DIM + (1 + pt.TASK_PRIV if pallets else 0)
        self.app = app
        self.rng = np.random.default_rng(seed)
        self.max_episode_steps = max_episode_steps
        self.goal_range = goal_range
        self.frames_per_step = int(frames_per_step)
        self.frame_callbacks = []
        self.episodes: list[deque] | None = None
        self.skipped_episodes = 0
        self.worker_vel = np.zeros((0, 2))
        self.n_workers = workers
        self._build_scene(list(robots), workers, seed)
        self.set_render(render)

    # ------------------------------------------------------------------ scene

    def _run(self, coro):
        from omni.kit.async_engine import run_coroutine

        task = run_coroutine(coro)
        while not task.done():
            self.app.update()
        if task.exception():
            raise task.exception()
        return task.result()

    def _build_scene(self, robot_types: list[str], workers: int, seed: int) -> None:
        import omni.anim.behavior.core as bh_core
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.storage.native import get_assets_root_path

        self._run(self._setup_ira(workers, seed))
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.set_looping(False)
        frame_hz = self.frames_per_step / CONTROL_DT
        self.timeline.set_time_codes_per_second(frame_hz)
        self.timeline.set_target_framerate(frame_hz)
        self.timeline.commit()
        self.bh = bh_core.acquire_interface()
        self.worker_paths = [str(p.GetPath()) for p in self.stage.Traverse() if p.GetTypeName() == "SkelRoot"]

        root = get_assets_root_path()
        counts: dict[str, int] = {}
        self.robots: list[Robot] = []
        for t in robot_types:
            if t not in ROBOT_SPECS:
                raise ValueError(f"unknown robot type {t!r}; choose from {sorted(ROBOT_SPECS)}")
            i = counts[t] = counts.get(t, -1) + 1
            self.robots.append(Robot(f"{t}_{i}", t, f"/World/Robots/{t}_{i}", root))
        for _ in range(5):
            self.app.update()
        for r in self.robots:
            r.measure_footprint(self.stage)
            r.prev_action = np.zeros(self.act_dim, np.float32)
        self._add_parking_floor()
        if self.task == "pallet":
            for k, r in enumerate(x for x in self.robots if x.type_name == "forklift"):
                pt.fit_fork_colliders(self.stage, r.prim_path)
                pt.create_pallet(self.stage, f"/World/Pallets/pallet_{k}", root)
                r.pallet = pt.Pallet(f"/World/Pallets/pallet_{k}")
                r.has_lift = True
            for _ in range(2):
                self.app.update()

        SimulationManager.set_physics_dt(PHYSICS_DT)
        self.timeline.play()
        self.app.update()
        if not os.environ.get("WAREHOUSE_NO_WORKER_FILTER"):
            self._filter_worker_contacts()
        for i, r in enumerate(self.robots):
            r.initialize()
            r.teleport((200.0 + 10.0 * i, 200.0), 0.0)
            if r.pallet is not None:
                r.pallet.initialize()
                self._park_pallet(r)
        self.app.update()
        for r in self.robots:
            r.measure_scan_body()
        self.map = self._build_map()
        self.map_image = self._map_image()

    def _add_parking_floor(self) -> None:
        """Static floor far outside the warehouse where idle robots and pallets wait."""
        from pxr import Gf, UsdGeom, UsdPhysics

        cube = UsdGeom.Cube.Define(self.stage, "/World/ParkingFloor")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(250.0, 0.0, -0.5))
        xf.AddScaleOp().Set(Gf.Vec3f(200.0, 500.0, 1.0))
        cube.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    def planner(self, r: Robot):
        """Hybrid-A* planner for this robot type (car-like robots only), built on first use."""
        from hybrid_astar import HybridAStar

        cache = self.__dict__.setdefault("_planners", {})
        if r.type_name not in cache:
            kappa = 0.85 * math.tan(r.spec.max_turn) / r.spec.controller["wheel_base"]
            cache[r.type_name] = HybridAStar(self.map, r.half, pt.FRONT_AXLE, kappa)
        return cache[r.type_name]

    def _filter_worker_contacts(self) -> None:
        """Workers' kinematic capsules would shove robots and tip pallets they walk through (the animated workers do
        not avoid robots); filter those contact pairs. Scene queries ignore the filter, so lidar still sees workers
        and contact is judged by distance in the reward."""
        from pxr import Usd, UsdPhysics

        bodies = []
        for root in ["/World/Robots", "/World/Pallets"]:
            prim = self.stage.GetPrimAtPath(root)
            if prim.IsValid():
                bodies += [p.GetPath() for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        n = 0
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/Characters")):
            if prim.GetName() == "BodyCollider" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(bodies)
                n += 1
        print(f"[warehouse_env] filtered {n} worker colliders against {len(bodies)} robot/pallet bodies", flush=True)

    def _park_pallet(self, r: Robot) -> None:
        i = self.robots.index(r)
        r.pallet.place((200.0 + 10.0 * i, -200.0), 0.0)

    async def _setup_ira(self, workers: int, seed: int) -> None:
        import yaml
        from isaacsim.replicator.agent.core import api as IRA

        cfg = {
            "isaacsim.replicator.agent": {
                "version": _ira_config_version(),
                "seed": int(seed),
                "simulation_duration": 86400,
                "environment": {"base_stage_asset_path": WAREHOUSE_USD},
                "character": {
                    "groups": {
                        "workers": {
                            "asset_path": CHARACTERS_ROOT,
                            "num": int(workers),
                            "routines": [
                                {
                                    "wander": {
                                        "walk": {"speed_range": [0.8, 1.4], "distance_range": [4.0, 12.0]},
                                        "idle": [{"animation": "idle", "time_range": [1.0, 4.0]}],
                                    }
                                }
                            ],
                        }
                    }
                },
            }
        }
        path = os.path.join(tempfile.mkdtemp(prefix="warehouse_amr_"), "ira.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        if not IRA.load_config_file(path):
            raise RuntimeError(f"IRA rejected config {path}")
        await IRA.setup_simulation()

    def _build_map(self) -> GridMap:
        import omni.physx
        import omni.usd
        from isaacsim.asset.gen.omap.bindings import _omap
        from pxr import Usd, UsdGeom

        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
            self.stage.GetPrimAtPath("/Root")
        ).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        import cv2
        import scipy.ndimage as ndi

        seeds = [p for p in (self.worker_position(w) for w in self.worker_paths) if p is not None]
        seeds = seeds or [np.array([0.0, 0.0])]
        origin = (float(seeds[0][0]), float(seeds[0][1]), 0.0)
        slices = {}

        def slice_at(z: float) -> np.ndarray:
            # generate2d samples a thin slice, so overhangs (shelf loads, forks) need their own slices.
            nonlocal min_b
            if z not in slices:
                gen = _omap.Generator(omni.physx.get_physx_interface(), omni.usd.get_context().get_stage_id())
                gen.update_settings(MAP_RES, 1.0, 0.0, 0.5)
                gen.set_transform(
                    origin, (mn[0] - origin[0], mn[1] - origin[1], z), (mx[0] - origin[0], mx[1] - origin[1], z + MAP_RES)
                )
                gen.generate2d()
                w, h, _ = gen.get_dimensions()
                slices[z] = np.array(gen.get_buffer(), dtype=np.float32).reshape(h, w) == 1.0
                min_b = gen.get_min_bound()
            return slices[z]

        min_b = None
        occupied = np.logical_or.reduce([slice_at(z) for z in MAP_SLICES])
        # What each robot's planar lidar can see, i.e. the map a SLAM run with that sensor would produce (for AMCL).
        self.loc_occupied = {}
        for r in self.robots:
            if r.type_name not in self.loc_occupied:
                zs = [round(z - MAP_RES / 2, 3) for z in r.spec.lidar_layers]
                self.loc_occupied[r.type_name] = np.logical_or.reduce([slice_at(z) for z in zs])
        min_b = self._calibrate_map_origin(occupied, (float(min_b[0]), float(min_b[1])))
        # The generator's own free-space fill is unreliable here; free space is the open region inside the walls
        # that the workers spawned in (walls are closed off with a small dilation first).
        sealed = cv2.dilate(occupied.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        labels, _ = ndi.label(~sealed)
        votes = [labels[int((s[1] - min_b[1]) / MAP_RES), int((s[0] - min_b[0]) / MAP_RES)] for s in seeds]
        votes = [v for v in votes if v > 0]
        region = labels == max(set(votes), key=votes.count) if votes else ~sealed
        region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        grid = GridMap(region & ~occupied, occupied, (min_b[0], min_b[1]), MAP_RES)
        print(
            f"[warehouse_env] map {grid.w}x{grid.h} @ {MAP_RES} m from {len(seeds)} seeds, free {grid.free.mean():.1%}, "
            f"occupied {grid.occupied.mean():.1%}",
            flush=True,
        )
        if os.environ.get("WAREHOUSE_AMR_MAP_DEBUG"):
            import cv2

            img = np.full(occupied.shape, 127, np.uint8)
            img[grid.free], img[grid.occupied] = 255, 0
            for s in seeds:
                c = ((np.asarray(s) - np.array(min_b[:2])) / MAP_RES).astype(int)
                cv2.circle(img, (int(c[0]), int(c[1])), 5, 60, 2)
            cv2.imwrite(os.environ["WAREHOUSE_AMR_MAP_DEBUG"], img[::-1])
        return grid

    def _calibrate_map_origin(self, occupied: np.ndarray, origin: tuple[float, float]) -> tuple[float, float]:
        """The generator's reported bounds can be several cells off from its buffer. Measure the offset by casting
        PhysX rays from open cells along +-x / +-y and comparing hit distances with the first occupied cell."""
        import carb
        import scipy.ndimage as ndi
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
        clear = np.argwhere(ndi.distance_transform_edt(~occupied) * MAP_RES > 0.6)
        rng = np.random.default_rng(0)
        h, w = occupied.shape
        errs = {d: [] for d in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        for rc in clear[rng.choice(len(clear), min(400, len(clear)), replace=False)]:
            p = np.array(origin) + (rc[::-1] + 0.5) * MAP_RES
            for d in errs:
                hit = sq.raycast_closest(carb.Float3(p[0], p[1], 0.35), carb.Float3(d[0], d[1], 0.0), 8.0)
                if not hit["hit"] or hit["collision"].startswith(("/World/Characters", "/World/Robots")):
                    continue
                for step in range(1, int(8.0 / (MAP_RES / 4))):
                    q = p + np.array(d) * step * MAP_RES / 4
                    c = ((q - origin) / MAP_RES).astype(int)
                    if not (0 <= c[0] < w and 0 <= c[1] < h):
                        break
                    if occupied[c[1], c[0]]:
                        errs[d].append(hit["distance"] - step * MAP_RES / 4)
                        break
        med = {d: float(np.median(e)) if len(e) > 10 else 0.0 for d, e in errs.items()}
        # Map content sitting +s too far along an axis shortens hits looking + and lengthens hits looking -.
        dx = 0.5 * (med[(1, 0)] - med[(-1, 0)])
        dy = 0.5 * (med[(0, 1)] - med[(0, -1)])
        dx, dy = round(dx / MAP_RES) * MAP_RES, round(dy / MAP_RES) * MAP_RES
        print(f"[warehouse_env] map origin correction ({dx:+.2f}, {dy:+.2f}) m from ray probes "
              f"(median hit-map: {', '.join(f'{d}:{v:+.2f}' for d, v in med.items())})", flush=True)
        return origin[0] + dx, origin[1] + dy

    def _map_image(self) -> np.ndarray:
        img = np.full(self.map.free.shape + (3,), 150, np.uint8)
        img[self.map.free] = (235, 235, 235)
        img[self.map.occupied] = (60, 60, 60)
        return img

    def set_render(self, enabled: bool) -> None:
        """Toggle viewport rendering and the PhysX-to-USD writeback that rendering needs.

        With writeback off (the fast default), robots still simulate and are read through the tensor API, but they
        do not move in rendered images.
        """
        import carb

        settings = carb.settings.get_settings()
        settings.set("/physics/updateToUsd", bool(enabled))
        settings.set("/physics/updateVelocitiesToUsd", False)
        try:
            from omni.kit.viewport.utility import get_active_viewport

            vp = get_active_viewport()
            if vp is not None:
                vp.updates_enabled = bool(enabled)
        except Exception:
            pass

    # ------------------------------------------------------------------ queries

    def worker_position(self, path: str) -> np.ndarray | None:
        agent = self.bh.get_agent(path)
        if agent is None:
            return None
        t = agent.get_world_translation()
        return np.array([t[0], t[1]])

    def worker_positions(self) -> np.ndarray:
        pts = [p for p in (self.worker_position(w) for w in self.worker_paths) if p is not None]
        return np.array(pts).reshape(-1, 2)

    def _track_workers(self) -> None:
        """Worker positions and finite-difference velocities (rows aligned with ``worker_paths``)."""
        pos = np.array([p if p is not None else (np.nan, np.nan) for p in map(self.worker_position, self.worker_paths)])
        pos = pos.reshape(-1, 2)
        prev = getattr(self, "_worker_pos", None)
        vel = np.zeros_like(pos) if prev is None or prev.shape != pos.shape else (pos - prev) / CONTROL_DT
        # Teleports / respawns of workers would show up as huge velocities.
        vel[~np.isfinite(vel).all(axis=1) | (np.linalg.norm(np.nan_to_num(vel), axis=1) > 3.0)] = 0.0
        self._worker_pos, self.worker_vel = pos, vel

    def _start_clear(self, robot: Robot, xy, yaw: float) -> bool:
        """No collider (props pushed around since the map was built, workers, robots) inside the start footprint."""
        import carb
        from omni.physx import get_physx_scene_query_interface

        half_z = 0.3 if robot.type_name == "carter" else 1.0
        hits = []

        own = (robot.prim_path,) + ((robot.pallet.path,) if robot.pallet is not None else ())

        def report(hit):
            # The robot's own pallet is moved to the new episode's pick pose anyway.
            if not any(hit.rigid_body == o or hit.rigid_body.startswith(o + "/") or hit.collision.startswith(o + "/")
                       for o in own):
                hits.append(hit.collision)
                return False
            return True

        q = quat_from_yaw(yaw)
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(robot.half[0] + 0.1, robot.half[1] + 0.1, half_z),
            carb.Float3(float(xy[0]), float(xy[1]), half_z + 0.06),
            carb.Float4(q[1], q[2], q[3], q[0]),
            report,
            False,
        )
        self._start_hits = hits
        return not hits

    def _sense(self, robot: Robot) -> None:
        """Planar scan from the footprint center (min over mount heights), clearance sectors, and the noisy scan."""
        import carb
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
        center, yaw, _ = robot.pose
        c, s = math.cos(yaw), math.sin(yaw)
        prefix = robot.prim_path + "/"
        ranges = np.full(SCAN_BEAMS, SCAN_RANGE, np.float32)
        pallet_ranges = ranges.copy()
        paths = [""] * SCAN_BEAMS
        # The robot's own task pallet is seen by the lidar but is not an obstacle for collisions / clearance.
        pallet = robot.pallet.path if robot.pallet is not None and robot.task is not None else None
        for k, (bx, by) in enumerate(robot.scan_dirs):
            dx, dy = c * bx - s * by, s * bx + c * by
            for z in robot.spec.lidar_layers:
                start = float(robot.scan_body[k]) + SCAN_START
                # Moving parts (steered wheels) can still poke past the measured extent: step over own hits.
                env_hit = None
                for _ in range(4):
                    if start > SCAN_RANGE - 0.05:
                        break
                    hit = sq.raycast_closest(
                        carb.Float3(center[0] + dx * start, center[1] + dy * start, z), carb.Float3(dx, dy, 0.0),
                        SCAN_RANGE - start,
                    )
                    if not hit["hit"]:
                        break
                    body = hit["rigidBody"] or hit["collision"]
                    if pallet and (body == pallet or body.startswith(pallet + "/")):
                        pallet_ranges[k] = min(pallet_ranges[k], hit["distance"] + start)
                    elif not body.startswith(prefix):
                        env_hit = hit
                        break
                    start += hit["distance"] + 0.05
                if env_hit is not None and env_hit["distance"] + start < ranges[k]:
                    ranges[k], paths[k] = env_hit["distance"] + start, env_hit["collision"]
        robot.scan = ranges
        per = SCAN_BEAMS // LIDAR_BEAMS
        clear = np.roll(np.clip(ranges - robot.scan_body, 0.0, LIDAR_RANGE), per // 2)
        robot.clearance = clear.reshape(LIDAR_BEAMS, per).min(axis=1)
        k = int(np.argmin(ranges - robot.scan_body))
        robot.closest_hit = paths[k]
        noise = self.rng.normal(0.0, SCAN_NOISE, SCAN_BEAMS).astype(np.float32)
        drop = self.rng.random(SCAN_BEAMS) < SCAN_DROPOUT
        noisy, filtered = np.minimum(ranges, pallet_ranges) + noise, ranges + noise
        noisy[drop] = filtered[drop] = SCAN_RANGE
        robot.scan_noisy = np.clip(noisy, robot.scan_body, SCAN_RANGE).astype(np.float32)
        robot.scan_filtered = np.clip(filtered, robot.scan_body, SCAN_RANGE).astype(np.float32)
        bins = np.clip(robot.scan_noisy - robot.scan_body, 0.0, LIDAR_RANGE)
        # Inverse-distance encoding: centimetres matter near the body, metres far away.
        robot.scan_hist.append(1.0 / (1.0 + 2.0 * bins.reshape(SCAN_BINS, SCAN_BEAMS // SCAN_BINS).min(axis=1)))

    def _update_estimators(self, robot: Robot, new_episode: bool) -> None:
        """Wheel odometry (scale bias + noise, integrated) and the localization error model."""
        if new_episode:
            robot.odom_scale = 1.0 + self.rng.normal(0.0, ODOM_SCALE_STD, 2)
            robot.loc_err = self.rng.normal(0.0, LOC_STD)
            robot.scan_hist.clear()
            return
        v = robot.vel * robot.odom_scale + self.rng.normal(0.0, ODOM_NOISE)
        robot.odom_vel = v
        x, y, th = robot.odom_pose
        th2 = th + v[1] * CONTROL_DT
        robot.odom_pose = np.array([x + v[0] * CONTROL_DT * math.cos(th + 0.5 * v[1] * CONTROL_DT),
                                    y + v[0] * CONTROL_DT * math.sin(th + 0.5 * v[1] * CONTROL_DT), wrap(th2)])
        a = math.exp(-CONTROL_DT / LOC_TAU)
        robot.loc_err = a * robot.loc_err + math.sqrt(1 - a * a) * self.rng.normal(0.0, LOC_STD)

    def _footprint_distance(self, robot: Robot, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros(0)
        center, yaw, _ = robot.pose
        c, s = math.cos(yaw), math.sin(yaw)
        d = pts - center[:2]
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        q = np.maximum(np.abs(local) - np.array(robot.half), 0.0)
        return np.linalg.norm(q, axis=1)

    # ------------------------------------------------------------------ episodes

    def _sample_pose(self, robot: Robot, avoid: list[np.ndarray], rng=None, humans=None) -> tuple[np.ndarray, float]:
        rng = self.rng if rng is None else rng
        nav = self.map.nav(robot.spec.inflation)
        humans = self.worker_positions() if humans is None else humans
        for _ in range(500):
            cell = nav["cells"][rng.integers(len(nav["cells"]))]
            xy = self.map.to_world(cell)
            yaw = float(rng.uniform(-math.pi, math.pi))
            if any(np.linalg.norm(xy - p) < 4.0 for p in avoid):
                continue
            if len(humans) and np.min(np.linalg.norm(humans - xy, axis=1)) < 2.5:
                continue
            if self.map.footprint_clear(xy, yaw, robot.half, margin=0.15) and (
                rng is not self.rng or self._start_clear(robot, xy, yaw)
            ):
                return xy, yaw
        raise RuntimeError(f"no free spawn pose for {robot.name}")

    def _sample_goal(self, robot: Robot, start, rng=None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        nav = self.map.nav(robot.spec.inflation)
        for _ in range(30):
            goal = self.map.to_world(nav["cells"][rng.integers(len(nav["cells"]))])
            if not (self.goal_range[0] <= np.linalg.norm(goal - start) <= self.goal_range[1]):
                continue
            field_ = self.map.distance_field(robot.spec.inflation, goal)
            geo = self.map.geodesic(robot.spec.inflation, field_, start)
            if math.isfinite(geo) and geo <= 2.0 * self.goal_range[1]:
                break
        return goal

    def _set_goal_keep(self, robot: Robot, goal) -> float:
        """New navigation goal (distance field, goal heading) without starting a new episode; returns geodesic."""
        start = robot.pose[0][:2]
        robot.goal = np.asarray(goal, dtype=np.float64)
        robot.field = self.map.distance_field(robot.spec.inflation, robot.goal)
        geo = self.map.geodesic(robot.spec.inflation, robot.field, start)
        # Heading of the shortest path where it enters the goal, for planners that need a goal orientation.
        before = self.map.lookahead(robot.spec.inflation, robot.field, start, max(geo - 2.0, 0.0))
        d = robot.goal - (before if geo > 2.5 else start)
        robot.goal_yaw = math.atan2(d[1], d[0])
        return geo

    def _set_goal(self, robot: Robot, goal) -> None:
        geo = self._set_goal_keep(robot, goal)
        robot.prev_geo = robot.best_geo = geo
        robot.steps = robot.since_progress = robot.blocked_steps = robot.recovery_steps = 0
        robot.ep_return = 0.0
        robot.prev_action[:] = 0.0
        robot.episode_id += 1
        robot.metrics = {"geodesic": geo, "path_length": 0.0, "min_human": math.inf, "near_human_steps": 0,
                         "passive_contacts": 0}

    def _place(self, robot: Robot, xy, yaw: float) -> None:
        robot.teleport(xy, yaw)
        robot.read_state()
        robot.pose = (np.array([xy[0], xy[1], 0.0]), yaw, 1.0)
        robot.vel = np.zeros(2)
        robot.teleported = True
        self._update_estimators(robot, new_episode=True)

    def _park(self, robot: Robot) -> None:
        i = self.robots.index(robot)
        robot.teleport((200.0 + 10.0 * i, 200.0), 0.0)
        robot.active = False
        robot.steps = -1
        if robot.pallet is not None and robot.task is not None:
            self._park_pallet(robot)
            robot.task = None

    def _task_avoid(self, robot: Robot) -> list:
        """Positions a new pallet task must keep clear of: other robots, their pallets and drop zones."""
        out = []
        for r in self.robots:
            if r is robot or not (r.active or r.pending):
                continue
            out.append(r.pose[0][:2])
            if r.task is not None:
                out += [r.task.pallet[:2], r.task.drop[:2]]
        return out

    def _pallet_spot_clear(self, robot: Robot, task: pt.Task, humans, keep_out: float) -> bool:
        p = task.pallet0[:2]
        if len(humans) and np.min(np.linalg.norm(humans - p, axis=1)) < keep_out:
            return False
        return not any(np.linalg.norm(p - q) < 4.0 for q in self._task_avoid(robot))

    def _begin_task(self, robot: Robot, task: pt.Task) -> None:
        robot.pallet.place(task.pallet0[:2], float(task.pallet0[2]))
        pt.begin(self, robot, task)

    def _start_episode(self, robot: Robot, respawn: bool) -> None:
        """Begin the robot's next episode: the next benchmark episode, else a new goal (plus respawn on failure)."""
        if self.episodes is not None:
            queue = self.episodes[self.robots.index(robot)]
            if not queue:
                self._park(robot)
                robot.pending = False
                return
            start, yaw, goal = queue[0]
            humans = self.worker_positions()
            others = [r.pose[0][:2] for r in self.robots if r is not robot and r.active]
            robot.pending_steps = robot.pending_steps + 1 if robot.pending else 0
            # Workers can idle next to a start indefinitely: relax the keep-out after 30 s, give up after 60 s.
            keep_out = 2.0 if robot.pending_steps < int(30 / CONTROL_DT) else 1.0
            if robot.pending_steps > int(60 / CONTROL_DT):
                print(f"[warehouse_env] skip {robot.name} episode: start not clear for 60 s", flush=True)
                queue.popleft()
                self.skipped_episodes += 1
                robot.pending = False
                self._start_episode(robot, respawn)
                return
            blocked = (len(humans) and np.min(np.linalg.norm(humans - np.asarray(start), axis=1)) < keep_out) or any(
                np.linalg.norm(np.asarray(start) - p) < 3.0 for p in others
            )
            task = pt.task_from_json(goal) if isinstance(goal, dict) else None
            if task is not None and not blocked:
                blocked = not self._pallet_spot_clear(robot, task, humans, keep_out)
            if not blocked and not self._start_clear(robot, start, yaw):
                # A prop has been pushed onto this start since the map was built: the episode is not runnable.
                print(f"[warehouse_env] skip {robot.name} episode: start blocked by {self._start_hits[:3]}", flush=True)
                queue.popleft()
                self.skipped_episodes += 1
                self._start_episode(robot, respawn)
                return
            if blocked:
                # Hold the episode (robot parked) until workers / other robots clear the start pose.
                self._park(robot)
                robot.pending = True
                return
            queue.popleft()
            robot.pending, robot.active = False, True
            self._place(robot, start, yaw)
            if task is not None:
                self._begin_task(robot, task)
            else:
                self._set_goal(robot, goal)
            robot.steps = 0
            return
        if respawn:
            others = [r.pose[0][:2] for r in self.robots if r is not robot and r.steps >= 0]
            if robot.pallet is not None:
                others += [p for r in self.robots if r is not robot and r.task is not None
                           for p in (r.task.pallet[:2], r.task.drop[:2])]
            self._place(robot, *self._sample_pose(robot, others))
        if robot.pallet is not None:
            self._begin_task(robot, pt.sample_task(self, robot, robot.pose[0][:2], self.rng,
                                                   avoid=self._task_avoid(robot)))
        else:
            self._set_goal(robot, self._sample_goal(robot, robot.pose[0][:2]))

    def make_episodes(self, per_robot: int, seed: int = 12345) -> list[list]:
        """Fixed (start xy, start yaw, goal xy) lists, one per robot, independent of the controller under test."""
        rng = np.random.default_rng(seed)
        out = []
        for r in self.robots:
            eps = []
            for _ in range(per_robot):
                xy, yaw = self._sample_pose(r, [], rng=rng, humans=np.zeros((0, 2)))
                if r.pallet is not None:
                    goal = pt.sample_task(self, r, xy, rng).to_json()
                else:
                    goal = self._sample_goal(r, xy, rng=rng).tolist()
                eps.append((xy.tolist(), yaw, goal))
            out.append(eps)
        return out

    def set_episodes(self, episodes: list[list] | None) -> None:
        """Switch to benchmark mode (robots run their lists, then park) or back to random episodes (None)."""
        self.episodes = None if episodes is None else [deque(e) for e in episodes]

    @property
    def benchmark_done(self) -> bool:
        return self.episodes is not None and not any(r.active or r.pending for r in self.robots)

    def _advance(self) -> None:
        for _ in range(self.frames_per_step):
            self.app.update()
            for cb in self.frame_callbacks:
                cb()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for r in self.robots:
            r.steps, r.active, r.pending, r.est_override = -1, False, False, None
        for r in self.robots:
            r.active = self.episodes is None
            self._start_episode(r, respawn=True)
            if r.active:
                r.steps = 0
        self._advance()
        for r in self.robots:
            if r.active:
                r.read_state()
                self._sense(r)
        return self.observations()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32).reshape(len(self.robots), -1)
        if actions.shape[1] < self.act_dim:
            # Navigation-only controllers (Nav2, older checkpoints): forks stay down.
            pad = np.full((len(self.robots), self.act_dim - actions.shape[1]), -1.0, np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        for r, a in zip(self.robots, actions):
            r.est_override = None
            r.teleported = False
            if r.active:
                r.apply(a)
            elif r.pending:
                self._start_episode(r, respawn=False)
            else:
                self._park(r)
        self._advance()
        self._track_workers()
        humans = self.worker_positions()
        rewards = np.zeros(len(self.robots), np.float32)
        terminated = np.zeros(len(self.robots), bool)
        truncated = np.zeros(len(self.robots), bool)
        infos = []
        for i, (r, a) in enumerate(zip(self.robots, actions)):
            if not r.active:
                infos.append({})
                continue
            prev_xy = r.pose[0][:2].copy()
            r.read_state()
            self._update_estimators(r, new_episode=False)
            self._sense(r)
            rew, event = self._reward(r, np.clip(a, -1, 1), humans)
            r.metrics["path_length"] += float(np.linalg.norm(r.pose[0][:2] - prev_xy))
            r.prev_action = np.clip(a, -1, 1).astype(np.float32)
            r.ep_return += rew
            r.steps += 1
            rewards[i] = rew
            info = {}
            if event is None and r.steps >= self.max_episode_steps * (2 if r.task is not None else 1):
                event = "timeout"
            if event is not None:
                terminated[i] = event in ("success", "collision", "human_contact", "tipped", "pallet_push",
                                          "pallet_drop")
                truncated[i] = not terminated[i]
                m = r.metrics
                info = {
                    "event": event, "episode_return": r.ep_return, "episode_length": r.steps, "robot": r.name,
                    "type": r.type_name, "episode_id": r.episode_id, "time": r.steps * CONTROL_DT,
                    "geodesic": m["geodesic"], "path_length": m["path_length"],
                    "min_human": m["min_human"] if math.isfinite(m["min_human"]) else None,
                    "near_human_time": m["near_human_steps"] * CONTROL_DT,
                    "passive_contact_time": m["passive_contacts"] * CONTROL_DT,
                }
                if event == "collision":
                    info["hit"] = r.closest_hit
                if event == "human_contact":
                    info["contact"] = getattr(r, "contact_dbg", None)
                if r.task is not None:
                    t = r.task
                    info.update({
                        "task": "pallet", "phase": t.phase, "picked": t.t_pick is not None, "pick_time": t.t_pick,
                        "place_err_m": None if t.place_err is None else round(t.place_err[0], 3),
                        "place_err_deg": None if t.place_err is None else round(math.degrees(t.place_err[1]), 2),
                        "repicks": t.repicks, "pallet": t.pallet0.tolist(), "drop": t.drop.tolist(),
                        "miss_err": t.miss_err,
                    })
                r.events[event] = r.events.get(event, 0) + 1
                self._start_episode(r, respawn=event not in ("success", "timeout"))
                if r.active:
                    self._sense(r)
            infos.append(info)
        return self.observations(), rewards, terminated, truncated, infos

    def _reward(self, r: Robot, a: np.ndarray, humans: np.ndarray) -> tuple[float, str | None]:
        center, yaw, up = r.pose
        if not (np.isfinite(center).all() and math.isfinite(yaw) and np.isfinite(r.vel).all()):
            # The articulation's physics state diverged; treat as a tip-over so the robot is respawned.
            r.pose, r.vel = (np.zeros(3), 0.0, 1.0), np.zeros(2)
            r.odom_vel = np.zeros(2)
            return -10.0, "tipped"
        rew = -0.005 - 0.02 * abs(float(a[1] - r.prev_action[1]))
        task_event = None
        if r.task is not None:
            task_rew, task_event = pt.update(self, r)
            rew += task_rew
        else:
            geo = self.map.geodesic(r.spec.inflation, r.field, center[:2])
            if not math.isfinite(geo):
                geo = r.prev_geo
            rew += float(np.clip(r.prev_geo - geo, -0.5, 0.5))
            r.prev_geo = geo
            if geo < r.best_geo - 0.5:
                r.best_geo, r.since_progress = geo, 0
            else:
                r.since_progress += 1
        human_d = self._footprint_distance(r, humans)
        if len(human_d):
            rew -= 0.05 * float(np.sum(np.clip(1.5 - human_d, 0.0, None)))
            r.metrics["min_human"] = min(r.metrics["min_human"], float(human_d.min()))
            r.metrics["near_human_steps"] += int(human_d.min() < 1.0)
        if up < 0.7 or center[2] < -1.0:
            return rew - 10.0, "tipped"
        if len(human_d) and human_d.min() < HUMAN_RADIUS + 0.05:
            # Workers do not see robots; contact only counts against a robot that is driving (a worker shoving a
            # stopped robot moves it too).
            if abs(r.vel[0]) > 0.15 and abs(r.v_cmd) > 0.15:
                k = int(np.argmin(human_d))
                c, s_ = math.cos(r.pose[1]), math.sin(r.pose[1])
                rel, wv = humans[k] - center[:2], self.worker_vel[k] if k < len(self.worker_vel) else np.zeros(2)
                r.contact_dbg = {"rel": [round(c * rel[0] + s_ * rel[1], 2), round(-s_ * rel[0] + c * rel[1], 2)],
                                 "wvel": [round(c * wv[0] + s_ * wv[1], 2), round(-s_ * wv[0] + c * wv[1], 2)],
                                 "v": round(float(r.vel[0]), 2), "v_cmd": round(float(r.v_cmd), 2),
                                 "dbg": getattr(r, "expert_dbg", None)}
                return rew - 10.0, "human_contact"
            r.metrics["passive_contacts"] += 1
        # Lidar contact with a worker is judged by the worker rule above, not as an obstacle collision.
        if r.clearance.min() < 0.03 and not r.closest_hit.startswith("/World/Characters/"):
            # Another robot (or the pallet it carries) running into a stopped robot is that robot's collision.
            passive = r.closest_hit.startswith(("/World/Robots/", "/World/Pallets/")) and \
                abs(r.vel[0]) < 0.15 and abs(r.v_cmd) < 0.15
            if not passive:
                return rew - 10.0, "collision"
        if r.task is not None:
            return rew, task_event
        if np.linalg.norm(center[:2] - r.goal) < r.spec.goal_tolerance:
            return rew + 10.0, "success"
        if r.since_progress > int(20.0 / CONTROL_DT):
            return rew - 2.0, "stuck"
        return rew, None

    def observations(self, est_poses=None) -> np.ndarray:
        """Sensor observations. ``est_poses`` (N x 3 x/y/yaw, NaN rows ignored) replaces the localization model."""
        if est_poses is not None:
            for r, p in zip(self.robots, np.asarray(est_poses, dtype=np.float64).reshape(-1, 3)):
                r.est_override = None if np.isnan(p).any() else p
        obs = np.stack([self._observe(r) if r.active else np.zeros(self.obs_dim) for r in self.robots])
        return np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    def privileged_observations(self) -> np.ndarray:
        humans = self.worker_positions()
        priv = np.stack([self._observe_privileged(r, humans) if r.active else np.zeros(self.priv_dim)
                         for r in self.robots])
        return np.nan_to_num(priv, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    def _observe(self, r: Robot) -> np.ndarray:
        x, y, yaw = r.est_pose
        c, s = math.cos(yaw), math.sin(yaw)
        d = r.goal - (x, y)
        goal_b = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        geo = self.map.geodesic(r.spec.inflation, r.field, (x, y))
        look = self.map.lookahead(r.spec.inflation, r.field, (x, y), r.spec.lookahead) - (x, y)
        look = np.array([c * look[0] + s * look[1], -s * look[0] + c * look[1]])
        look = look / max(np.linalg.norm(look), 1e-6)
        now = r.scan_hist[-1] if r.scan_hist else np.full(SCAN_BINS, 1.0 / (1.0 + 2.0 * LIDAR_RANGE))
        old = r.scan_hist[0] if r.scan_hist else now
        one_hot = np.array([r.type_name == t for t in TYPE_NAMES], np.float32)
        return np.concatenate(
            [
                np.clip(goal_b / 10.0, -3, 3),
                [min(geo / 20.0, 3.0) if math.isfinite(geo) else 3.0],
                look,
                [r.odom_vel[0] / r.spec.max_speed, r.odom_vel[1] / 1.5],
                r.prev_action,
                now,
                old,
                one_hot,
                *([pt.observe(self, r)] if self.task == "pallet" else []),
            ]
        )

    def _observe_privileged(self, r: Robot, humans: np.ndarray) -> np.ndarray:
        center, yaw, _ = r.pose
        c, s = math.cos(yaw), math.sin(yaw)

        def body(p):
            d = np.asarray(p)[..., :2] - center[:2]
            return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)

        goal_b = body(r.goal)
        geo = self.map.geodesic(r.spec.inflation, r.field, center[:2])
        look = body(self.map.lookahead(r.spec.inflation, r.field, center[:2], r.spec.lookahead))
        look = look / max(np.linalg.norm(look), 1e-6)
        near = np.tile([0.0, 0.0, 1.0], 2)
        if len(humans):
            hd = np.linalg.norm(humans - center[:2], axis=1)
            for j, k in enumerate(np.argsort(hd)[:2]):
                near[3 * j : 3 * j + 3] = [*np.clip(body(humans[k]) / 10.0, -1, 1), min(hd[k] / 10.0, 1.0)]
        one_hot = np.array([r.type_name == t for t in TYPE_NAMES], np.float32)
        return np.concatenate(
            [
                np.clip(goal_b / 10.0, -3, 3),
                [min(geo / 20.0, 3.0) if math.isfinite(geo) else 3.0],
                look,
                [r.vel[0] / r.spec.max_speed, r.vel[1] / 1.5],
                r.prev_action,
                np.clip(r.clearance / LIDAR_RANGE, 0, 1),
                near,
                one_hot,
                *([pt.observe_privileged(self, r)] if self.task == "pallet" else []),
            ]
        )

    # ------------------------------------------------------------------ expert

    def expert_actions(self) -> np.ndarray:
        """Scripted baseline: geodesic path following with lidar/worker-aware slowdown, normalized actions."""
        humans = self.worker_positions()
        return np.stack(
            [self._expert(r, humans) if r.active else np.zeros(self.act_dim) for r in self.robots]
        ).astype(np.float32)

    def _expert(self, r: Robot, humans: np.ndarray) -> np.ndarray:
        if r.task is not None:
            return pt.expert(self, r, humans)
        a = self._expert_nav(r, humans)
        return a if self.act_dim == ACT_DIM else np.array([a[0], a[1], -1.0], np.float32)

    def _human_conflict(self, r: Robot, speed: float) -> tuple[float, bool]:
        """Yield to workers: predict both motions over a short horizon at normalized ``speed``; returns the speed
        scale that stops the robot before the first predicted conflict (workers do not avoid robots)."""
        center, yaw, _ = r.pose
        pos, vel = getattr(self, "_worker_pos", np.zeros((0, 2))), self.worker_vel
        ok = np.isfinite(pos).all(axis=1) if len(pos) else np.zeros(0, bool)
        if not ok.any():
            return 1.0, False
        v_cmd = speed * r.spec.max_speed
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        c, s = math.cos(yaw), math.sin(yaw)
        margin = 0.5 if r.type_name == "carter" else 0.8
        for t in np.arange(0.0, 2.6, 0.25):
            rel = pos[ok] + vel[ok] * t - (center[:2] + fwd * v_cmd * t)
            local = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], axis=1)
            gap = np.linalg.norm(np.maximum(np.abs(local) - np.array(r.half), 0.0), axis=1) - HUMAN_RADIUS
            if gap.min() < margin:
                return float(np.clip((t - 0.5) / 2.0, 0.0, 1.0)), True
        return 1.0, False

    def _human_scale(self, r: Robot, speed: float) -> float:
        return self._human_conflict(r, speed)[0]

    def _expert_nav(self, r: Robot, humans: np.ndarray) -> np.ndarray:
        center, yaw, _ = r.pose
        target = self.map.lookahead(r.spec.inflation, r.field, center[:2], r.spec.lookahead)
        d = target - center[:2]
        if np.linalg.norm(r.goal - center[:2]) < r.spec.lookahead:
            d = r.goal - center[:2]
        alpha = wrap(math.atan2(d[1], d[0]) - yaw)
        dist = max(float(np.linalg.norm(d)), 1e-3)
        front = np.abs(np.array([wrap(a) for a in r.beam_angles])) < math.radians(35)
        rear = np.abs(np.array([wrap(a - math.pi) for a in r.beam_angles])) < math.radians(35)
        if r.spec.kind == "differential":
            turn = float(np.clip(2.0 * alpha / r.spec.max_turn, -1, 1))
            speed = 0.0 if abs(alpha) > 1.0 else math.cos(alpha) * min(1.0, dist / 1.0 + 0.3)
            sector = front
        else:
            reverse = abs(alpha) > math.radians(110)
            a_eff = wrap(alpha - math.pi) if reverse else alpha
            steer = math.atan2(2.0 * r.spec.controller["wheel_base"] * math.sin(a_eff), dist)
            # yaw rate = v * tan(steer) / wheel_base, so reversing needs the opposite steering sign
            turn = float(np.clip((-steer if reverse else steer) / r.spec.max_turn, -1, 1))
            speed = (0.4 if abs(steer) > 0.35 else 1.0) * (-0.5 if reverse else 1.0)
            sector = rear if reverse else front
        if r.recovery_steps > 0:
            # Reverse with opposite lock so the nose swings toward the path (a K-turn for the forklift).
            r.recovery_steps -= 1
            if r.clearance[rear].min() > 0.3:
                r.expert_dbg = {"mode": "recovery"}
                return np.array([-0.4, -math.copysign(1.0, alpha) if r.spec.kind == "ackermann" else 0.0], np.float32)
            r.recovery_steps = 0
        if r.spec.kind == "differential":
            # Distance to the nearest return inside the swept corridor ahead; a cone would also see the shelves
            # beside the robot in narrow aisles and stall it.
            px, py = r.scan * np.cos(r.scan_angles), r.scan * np.sin(r.scan_angles)
            ahead = (px > 0) & (np.abs(py) < r.half[1] + 0.15)
            clear = float((px[ahead] - r.half[0]).min()) if ahead.any() else SCAN_RANGE
            if r.clearance.min() < 0.05:
                clear = min(clear, float(r.clearance[sector].min()))
        else:
            clear = float(r.clearance[sector].min())
        scale = float(np.clip((clear - r.spec.stop_distance) / r.spec.stop_distance, 0.0, 1.0))
        hs, human_block = self._human_conflict(r, speed * scale)
        scale = min(scale, hs)
        blocked = scale == 0.0 and abs(speed) > 0 and not human_block
        r.blocked_steps = r.blocked_steps + 1 if blocked else 0
        if r.blocked_steps > int(1.5 / CONTROL_DT) and speed > 0:
            r.blocked_steps = 0
            r.recovery_steps = int((2.5 if r.spec.kind == "ackermann" else 1.0) / CONTROL_DT)
        r.expert_dbg = {
            "mode": "drive",
            "speed": round(speed, 3),
            "alpha": round(alpha, 3),
            "clear": round(clear, 3),
            "scale_clear": round(float(np.clip((clear - r.spec.stop_distance) / r.spec.stop_distance, 0.0, 1.0)), 3),
            "scale": round(scale, 3),
            "human_block": human_block,
        }
        return np.array([speed * scale, turn], np.float32)

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        return {
            "robots": [
                {
                    "name": r.name,
                    "type": r.type_name,
                    "xy": r.pose[0][:2].round(2).tolist(),
                    "yaw": round(r.pose[1], 3),
                    "goal": np.asarray(r.goal).round(2).tolist(),
                    "events": dict(r.events),
                    "active": r.active,
                    "expert": getattr(r, "expert_dbg", None),
                    "speed": round(float(r.odom_vel[0]), 3),
                    "lift": round(r.lift, 3),
                    "phase": None if r.task is None else r.task.phase,
                }
                for r in self.robots
            ],
            "workers": self.worker_positions().round(2).tolist(),
            "sim_time": self.timeline.get_current_time(),
            "benchmark_done": self.benchmark_done,
            "skipped_episodes": self.skipped_episodes,
        }

    def sensor_state(self) -> dict:
        """What a ROS stack on each robot would receive (noisy scan + odometry), plus truth for initial poses."""
        return {
            "sim_time": float(self.timeline.get_current_time()),
            "robots": [
                {
                    "active": r.active,
                    "scan": r.scan_noisy,
                    "odom_pose": r.odom_pose.copy(),
                    "odom_vel": r.odom_vel.copy(),
                    "true_pose": np.array([r.pose[0][0], r.pose[0][1], r.pose[1]]),
                    "goal": np.array([r.goal[0], r.goal[1], r.goal_yaw]),
                    "episode_id": r.episode_id,
                    "teleported": r.teleported,
                    "scan_filtered": r.scan_filtered,
                    "lift": r.lift,
                    "task": None if r.task is None else self._task_sensors(r),
                }
                for r in self.robots
            ],
        }

    def _task_sensors(self, r: Robot) -> dict:
        """Pallet task inputs for an external stack: phase, load sensor, the pallet detector output (target pose
        relative to the robot, None when the face is not in view this step), drop and retreat targets."""
        t = r.task
        return {
            "phase": t.phase,
            "loaded": t.loaded,
            "lifted": t.lifted,
            "detection": rel_list(pt.observed_target(self, r)[0]) if t.phase == 0 and t.det_age < 0.05 else None,
            # Last detected target, carried along by odometry (odom frame); None until the face was seen.
            "det_odom": None if t.det_odom is None or t.phase != 0 else t.det_odom.tolist(),
            "prior_target": pt.target_from_pallet(t.prior, t.side).tolist(),
            # Drop pose for the forklift given where the pallet sits on the forks (own estimate from the detector).
            "drop_target": pt.drop_target(t, t.carry_est).tolist(),
            "retreat_odom": None if t.retreat_odom is None else t.retreat_odom.tolist(),
            "repicks": t.repicks,
        }

    def robot_specs(self) -> list[dict]:
        """Static per-robot geometry and limits for external stacks (footprint, scan layout, speed limits)."""
        return [
            {
                "name": r.name,
                "type": r.type_name,
                "kind": r.spec.kind,
                "half": [float(r.half[0]), float(r.half[1])],
                "max_speed": r.spec.max_speed,
                "max_turn": r.spec.max_turn,
                "goal_tolerance": r.spec.goal_tolerance,
                "inflation": r.spec.inflation,
                "wheel_base": r.spec.controller.get("wheel_base"),
                "max_lateral_accel": r.spec.max_lateral_accel,
                "scan_angle_min": float(r.scan_angles[0]),
                "scan_angle_increment": float(2.0 * math.pi / SCAN_BEAMS),
                "scan_range_max": SCAN_RANGE,
                "scan_range_min": float(r.scan_body.min()),
            }
            for r in self.robots
        ]


def rel_list(p) -> list[float]:
    return [float(x) for x in p]


def _ira_config_version() -> str:
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    ext_id = mgr.get_enabled_extension_id("isaacsim.replicator.agent.core")
    version = ext_id.split("-")[-1].split("+")[0]
    major, minor = version.split(".")[:2]
    return f"{major}.{minor}.0"
