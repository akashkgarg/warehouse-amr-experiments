"""ROS 2 bridge: drive warehouse_sim robots with Nav2, or with a learned policy localized by AMCL.

Runs under the system ROS 2 Jazzy Python (not Isaac Sim's), against a running ``warehouse_sim.py serve``:

    source /opt/ros/jazzy/setup.bash
    python3.12 tools/warehouse_amr/nav2_bridge.py nav2 --address 127.0.0.1:47800 --episodes 10 \
        --out ~/warehouse_amr/bench/nav2.json
    python3.12 tools/warehouse_amr/nav2_bridge.py policy --policy ~/warehouse_amr/runs/ppo1/policy.npz \
        --address 127.0.0.1:47800 --episodes 10 --out ~/warehouse_amr/bench/policy_amcl.json

Per robot (ROS namespace = robot name) the bridge publishes ``scan`` (the sim's noisy planar lidar), ``odom`` and
``tf`` (odom -> base_link from noisy wheel odometry), ``tf_static`` (base_link -> base_scan) and ``initialpose`` at
each episode start, plus a global ``/clock``. ``nav2`` mode launches a full Nav2 stack per robot (AMCL, planner,
MPPI controller, behaviors, velocity smoother, collision monitor) and sends each benchmark goal as a
NavigateToPose action; ``cmd_vel`` is converted back to the sim's normalized actions. ``policy`` mode launches the
same stack but sends no goals, and feeds the AMCL pose estimate to the policy's observation.

AMCL's map_server gets the map at each robot type's lidar heights (what SLAM with that sensor would build); the global
costmap gets the full-height planning map on ``<robot>/planning_map``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils  # noqa: E402
from warehouse_client import WarehouseClient  # noqa: E402

NAV2_PARAMS = "/opt/ros/jazzy/share/nav2_bringup/params/nav2_params.yaml"
PALLET_STAGE = 4.0  # staging distance in front of the fork-insertion pose (pallet_task.STAGE)
LIFT_TIMEOUT_STEPS = 60
CONTROL_DT = 0.1
AMCL_TF_TOLERANCE = 0.5
RESTAGE_TOL = (0.12, math.radians(6.0))  # staging offset from the detected insertion line (lateral m, yaw rad)
# Fork tips reach the pallet face inside this distance-to-go window; the tips must line up with the pockets there.
POCKET_ENTRY = (0.2, 1.25)
POCKET_TOL = (0.07, math.radians(4.0))
FORK_TIP = 1.867  # forklift center -> fork tip (pallet_task.FORK_TIP)
MAX_RESTAGES = 3
DOCK_DEBUG = bool(os.environ.get("NAV2_DOCK_DEBUG"))


def pt_rel(ref: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pose ``p`` expressed in the frame of ``ref`` (x, y, yaw)."""
    c, s = math.cos(ref[2]), math.sin(ref[2])
    d = p[:2] - ref[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], math.atan2(math.sin(p[2] - ref[2]), math.cos(p[2] - ref[2]))])


# ---------------------------------------------------------------------------------------------- map + params


def map_layers(m: dict, robot_type: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(free, occupied) of the planning map, or of the map that robot type's lidar sees (for AMCL)."""
    free, occ = np.asarray(m["free"]), np.asarray(m["occupied"])
    if robot_type is None:
        return free, occ
    loc = np.asarray(m["loc_occupied"][robot_type])
    return (free | occ) & ~loc, loc


def write_map(m: dict, out_dir: str, robot_type: str | None = None) -> str:
    free, occ = map_layers(m, robot_type)
    img = np.full(free.shape, 205, np.uint8)
    img[free] = 254
    img[occ] = 0
    img = img[::-1]
    stem = f"warehouse_map_{robot_type}" if robot_type else "warehouse_map"
    pgm = os.path.join(out_dir, f"{stem}.pgm")
    with open(pgm, "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    path = os.path.join(out_dir, f"{stem}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump({"image": pgm, "mode": "trinary", "resolution": float(m["res"]),
                        "origin": [float(m["origin"][0]), float(m["origin"][1]), 0.0], "negate": 0,
                        "occupied_thresh": 0.65, "free_thresh": 0.25}, f)
    return path


def nav2_params(spec: dict, base: dict, pallet: bool = False) -> dict:
    """Stock Nav2 Jazzy params adapted to one robot: frames, footprint, limits, planner/controller per drive type.
    ``pallet``: the forklift also gets docking-server dock types for fork insertion (pallet detector) and drops."""
    p = copy.deepcopy(base)
    p.pop("loopback_simulator", None)
    hx, hy = spec["half"]
    footprint = f"[[{hx:.3f}, {hy:.3f}], [{hx:.3f}, {-hy:.3f}], [{-hx:.3f}, {-hy:.3f}], [{-hx:.3f}, {hy:.3f}]]"
    ackermann = spec["kind"] == "ackermann"
    vmax = float(spec["max_speed"])
    # MPPI's footprint collision checking wants inflation >= the circumscribed radius; a steep decay keeps narrow
    # aisles traversable.
    inflation = math.hypot(hx, hy) + 0.05
    if ackermann:
        r_min = spec["wheel_base"] / math.tan(spec["max_turn"])
        wmax, vrev, scaling, local_size = vmax / r_min, 0.5, 4.0, 10
    else:
        r_min, wmax, vrev, scaling, local_size = 0.0, float(spec["max_turn"]), 0.35, 3.0, 6

    amcl = p["amcl"]["ros__parameters"]
    amcl.update(base_frame_id="base_link", laser_max_range=19.9, laser_min_range=float(spec["scan_range_min"]),
                set_initial_pose=False, transform_tolerance=AMCL_TF_TOLERANCE, alpha1=0.1, alpha2=0.1, alpha3=0.1, alpha4=0.1, alpha5=0.1,
                # Repetitive shelving aliases along the aisles: use more beams and update often.
                max_beams=120, update_min_d=0.1, update_min_a=0.1, sigma_hit=0.1, min_particles=1000,
                max_particles=4000, laser_likelihood_max_dist=1.0)

    bt = p["bt_navigator"]["ros__parameters"]
    bt.update(robot_base_frame="base_link", odom_topic="odom")

    cs = p["controller_server"]["ros__parameters"]
    cs["general_goal_checker"].update(xy_goal_tolerance=0.5 * spec["goal_tolerance"], yaw_goal_tolerance=6.28)
    mppi = cs["FollowPath"]
    mppi.update(vx_max=vmax, vx_min=-vrev, wz_max=wmax, vy_max=0.0, visualize=False,
                motion_model="Ackermann" if ackermann else "DiffDrive")
    mppi["AckermannConstraints"]["min_turning_r"] = float(max(r_min, 0.2))
    mppi["CostCritic"]["consider_footprint"] = True

    def costmap(c: dict, local: bool) -> None:
        c.pop("robot_radius", None)
        # The warm-up sim clock runs much faster than wall time, so the default 60 s TF wait can lapse at bringup.
        c.update(robot_base_frame="base_link", footprint=footprint, footprint_padding=0.05,
                 initial_transform_timeout=3600.0)
        obstacle = {
            "plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True, "observation_sources": "scan",
            "scan": {"topic": "scan", "max_obstacle_height": 2.5, "clearing": True, "marking": True,
                     "data_type": "LaserScan", "raytrace_max_range": 10.0, "raytrace_min_range": 0.0,
                     "obstacle_max_range": 8.0, "obstacle_min_range": 0.0},
        }
        c["obstacle_layer"] = obstacle
        c.pop("voxel_layer", None)
        c["inflation_layer"].update(inflation_radius=inflation, cost_scaling_factor=scaling)
        if local:
            c.update(width=local_size, height=local_size, resolution=0.05,
                     plugins=["obstacle_layer", "inflation_layer"])
        else:
            c.update(resolution=0.1, plugins=["static_layer", "obstacle_layer", "inflation_layer"])
            # map_server serves the lidar-height map for AMCL; planning uses the full-height map (overhangs).
            c["static_layer"]["map_topic"] = f"/{spec['name']}/planning_map"

    costmap(p["local_costmap"]["local_costmap"]["ros__parameters"], local=True)
    costmap(p["global_costmap"]["global_costmap"]["ros__parameters"], local=False)

    planner = p["planner_server"]["ros__parameters"]
    if ackermann:
        planner["GridBased"] = {
            "plugin": "nav2_smac_planner::SmacPlannerHybrid", "downsample_costmap": False, "tolerance": 0.5,
            "allow_unknown": False, "max_iterations": 1000000, "max_on_approach_iterations": 1000,
            "max_planning_time": 5.0, "motion_model_for_search": "REEDS_SHEPP", "angle_quantization_bins": 72,
            "analytic_expansion_ratio": 3.5, "analytic_expansion_max_length": 3.0,
            "minimum_turning_radius": float(r_min), "reverse_penalty": 2.0, "change_penalty": 0.0,
            "non_straight_penalty": 1.2, "cost_penalty": 2.0, "retrospective_penalty": 0.015,
            "lookup_table_size": 20.0, "cache_obstacle_heuristic": False, "smooth_path": True,
        }
    else:
        planner["GridBased"].update(allow_unknown=False)

    beh = p["behavior_server"]["ros__parameters"]
    beh.update(robot_base_frame="base_link", max_rotational_vel=min(1.0, wmax))

    vs = p["velocity_smoother"]["ros__parameters"]
    acc = (1.0, 1.5) if ackermann else (1.5, 3.0)
    vs.update(max_velocity=[vmax, 0.0, wmax], min_velocity=[-vrev, 0.0, -wmax], max_accel=[acc[0], 0.0, acc[1]],
              max_decel=[-acc[0], 0.0, -acc[1]])

    cm = p["collision_monitor"]["ros__parameters"]
    cm.update(base_frame_id="base_link")
    cm["FootprintApproach"]["footprint_topic"] = "local_costmap/published_footprint"
    # The usual people-sharing setup: slow down while anything is inside a zone biased toward the direction of travel.
    front, side, back = 0.8, 0.4, 0.2
    cm["polygons"] = ["PolygonSlow", "FootprintApproach"]
    cm["PolygonSlow"] = {
        "type": "polygon", "action_type": "slowdown", "slowdown_ratio": 0.4, "min_points": 4, "visualize": False,
        "enabled": True,
        "points": f"[[{hx + front:.2f}, {hy + side:.2f}], [{hx + front:.2f}, {-hy - side:.2f}], "
                  f"[{-hx - back:.2f}, {-hy - side:.2f}], [{-hx - back:.2f}, {hy + side:.2f}]]",
    }
    cm["scan"]["topic"] = "scan"

    ds = p["docking_server"]["ros__parameters"]
    ds["base_frame"] = "base_link"
    if pallet:
        # Staging needs the heading too: the docking controller only corrects small offsets. Regulated Pure Pursuit
        # tracks the Smac Hybrid (Reeds-Shepp) path, so the truck arrives aligned; MPPI does not converge on a
        # heading-constrained goal with car-like kinematics.
        cs["general_goal_checker"].update(xy_goal_tolerance=0.15, yaw_goal_tolerance=0.1)
        cs["FollowPath"] = {
            "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
            "desired_linear_vel": 1.0, "lookahead_dist": 1.5, "min_lookahead_dist": 0.8, "max_lookahead_dist": 3.0,
            "lookahead_time": 1.5, "use_velocity_scaled_lookahead_dist": True, "transform_tolerance": 0.2,
            "min_approach_linear_velocity": 0.1, "approach_velocity_scaling_dist": 1.0,
            "use_collision_detection": True, "max_allowed_time_to_collision_up_to_carrot": 1.0,
            "use_regulated_linear_velocity_scaling": True, "regulated_linear_scaling_min_radius": 2.5,
            "regulated_linear_scaling_min_speed": 0.25, "use_cost_regulated_linear_velocity_scaling": False,
            "use_rotate_to_heading": False, "allow_reversing": True, "max_robot_pose_search_dist": 10.0,
            "use_fixed_curvature_lookahead": False, "interpolate_curvature_after_goal": False,
        }
        dock = {"plugin": "opennav_docking::SimpleNonChargingDock", "docking_threshold": 0.04,
                "staging_x_offset": -PALLET_STAGE, "staging_yaw_offset": 0.0, "use_stall_detection": False,
                "external_detection_timeout": 1.0, "external_detection_translation_x": 0.0,
                "external_detection_translation_y": 0.0, "external_detection_rotation_yaw": 0.0,
                "external_detection_rotation_pitch": 0.0, "external_detection_rotation_roll": 0.0,
                "filter_coef": 0.1}
        ds.update(dock_plugins=["pallet_dock", "drop_dock"], dock_approach_timeout=90.0, max_retries=2,
                  controller_frequency=20.0, initial_perception_timeout=5.0, dock_backwards=False,
                  dock_prestaging_tolerance=0.08,
                  pallet_dock={**dock, "use_external_detection_pose": True},
                  drop_dock={**dock, "use_external_detection_pose": False})
        ds.pop("simple_charging_dock", None)
        # The fork pockets leave ~6 cm per side: approach slowly with stiff heading gains.
        ds["controller"].update(v_linear_min=0.05, v_linear_max=0.2, v_angular_max=0.5, k_phi=5.0, k_delta=3.0,
                                use_collision_detection=True, dock_collision_threshold=0.3)
    return p


# ---------------------------------------------------------------------------------------------- bridge


class Bridge:
    def __init__(self, args):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile

        from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import DockRobot, NavigateToPose, UndockRobot
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from tf2_msgs.msg import TFMessage

        self.msgs = dict(PoseWithCovarianceStamped=PoseWithCovarianceStamped, TransformStamped=TransformStamped,
                         Odometry=Odometry, Clock=Clock, LaserScan=LaserScan, TFMessage=TFMessage,
                         NavigateToPose=NavigateToPose, DockRobot=DockRobot, UndockRobot=UndockRobot,
                         PoseStamped=PoseStamped)
        self.args = args
        self.env = WarehouseClient(args.address)
        self.specs = self.env.spec["robot_specs"]
        self.names = [s["name"] for s in self.specs]
        self.n = len(self.names)
        self.lock = threading.Lock()
        self.cmd = np.zeros((self.n, 2))
        self.cmd_ok = [False] * self.n
        self.map_odom = [None] * self.n
        self.map_odom_after = [0.0] * self.n
        self.goal_handle = [None] * self.n
        self.goal_episode = [-1] * self.n
        self.resend_at = [None] * self.n
        self.episode = [-1] * self.n
        self.task = self.env.spec.get("task", "nav")
        # Forklifts on the pallet task run the docking-server pipeline instead of a single NavigateToPose.
        self.pallet = [self.task == "pallet" and s["type"] == "forklift" for s in self.specs]
        self.pstate = [None] * self.n
        self.loc_start = [None] * self.n
        self.amcl_pose = [None] * self.n
        self.loc_sent = [0.0] * self.n
        self.lift_cmd = np.full(self.n, -1.0, np.float32)
        self.dock_fb = [0] * self.n
        self.policy = None
        if args.mode == "policy":
            from policy_np import NumpyPolicy

            self.policy = NumpyPolicy(os.path.expanduser(args.policy))

        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("warehouse_bridge")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_clock = self.node.create_publisher(Clock, "/clock", 10)
        self.pub = []
        for i, name in enumerate(self.names):
            ns = f"/{name}"
            self.pub.append(dict(
                scan=self.node.create_publisher(LaserScan, f"{ns}/scan", 10),
                odom=self.node.create_publisher(Odometry, f"{ns}/odom", 10),
                tf=self.node.create_publisher(TFMessage, f"{ns}/tf", 50),
                tf_static=self.node.create_publisher(TFMessage, f"{ns}/tf_static", latched),
                initialpose=self.node.create_publisher(PoseWithCovarianceStamped, f"{ns}/initialpose", 10),
                detected=self.node.create_publisher(PoseStamped, f"{ns}/detected_dock_pose", 10),
            ))
            self.node.create_subscription(Twist, f"{ns}/cmd_vel", lambda m, i=i: self._on_cmd(i, m), 10)
            self.node.create_subscription(TFMessage, f"{ns}/tf", lambda m, i=i: self._on_tf(i, m), 50)
            self.node.create_subscription(PoseWithCovarianceStamped, f"{ns}/amcl_pose",
                                          lambda m, i=i: self._on_amcl(i, m), 10)
        self.nav = [ActionClient(self.node, NavigateToPose, f"/{n}/navigate_to_pose") for n in self.names]
        self.dock = [ActionClient(self.node, DockRobot, f"/{n}/dock_robot") if pl else None
                     for n, pl in zip(self.names, self.pallet)]
        self.undock = [ActionClient(self.node, UndockRobot, f"/{n}/undock_robot") if pl else None
                       for n, pl in zip(self.names, self.pallet)]
        from std_srvs.srv import Trigger

        self.Trigger = Trigger
        manager = "lifecycle_manager_navigation" if args.mode == "nav2" else "lifecycle_manager_localization"
        self.lifecycle = [self.node.create_client(Trigger, f"/{n}/{manager}/is_active") for n in self.names]
        self.stack_active = [False] * self.n
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.procs = []

    # ------------------------------------------------------------------ ROS callbacks

    def _on_cmd(self, i: int, m) -> None:
        with self.lock:
            if self.cmd_ok[i]:
                self.cmd[i] = (m.linear.x, m.angular.z)

    def _on_amcl(self, i: int, m) -> None:
        stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        with self.lock:
            self.amcl_pose[i] = (m.pose.pose.position.x, m.pose.pose.position.y, stamp)

    def _on_tf(self, i: int, m) -> None:
        for t in m.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                stamp = t.header.stamp.sec + 1e-9 * t.header.stamp.nanosec
                q = t.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                with self.lock:
                    # AMCL future-dates its TF by transform_tolerance; older ones predate the last initial pose.
                    if stamp >= self.map_odom_after[i]:
                        self.map_odom[i] = (t.transform.translation.x, t.transform.translation.y, yaw)

    # ------------------------------------------------------------------ launch

    def launch(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        m = self.env.map()
        self._publish_planning_map(m)
        with open(NAV2_PARAMS) as f:
            base = yaml.safe_load(f)
        for spec, pallet in zip(self.specs, self.pallet):
            map_yaml = write_map(m, out_dir, spec["type"])
            params = os.path.join(out_dir, f"nav2_{spec['name']}.yaml")
            with open(params, "w") as f:
                yaml.safe_dump(nav2_params(spec, base, pallet), f, sort_keys=False)
            # /usr/bin/python3 may not be the 3.12 that Jazzy is built for, and `ros2` hardcodes it.
            ros2 = [sys.executable, "/opt/ros/jazzy/bin/ros2"]
            # Policy mode also uses the full bringup (navigation stays idle without goals): localization_launch.py
            # does not push the namespace, so per-robot map_server/amcl instances would collide.
            cmd = [*ros2, "launch", "nav2_bringup", "bringup_launch.py", f"namespace:={spec['name']}",
                   "use_namespace:=True", f"map:={map_yaml}", f"params_file:={params}", "use_sim_time:=True",
                   "autostart:=True", "use_composition:=True", "container_name:=nav2_container"]
            log = open(os.path.join(out_dir, f"ros_{spec['name']}.log"), "w")
            self.procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        print(f"[bridge] launched {len(self.procs)} {self.args.mode} stacks, logs in {out_dir}", flush=True)

    def _publish_planning_map(self, m: dict) -> None:
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        free, occ = map_layers(m)
        grid = np.full(free.shape, -1, np.int8)
        grid[free], grid[occ] = 0, 100
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.info.resolution = float(m["res"])
        msg.info.height, msg.info.width = grid.shape
        msg.info.origin.position.x, msg.info.origin.position.y = float(m["origin"][0]), float(m["origin"][1])
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pubs = []
        for name in self.names:
            pub = self.node.create_publisher(OccupancyGrid, f"/{name}/planning_map", qos)
            pub.publish(msg)
            self.map_pubs.append(pub)

    def shutdown(self) -> None:
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        self.executor.shutdown()
        self.node.destroy_node()
        self.rclpy.shutdown()

    # ------------------------------------------------------------------ publishing

    def _stamp(self, t: float):
        from builtin_interfaces.msg import Time

        sec = int(t)
        return Time(sec=sec, nanosec=int((t - sec) * 1e9))

    def _tf(self, stamp, parent, child, x, y, yaw):
        tf = self.msgs["TransformStamped"]()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, parent, child
        tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return tf

    def publish(self, sensors: dict) -> None:
        t = sensors["sim_time"]
        stamp = self._stamp(t)
        clock = self.msgs["Clock"]()
        clock.clock = stamp
        self.pub_clock.publish(clock)
        for i, (spec, s) in enumerate(zip(self.specs, sensors["robots"])):
            if not s["active"]:
                continue
            x, y, th = s["odom_pose"]
            self.pub[i]["tf"].publish(self.msgs["TFMessage"](transforms=[self._tf(stamp, "odom", "base_link", x, y, th)]))
            odom = self.msgs["Odometry"]()
            odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_link"
            odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            odom.twist.twist.linear.x, odom.twist.twist.angular.z = float(s["odom_vel"][0]), float(s["odom_vel"][1])
            self.pub[i]["odom"].publish(odom)
            scan = self.msgs["LaserScan"]()
            scan.header.stamp, scan.header.frame_id = stamp, "base_scan"
            scan.angle_min = spec["scan_angle_min"]
            scan.angle_increment = spec["scan_angle_increment"]
            scan.angle_max = spec["scan_angle_min"] + spec["scan_angle_increment"] * (len(s["scan"]) - 1)
            scan.scan_time, scan.time_increment = CONTROL_DT, 0.0
            scan.range_min, scan.range_max = spec["scan_range_min"], spec["scan_range_max"]
            # The forklift's own pallet is not an obstacle once the forks go in (fork insertion and carrying).
            ranges = s["scan_filtered"] if self._own_pallet_filtered(i, s) else s["scan"]
            scan.ranges = np.minimum(ranges, spec["scan_range_max"] - 0.01).astype(np.float32).tolist()
            self.pub[i]["scan"].publish(scan)

    def _own_pallet_filtered(self, i: int, s: dict) -> bool:
        st, task = self.pstate[i], s.get("task")
        if not self.pallet[i] or st is None or task is None:
            return False
        controlling = st["state"] == "picking" and self.dock_fb[i] == self.msgs["DockRobot"].Feedback.CONTROLLING
        if task["phase"] >= 1 or controlling or st["state"] in ("lift", "unpick", "unpicking"):
            return True
        if task["det_odom"] is None:
            return False
        # Re-staging with the forks at or in the pockets: the pallet would otherwise sit inside the footprint.
        rel = pt_rel(np.array(task["det_odom"]), np.array(s["odom_pose"]))
        return -rel[0] < 2.5 and abs(rel[1]) < 0.6 and abs(rel[2]) < 0.5

    def publish_static(self) -> None:
        stamp = self._stamp(0.0)
        for i in range(self.n):
            self.pub[i]["tf_static"].publish(
                self.msgs["TFMessage"](transforms=[self._tf(stamp, "base_link", "base_scan", 0.0, 0.0, 0.0)]))

    def initial_pose(self, i: int, pose, t: float) -> None:
        msg = self.msgs["PoseWithCovarianceStamped"]()
        msg.header.stamp, msg.header.frame_id = self._stamp(t), "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(pose[0]), float(pose[1])
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
        cov = np.zeros(36)
        cov[0] = cov[7] = 0.05
        cov[35] = 0.02
        msg.pose.covariance = cov.tolist()
        self.pub[i]["initialpose"].publish(msg)

    # ------------------------------------------------------------------ goals

    def send_goal(self, i: int, goal, t: float) -> None:
        g = self.msgs["NavigateToPose"].Goal()
        g.pose.header.frame_id, g.pose.header.stamp = "map", self._stamp(t)
        g.pose.pose.position.x, g.pose.pose.position.y = float(goal[0]), float(goal[1])
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
        ep = self.episode[i]
        fut = self.nav[i].send_goal_async(g)
        fut.add_done_callback(lambda f, i=i, ep=ep: self._on_accept(i, ep, f))

    def _on_accept(self, i: int, ep: int, fut) -> None:
        handle = fut.result()
        if not handle.accepted or ep != self.episode[i]:
            if handle.accepted:
                handle.cancel_goal_async()
            elif ep == self.episode[i]:
                with self.lock:
                    self.resend_at[i] = time.time() + 1.0
            return
        with self.lock:
            self.goal_handle[i], self.goal_episode[i], self.cmd_ok[i] = handle, ep, True
        handle.get_result_async().add_done_callback(lambda f, i=i, ep=ep: self._on_result(i, ep, f))

    def _on_result(self, i: int, ep: int, fut) -> None:
        # Nav2 gave up (or believes it arrived while the sim disagrees): retry after 1 s until the sim ends it.
        if ep == self.episode[i]:
            with self.lock:
                self.resend_at[i] = time.time() + 1.0

    def _pose_stamped(self, frame: str, pose, t: float):
        msg = self.msgs["PoseStamped"]()
        msg.header.frame_id, msg.header.stamp = frame, self._stamp(t)
        msg.pose.position.x, msg.pose.position.y = float(pose[0]), float(pose[1])
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
        return msg

    def send_dock(self, i: int, dock_type: str, pose, t: float) -> None:
        g = self.msgs["DockRobot"].Goal()
        g.use_dock_id, g.dock_type, g.navigate_to_staging_pose = False, dock_type, True
        # Stamp 0: transform the dock pose with the latest map -> odom, not at a stamp AMCL may not have yet.
        g.dock_pose = self._pose_stamped("map", pose, 0.0)
        self._send_task_goal(i, self.dock[i], g, feedback=True)

    def send_undock(self, i: int, dock_type: str) -> None:
        g = self.msgs["UndockRobot"].Goal()
        g.dock_type, g.max_undocking_time = dock_type, 60.0
        self._send_task_goal(i, self.undock[i], g)

    def _send_task_goal(self, i: int, client, goal, feedback: bool = False) -> None:
        ep, st = self.episode[i], self.pstate[i]
        st["result"] = None
        # Results of a goal cancelled for a re-send must not leak into the new one.
        seq = st["seq"] = st.get("seq", 0) + 1
        self.dock_fb[i] = 0
        fb = (lambda m, i=i: self.dock_fb.__setitem__(i, int(m.feedback.state)) if st["seq"] == seq else None) \
            if feedback else None
        fut = client.send_goal_async(goal, feedback_callback=fb)

        def accepted(f, i=i, ep=ep):
            handle = f.result()
            if ep != self.episode[i] or st["seq"] != seq:
                if handle.accepted:
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                st["result"] = False
                return
            with self.lock:
                self.goal_handle[i], self.goal_episode[i], self.cmd_ok[i] = handle, ep, True
            handle.get_result_async().add_done_callback(lambda r, i=i, ep=ep: done(r, i, ep))

        def done(f, i, ep):
            if ep != self.episode[i] or st["seq"] != seq:
                return
            with self.lock:
                self.cmd_ok[i], self.goal_handle[i] = False, None
                self.cmd[i] = 0.0
            res = f.result().result
            st["result"] = bool(res.success)
            if not res.success and res.error_code != 0:
                print(f"[bridge] {self.names[i]} {st['state']} failed: {res.error_code} {res.error_msg}", flush=True)

        fut.add_done_callback(accepted)

    def tick_pallet(self, sensors: dict, step: int) -> None:
        """Forklift pallet pipeline: dock into the pallet (Nav2 to staging + docking controller on the detector),
        lift, dock at the drop pose (AMCL), lower, undock. Each docking failure is retried."""
        t = sensors["sim_time"]
        now = time.time()
        for i, s in enumerate(sensors["robots"]):
            st, task = self.pstate[i], s.get("task")
            if not self.pallet[i] or not s["active"] or st is None or task is None:
                continue
            if task["det_odom"] is not None:
                self.pub[i]["detected"].publish(self._pose_stamped("odom", task["det_odom"], t))
            phase, state = task["phase"], st["state"]
            ready = now >= st["due"]

            def go(new, delay=0.0):
                print(f"[bridge] {self.names[i]} t={t:.1f} {st['state']} -> {new} (phase {phase})", flush=True)
                st["state"], st["due"], st["since"] = new, now + delay, step

            if not self.localized(i, s):
                continue
            if DOCK_DEBUG and state == "picking" and task["det_odom"] is not None and step % 5 == 0:
                rel = pt_rel(np.array(task["det_odom"]), np.array(s["odom_pose"]))
                print(f"[dock] {self.names[i]} t={t:.1f} fb={self.dock_fb[i]} rel=({rel[0]:+.2f},{rel[1]:+.2f},"
                      f"{np.degrees(rel[2]):+.1f}) cmd={self.cmd[i]}", flush=True)
            if state == "pick" and ready and self.dock[i].server_is_ready():
                # After a missed drop the pallet sits near the drop pose; the detector refines either estimate.
                target = task["prior_target"] if task["repicks"] == 0 else task["drop_target"]
                if st.get("restage") is not None and st["restage"][0] == task["repicks"]:
                    target = st["restage"][1]
                self.lift_cmd[i] = -1.0
                self.send_dock(i, "pallet_dock", target, t)
                go("picking")
            elif state == "picking":
                if phase >= 1:
                    self.cancel(i)
                    go("drop")
                elif st["result"] is True:
                    go("lift")
                elif st["result"] is False:
                    go("pick", 1.0)
                elif self.dock_fb[i] == self.msgs["DockRobot"].Feedback.CONTROLLING and task["det_odom"] is not None:
                    # Staging came from the pallet prior and the docking controller cannot remove a large lateral or
                    # heading offset over the staging distance; nor does it check alignment before the fork tips
                    # enter the pockets. Re-stage on the detection in either case.
                    det = np.array(task["det_odom"])
                    rel = pt_rel(det, np.array(s["odom_pose"]))
                    along, tip_lat = -rel[0], rel[1] + FORK_TIP * math.sin(rel[2])
                    staged = st.get("checked") != st["seq"]
                    st["checked"] = st["seq"]
                    bad = (abs(rel[1]) > RESTAGE_TOL[0] or abs(rel[2]) > RESTAGE_TOL[1]) if staged else \
                        POCKET_ENTRY[0] < along < POCKET_ENTRY[1] and \
                        (abs(tip_lat) > POCKET_TOL[0] or abs(rel[2]) > POCKET_TOL[1])
                    if bad and st.get("restages", 0) < MAX_RESTAGES:
                        st["restages"] = st.get("restages", 0) + 1
                        mo = self.map_odom[i]
                        c, sn = math.cos(mo[2]), math.sin(mo[2])
                        st["restage"] = (task["repicks"], [mo[0] + c * det[0] - sn * det[1],
                                                           mo[1] + sn * det[0] + c * det[1], mo[2] + det[2]])
                        print(f"[bridge] {self.names[i]} re-staging: offset lat {rel[1]:+.2f} m "
                              f"yaw {np.degrees(rel[2]):+.1f} deg", flush=True)
                        self.cancel(i)
                        go("pick", 0.5)
            elif state == "lift":
                self.lift_cmd[i] = 0.0
                if phase >= 1:
                    go("drop")
                elif step - st["since"] > LIFT_TIMEOUT_STEPS:
                    self.lift_cmd[i] = -1.0
                    go("unpick", 1.5)
            elif state == "unpick" and ready and self.undock[i].server_is_ready():
                self.send_undock(i, "pallet_dock")
                go("unpicking")
            elif state == "unpicking" and st["result"] is not None:
                go("pick", 0.5)
            elif state == "drop" and ready and self.dock[i].server_is_ready():
                self.lift_cmd[i] = 0.0
                self.send_dock(i, "drop_dock", task["drop_target"], t)
                go("dropping")
            elif state == "dropping":
                if phase == 0:
                    go("unpick")
                elif st["result"] is True:
                    go("lower")
                elif st["result"] is False:
                    go("drop", 1.0)
            elif state == "lower":
                self.lift_cmd[i] = -1.0
                if phase == 2:
                    go("undock")
                elif phase == 0 and step - st["since"] > 15:
                    go("unpick")
            elif state == "undock" and ready and self.undock[i].server_is_ready():
                self.send_undock(i, "drop_dock")
                go("undocking")
            elif state == "undocking":
                if phase == 0:
                    go("unpick", 0.5)
                elif phase == 1:
                    go("drop", 0.5)

    def localized(self, i: int, s: dict) -> bool:
        """AMCL has taken this episode's initial pose (an amcl_pose near the start, published after it, and a
        map -> odom TF); re-sends the initial pose if not."""
        with self.lock:
            mo, start, amcl = self.map_odom[i], self.loc_start[i], self.amcl_pose[i]
        if mo is None:
            return False
        if start is None:
            return True
        x, y, _ = s["odom_pose"]
        c, sn = math.cos(mo[2]), math.sin(mo[2])
        tf_ok = math.hypot(mo[0] + c * x - sn * y - start[0], mo[1] + sn * x + c * y - start[1]) < 1.0
        if tf_ok and amcl is not None and amcl[2] >= start[2] and \
                math.hypot(amcl[0] - start[0], amcl[1] - start[1]) < 1.0:
            self.loc_start[i] = None
            return True
        now = time.time()
        if now - self.loc_sent[i] > 1.0:
            self.loc_sent[i] = now
            self.initial_pose(i, s["true_pose"], 0.0)
        return False

    def cancel(self, i: int) -> None:
        with self.lock:
            handle, self.goal_handle[i], self.cmd_ok[i] = self.goal_handle[i], None, False
            self.cmd[i] = 0.0
            self.resend_at[i] = None
        if handle is not None:
            handle.cancel_goal_async()

    # ------------------------------------------------------------------ actions / estimates

    def actions_from_cmd(self) -> np.ndarray:
        with self.lock:
            cmd = self.cmd.copy()
        act = np.zeros((self.n, 2), np.float32)
        for i, (spec, (v, w)) in enumerate(zip(self.specs, cmd)):
            if spec["kind"] == "ackermann":
                if abs(v) < 0.05:
                    steer = math.copysign(spec["max_turn"], w) if abs(w) > 0.05 else 0.0
                else:
                    steer = math.atan(spec["wheel_base"] * w / v)
                act[i] = (v / spec["max_speed"], steer / spec["max_turn"])
            else:
                act[i] = (v / spec["max_speed"], w / spec["max_turn"])
        return np.clip(act, -1, 1)

    def estimates(self, sensors: dict, init_poses) -> np.ndarray:
        est = np.full((self.n, 3), np.nan)
        with self.lock:
            mo = list(self.map_odom)
        for i, s in enumerate(sensors["robots"]):
            if not s["active"]:
                continue
            if mo[i] is None:
                est[i] = init_poses[i]
                continue
            ox, oy, oth = mo[i]
            x, y, th = s["odom_pose"]
            c, sn = math.cos(oth), math.sin(oth)
            est[i] = (ox + c * x - sn * y, oy + sn * x + c * y, math.atan2(math.sin(oth + th), math.cos(oth + th)))
        return est

    # ------------------------------------------------------------------ main loop

    def new_episodes(self, sensors: dict, step: int, init_poses, send_goals: bool) -> None:
        for i, s in enumerate(sensors["robots"]):
            if s["episode_id"] == self.episode[i]:
                continue
            self.cancel(i)
            self.pstate[i] = None
            self.episode[i] = s["episode_id"]
            if not s["active"]:
                continue
            if s["teleported"] or init_poses[i] is None:
                init_poses[i] = s["true_pose"].copy()
                with self.lock:
                    self.map_odom[i] = None
                    self.map_odom_after[i] = sensors["sim_time"] + AMCL_TF_TOLERANCE + 0.05
                self.initial_pose(i, s["true_pose"], sensors["sim_time"])
                self.loc_start[i] = (*s["true_pose"][:2], sensors["sim_time"])
                self.loc_sent[i] = time.time()
            if send_goals and self.pallet[i] and s.get("task") is not None:
                self.pstate[i] = {"state": "pick", "due": time.time() + 0.3, "since": step, "result": None}
                self.dock_fb[i] = 0
                self.lift_cmd[i] = -1.0
            elif send_goals:
                with self.lock:
                    self.resend_at[i] = time.time() + 0.3

    def tick_goals(self, sensors: dict) -> None:
        now = time.time()
        for i, s in enumerate(sensors["robots"]):
            with self.lock:
                due = self.resend_at[i] is not None and now >= self.resend_at[i]
                if due:
                    self.resend_at[i] = None
            # Wait for AMCL to localize at the new start (map -> odom after the initial pose); planning from the
            # previous episode's pose fails or sends the robot the wrong way.
            if due and s["active"] and self.nav[i].server_is_ready() and self.localized(i, s):
                self.send_goal(i, s["goal"], sensors["sim_time"])
            elif due and s["active"]:
                with self.lock:
                    self.resend_at[i] = now + 1.0

    def run_steps(self, sensors, init_poses, until, on_info=None, label="", controller="zero") -> tuple[dict, int]:
        env, step, t_last = self.env, 0, time.time()
        while not until(step, sensors):
            self.publish(sensors)
            self.new_episodes(sensors, step, init_poses, send_goals=controller == "nav2")
            if controller == "zero" and step % 20 == 10:
                # Re-send initial poses AMCL has not picked up yet (it may not have been subscribed).
                for i, s in enumerate(sensors["robots"]):
                    if s["active"] and self.map_odom[i] is None:
                        self.initial_pose(i, s["true_pose"], sensors["sim_time"])
            if controller == "nav2":
                self.tick_goals(sensors)
                self.tick_pallet(sensors, step)
            # Real-time pacing: Nav2's timers and controllers run on wall clock rates, as on a robot.
            wait = CONTROL_DT - (time.time() - t_last)
            if wait > 0:
                time.sleep(wait)
            t_last = time.time()
            est = self.estimates(sensors, init_poses)
            if controller == "policy":
                act = self.policy(env.observe(est))
            elif controller == "nav2":
                act = self.actions_from_cmd()
                if self.task == "pallet":
                    act = np.concatenate([act, self.lift_cmd[:, None]], 1)
            else:
                act = np.zeros((self.n, 2), np.float32)
            truth = np.array([s["true_pose"] for s in sensors["robots"]])
            obs, rew, term, trunc, infos = env.step(act, caption=label, sensors=True)
            for i in range(self.n):
                if sensors["robots"][i]["active"] and not np.isnan(est[i]).any():
                    self.loc_err[i].append(float(np.hypot(*(est[i][:2] - truth[i][:2]))))
            sensors = env.sensors
            step += 1
            if on_info:
                for i, info in enumerate(infos):
                    if info:
                        on_info(i, info)
        return sensors, step

    def run(self) -> None:
        args, env = self.args, self.env
        out_dir = os.path.expanduser(args.ros_dir)
        self.loc_err = [[] for _ in range(self.n)]
        env.episodes(episodes=None)
        env.reset(seed=args.seed, sensors=True)
        sensors = env.sensors
        self.publish_static()
        self.launch(out_dir)
        init_poses = [None] * self.n

        # Warm up: robots hold still while the stacks configure, localize and (nav2) bring up their action servers.
        t0 = time.time()

        def ready(step, s):
            if time.time() - t0 > args.startup_timeout:
                raise RuntimeError("ROS stacks did not come up; see logs in " + out_dir)
            with self.lock:
                localized = all(m is not None for m in self.map_odom)
            if step % 10 == 0:
                for i, cli in enumerate(self.lifecycle):
                    if not self.stack_active[i] and cli.service_is_ready():
                        cli.call_async(self.Trigger.Request()).add_done_callback(
                            lambda f, i=i: self.stack_active.__setitem__(i, bool(f.result().success)))
            active = all(self.stack_active)
            if step % 20 == 0:
                print(f"[bridge] warmup {time.time() - t0:.0f}s localized={localized} "
                      f"active={sum(self.stack_active)}/{self.n}", flush=True)
            return localized and active and step > 20

        mode = args.mode
        sensors, _ = self.run_steps(sensors, init_poses, ready, label="warmup")
        print(f"[bridge] stacks ready after {time.time() - t0:.0f}s", flush=True)

        episodes = env.episodes(per_robot=args.episodes, seed=args.bench_seed)
        total = sum(len(e) for e in episodes)
        if args.record:
            env.record(os.path.abspath(os.path.expanduser(args.record)), follow=args.follow)
        env.reset(seed=args.seed, sensors=True)
        sensors = env.sensors
        self.episode = [-1] * self.n
        self.loc_err = [[] for _ in range(self.n)]
        records, t_bench = [], time.time()
        label = "Nav2 (AMCL)" if mode == "nav2" else f"policy (AMCL) {os.path.basename(args.policy)}"

        def on_info(i, info):
            errs = self.loc_err[i]
            info["loc_err_mean"] = float(np.mean(errs)) if errs else None
            self.loc_err[i] = []
            records.append(info)
            print(f"[{mode}] {len(records)}/{total} {info['robot']}: {info['event']} in {info['time']:.1f}s"
                  f" (loc err {info['loc_err_mean'] or 0:.2f} m)", flush=True)

        sensors, steps = self.run_steps(
            sensors, init_poses,
            lambda step, s: len(records) >= total or step >= args.max_steps
            or (step % 50 == 0 and env.status()["benchmark_done"]),
            on_info, label=label, controller=mode)
        skipped = env.status()["skipped_episodes"]
        if args.record:
            env.record(None)
        env.episodes(episodes=None)
        meta = {"controller": label, "bench_seed": args.bench_seed, "steps": steps, "skipped_episodes": skipped,
                "wall_s": round(time.time() - t_bench, 1)}
        out = os.path.expanduser(args.out) if args.out else None
        summary = bench_utils.write(out, records, meta) if out else bench_utils.summarize(records)
        print(json.dumps({**meta, "summary": summary}, indent=1), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("nav2", "policy"))
    p.add_argument("--address", default="127.0.0.1:47800")
    p.add_argument("--policy", default=None, help="policy mode: exported policy.npz")
    p.add_argument("--episodes", type=int, default=10, help="benchmark episodes per robot")
    p.add_argument("--bench-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--startup-timeout", type=float, default=240.0)
    p.add_argument("--out", default=None)
    p.add_argument("--record", default=None)
    p.add_argument("--follow", type=int, default=0)
    p.add_argument("--ros-dir", default="~/warehouse_amr/ros", help="generated map / params / ROS logs")
    args = p.parse_args()
    if args.mode == "policy" and not args.policy:
        p.error("policy mode needs --policy")
    bridge = Bridge(args)
    try:
        bridge.run()
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
