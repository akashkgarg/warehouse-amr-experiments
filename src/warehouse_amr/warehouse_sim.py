"""Warehouse AMR sim: full warehouse, walking workers, a drivable ForkliftC and Nova Carters.

Run a local demo with a built-in policy (optionally recording an MP4):
    ./python.sh tools/warehouse_amr/warehouse_sim.py demo --robots forklift,carter,carter --workers 8 \
        --policy expert --steps 900 --record ~/warehouse_amr/demo.mp4

Serve the environment to an external trainer (see train_ppo.py / warehouse_client.py):
    ./python.sh tools/warehouse_amr/warehouse_sim.py serve --address 127.0.0.1:47800

Server protocol (multiprocessing.connection, authkey b"warehouse-amr"), one dict per request:
    {"cmd": "spec"}                         -> {"obs_dim", "priv_dim", "act_dim", "robots", "robot_specs", ...}
    {"cmd": "map"}                          -> {"free", "occupied", "origin", "res"} static occupancy grid, plus
                                               "loc_occupied" {robot type: grid at that type's lidar heights}
    {"cmd": "reset", "seed": int|None}      -> {"obs", "priv"}
    {"cmd": "step", "actions": (N, act_dim), "sensors": bool}
                                            -> {"obs", "priv", "reward", "terminated", "truncated", "infos"
                                                [, "sensors": per-robot noisy scan / odometry / goal]}
    {"cmd": "observe", "est_poses": (N, 3)} -> {"obs"} recomputed from external pose estimates (e.g. AMCL)
    {"cmd": "expert"}                       -> {"actions": (N, act_dim)} scripted baseline for the current state
    {"cmd": "episodes", "per_robot": int, "seed": int} | {"cmd": "episodes", "episodes": list|None}
                                            -> {"episodes"} fixed benchmark episodes (apply on next reset)
    {"cmd": "record", "path": str|None, "follow": int}  start (path) or stop (None) MP4 recording
    {"cmd": "status"}                       -> robot poses, goals, event counts, worker positions
    {"cmd": "close"}                        ends the session; the server keeps listening
Robots auto-reset individually, so ``step`` can be called indefinitely after one ``reset``. In benchmark mode
each robot runs its episode list and then parks (``status()["benchmark_done"]``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("mode", choices=("demo", "serve"))
parser.add_argument("--robots", default="forklift,carter,carter", help="comma-separated robot types: forklift, carter")
parser.add_argument("--workers", type=int, default=8, help="number of walking warehouse workers")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-episode-steps", type=int, default=1500, help="control steps (0.1 s) before timeout")
parser.add_argument("--frames-per-step", type=int, default=None,
                    help="rendered frames per 0.1 s control step (default: 3 when recording a demo, else 1)")
parser.add_argument("--policy", choices=("expert", "random", "zero"), default="expert", help="demo mode policy")
parser.add_argument("--steps", type=int, default=600, help="demo mode control steps")
parser.add_argument("--record", default=None, help="demo mode: write an MP4 here")
parser.add_argument("--follow", type=int, default=0, help="robot index the recording camera follows")
parser.add_argument("--address", default="127.0.0.1:47800", help="serve mode listen address")
parser.add_argument("--task", choices=("nav", "pallet"), default="nav",
                    help="pallet: forklifts run pallet pick-and-place episodes (see pallet_task.py)")
parser.add_argument("--status-file", default=None, help="write a JSON status snapshot here every 50 steps")
args, _ = parser.parse_known_args()

from warehouse_env import CONTROL_DT, launch_app

app = launch_app(gpu=args.gpu, headless=True)

import numpy as np
from recorder import Recorder
from warehouse_env import WarehouseEnv


def make_env() -> WarehouseEnv:
    t0 = time.time()
    env = WarehouseEnv(
        app,
        robots=[r.strip() for r in args.robots.split(",") if r.strip()],
        workers=args.workers,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        frames_per_step=args.frames_per_step or (3 if args.mode == "demo" and args.record else 1),
        task=args.task,
    )
    print(f"[warehouse_sim] scene ready in {time.time() - t0:.0f}s: robots {[r.name for r in env.robots]}, "
          f"{len(env.worker_paths)} workers", flush=True)
    return env


def write_status(env: WarehouseEnv, extra: dict) -> None:
    if args.status_file:
        with open(args.status_file, "w") as f:
            json.dump({**env.status(), **extra}, f)


def run_demo(env: WarehouseEnv) -> None:
    rng = np.random.default_rng(args.seed)
    recorder = None
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        recorder = Recorder(env, args.record, follow=args.follow)
    obs = env.reset(seed=args.seed)
    totals = np.zeros(len(env.robots))
    t0 = time.time()
    for step in range(args.steps):
        if args.policy == "expert":
            actions = env.expert_actions()
        elif args.policy == "random":
            actions = rng.uniform(-1, 1, (len(env.robots), env.act_dim))
        else:
            actions = np.zeros((len(env.robots), env.act_dim))
        if recorder:
            recorder.caption = f"t={step * CONTROL_DT:5.1f}s  policy={args.policy}"
        obs, rew, term, trunc, infos = env.step(actions)
        totals += rew
        for info in infos:
            if info:
                print(f"[step {step}] {info['robot']}: {info['event']} after {info['episode_length']} steps, "
                      f"return {info['episode_return']:.1f}" + (f", hit {info['hit']}" if "hit" in info else ""),
                      flush=True)
        if step % 50 == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"[step {step}] {rate:.1f} control steps/s, reward so far {np.round(totals, 1).tolist()}", flush=True)
            write_status(env, {"step": step})
    if recorder:
        recorder.close()
    print("[warehouse_sim] events:", json.dumps({r.name: r.events for r in env.robots}), flush=True)


def serve(env: WarehouseEnv) -> None:
    from multiprocessing.connection import Listener

    host, port = args.address.split(":")
    listener = Listener((host, int(port)), authkey=b"warehouse-amr")
    print(f"[warehouse_sim] serving on {args.address}", flush=True)
    recorder = None
    steps = 0
    while True:
        conn = listener.accept()
        print(f"[warehouse_sim] client connected from {listener.last_accepted}", flush=True)
        try:
            while True:
                msg = conn.recv()
                cmd = msg.get("cmd")
                if cmd == "spec":
                    conn.send({
                        "obs_dim": env.obs_dim, "priv_dim": env.priv_dim, "act_dim": env.act_dim,
                        "control_dt": CONTROL_DT, "task": env.task,
                        "robots": [r.name for r in env.robots], "types": [r.type_name for r in env.robots],
                        "robot_specs": env.robot_specs(), "workers": len(env.worker_paths),
                    })
                elif cmd == "map":
                    m = env.map
                    conn.send({"free": m.free, "occupied": m.occupied, "origin": m.origin.tolist(), "res": m.res,
                               "loc_occupied": env.loc_occupied})
                elif cmd == "reset":
                    obs = env.reset(seed=msg.get("seed"))
                    reply = {"obs": obs, "priv": env.privileged_observations()}
                    if msg.get("sensors"):
                        reply["sensors"] = env.sensor_state()
                    conn.send(reply)
                elif cmd == "step":
                    if recorder:
                        recorder.caption = msg.get("caption", "")
                    obs, rew, term, trunc, infos = env.step(np.asarray(msg["actions"], np.float32))
                    steps += 1
                    if steps % 50 == 0:
                        write_status(env, {"step": steps})
                    reply = {"obs": obs, "priv": env.privileged_observations(), "reward": rew, "terminated": term,
                             "truncated": trunc, "infos": infos}
                    if msg.get("sensors"):
                        reply["sensors"] = env.sensor_state()
                    conn.send(reply)
                elif cmd == "observe":
                    conn.send({"obs": env.observations(est_poses=msg.get("est_poses"))})
                elif cmd == "expert":
                    conn.send({"actions": env.expert_actions()})
                elif cmd == "episodes":
                    if "episodes" in msg:
                        episodes = msg["episodes"]
                    else:
                        episodes = env.make_episodes(int(msg["per_robot"]), seed=int(msg.get("seed", 12345)))
                    env.set_episodes(episodes)
                    conn.send({"episodes": episodes})
                elif cmd == "record":
                    if recorder:
                        recorder.close()
                        recorder = None
                    if msg.get("path"):
                        os.makedirs(os.path.dirname(os.path.abspath(msg["path"])), exist_ok=True)
                        recorder = Recorder(env, msg["path"], follow=int(msg.get("follow", 0)))
                    conn.send({"ok": True})
                elif cmd == "status":
                    conn.send(env.status())
                elif cmd == "close":
                    conn.send({"ok": True})
                    break
                else:
                    conn.send({"error": f"unknown cmd {cmd!r}"})
        except (EOFError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            if recorder:
                recorder.close()
                recorder = None
            print("[warehouse_sim] client disconnected", flush=True)


try:
    env = make_env()
    run_demo(env) if args.mode == "demo" else serve(env)
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
