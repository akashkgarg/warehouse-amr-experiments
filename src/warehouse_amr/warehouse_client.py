"""Client for ``warehouse_sim.py serve``; needs only numpy, so it runs from any Python environment.

    from warehouse_client import WarehouseClient
    env = WarehouseClient("127.0.0.1:47800")
    obs = env.reset(seed=0)                       # (N robots, obs_dim) sensor observations
    obs, rew, term, trunc, infos = env.step(env.expert())
    env.priv                                      # (N, priv_dim) privileged observations of the same state
"""

from __future__ import annotations

import time
from multiprocessing.connection import Client

import numpy as np

AUTHKEY = b"warehouse-amr"


class WarehouseClient:
    def __init__(self, address: str = "127.0.0.1:47800", connect_timeout: float = 600.0):
        host, port = address.split(":")
        deadline = time.time() + connect_timeout
        while True:
            try:
                self.conn = Client((host, int(port)), authkey=AUTHKEY)
                break
            except (ConnectionRefusedError, OSError):
                if time.time() > deadline:
                    raise
                time.sleep(2.0)
        self.address = address
        self.spec = self._call({"cmd": "spec"})
        self.num_robots = len(self.spec["robots"])
        self.obs_dim = self.spec["obs_dim"]
        self.priv_dim = self.spec["priv_dim"]
        self.act_dim = self.spec["act_dim"]
        self.priv = None
        self.sensors = None

    def _call(self, msg: dict) -> dict:
        self.conn.send(msg)
        reply = self.conn.recv()
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def reset(self, seed: int | None = None, sensors: bool = False) -> np.ndarray:
        r = self._call({"cmd": "reset", "seed": seed, "sensors": sensors})
        self.priv, self.sensors = r["priv"], r.get("sensors")
        return r["obs"]

    def send_step(self, actions, caption: str = "", sensors: bool = False) -> None:
        self.conn.send({"cmd": "step", "actions": np.asarray(actions, np.float32), "caption": caption,
                        "sensors": sensors})

    def recv_step(self):
        r = self.conn.recv()
        if "error" in r:
            raise RuntimeError(r["error"])
        self.priv, self.sensors = r["priv"], r.get("sensors")
        return r["obs"], r["reward"], r["terminated"], r["truncated"], r["infos"]

    def step(self, actions, caption: str = "", sensors: bool = False):
        self.send_step(actions, caption, sensors)
        return self.recv_step()

    def observe(self, est_poses) -> np.ndarray:
        return self._call({"cmd": "observe", "est_poses": np.asarray(est_poses, np.float64)})["obs"]

    def expert(self) -> np.ndarray:
        return self._call({"cmd": "expert"})["actions"]

    def episodes(self, per_robot: int | None = None, seed: int = 12345, episodes=None) -> list:
        msg = {"cmd": "episodes", "seed": seed}
        if per_robot is None:
            msg["episodes"] = episodes
        else:
            msg["per_robot"] = per_robot
        return self._call(msg)["episodes"]

    def map(self) -> dict:
        return self._call({"cmd": "map"})

    def record(self, path: str | None, follow: int = 0) -> None:
        self._call({"cmd": "record", "path": path, "follow": follow})

    def status(self) -> dict:
        return self._call({"cmd": "status"})

    def close(self) -> None:
        try:
            self._call({"cmd": "close"})
        finally:
            self.conn.close()


class FleetClient:
    """Several sim servers stepped in parallel and concatenated along the robot axis."""

    def __init__(self, addresses: list[str]):
        self.envs = [WarehouseClient(a) for a in addresses]
        self.sizes = [e.num_robots for e in self.envs]
        self.num_robots = sum(self.sizes)
        self.obs_dim = self.envs[0].obs_dim
        self.priv_dim = self.envs[0].priv_dim
        self.act_dim = self.envs[0].act_dim
        self.robot_names = [f"{i}:{n}" for i, e in enumerate(self.envs) for n in e.spec["robots"]]
        self.robot_types = [t for e in self.envs for t in e.spec["types"]]

    @property
    def priv(self) -> np.ndarray:
        return np.concatenate([e.priv for e in self.envs])

    def _split(self, actions):
        return np.split(np.asarray(actions, np.float32), np.cumsum(self.sizes)[:-1])

    def reset(self, seed: int = 0) -> np.ndarray:
        return np.concatenate([e.reset(seed + 1000 * i) for i, e in enumerate(self.envs)])

    def step(self, actions, caption: str = ""):
        for e, a in zip(self.envs, self._split(actions)):
            e.send_step(a, caption)
        out = [e.recv_step() for e in self.envs]
        obs, rew, term, trunc = (np.concatenate([o[k] for o in out]) for k in range(4))
        infos = [i for o in out for i in o[4]]
        return obs, rew, term, trunc, infos

    def expert(self) -> np.ndarray:
        for e in self.envs:
            e.conn.send({"cmd": "expert"})
        return np.concatenate([e.conn.recv()["actions"] for e in self.envs])

    def episodes(self, per_robot: int, seed: int = 12345) -> list:
        return [e.episodes(per_robot, seed=seed + 1000 * i) for i, e in enumerate(self.envs)]

    def status(self) -> list[dict]:
        return [e.status() for e in self.envs]

    def close(self) -> None:
        for e in self.envs:
            e.close()
