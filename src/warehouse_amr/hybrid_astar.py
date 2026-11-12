"""Small Hybrid-A* planner for car-like robots (forward + reverse arcs) on the warehouse occupancy grid.

Plans for the fixed (non-steered) axle point, which moves along the heading with curvature in
``[-kappa_max, kappa_max]``. The footprint is covered by circles along the body axis and checked against the
distance transform of the static map plus optional extra obstacle points (e.g. the current lidar returns).
"""

from __future__ import annotations

import heapq
import math

import numpy as np


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class HybridAStar:
    def __init__(self, grid, half: tuple[float, float], axle_offset: float, kappa_max: float, margin: float = 0.03,
                 circles: int = 8):
        """``axle_offset``: fixed axle position ahead of the footprint center along the body x axis."""
        import scipy.ndimage as ndi

        self.grid = grid
        self.dist = (ndi.distance_transform_edt(grid.free) * grid.res).astype(np.float32)
        seg = 2.0 * half[0] / circles
        # circle centers relative to the axle point, along the heading
        self.cx = (-half[0] + seg * (np.arange(circles) + 0.5) - axle_offset).astype(np.float32)
        self.radius = math.hypot(half[1], seg / 2.0) + margin
        self.kappa = kappa_max
        self.step = 0.8
        self.sub = 4
        self.prims = []  # (direction, curvature, local poses (sub, 3))
        for d in (1, -1):
            for k in (-1.0, -0.5, 0.0, 0.5, 1.0):
                kap = k * kappa_max
                s = d * self.step * np.arange(1, self.sub + 1) / self.sub
                if abs(kap) < 1e-9:
                    loc = np.stack([s, np.zeros_like(s), np.zeros_like(s)], 1)
                else:
                    th = kap * s
                    loc = np.stack([np.sin(th) / kap, (1.0 - np.cos(th)) / kap, th], 1)
                self.prims.append((d, k, loc))

    def _free(self, poses: np.ndarray, obstacles: np.ndarray | None) -> bool:
        """poses: (n, 3) axle poses; True if every footprint circle is clear."""
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        px = poses[:, :1] + c[:, None] * self.cx[None, :]
        py = poses[:, 1:2] + s[:, None] * self.cx[None, :]
        g = self.grid
        cols = ((px - g.origin[0]) / g.res).astype(np.int64)
        rows = ((py - g.origin[1]) / g.res).astype(np.int64)
        if (cols < 0).any() or (rows < 0).any() or (cols >= g.w).any() or (rows >= g.h).any():
            return False
        if (self.dist[rows, cols] < self.radius).any():
            return False
        if obstacles is not None and len(obstacles):
            d2 = (px.reshape(-1, 1) - obstacles[None, :, 0]) ** 2 + (py.reshape(-1, 1) - obstacles[None, :, 1]) ** 2
            if d2.min() < (self.radius - 0.05) ** 2:
                return False
        return True

    def plan(self, start, goal, heuristic, obstacles=None, max_expansions: int = 8000,
             lane: tuple[float, float] = (0.0, 0.0), tol: tuple[float, float] = (0.3, 0.15), weight: float = 1.5):
        """Axle poses ``start`` -> ``goal`` (x, y, yaw). Any pose on the goal's approach line between ``lane[0]`` and
        ``lane[1]`` m behind the goal, within ``tol`` (lateral m, heading rad), also counts as arrived.
        ``heuristic(x, y)`` estimates the remaining distance to the goal position.
        Returns an (n, 4) array of (x, y, yaw, direction) at ~0.2 m spacing, or None."""
        start = np.asarray(start, np.float64)
        goal = np.asarray(goal, np.float64)
        gd = np.array([math.cos(goal[2]), math.sin(goal[2])])
        res_xy, res_th = 0.3, 2.0 * math.pi / 72

        def h(p):
            return heuristic(p[0], p[1]) + 1.5 * abs(wrap(p[2] - goal[2]))

        nodes = [(start, 0.0, -1, 1, -1)]  # pose, g, parent, direction, primitive
        key0 = (int(start[0] // res_xy), int(start[1] // res_xy), int((start[2] % (2 * math.pi)) // res_th))
        best = {key0: 0.0}
        heap = [(weight * h(start), 0, 0.0)]
        found = None
        exp = 0
        while heap and exp < max_expansions:
            f, i, g0 = heapq.heappop(heap)
            pose, g, _, dprev, _ = nodes[i]
            if g > g0 + 1e-9:
                continue
            exp += 1
            rel = pose[:2] - goal[:2]
            behind, lat = -(rel @ gd), rel[0] * -gd[1] + rel[1] * gd[0]
            yerr = abs(wrap(pose[2] - goal[2]))
            if yerr < tol[1] and abs(lat) < tol[0] and lane[0] - tol[0] <= behind <= lane[1] + tol[0]:
                found = i
                break
            # Analytic shot: a straight segment onto the goal's approach line.
            dist = math.hypot(rel[0], rel[1])
            if dist < 8.0 and yerr < tol[1]:
                c, s = math.cos(pose[2]), math.sin(pose[2])
                lon = c * -rel[0] + s * -rel[1]
                lat_err = -s * -rel[0] + c * -rel[1]
                if abs(lat_err) < tol[0]:
                    lon = lon - min(max(lon, 0.0), lane[1]) if lon > 0 else lon
                    lon = lon if abs(lon) > 0.05 else 0.0
                    n = max(2, int(abs(lon) / 0.2))
                    t = np.linspace(0, lon, n)[1:]
                    seg = np.stack([pose[0] + c * t, pose[1] + s * t, np.full_like(t, pose[2])], 1)
                    if abs(lon) > 0 and self._free(seg, obstacles):
                        d = 1 if lon >= 0 else -1
                        nodes.append((seg[-1], g + abs(lon), i, d, ("line", lon)))
                        found = len(nodes) - 1
                        break
            c, s = math.cos(pose[2]), math.sin(pose[2])
            for pi, (d, k, loc) in enumerate(self.prims):
                world = np.empty_like(loc)
                world[:, 0] = pose[0] + c * loc[:, 0] - s * loc[:, 1]
                world[:, 1] = pose[1] + s * loc[:, 0] + c * loc[:, 1]
                world[:, 2] = pose[2] + loc[:, 2]
                end = world[-1]
                key = (int(end[0] // res_xy), int(end[1] // res_xy), int((end[2] % (2 * math.pi)) // res_th))
                cost = g + self.step * (1.0 if d > 0 else 2.0) + 0.05 * abs(k) + (2.0 if d != dprev and i else 0.0)
                if cost >= best.get(key, math.inf):
                    continue
                if not self._free(world, obstacles):
                    continue
                best[key] = cost
                nodes.append((end.copy(), cost, i, d, pi))
                heapq.heappush(heap, (cost + weight * h(end), len(nodes) - 1, cost))
        if found is None:
            return None
        chain = []
        i = found
        while nodes[i][2] >= 0:
            chain.append(i)
            i = nodes[i][2]
        chain.reverse()
        out = [np.array([*start, nodes[chain[0]][3] if chain else 1])]
        for j in chain:
            pose_parent = nodes[nodes[j][2]][0]
            d, prim = nodes[j][3], nodes[j][4]
            c, s = math.cos(pose_parent[2]), math.sin(pose_parent[2])
            if isinstance(prim, tuple):
                lon = prim[1]
                t = np.linspace(0, lon, max(2, int(abs(lon) / 0.2)))[1:]
                pts = np.stack([pose_parent[0] + c * t, pose_parent[1] + s * t, np.full_like(t, pose_parent[2])], 1)
            else:
                dd, k, _ = self.prims[prim]
                kap = k * self.kappa
                sarr = dd * self.step * np.arange(1, 5) / 4
                if abs(kap) < 1e-9:
                    loc = np.stack([sarr, np.zeros_like(sarr), np.zeros_like(sarr)], 1)
                else:
                    th = kap * sarr
                    loc = np.stack([np.sin(th) / kap, (1.0 - np.cos(th)) / kap, th], 1)
                pts = np.stack([pose_parent[0] + c * loc[:, 0] - s * loc[:, 1],
                                pose_parent[1] + s * loc[:, 0] + c * loc[:, 1], pose_parent[2] + loc[:, 2]], 1)
            for p in pts:
                out.append(np.array([p[0], p[1], wrap(p[2]), d]))
        path = np.array(out)
        if len(path) > 1:
            path[0, 3] = path[1, 3]
        return path
