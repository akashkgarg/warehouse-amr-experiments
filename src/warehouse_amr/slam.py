"""Multi-robot 2D pose-graph SLAM over ``collect_recon.py`` captures (no ground truth used).

One graph holds every capture of every robot in every run (they all map the same building):

* nodes: robot pose (x, y, yaw) at each capture;
* odometry edges between consecutive captures of a robot (same respawn segment) from its drifting wheel odometry;
* scan-matching edges from a planar point-to-plane ICP between lidar sweeps (sensor frame, with range noise):
  sequential (next two captures) and loop closures (any robot / run whose current estimate is nearby), with the
  ICP Hessian as the information matrix so a corridor only constrains the directions it can;
* weak priors from each robot's own localization estimate (``priors="all"``); or only each segment's first pose
  (``"first"``); or only each robot's very first pose per run (``"kidnapped"``: every respawn is unknown);
* place-recognition closures (``place=True``): Scan Context descriptors over each sweep, ring-key retrieval,
  multi-start ICP verification and pairwise-consistency filtering per pair of chains, so captures can be tied
  together no matter how far the current estimate has drifted;

* landmarks (``landmarks=True``): thin vertical poles (rack uprights, building columns) detected in each 3D sweep,
  associated in the world once the trajectory is good, and added as point nodes with position-only edges;

solved by sparse Gauss-Newton with a Cauchy kernel on the scan edges, re-searching closures as the estimate tightens.
"""

from __future__ import annotations

import math
import os
import time
import zlib

import numpy as np
from scipy.spatial import cKDTree

LIDAR_RANGE_STD = 0.015
SWEEP_VOXEL = 0.08
SC_RINGS, SC_SECTORS, SC_RANGE, SC_ZMAX = 20, 60, 30.0, 3.0
USE_MAST = not os.environ.get("RECON_NO_MAST")


# ----------------------------------------------------------------------------------------------------------- SE(2)
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose of ``b`` in the frame of ``a`` (rows x, y, yaw; broadcasts)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(b[:, 2] - a[:, 2])], 1)


def compose(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    a, d = np.atleast_2d(a), np.atleast_2d(d)
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    return np.stack([a[:, 0] + c * d[:, 0] - s * d[:, 1], a[:, 1] + s * d[:, 0] + c * d[:, 1],
                     wrap(a[:, 2] + d[:, 2])], 1)


# ---------------------------------------------------------------------------------------------------------- sweeps
def noisy_world_sweep(d: str, rec: dict, source: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Static lidar returns (world, as logged) with Gaussian range noise along each beam, and the beam origins.
    ``source``: "auto" (the forklift's mast lidar when it has one), "mast" or "base" (its planar scanner)."""
    z = np.load(os.path.join(d, "lidar", f"{rec['capture']:06d}_{rec['robot']}.npz"))
    mast = (source != "base" and USE_MAST and "mast_points" in z.files and "mast_lidar_origin" in rec)
    if mast:
        p = z["mast_points"][~z["mast_dynamic"]].astype(np.float64)
        o = np.array(rec["mast_lidar_origin"], np.float64)
    else:
        p = z["points"][~z["dynamic"]].astype(np.float64)
        o = np.array(rec["lidar_origin"], np.float64)
    if rec["type"] == "carter" or mast:
        origins = np.broadcast_to(o, p.shape)
    else:
        origins = np.c_[np.full(len(p), o[0]), np.full(len(p), o[1]), p[:, 2]]
    ray = p - origins
    rng = np.random.default_rng([rec["capture"], zlib.crc32(d.encode()), int(mast)])
    rng_m = np.linalg.norm(ray, axis=1, keepdims=True)
    p = origins + ray * (1.0 + rng.normal(0.0, LIDAR_RANGE_STD, (len(p), 1)) / np.maximum(rng_m, 1e-6))
    return p, origins


def to_local(p: np.ndarray, gt_pose) -> np.ndarray:
    """World points placed with the true pose back into the robot frame (what the sensor actually measured)."""
    x, y, z, yaw = gt_pose
    c, s = math.cos(yaw), math.sin(yaw)
    q = p - (x, y, z)
    return np.c_[c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1], q[:, 2]]


def voxel_down(p: np.ndarray, voxel: float) -> np.ndarray:
    if not len(p):
        return p
    k = np.floor(p / voxel).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv.ravel(), p)
    return out / cnt[:, None]


def normals(p: np.ndarray, planar: bool) -> np.ndarray:
    """PCA normals; for the forklift's horizontal scan lines, per-layer 2D line normals."""
    n = np.zeros_like(p)
    if planar:
        layers = np.round(p[:, 2] / 0.2)
        for lz in np.unique(layers):
            idx = np.nonzero(layers == lz)[0]
            if len(idx) < 4:
                continue
            q = p[idx, :2]
            _, nb = cKDTree(q).query(q, k=min(6, len(idx)), distance_upper_bound=0.5)
            ok = nb < len(idx)
            nbq = q[np.where(ok, nb, 0)]
            cnt = ok.sum(1, keepdims=True)
            mu = (nbq * ok[..., None]).sum(1) / cnt
            dq = (nbq - mu[:, None]) * ok[..., None]
            cov = np.einsum("nki,nkj->nij", dq, dq)
            w, v = np.linalg.eigh(cov)
            good = (cnt[:, 0] >= 3) & (w[:, 0] < 0.2 * w[:, 1])
            n[idx[good], :2] = v[good, :, 0]
        return n
    _, nb = cKDTree(p).query(p, k=min(10, len(p)), distance_upper_bound=0.4)
    ok = nb < len(p)
    nbq = p[np.where(ok, nb, 0)]
    cnt = ok.sum(1, keepdims=True)
    mu = (nbq * ok[..., None]).sum(1) / cnt
    dq = (nbq - mu[:, None]) * ok[..., None]
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", dq, dq))
    good = (cnt[:, 0] >= 5) & (w[:, 0] < 0.3 * w[:, 1])
    n[good] = v[good, :, 0]
    return n


def is_planar(d: str, rec: dict) -> bool:
    """Only the forklift's stock scanner is planar; its mast lidar (v3 data) is a full 3D sweep."""
    if rec["type"] == "carter":
        return False
    return not (USE_MAST and "mast_lidar_origin" in rec)


def scan_context(p: np.ndarray) -> np.ndarray:
    """Scan Context (Kim & Kim 2018): max height per (ring, sector) bin of a robot-frame sweep, heights capped so
    a low and a tall sensor describe the same place alike."""
    r = np.hypot(p[:, 0], p[:, 1])
    ok = (r > 0.5) & (r < SC_RANGE) & (p[:, 2] > 0.05)
    r, th, z = r[ok], np.arctan2(p[ok, 1], p[ok, 0]), np.minimum(p[ok, 2], SC_ZMAX)
    ri = np.minimum((r / SC_RANGE * SC_RINGS).astype(int), SC_RINGS - 1)
    si = np.minimum(((th + np.pi) / (2 * np.pi) * SC_SECTORS).astype(int), SC_SECTORS - 1)
    sc = np.zeros((SC_RINGS, SC_SECTORS), np.float32)
    np.maximum.at(sc, (ri, si), z.astype(np.float32))
    return sc


def detect_poles(p: np.ndarray, cell: float = 0.05, zlo: float = 0.3, zhi: float = 2.6, hbin: float = 0.15,
                 min_h: int = 6, max_ext: float = 0.2, iso: float = 0.35, rmax: float = 12.0) -> np.ndarray:
    """Pole landmarks in a robot-frame 3D sweep: 5 cm columns of returns spanning at least ``min_h`` height bins,
    at most ``max_ext`` across and with no other tall structure within ``iso``. Returns (K, 2) xy centroids."""
    from scipy import ndimage

    r = np.hypot(p[:, 0], p[:, 1])
    q = p[(p[:, 2] > zlo) & (p[:, 2] < zhi) & (r < rmax) & (r > 0.6)]
    if len(q) < 10:
        return np.zeros((0, 2))
    ij = np.floor(q[:, :2] / cell).astype(np.int64)
    ij -= ij.min(0)
    shape = ij.max(0) + 1
    hb = np.floor((q[:, 2] - zlo) / hbin).astype(np.int64)
    occ = np.zeros((shape[0], shape[1], int(np.ceil((zhi - zlo) / hbin))), bool)
    occ[ij[:, 0], ij[:, 1], hb] = True
    lab, k = ndimage.label(occ.sum(2) >= min_h, np.ones((3, 3)))
    out = []
    ri = int(round(iso / cell))
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    for l, sl in enumerate(ndimage.find_objects(lab), 1):
        if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * cell > max_ext:
            continue
        win = lab[max(sl[0].start - ri, 0):sl[0].stop + ri, max(sl[1].start - ri, 0):sl[1].stop + ri]
        if ((win > 0) & (win != l)).any():
            continue
        cells = np.argwhere(lab == l)
        out.append(q[np.isin(flat, cells[:, 0] * shape[1] + cells[:, 1]), :2].mean(0))
    return np.array(out).reshape(-1, 2)


MAST_ICP_BAND = (0.2, 4.0)


def _sweep_job(job):
    d, rec = job
    p, _ = noisy_world_sweep(d, rec)
    p = to_local(p, rec["gt_pose"])
    sc = scan_context(p)
    poles = np.zeros((0, 2)) if is_planar(d, rec) else detect_poles(p)
    if rec["type"] != "carter" and not is_planar(d, rec):
        # Forklift with a mast lidar: for matching, its low band plus the planar scanner. The mast alone matches
        # badly now and then (it sees little up close, under its 45 deg cone); the planar layers alone are fine but
        # too sparse to describe a place.
        band = p[(p[:, 2] > MAST_ICP_BAND[0]) & (p[:, 2] < MAST_ICP_BAND[1])]
        pm = voxel_down(band, SWEEP_VOXEL)
        nm = normals(pm, planar=False)
        pb, _ = noisy_world_sweep(d, rec, source="base")
        pb = voxel_down(to_local(pb, rec["gt_pose"]), SWEEP_VOXEL)
        nb = normals(pb, planar=True)
        p, n = np.concatenate([pm, pb]), np.concatenate([nm, nb])
    else:
        p = voxel_down(p, SWEEP_VOXEL)
        n = normals(p, planar=is_planar(d, rec))
    keep = np.abs(n).sum(1) > 0
    return p.astype(np.float32), n.astype(np.float32), keep, sc, poles


# ------------------------------------------------------------------------------------------------------------- ICP
def icp2d(src, tgt, tgt_n, tree, x0, radii=(0.5, 0.25, 0.12), iters=8):
    """Planar point-to-plane ICP of ``src`` onto ``tgt`` (both sensor frames) from initial pose ``x0`` of src in tgt.

    Returns (pose, fitness, rmse, information 3x3)."""
    x = np.array(x0, np.float64)
    for rad in radii:
        for _ in range(iters):
            c, s = math.cos(x[2]), math.sin(x[2])
            q = np.c_[c * src[:, 0] - s * src[:, 1] + x[0], s * src[:, 0] + c * src[:, 1] + x[1], src[:, 2]]
            dist, idx = tree.query(q, distance_upper_bound=rad)
            ok = np.isfinite(dist)
            if ok.sum() < 30:
                return x, 0.0, 1.0, np.zeros((3, 3))
            qo, t, n = q[ok], tgt[idx[ok]], tgt_n[idx[ok]]
            r = np.einsum("ij,ij->i", n, qo - t)
            J = np.c_[n[:, 0], n[:, 1], -n[:, 0] * qo[:, 1] + n[:, 1] * qo[:, 0]]
            hub = 0.5 * rad
            w = np.where(np.abs(r) < hub, 1.0, hub / np.maximum(np.abs(r), 1e-9))
            H = (J * w[:, None]).T @ J + 1e-6 * np.eye(3)
            dx = -np.linalg.solve(H, (J * w[:, None]).T @ r)
            cd, sd = math.cos(dx[2]), math.sin(dx[2])
            x = np.array([cd * x[0] - sd * x[1] + dx[0], sd * x[0] + cd * x[1] + dx[1], x[2] + dx[2]])
            if np.abs(dx[:2]).max() < 1e-4 and abs(dx[2]) < 1e-5:
                break
    c, s = math.cos(x[2]), math.sin(x[2])
    q = np.c_[c * src[:, 0] - s * src[:, 1] + x[0], s * src[:, 0] + c * src[:, 1] + x[1], src[:, 2]]
    dist, idx = tree.query(q, distance_upper_bound=0.1)
    ok = np.isfinite(dist)
    if ok.sum() < 30:
        return x, 0.0, 1.0, np.zeros((3, 3))
    qo, t, n = q[ok], tgt[idx[ok]], tgt_n[idx[ok]]
    r = np.einsum("ij,ij->i", n, qo - t)
    J = np.c_[n[:, 0], n[:, 1], -n[:, 0] * qo[:, 1] + n[:, 1] * qo[:, 0]]
    # Neighboring returns are not independent: scale the Hessian to ~N_EFF independent points at the noise level.
    n_eff = 60.0
    info = (J.T @ J) / ok.sum() * n_eff / 0.03 ** 2
    # J is w.r.t. a perturbation applied in the target frame (left); re-express for the pose vector itself.
    B = np.array([[1.0, 0.0, -x[1]], [0.0, 1.0, x[0]], [0.0, 0.0, 1.0]])
    Bi = np.linalg.inv(B)
    info = Bi.T @ info @ Bi
    return x, float(ok.mean()), float(np.sqrt(np.mean(r ** 2))), info


_W: dict = {}


def _init_worker(path: str) -> None:
    for name in ("pts", "nrm", "off"):
        _W[name] = np.load(f"{path}/{name}.npy", mmap_mode="r")
    _W["n_pts"] = np.load(f"{path}/n_pts.npy")
    _W["cache"] = {}


def _sweep(i: int):
    c = _W["cache"]
    if i not in c:
        if len(c) > 400:
            c.clear()
        o = _W["off"]
        p, n = np.array(_W["pts"][o[i]:o[i + 1]], np.float64), np.array(_W["nrm"][o[i]:o[i + 1]], np.float64)
        c[i] = (p, n, cKDTree(p))
    return c[i]


def _pair_job(job):
    i, j, x0 = job
    # Target is the denser sweep (Carter 3D) so a forklift's few scan lines register onto it, not the reverse.
    swap = _W["n_pts"][j] > _W["n_pts"][i]
    if swap:
        i, j, x0 = j, i, between(np.asarray(x0)[None], np.zeros((1, 3)))[0]
    tp, tn, tree = _sweep(i)
    sp, _, _ = _sweep(j)
    x, fit, rmse, info = icp2d(sp, tp, tn, tree, x0)
    if swap:
        # Edge stays (orig i -> orig j): invert the pose and rotate the information into the other frame.
        xi = between(x[None], np.zeros((1, 3)))[0]
        c, s = math.cos(x[2]), math.sin(x[2])
        # d(inv)/d(x) Jacobian for SE(2) inversion.
        Jinv = np.array([[-c, -s, s * x[0] - c * x[1]], [s, -c, c * x[0] + s * x[1]], [0, 0, -1.0]])
        A = np.linalg.inv(Jinv)
        info = A.T @ info @ A
        x = xi
    return x, fit, rmse, info


def _pr_job(job):
    """Verify a place-recognition candidate: ICP from the descriptor's yaw and a few translation starts."""
    i, j, yaw = job
    swap = _W["n_pts"][j] > _W["n_pts"][i]
    x0 = np.array([0.0, 0.0, yaw])
    if swap:
        i, j, x0 = j, i, between(x0[None], np.zeros((1, 3)))[0]
    tp, tn, tree = _sweep(i)
    sp, _, _ = _sweep(j)
    best = (x0, 0.0, 1.0, np.zeros((3, 3)))
    for dx, dy in ((0, 0), (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)):
        for dyaw in (0.0, 0.1, -0.1):
            res = icp2d(sp, tp, tn, tree, x0 + (dx, dy, dyaw), radii=(2.0, 1.0, 0.5, 0.25, 0.12), iters=6)
            if res[1] > best[1]:
                best = res
        if best[1] > 0.7:
            break
    x, fit, rmse, info = best
    if swap:
        xi = between(x[None], np.zeros((1, 3)))[0]
        c, s = math.cos(x[2]), math.sin(x[2])
        Jinv = np.array([[-c, -s, s * x[0] - c * x[1]], [s, -c, c * x[0] + s * x[1]], [0, 0, -1.0]])
        A = np.linalg.inv(Jinv)
        info = A.T @ info @ A
        x = xi
    return x, fit, rmse, info


def sc_candidates(sc: np.ndarray, chain: np.ndarray, pos: np.ndarray, query_every: int = 2, k_ring: int = 25,
                  per_query: int = 2, max_dist: float = 0.25, min_gap: int = 50):
    """Scan Context retrieval: ring-key kNN, then the column-shift distance. Returns (i, j, yaw_of_j_in_i, dist)."""
    n = len(sc)
    valid = sc > 0
    ring = sc.mean(2)
    tree = cKDTree(ring)
    norm = np.linalg.norm(sc, axis=1, keepdims=True)
    scn = np.where(norm > 0, sc / np.maximum(norm, 1e-9), 0.0)
    col_ok = norm[:, 0] > 0
    sector = 2 * np.pi / SC_SECTORS
    shift_idx = (np.arange(SC_SECTORS)[None, :] - np.arange(SC_SECTORS)[:, None]) % SC_SECTORS  # [s, c] -> c - s
    out = []
    queries = np.arange(0, n, query_every)
    _, nb = tree.query(ring[queries], k=min(k_ring + 1, n))
    for q, cand in zip(queries, nb):
        cand = cand[(cand != q) & ~((chain[cand] == chain[q]) & (np.abs(pos[cand] - pos[q]) < min_gap))]
        if not len(cand):
            continue
        M = np.einsum("rc,nrd->ncd", scn[q], scn[cand])  # cos(A col c, B col d)
        S = M[:, np.arange(SC_SECTORS)[None, :], shift_idx].sum(2)  # [cand, s]
        cnt = (col_ok[q][None, None, :] & col_ok[cand][:, shift_idx]).sum(2)
        dist = 1.0 - S / np.maximum(cnt, 1)
        best_s = dist.argmin(1)
        best_d = dist[np.arange(len(cand)), best_s]
        for o in np.argsort(best_d)[:per_query]:
            if best_d[o] < max_dist:
                yaw = float(wrap(best_s[o] * sector))
                out.append((int(q), int(cand[o]), yaw, float(best_d[o])))
    del valid
    return out


def pairwise_consistent(pairs: np.ndarray, z: np.ndarray, x: np.ndarray, chain: np.ndarray, min_group: int = 2):
    """Pairwise consistency (PCM-style) per pair of chains: keep the largest set of closures that agree with each
    other through each chain's own (locally accurate) trajectory. Returns a keep mask."""
    keep = np.zeros(len(pairs), bool)
    groups: dict = {}
    for k, (a, b) in enumerate(pairs):
        groups.setdefault((chain[a], chain[b]), []).append(k)
    for ks in groups.values():
        if len(ks) < min_group:
            continue
        ks = np.array(ks)
        a, b = pairs[ks, 0], pairs[ks, 1]
        m = len(ks)
        ok = np.eye(m, dtype=bool)
        for u in range(m):
            dA = between(np.repeat(x[a[u]][None], m, 0), x[a])          # a_u -> a_v
            dB = between(np.repeat(x[b[u]][None], m, 0), x[b])          # b_u -> b_v
            via_b = compose(np.repeat(z[ks[u]][None], m, 0), dB)        # b_v in a_u, through closure u
            via_a = compose(dA, z[ks])                                  # b_v in a_u, through closure v
            et = np.linalg.norm(via_b[:, :2] - via_a[:, :2], axis=1)
            er = np.abs(wrap(via_b[:, 2] - via_a[:, 2]))
            span = np.linalg.norm(dA[:, :2], axis=1) + np.linalg.norm(dB[:, :2], axis=1)
            ok[u] |= (et < 0.25 + 0.03 * span) & (er < math.radians(2.0) + 0.003 * span)
        ok &= ok.T
        # Greedy maximum clique.
        members = list(np.argsort(-ok.sum(1)))
        clique = []
        for u in members:
            if all(ok[u, v] for v in clique):
                clique.append(u)
        if len(clique) >= min_group:
            keep[ks[clique]] = True
    return keep


# ---------------------------------------------------------------------------------------------------------- solver
class Graph:
    def __init__(self, n: int):
        self.n = n
        self.edges: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        self.priors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def add(self, i, j, z, info, robust: float = 0.0):
        """Relative-pose edges (arrays): i, j (E,), z (E,3), info (E,3,3), Cauchy scale (0 = least squares)."""
        self.edges.append((np.asarray(i), np.asarray(j), np.asarray(z, np.float64), np.asarray(info), robust))

    def solve(self, x: np.ndarray, iters: int = 15, verbose: bool = False) -> np.ndarray:
        import scipy.sparse as sp

        try:
            from sksparse.cholmod import cho_solve

            def spsolve(H, b):
                return cho_solve(H, b)
        except ImportError:
            from scipy.sparse.linalg import spsolve

        x = x.copy()
        for it in range(iters):
            rows, cols, vals = [], [], []
            b = np.zeros(3 * self.n)
            cost = 0.0
            for i, j, z, info, robust in self.edges:
                xi, xj = x[i], x[j]
                c, s = np.cos(xi[:, 2]), np.sin(xi[:, 2])
                dx, dy = xj[:, 0] - xi[:, 0], xj[:, 1] - xi[:, 1]
                e = np.stack([c * dx + s * dy - z[:, 0], -s * dx + c * dy - z[:, 1],
                              wrap(xj[:, 2] - xi[:, 2] - z[:, 2])], 1)
                E = len(i)
                Ji = np.zeros((E, 3, 3))
                Ji[:, 0, 0], Ji[:, 0, 1], Ji[:, 0, 2] = -c, -s, -s * dx + c * dy
                Ji[:, 1, 0], Ji[:, 1, 1], Ji[:, 1, 2] = s, -c, -c * dx - s * dy
                Ji[:, 2, 2] = -1.0
                Jj = np.zeros((E, 3, 3))
                Jj[:, 0, 0], Jj[:, 0, 1] = c, s
                Jj[:, 1, 0], Jj[:, 1, 1] = -s, c
                Jj[:, 2, 2] = 1.0
                chi2 = np.einsum("ei,eij,ej->e", e, info, e)
                w = 1.0 / (1.0 + chi2 / robust ** 2) if robust else np.ones(E)
                cost += float((np.log1p(chi2 / robust ** 2) * robust ** 2).sum() if robust else chi2.sum())
                Wi = info * w[:, None, None]
                for (a, Ja), (bb, Jb) in (((i, Ji), (i, Ji)), ((i, Ji), (j, Jj)), ((j, Jj), (i, Ji)),
                                          ((j, Jj), (j, Jj))):
                    blk = np.einsum("eki,ekl,elj->eij", Ja, Wi, Jb)
                    r_idx = (3 * a)[:, None, None] + np.arange(3)[None, :, None]
                    c_idx = (3 * bb)[:, None, None] + np.arange(3)[None, None, :]
                    rows.append(np.broadcast_to(r_idx, blk.shape).ravel())
                    cols.append(np.broadcast_to(c_idx, blk.shape).ravel())
                    vals.append(blk.ravel())
                for a, Ja in ((i, Ji), (j, Jj)):
                    g = np.einsum("eki,ekl,el->ei", Ja, Wi, e)
                    np.add.at(b, (3 * a)[:, None] + np.arange(3)[None], g)
            if self.priors is not None:
                k, z, info = self.priors
                e = x[k] - z
                e[:, 2] = wrap(e[:, 2])
                cost += float(np.einsum("ei,eij,ej->e", e, info, e).sum())
                r_idx = (3 * k)[:, None, None] + np.arange(3)[None, :, None]
                c_idx = (3 * k)[:, None, None] + np.arange(3)[None, None, :]
                rows.append(np.broadcast_to(r_idx, info.shape).ravel())
                cols.append(np.broadcast_to(c_idx, info.shape).ravel())
                vals.append(info.ravel())
                np.add.at(b, (3 * k)[:, None] + np.arange(3)[None], np.einsum("eij,ej->ei", info, e))
            H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(3 * self.n, 3 * self.n)).tocsc()
            H = H + sp.identity(3 * self.n, format="csc") * 1e-6
            dx = -spsolve(H, b).reshape(-1, 3)
            x += dx
            x[:, 2] = wrap(x[:, 2])
            step = float(np.abs(dx[:, :2]).max())
            if verbose:
                print(f"[slam]   GN {it}: cost {cost:.4g}, max step {step:.4f} m", flush=True)
            if step < 1e-4:
                break
        return x


def solve_anchored(g: Graph, x: np.ndarray, chain_ok: np.ndarray, chain: np.ndarray) -> np.ndarray:
    """Solve only over chains tied to the map; the rest (unlocalized respawns) would make the system singular."""
    live = np.ones(len(x), bool)
    live[:len(chain)] = chain_ok[chain]
    return solve_live(g, x, live)


def solve_live(g: Graph, x: np.ndarray, live: np.ndarray) -> np.ndarray:
    if live.all():
        return g.solve(x)
    idx = np.nonzero(live)[0]
    remap = -np.ones(len(x), np.int64)
    remap[idx] = np.arange(len(idx))
    sub = Graph(len(idx))
    for i, j, z, info, rb in g.edges:
        ok = live[i] & live[j]
        sub.add(remap[i[ok]], remap[j[ok]], z[ok], info[ok], rb)
    if g.priors is not None:
        k, z, info = g.priors
        ok = live[k]
        sub.priors = (remap[k[ok]], z[ok], info[ok])
    out = x.copy()
    out[idx] = sub.solve(x[idx])
    return out


def add_landmarks(g: Graph, x: np.ndarray, poles: list, live: np.ndarray, chain: np.ndarray, log,
                  radius: float = 0.15, min_obs: int = 5):
    """Associate pole detections in the world (current poses) and add each cluster as a point node. Returns the
    graph over n + m nodes, the extended state and the number of landmarks."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg

    n = len(x)
    cap = np.concatenate([np.full(len(o), k) for k, o in enumerate(poles) if live[k] and len(o)] or [np.zeros(0, int)])
    obs = np.concatenate([o for k, o in enumerate(poles) if live[k] and len(o)] or [np.zeros((0, 2))])
    if not len(obs):
        return g, x, 0
    w = compose(x[cap], np.c_[obs, np.zeros(len(obs))])[:, :2]
    pairs = cKDTree(w).query_pairs(radius, output_type="ndarray")
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(w),) * 2)
    _, lab = cg.connected_components(A, directed=False)
    sizes = np.bincount(lab)
    n_chains = np.array([len(set(chain[cap[lab == c]].tolist())) for c in range(len(sizes))])
    good = (sizes >= min_obs) & (n_chains >= 2)
    lm_id = -np.ones(len(sizes), np.int64)
    lm_id[good] = np.arange(good.sum())
    m = int(good.sum())
    use = lm_id[lab] >= 0
    centers = np.zeros((m, 3))
    np.add.at(centers[:, :2], lm_id[lab[use]], w[use])
    centers[:, :2] /= np.bincount(lm_id[lab[use]], minlength=m)[:, None]
    g2 = Graph(n + m)
    g2.edges = list(g.edges)
    g2.priors = g.priors
    rng_m = np.linalg.norm(obs[use], axis=1)
    sig = 0.03 + 0.005 * rng_m
    info = np.zeros((use.sum(), 3, 3))
    info[:, 0, 0] = info[:, 1, 1] = 1.0 / sig ** 2
    g2.add(cap[use], n + lm_id[lab[use]], np.c_[obs[use], np.zeros(use.sum())], info, 2.0)
    log(f"[slam] landmarks: {m} poles from {use.sum()}/{len(obs)} detections")
    return g2, np.r_[x, centers], m


def place_chains(x, order, chain, anchored, pairs, z, odom, log, min_support: int = 2):
    """Kidnapped mode: place unanchored chains from place-recognition closures to already placed chains. Each
    closure implies a pose for the chain's start; the largest agreeing group (within 0.5 m / 5 deg) wins, best
    supported chains first, and the chain keeps its own (scan-matched) shape."""
    n_chain = len(anchored)
    placed = anchored.copy()
    members = [[] for _ in range(n_chain)]
    for k in order:
        members[chain[k]].append(k)
    x = x.copy()
    inv = between(z, np.zeros_like(z))
    while True:
        best = None
        votes: dict = {}
        for (a, b), za, zb in zip(pairs, z, inv):
            for p_, q_, zq in ((a, b, za), (b, a, zb)):
                c = chain[q_]
                if placed[chain[p_]] and not placed[c]:
                    xq = compose(x[p_][None], zq[None])[0]
                    start = compose(xq[None], between(x[q_][None], x[members[c][0]][None]))[0]
                    votes.setdefault(c, []).append(start)
        for c, v in votes.items():
            v = np.array(v)
            d = np.linalg.norm(v[:, None, :2] - v[None, :, :2], axis=2)
            dr = np.abs(wrap(v[:, None, 2] - v[None, :, 2]))
            agree = (d < 0.5) & (dr < math.radians(5.0))
            sup = agree.sum(1)
            u = int(sup.argmax())
            if sup[u] >= min_support and (best is None or sup[u] > best[1]):
                grp = v[agree[u]]
                best = (c, int(sup[u]), np.r_[grp[:, :2].mean(0), math.atan2(np.sin(grp[:, 2]).mean(),
                                                                              np.cos(grp[:, 2]).mean())])
        if best is None:
            break
        c, _, start = best
        mem = members[c]
        rel = between(np.repeat(x[mem[0]][None], len(mem), 0), x[mem])
        x[mem] = compose(np.repeat(start[None], len(mem), 0), rel)
        placed[c] = True
    log(f"[slam] kidnapped: {placed.sum()}/{n_chain} chains placed ({anchored.sum()} anchored)")
    return x, placed[chain]


# ------------------------------------------------------------------------------------------------------------ main
def linked_chains(closures: np.ndarray, chain: np.ndarray, n_chains: int) -> np.ndarray:
    """Chains with at least one loop closure to a different chain. Without priors on every pose, any other chain
    rests only on its start estimate (decimeters off), so it is not trusted as part of the map."""
    if not len(closures):
        return np.zeros(n_chains, bool)
    a, b = chain[closures[:, 0]], chain[closures[:, 1]]
    out = np.zeros(n_chains, bool)
    out[a[a != b]] = out[b[a != b]] = True
    return out


def run(recs: list[dict], priors="all", closure_rounds: int = 3, procs: int = 64, place: bool = False,
        landmarks: bool = False, log=print) -> np.ndarray:
    """Returns (N, 3) SLAM poses for ``recs`` (any order) without reading ``gt_pose`` except to undo the logged
    world placement of the lidar points (i.e. recover the sensor-frame measurement).

    ``priors``: "all" (every capture, loose), "first" (first capture of each respawn segment) or "kidnapped"
    (first capture of each robot per run only; later segments must be found by place recognition). Sets
    ``run.localized`` (captures tied to the map) alongside ``run.stages`` and ``run.closures``."""
    if priors is True:
        priors = "all"
    elif priors is False:
        priors = "first"
    import shutil
    from multiprocessing import get_context

    t0 = time.time()
    n = len(recs)
    order = sorted(range(n), key=lambda k: (recs[k]["key"], recs[k]["capture"]))
    # Chains: consecutive captures of one robot without a respawn in between.
    chain = np.zeros(n, np.int64)
    prev = None
    cid = -1
    for k in order:
        r = recs[k]
        tag = (r["key"], r["segment"])
        if tag != prev:
            cid += 1
            prev = tag
        chain[k] = cid
    odom = np.array([r["odom_pose"] for r in recs], np.float64)
    est = np.array([r["est_pose"] for r in recs], np.float64)
    carter = np.array([r["type"] == "carter" for r in recs])

    os.environ["OMP_NUM_THREADS"] = "1"
    ctx = get_context("spawn")
    shared = f"/dev/shm/slam_{os.getpid()}"
    os.makedirs(shared, exist_ok=True)
    try:
        with ctx.Pool(procs) as pool:
            sweeps = pool.map(_sweep_job, [(r["dir"], r) for r in recs], chunksize=16)
        pts = [sw[0][sw[2]] for sw in sweeps]
        nrm = [sw[1][sw[2]] for sw in sweeps]
        sc = np.stack([sw[3] for sw in sweeps])
        poles = [sw[4] for sw in sweeps]
        off = np.cumsum([0] + [len(p) for p in pts])
        np.save(f"{shared}/pts.npy", np.concatenate(pts))
        np.save(f"{shared}/nrm.npy", np.concatenate(nrm))
        np.save(f"{shared}/off.npy", off)
        np.save(f"{shared}/n_pts.npy", np.diff(off))
        log(f"[slam] {n} sweeps, {off[-1]} points, {cid + 1} chains ({time.time() - t0:.0f} s)")

        # Odometry dead-reckoning from each chain's first localization estimate.
        x = np.zeros((n, 3))
        seq_i, seq_j = [], []
        first = {}
        first_of_key = {}
        for pos, k in enumerate(order):
            c = chain[k]
            if c not in first:
                first[c] = k
                first_of_key.setdefault(recs[k]["key"], k)
                x[k] = est[k]
            else:
                kp = order[pos - 1]
                x[k] = compose(x[kp][None], between(odom[kp][None], odom[k][None]))[0]
                seq_i.append(kp)
                seq_j.append(k)
        seq_i, seq_j = np.array(seq_i), np.array(seq_j)
        g = Graph(n)
        z_odom = between(odom[seq_i], odom[seq_j])
        dist = np.linalg.norm(z_odom[:, :2], axis=1)
        sig_t = 0.03 + 0.05 * dist
        sig_r = 0.01 + 0.05 * np.abs(z_odom[:, 2]) + 0.01 * dist
        info_o = np.zeros((len(seq_i), 3, 3))
        info_o[:, 0, 0] = info_o[:, 1, 1] = 1 / sig_t ** 2
        info_o[:, 2, 2] = 1 / sig_r ** 2
        g.add(seq_i, seq_j, z_odom, info_o)
        if priors == "all":
            k = np.arange(n)
        elif priors == "first":
            k = np.array(sorted(first.values()))
        else:
            k = np.array(sorted(first_of_key.values()))
        s_t, s_r = 0.3, math.radians(5.0)
        anchored = np.zeros(cid + 1, bool)
        anchored[chain[k]] = True
        info_p = np.zeros((len(k), 3, 3))
        info_p[:, 0, 0] = info_p[:, 1, 1] = 1 / s_t ** 2
        info_p[:, 2, 2] = 1 / s_r ** 2
        g.priors = (k, est[k], info_p)

        # Scan-matching edges: next and second-next capture in the chain.
        pos_of = {k: p for p, k in enumerate(order)}
        pairs = [(a, b) for a, b in zip(seq_i, seq_j)]
        for p_, k in enumerate(order[:-2]):
            k2 = order[p_ + 2]
            if chain[k2] == chain[k]:
                pairs.append((k, k2))
        pairs = np.array(pairs)
        done = set()
        closures = []
        with ctx.Pool(procs, initializer=_init_worker, initargs=(shared,)) as pool:

            def match(pairs, x, robust, fit_min, rmse_max, tag):
                z0 = between(x[pairs[:, 0]], x[pairs[:, 1]])
                res = pool.map(_pair_job, [(int(a), int(b), z) for (a, b), z in zip(pairs, z0)], chunksize=64)
                z = np.array([r[0] for r in res])
                fit = np.array([r[1] for r in res])
                rmse = np.array([r[2] for r in res])
                info = np.array([r[3] for r in res])
                jump = np.linalg.norm(z[:, :2] - z0[:, :2], axis=1)
                ok = (fit > fit_min) & (rmse < rmse_max) & (np.linalg.eigvalsh(info)[:, 0] > 0)
                if tag != "seq":
                    ok &= jump < 1.0
                g.add(pairs[ok, 0], pairs[ok, 1], z[ok], info[ok], robust)
                if tag != "seq":
                    closures.append(pairs[ok])
                done.update(map(tuple, np.sort(pairs, 1).tolist()))
                log(f"[slam] {tag}: {ok.sum()}/{len(pairs)} ICP edges accepted (median fitness "
                    f"{np.median(fit):.2f}) ({time.time() - t0:.0f} s)")
                return pairs[ok], z[ok]

            yield_stage = [("odometry", x.copy())]
            match(pairs, x, 3.0, 0.5, 0.05, "seq")
            if priors == "kidnapped":
                # Refine every chain's own shape (each pinned at an arbitrary start) so the consistency check and
                # the placement below see scan-matched trajectories, not raw odometry.
                kept = g.priors
                kf = np.array(sorted(first.values()))
                g.priors = (kf, x[kf], np.repeat(info_p[:1], len(kf), 0))
                x = g.solve(x)
                g.priors = kept
            else:
                x = solve_anchored(g, x, anchored, chain)
            yield_stage.append(("odometry+seq", x.copy()))
            localized = anchored[chain].copy()
            if place:
                pos_arr = np.array([pos_of[k] for k in range(n)])
                cand = sc_candidates(sc, chain, pos_arr)
                log(f"[slam] place recognition: {len(cand)} descriptor candidates ({time.time() - t0:.0f} s)")
                res = pool.map(_pr_job, [(a, b, yaw) for a, b, yaw, _ in cand], chunksize=32)
                pr_pairs = np.array([(a, b) for a, b, _, _ in cand], np.int64).reshape(-1, 2)
                z = np.array([r[0] for r in res]).reshape(-1, 3)
                fit = np.array([r[1] for r in res])
                rmse = np.array([r[2] for r in res])
                info = np.array([r[3] for r in res]).reshape(-1, 3, 3)
                ok = (fit > 0.6) & (rmse < 0.04) & (np.linalg.eigvalsh(info)[:, 0] > 0) if len(res) else np.zeros(0, bool)
                verified = ok.sum()
                keep = np.zeros(len(ok), bool)
                keep[ok] = pairwise_consistent(pr_pairs[ok], z[ok], x, chain)
                log(f"[slam] place recognition: {verified} ICP-verified, {keep.sum()} pairwise-consistent "
                    f"({time.time() - t0:.0f} s)")
                done.update(map(tuple, np.sort(pr_pairs, 1).tolist()))
                pr_pairs, z, info = pr_pairs[keep], z[keep], info[keep]
                if priors == "kidnapped":
                    x, localized = place_chains(x, order, chain, anchored, pr_pairs, z, odom, log)
                g.add(pr_pairs[:, 0], pr_pairs[:, 1], z, info, 1.0)
                closures.append(pr_pairs)
                x = solve_anchored(g, x, localized[[first[c] for c in range(cid + 1)]], chain)
                # A wrong closure that survived the consistency check shows up as a large residual: drop and re-solve.
                i_, j_, z_, info_, rb = g.edges[-1]
                e = between(x[i_], x[j_]) - z_
                e[:, 2] = wrap(e[:, 2])
                chi2 = np.einsum("ei,eij,ej->e", e, info_, e)
                good = chi2 < 50.0
                g.edges[-1] = (i_[good], j_[good], z_[good], info_[good], rb)
                closures[-1] = closures[-1][good]
                log(f"[slam] place recognition: {good.sum()}/{len(good)} closures kept after the residual check")
                x = solve_anchored(g, x, localized[[first[c] for c in range(cid + 1)]], chain)
                yield_stage.append(("place recognition", x.copy()))
            run.localized = localized
            rng = np.random.default_rng(0)
            for rnd in range(closure_rounds):
                # Drifted chains (no priors, no place recognition) are still far off; keep searching wide.
                radius = (3.0, 2.0, 1.5, 1.5)[min(rnd, 3)] if (priors == "all" or place) else 3.0
                tree = cKDTree(x[:, :2])
                cand = []
                for a in rng.permutation(n):
                    if not localized[a]:
                        continue
                    nb = tree.query_ball_point(x[a, :2], radius)
                    nb = [b for b in nb if b != a and localized[b] and not (chain[b] == chain[a] and
                                                          abs(pos_of[b] - pos_of[a]) < 15)]
                    if not nb:
                        continue
                    for b in rng.choice(nb, size=min(4, len(nb)), replace=False):
                        key = (min(a, b), max(a, b))
                        if key not in done:
                            done.add(key)
                            cand.append(key)
                if not cand:
                    break
                match(np.array(cand), x, 3.0, 0.45, 0.05, f"closures round {rnd}")
                x = solve_anchored(g, x, localized[[first[c] for c in range(cid + 1)]], chain)
                yield_stage.append((f"closures {rnd}", x.copy()))
            if place and cid > 0:
                localized &= linked_chains(np.concatenate(closures), chain, cid + 1)[chain]
                log(f"[slam] {localized.sum()}/{n} captures in chains with a closure to another chain")
                run.localized = localized
            if landmarks:
                live = localized.copy()
                g2, xl, m = add_landmarks(g, x, poles, live, chain, log)
                if m:
                    xl = solve_live(g2, xl, np.r_[live, np.ones(m, bool)])
                    x = xl[:n]
                    run.landmarks = xl[n:, :2]
                yield_stage.append(("landmarks", x.copy()))
    finally:
        shutil.rmtree(shared, ignore_errors=True)
    run.stages = yield_stage
    if not hasattr(run, "localized") or len(run.localized) != n:
        run.localized = np.ones(n, bool)
    run.closures = np.concatenate(closures) if closures else np.zeros((0, 2), int)
    return x
