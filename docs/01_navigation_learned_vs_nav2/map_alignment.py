"""Lidar hits over the occupancy map, with the generator's raw origin (0.7 m off) and the calibrated one.
Run with Isaac Sim's python.sh against a free warehouse_sim server."""

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/isaacsim/tools/warehouse_amr"))
from warehouse_client import WarehouseClient

c = WarehouseClient(sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:47800")
m = c.map()
origin, res = np.asarray(m["origin"], float), float(m["res"])
occ = np.asarray(m["loc_occupied"]["carter"])
c.episodes(episodes=None)
c.reset(seed=21, sensors=True)
for _ in range(5):
    c.step(c.expert(), sensors=True)
angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
s = max((r for r in c.sensors["robots"][1:] if r["active"]),
        key=lambda r: np.sum(np.asarray(r["scan"]) < 6.0))
x, y, th = s["true_pose"]
scan = np.asarray(s["scan"])
pts = np.stack([x + scan * np.cos(th + angles), y + scan * np.sin(th + angles)], 1)[scan < 19.5]


def panel(map_origin, title):
    scale, half = 10, 7.0
    lo = np.array([x - half, y - half])
    img = np.full((int(2 * half / res * scale),) * 2 + (3,), 255, np.uint8)
    r0, c0 = ((lo - map_origin) / res).astype(int)[::-1]
    n = int(2 * half / res)
    sub = np.zeros((n, n), bool)
    rr, cc = np.mgrid[0:n, 0:n]
    ok = (rr + r0 >= 0) & (rr + r0 < occ.shape[0]) & (cc + c0 >= 0) & (cc + c0 < occ.shape[1])
    sub[ok] = occ[(rr + r0)[ok], (cc + c0)[ok]]
    big = cv2.resize(sub.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
    img[big] = (70, 70, 70)
    px = lambda p: (int((p[0] - lo[0]) / res * scale), int((p[1] - lo[1]) / res * scale))
    for p in pts:
        cv2.circle(img, px(p), 4, (0, 0, 230), -1)
    cv2.circle(img, px((x, y)), 9, (230, 120, 0), -1)
    cv2.line(img, px((x, y)), px((x + 1.2 * math.cos(th), y + 1.2 * math.sin(th))), (230, 120, 0), 3)
    img = np.ascontiguousarray(img[::-1])
    cv2.putText(img, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


raw = panel(origin + np.array([0.7, 0.0]), "generator bounds as reported")
fixed = panel(origin, "after ray-probe calibration")
gap = np.full((raw.shape[0], 20, 3), 255, np.uint8)
out = os.path.expanduser("~/warehouse_amr/report/media/map_alignment.png")
cv2.imwrite(out, np.hstack([raw, gap, fixed]))
print(out)
