"""Benchmark bookkeeping shared by train_ppo.py (expert / learned policy) and nav2_bridge.py (Nav2)."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

FAILURES = ("collision", "human_contact", "tipped", "stuck", "pallet_push", "pallet_drop")


def summarize(records: list[dict]) -> dict:
    """Per robot type: success / failure-mode rates, time, SPL, proximity to workers, and pallet-task metrics."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    out = {}
    for t, rs in sorted(by_type.items()):
        n = len(rs)
        ok = [r for r in rs if r["event"] == "success"]
        spl = [
            (r["event"] == "success") * r["geodesic"] / max(r["path_length"], r["geodesic"], 1e-6)
            for r in rs
            if r.get("geodesic") and math.isfinite(r["geodesic"]) and r.get("task") != "pallet"
        ]
        mins = [r["min_human"] for r in rs if r.get("min_human") is not None]
        out[t] = {
            "episodes": n,
            **{f"{e}_rate": round(sum(r["event"] == e for r in rs) / n, 3)
               for e in ("success", *FAILURES, "timeout")},
            "time_to_goal_s": round(float(np.mean([r["time"] for r in ok])), 1) if ok else None,
            "spl": round(float(np.mean(spl)), 3) if spl else None,
            "min_worker_dist_m": round(float(np.percentile(mins, 10)), 2) if mins else None,
            "near_worker_time_s": round(float(np.mean([r["near_human_time"] for r in rs])), 2),
        }
        if any(r.get("task") == "pallet" for r in rs):
            placed = [r for r in ok if r.get("place_err_m") is not None]
            out[t].update({
                "pick_rate": round(sum(bool(r.get("picked")) for r in rs) / n, 3),
                "time_to_pick_s": round(float(np.mean([r["pick_time"] for r in rs if r.get("picked")])), 1)
                if any(r.get("picked") for r in rs) else None,
                "place_err_m": round(float(np.mean([r["place_err_m"] for r in placed])), 3) if placed else None,
                "place_err_deg": round(float(np.mean([r["place_err_deg"] for r in placed])), 2) if placed else None,
                "repicks": round(float(np.mean([r.get("repicks", 0) for r in rs])), 2),
            })
    return out


def write(path: str, records: list[dict], meta: dict) -> dict:
    summary = summarize(records)
    with open(path, "w") as f:
        json.dump({**meta, "summary": summary, "episodes": records}, f, indent=1)
    return summary
