"""Stereo depth for ``collect_recon.py`` captures with FoundationStereo (learned, zero-shot).

For every rectified pair (``<stem>.jpg`` left, ``<stem>_r.jpg`` right) writes ``<stem>_s.png``: 16-bit depth in mm
from disparity (``fx * baseline / d``), zero where the match is unusable (disparity under ``--min-disp``, pixels the
right camera cannot see, depth discontinuities). With ``--lr`` the right-view disparity is also computed (the
mirrored pair) and left pixels whose match does not map back within ``--lr-tol`` px are dropped as well: that removes
occlusions and the smear off thin foreground structure that the forward pass alone cannot detect.

Usage (FoundationStereo venv; one process per GPU with ``--shard k/n``)::

    ~/.venvs/stereo/bin/python tools/warehouse_amr/stereo_depth.py ~/warehouse_amr/recon/v2/data_s21 --shard 0/2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

FS_ROOT = os.path.expanduser("~/workspace/stereo/FoundationStereo")


def load_model(ckpt: str, iters: int):
    sys.path.insert(0, FS_ROOT)
    from omegaconf import OmegaConf

    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "cfg.yaml"))
    cfg.setdefault("vit_size", "vitl")
    cfg["valid_iters"] = iters
    model = FoundationStereo(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return model.cuda().eval()


def disparity(model, left: np.ndarray, right: np.ndarray, iters: int) -> np.ndarray:
    from core.utils.utils import InputPadder

    a = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    b = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = padder.pad(a, b)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        d = model.forward(a, b, iters=iters, test_mode=True)
    return padder.unpad(d.float()).cpu().numpy()[0, 0]


def lr_consistent(disp_l: np.ndarray, disp_r: np.ndarray, tol: float) -> np.ndarray:
    """Left pixels whose disparity agrees with the right view's disparity at the matched pixel."""
    h, w = disp_l.shape
    xr = np.arange(w, dtype=np.float32)[None, :] - disp_l
    back = cv2.remap(disp_r.astype(np.float32), xr.astype(np.float32),
                     np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=-1e3)
    return np.abs(back - disp_l) <= tol


def to_depth(disp: np.ndarray, fx: float, baseline: float, min_disp: float, max_depth: float,
             keep: np.ndarray | None = None) -> np.ndarray:
    h, w = disp.shape
    depth = np.where(disp > min_disp, fx * baseline / np.maximum(disp, 1e-6), 0.0)
    if keep is not None:
        depth[~keep] = 0
    # Left pixels whose match falls off the right image were never seen by both cameras.
    depth[np.arange(w)[None, :] - disp < 0] = 0
    # Flying pixels at depth edges: relative depth jump to a neighbor above 5%.
    z = np.where(depth > 0, depth, np.nan)
    jump = np.zeros_like(depth, bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(z, (dy, dx), (0, 1))
        jump |= np.abs(nb - z) > 0.05 * z
    depth[jump] = 0
    depth[depth > max_depth] = 0
    return depth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ckpt", default=os.path.join(FS_ROOT, "pretrained_models/23-51-11/model_best_bp2.pth"))
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--min-disp", type=float, default=2.0)
    ap.add_argument("--max-depth", type=float, default=30.0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="_s")
    ap.add_argument("--lr", action="store_true", help="left-right consistency check (twice the compute)")
    ap.add_argument("--lr-tol", type=float, default=1.0)
    ap.add_argument("--plain-suffix", default="", help="with --lr, also write the unchecked depth under this suffix")
    args = ap.parse_args()

    k, n = map(int, args.shard.split("/"))
    jobs = []
    for d in args.dirs:
        for line in open(os.path.join(d, "captures.jsonl")):
            r = json.loads(line)
            for c in r.get("cams", []):
                stem = os.path.join(d, "frames", c["stem"])
                if os.path.exists(stem + "_r.jpg"):
                    jobs.append((stem, c["K"][0], c["baseline"]))
    jobs = jobs[k::n]
    if args.limit:
        jobs = jobs[:args.limit]
    model = load_model(args.ckpt, args.iters)
    t0 = time.time()
    done = 0
    for i, (stem, fx, baseline) in enumerate(jobs):
        if os.path.exists(stem + args.suffix + ".png") and (
                not args.plain_suffix or os.path.exists(stem + args.plain_suffix + ".png")):
            continue
        left = cv2.imread(stem + ".jpg")[..., ::-1].copy()
        right = cv2.imread(stem + "_r.jpg")[..., ::-1].copy()
        disp = disparity(model, left, right, args.iters)
        keep = None
        if args.lr:
            disp_r = disparity(model, right[:, ::-1].copy(), left[:, ::-1].copy(), args.iters)[:, ::-1]
            keep = lr_consistent(disp, disp_r, args.lr_tol)
        depth = to_depth(disp, fx, baseline, args.min_disp, args.max_depth, keep)
        if args.lr and args.plain_suffix:
            plain = to_depth(disp, fx, baseline, args.min_disp, args.max_depth)
            cv2.imwrite(stem + args.plain_suffix + ".png", np.round(plain * 1000).astype(np.uint16))
        cv2.imwrite(stem + args.suffix + ".png", np.round(depth * 1000).astype(np.uint16))
        done += 1
        if done % 200 == 0:
            print(f"[stereo] {args.shard}: {i + 1}/{len(jobs)} ({done / (time.time() - t0):.1f} pairs/s)", flush=True)
    print(f"[stereo] {args.shard}: done {done} pairs in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
