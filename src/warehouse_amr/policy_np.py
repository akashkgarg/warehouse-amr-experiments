"""Torch-free inference for an exported actor (``policy.npz`` from train_ppo.py), e.g. inside the ROS bridge."""

from __future__ import annotations

import numpy as np


class NumpyPolicy:
    def __init__(self, path: str):
        d = np.load(path)
        self.mean, self.var = d["mean"], d["var"]
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -8, 8)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return np.clip(x, -1, 1).astype(np.float32)
