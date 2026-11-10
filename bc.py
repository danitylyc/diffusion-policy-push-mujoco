"""
Simple MLP behavior-cloning policy (numpy, no torch).

This is the BC baseline from the Diffusion Policy paper (Zhao et al., RSS 2023).
It directly regresses the action from the low-dimensional state. For a 2D push
task this is enough to demonstrate the full pipeline: collect -> train ->
deploy. The full diffusion-based image-conditioned model lives in policy.py.
"""
from __future__ import annotations

import numpy as np
import pickle
from pathlib import Path


class MLPBC:
    def __init__(self, obs_dim: int = 8, action_dim: int = 2, hidden: int = 256,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2.0 / obs_dim), (obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0, 0.01, (hidden, hidden))
        self.b3 = np.zeros(hidden)
        self.w4 = rng.normal(0, 0.01, (hidden, action_dim))
        self.b4 = np.zeros(action_dim)
        self.obs_mean = None
        self.obs_std = None

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.obs_mean is None:
            return x
        return (x - self.obs_mean) / (self.obs_std + 1e-6)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize(x)
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        h = np.maximum(0.0, h @ self.w2 + self.b2)
        h = np.maximum(0.0, h @ self.w3 + self.b3)
        return h @ self.w4 + self.b4

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float):
        x = self.normalize(x)
        # forward
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0.0, z2)
        z3 = h2 @ self.w3 + self.b3
        h3 = np.maximum(0.0, z3)
        pred = h3 @ self.w4 + self.b4
        diff = pred - y
        loss = float(np.mean(diff ** 2))
        # backward
        B = x.shape[0]
        d4 = (2.0 / B / y.shape[1]) * diff
        gw4 = h3.T @ d4; gb4 = d4.sum(0)
        dh3 = d4 @ self.w4.T
        dz3 = dh3 * (z3 > 0)
        gw3 = h2.T @ dz3; gb3 = dz3.sum(0)
        dh2 = dz3 @ self.w3.T
        dz2 = dh2 * (z2 > 0)
        gw2 = h1.T @ dz2; gb2 = dz2.sum(0)
        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (z1 > 0)
        gw1 = x.T @ dz1; gb1 = dz1.sum(0)
        for w, g in [(self.w1, gw1), (self.b1, gb1), (self.w2, gw2),
                     (self.b2, gb2), (self.w3, gw3), (self.b3, gb3),
                     (self.w4, gw4), (self.b4, gb4)]:
            w -= lr * g
        return loss

    def predict(self, obs: np.ndarray) -> np.ndarray:
        a = self.forward(obs[None, :])[0]
        return np.clip(a, -1.0, 1.0)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        p = cls(obs_dim=d["w1"].shape[0], action_dim=d["w3"].shape[1],
                hidden=d["w1"].shape[1])
        p.__dict__.update(d)
        return p
