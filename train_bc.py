"""Train the MLP BC policy on collected demonstrations."""
from __future__ import annotations

import argparse
import numpy as np
from bc import MLPBC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/push_demos.npz")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/bc_push.pkl")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    states = d["states"].astype(np.float32)
    actions = d["actions"].astype(np.float32)
    print(f"[bc] data: states {states.shape}, actions {actions.shape}")

    model = MLPBC(obs_dim=states.shape[1], action_dim=actions.shape[1], hidden=128)
    # fit normalization
    model.obs_mean = states.mean(axis=0)
    model.obs_std = states.std(axis=0)
    N = states.shape[0]
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        idx = rng.permutation(N)
        total = 0.0; nb = 0
        for s in range(0, N - args.batch_size, args.batch_size):
            bi = idx[s: s + args.batch_size]
            loss = model.train_step(states[bi], actions[bi], args.lr)
            total += loss; nb += 1
        if ep % 100 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep:3d} loss={total/nb:.4f}")

    model.save(args.out)
    print(f"[bc] saved -> {args.out}")


if __name__ == "__main__":
    main()
