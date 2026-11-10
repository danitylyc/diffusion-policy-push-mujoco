"""Train Diffusion Policy on scripted push demonstrations."""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from policy import DiffusionPolicy


class PushDataset(Dataset):
    def __init__(self, npz_path: str, horizon: int = 8, stride: int = 2):
        d = np.load(npz_path, allow_pickle=True)
        self.images = d["images"]            # (T, H, W, 3) uint8
        self.states = d["states"].astype(np.float32)
        self.actions = d["actions"].astype(np.float32)
        self.ep_ends = list(d["ep_ends"])
        self.horizon = horizon
        self.starts = []
        prev = 0
        for end in self.ep_ends:
            for s in range(prev, end - horizon, stride):
                self.starts.append(s)
            prev = end

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        h = self.horizon
        img = torch.from_numpy(self.images[s]).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(self.states[s])
        act = torch.from_numpy(self.actions[s: s + h])
        return img, state, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/push_demos.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/dp_push.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    ds = PushDataset(args.data, horizon=args.horizon)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[train] dataset size={len(ds)}")

    model = DiffusionPolicy(horizon=args.horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for img, state, act in dl:
            img = img.to(device); state = state.to(device); act = act.to(device)
            loss = model.loss(img, state, act)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * img.size(0)
            n += img.size(0)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[train] epoch {ep:3d} loss={total/n:.4f}")

    torch.save({"model": model.state_dict(), "horizon": args.horizon}, args.out)
    print(f"[train] saved -> {args.out}")


if __name__ == "__main__":
    main()
