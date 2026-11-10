"""
Mini Diffusion Policy (Zhao et al., RSS 2023).

Architecture (deliberately tiny so it trains on CPU):
  - image encoder: 4-layer CNN, 48x48x3 -> 64-d
  - state encoder: MLP 8 -> 32
  - condition: concat -> 128
  - 1D U-Net over action chunk (horizon=8, action_dim=2)
  - DDPM 50 train steps, DDIM 5 inference steps
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),   # 24
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 12
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), # 6
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, x):  # x: (B, 3, H, W) in [0,1]
        return self.net(x)


class UNet1d(nn.Module):
    def __init__(self, action_dim=2, horizon=8, cond_dim=128, hidden=32):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden))
        self.cond_proj = nn.Linear(cond_dim + hidden, hidden)
        self.in_proj = nn.Conv1d(action_dim, hidden, 1)
        self.down1 = self._block(hidden, hidden)
        self.down2 = self._block(hidden, hidden * 2)
        self.up2 = self._block(hidden * 2 + hidden, hidden)
        self.out = nn.Conv1d(hidden, action_dim, 1)

    def _block(self, cin, cout):
        return nn.Sequential(
            nn.Conv1d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(),
        )

    def forward(self, noisy_actions, t, cond):
        # noisy_actions: (B, H, A)
        x = noisy_actions.transpose(1, 2)               # (B, A, H)
        x = self.in_proj(x)                              # (B, hidden, H)
        temb = self.time_mlp(t.view(-1, 1))
        c = self.cond_proj(torch.cat([cond, temb], dim=-1))
        x = x + c.unsqueeze(-1)
        h1 = self.down1(x)
        h2 = self.down2(F.avg_pool1d(h1, 2))
        u2 = self.up2(torch.cat([F.interpolate(h2, scale_factor=2, mode="nearest"), h1], dim=1))
        return self.out(u2).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=2, horizon=8, n_train_steps=50):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_train_steps = n_train_steps
        self.img_enc = ImageEncoder(64)
        self.state_enc = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.cond_net = nn.Sequential(nn.Linear(64 + 32, 128), nn.ReLU())
        self.unet = UNet1d(action_dim, horizon, 128)

        betas = torch.linspace(1e-4, 0.02, n_train_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    def cond(self, image, state):
        zi = self.img_enc(image)
        zs = self.state_enc(state)
        return self.cond_net(torch.cat([zi, zs], dim=-1))

    def loss(self, image, state, action_chunk):
        c = self.cond(image, state)
        B = action_chunk.shape[0]
        t = torch.randint(0, self.n_train_steps, (B,), device=action_chunk.device)
        noise = torch.randn_like(action_chunk)
        ab = self.alpha_bars[t].view(B, 1, 1)
        noisy = torch.sqrt(ab) * action_chunk + torch.sqrt(1 - ab) * noise
        pred = self.unet(noisy, t.float() / self.n_train_steps, c)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict(self, image, state, n_infer_steps=5):
        c = self.cond(image, state)
        B = c.shape[0]
        device = c.device
        x = torch.randn(B, self.horizon, self.action_dim, device=device)
        ts = torch.linspace(0, self.n_train_steps - 1, n_infer_steps).long()
        for i in reversed(range(n_infer_steps)):
            t = ts[i].expand(B).to(device)
            eps = self.unet(x, t.float() / self.n_train_steps, c)
            ab = self.alpha_bars[t].view(B, 1, 1)
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            if i > 0:
                ab_prev = self.alpha_bars[ts[i - 1]].view(B, 1, 1)
                x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
            else:
                x = x0
        return x.clamp(-1.0, 1.0)
