"""Evaluate a trained Diffusion Policy in MuJoCo and save a video."""
from __future__ import annotations

import argparse
import numpy as np
import torch
import cv2
from env import PushEnv
from policy import DiffusionPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dp_push.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out-video", default="videos/eval.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = DiffusionPolicy(horizon=ckpt["horizon"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = PushEnv(randomize=True)
    successes = 0
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            img = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            state = torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
            with torch.no_grad():
                chunk = model.predict(img, state, n_infer_steps=20)[0].cpu().numpy()
            act = chunk[0]
            obs, r, term, trunc, info = env.step(act)
            # record first episode only for the video
            if ep == 0:
                frame = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame, f"d={np.linalg.norm(env._cube_xy()-env.goal_xy):.2f}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(frame)
            done = term or trunc
        successes += int(info["success"])
        if ep % 5 == 0:
            print(f"[eval] ep {ep} success={info['success']}")

    sr = successes / args.episodes
    print(f"[eval] success rate = {sr:.2f} over {args.episodes} episodes")

    if frames:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.out_video, fourcc, 10.0, (256, 256))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[eval] video saved -> {args.out_video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
