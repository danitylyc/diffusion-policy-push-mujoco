"""Evaluate MLP BC policy and save a rollout video."""
from __future__ import annotations

import argparse
import numpy as np
import cv2
from env import PushEnv
from bc import MLPBC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_push.pkl")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out-video", default="videos/eval_bc.mp4")
    args = ap.parse_args()

    model = MLPBC.load(args.ckpt)
    env = PushEnv(randomize=True)
    successes = 0
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        while not done:
            act = model.predict(obs["state"])
            obs, r, term, trunc, info = env.step(act)
            if ep == 0:
                frame = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                d = float(np.linalg.norm(env._cube_xy() - env.goal_xy))
                cv2.putText(frame, f"d={d:.2f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(frame)
            done = term or trunc
        successes += int(info["success"])
        if ep % 5 == 0:
            d = float(np.linalg.norm(env._cube_xy() - env.goal_xy))
            print(f"[eval] ep {ep} success={info['success']} d={d:.3f}")

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
