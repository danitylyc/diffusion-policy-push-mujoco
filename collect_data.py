"""
Collect scripted demonstrations for the push task.

Scripted policy:
  phase 1: move ee to a fixed approach point behind the cube (computed once
           at episode start, so we don't chase a moving target)
  phase 2: once close to the approach point, drive ee straight toward the goal
           -- this pushes the cube all the way there.
"""
from __future__ import annotations

import argparse
import numpy as np
from env import PushEnv


class _ScriptedPolicy:
    def __init__(self):
        self.approach = None
        self.phase = 0

    def reset(self):
        self.approach = None
        self.phase = 0

    def __call__(self, env: PushEnv) -> np.ndarray:
        cube = env._cube_xy()
        ee = env._ee_xy()
        goal = env.goal_xy
        d = goal - cube
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        d_hat = d / dist

        rest = 0.07
        if self.phase == 0:
            if self.approach is None:
                self.approach = cube - d_hat * rest
            if np.linalg.norm(ee - self.approach) < 0.02:
                self.phase = 1
            target = self.approach
        else:
            # drive straight through the cube toward goal
            target = goal

        delta = target - ee
        delta = np.clip(delta / 0.03, -1.0, 1.0)
        return delta.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", default="data/push_demos.npz")
    args = ap.parse_args()

    env = PushEnv(randomize=True)
    images, states, actions, rewards, dones = [], [], [], [], []
    ep_ends = []
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        policy = _ScriptedPolicy()
        done = False
        ep_len = 0
        while not done:
            a = policy(env)
            images.append(obs["image"])
            states.append(obs["state"])
            actions.append(a)
            obs, r, term, trunc, info = env.step(a)
            rewards.append(r)
            dones.append(term or trunc)
            ep_len += 1
            done = term or trunc
        successes += int(info["success"])
        ep_ends.append(len(actions))
        if ep % 10 == 0:
            print(f"[collect] ep {ep}/{args.episodes} success={info['success']} len={ep_len}")

    np.savez_compressed(
        args.out,
        images=np.stack(images),
        states=np.stack(states),
        actions=np.stack(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
        ep_ends=np.array(ep_ends),
        success_rate=np.array([successes / args.episodes]),
    )
    print(f"[collect] saved {args.out}, total_steps={len(actions)}, "
          f"success_rate={successes/args.episodes:.2f}")


if __name__ == "__main__":
    main()
