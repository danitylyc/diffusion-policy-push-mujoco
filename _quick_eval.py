"""Quick headless eval: no rendering, just success rate."""
import numpy as np
from env import PushEnv
from bc import MLPBC

m = MLPBC.load("checkpoints/bc_push.pkl")
e = PushEnv(randomize=True)
succ = 0
for ep in range(20):
    obs, _ = e.reset(seed=3000 + ep)
    done = False
    steps = 0
    while not done:
        a = m.predict(obs["state"])
        obs, r, term, trunc, info = e.step(a)
        done = term or trunc
        steps += 1
    succ += int(info["success"])
    d = float(np.linalg.norm(e._cube_xy() - e.goal_xy))
    print(f"ep {ep:2d} success={info['success']} d={d:.3f} steps={steps}")
print(f"success rate = {succ/20:.2f}")
