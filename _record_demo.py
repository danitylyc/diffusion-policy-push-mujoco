"""Record one successful rollout as a short mp4."""
import numpy as np
import cv2
from env import PushEnv
from bc import MLPBC

m = MLPBC.load("checkpoints/bc_push.pkl")
e = PushEnv(randomize=True)
# find a successful seed
for seed in range(20):
    obs, _ = e.reset(seed=4000 + seed)
    done = False
    frames = []
    while not done:
        a = m.predict(obs["state"])
        obs, r, term, trunc, info = e.step(a)
        frame = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
        d = float(np.linalg.norm(e._cube_xy() - e.goal_xy))
        cv2.putText(frame, f"d={d:.2f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frames.append(frame)
        done = term or trunc
    if info["success"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter("videos/demo_success.mp4", fourcc, 10.0, (256, 256))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"seed {seed} success, {len(frames)} frames -> videos/demo_success.mp4")
        break
    else:
        print(f"seed {seed} failed, skip")
