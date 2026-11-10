"""
MuJoCo 2D push task: a point end-effector pushes a square to a goal position.

This is the minimal version of the classic Diffusion Policy push benchmark
(Zhao et al., RSS 2023). We use a planar point EE instead of a 6-DoF arm so
that a scripted PD controller can produce clean demonstrations on CPU.

Observation:
    image  : top-down RGB (H, W, 3) uint8
    state  : (ee_x, ee_y, cube_x, cube_y, cube_cos, cube_sin, goal_x, goal_y)

Action:
    (dx, dy) target end-effector displacement in [-1, 1], scaled to ~2 cm/step.
"""
from __future__ import annotations

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

XML = """
<mujoco model="push">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01"/>
  <default>
    <geom condim="4" friction="0.6 0.1 0.01"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="table" type="plane" size="0.6 0.6 0.02" rgba="0.3 0.3 0.35 1"/>
    <!-- goal marker (visual) -->
    <geom name="goal" type="cylinder" size="0.08 0.005" pos="0.3 0.3 0.005" rgba="0.2 0.9 0.3 0.6"/>
    <!-- end-effector: a small cylinder on a prismatic xy joint -->
    <body name="ee" pos="0 0 0.03">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>
      <geom name="ee_g" type="cylinder" size="0.03 0.03" rgba="0.9 0.3 0.3 1"/>
    </body>
    <!-- cube: flat cylinder so it doesn't spin when pushed -->
    <body name="cube" pos="-0.2 -0.2 0.025">
      <freejoint name="cube_joint"/>
      <geom name="cube_g" type="cylinder" size="0.04 0.025" rgba="0.2 0.5 0.9 1" mass="0.03"/>
    </body>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0" fovy="40"/>
  </worldbody>
  <actuator>
    <position name="act_x" joint="ee_x" kp="500" damping="20"/>
    <position name="act_y" joint="ee_y" kp="500" damping="20"/>
  </actuator>
</mujoco>
"""


class PushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, image_size: int = 48, max_steps: int = 150,
                 randomize: bool = True, seed: int | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.max_steps = max_steps
        self.randomize = randomize
        self._rng = np.random.default_rng(seed)
        self._step = 0

        self.ee_x_adrs = self.model.joint("ee_x").qposadr[0]
        self.ee_y_adrs = self.model.joint("ee_y").qposadr[0]
        self.cube_qpos_adrs = self.model.joint("cube_joint").qposadr[0]

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._renderer = None
        self.goal_xy = np.array([0.3, 0.3])

    # ------------------------------------------------------------------ helpers
    def _render_image(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._renderer.update_scene(self.data, camera="top")
        return self._renderer.render()

    def _get_state(self) -> np.ndarray:
        q = self.data.qpos
        ee = np.array([q[self.ee_x_adrs], q[self.ee_y_adrs]])
        cube_xy = q[self.cube_qpos_adrs: self.cube_qpos_adrs + 2]
        quat = q[self.cube_qpos_adrs + 3: self.cube_qpos_adrs + 7]
        # quat -> yaw (cos, sin)
        w, x, y, z = quat
        yaw = 2 * np.arctan2(z, w)
        return np.concatenate([ee, cube_xy, [np.cos(yaw), np.sin(yaw)], self.goal_xy]).astype(np.float32)

    def _cube_xy(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_adrs: self.cube_qpos_adrs + 2].copy()

    def _ee_xy(self) -> np.ndarray:
        q = self.data.qpos
        return np.array([q[self.ee_x_adrs], q[self.ee_y_adrs]])

    def _reward(self):
        d = float(np.linalg.norm(self._cube_xy() - self.goal_xy))
        return -d, d < 0.15

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.ee_x_adrs] = -0.3
        self.data.qpos[self.ee_y_adrs] = -0.3
        if self.randomize:
            # random cube start
            self.data.qpos[self.cube_qpos_adrs] = self._rng.uniform(-0.3, 0.0)
            self.data.qpos[self.cube_qpos_adrs + 1] = self._rng.uniform(-0.3, 0.0)
            self.data.qpos[self.cube_qpos_adrs + 2] = 0.03
            # random goal
            self.goal_xy = np.array([self._rng.uniform(0.15, 0.4),
                                     self._rng.uniform(0.15, 0.4)])
            self.model.geom("goal").pos[0] = self.goal_xy[0]
            self.model.geom("goal").pos[1] = self.goal_xy[1]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        obs = {"image": self._render_image(), "state": self._get_state()}
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        # action = desired ee displacement (~3 cm per step)
        cur = self._ee_xy()
        target = cur + action * 0.03
        target = np.clip(target, -0.5, 0.5)
        self.data.ctrl[0] = target[0]
        self.data.ctrl[1] = target[1]
        for _ in range(10):  # 10 substeps @ 100Hz = 10Hz policy
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        reward, success = self._reward()
        terminated = success
        truncated = self._step >= self.max_steps
        obs = {"image": self._render_image(), "state": self._get_state()}
        info = {"success": success, "cube_xy": self._cube_xy(), "ee_xy": self._ee_xy()}
        return obs, reward, terminated, truncated, info
