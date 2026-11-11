# 手把手复现 Diffusion Policy（MuJoCo Push）

> 目标：从零理解并跑通「数据采集 → BC baseline → 扩散策略训练 → 部署 rollout」全流程。
> 本文按代码执行顺序逐文件、逐行讲解，每段代码先讲「这行在干什么」，再讲「为什么这么写」。

---

## 0. 先看全景：一个具身智能项目的数据流

```
                ┌──────────────────────────────────────────────┐
                │              MuJoCo 仿真环境 env.py            │
                │   观测 obs = {image(48×48×3), state(8,)}      │
                │   动作 action = (dx, dy) ∈ [-1,1]²            │
                └───────────────┬──────────────────────────────┘
                                │ 脚本策略 collect_data.py
                                ▼
                    data/push_demos.npz （60 episodes）
                    images / states / actions / ep_ends
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                   ▼
   BC baseline: bc.py + train_bc.py     Diffusion Policy: policy.py + train.py
   状态 → 动作 直接 MLP 回归            观测 → 条件 → 去噪出动作序列
              │                                   │
              ▼                                   ▼
   checkpoints/bc_push.pkl              checkpoints/dp_push.pt
   40% 成功率                           20% 成功率
              └─────────────────┬─────────────────┘
                                ▼
                    evaluate.py / evaluate_bc.py
                    MuJoCo 实时 rollout + 录 mp4
```

**两条路线对照理解**：
- **BC（行为克隆）**：给定状态，直接回归出一个动作 `a = f(state)`。简单，但一个观测只能对应一个动作，遇到多模态分布就崩。
- **Diffusion Policy（扩散策略）**：把动作序列看作要「生成」的数据，从噪声出发，以观测为条件一步步去噪，生成一整段动作 chunk。能建模「同一个观测下多种合理动作」的多模态分布。

---

## 1. 前置知识：DDPM 扩散到底在做什么

### 1.1 正向加噪（训练时）

给定一段真实动作序列 `x0`（形状 `(H=8, A=2)`），我们逐步往里面加高斯噪声，直到第 `T=50` 步变成纯噪声：

```
x_t = √(ᾱ_t) · x0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)
```

- `ᾱ_t` 是逐步累积的「保留信号比例」，从接近 1 单调降到 0。
- 这个式子可以**直接跳到任意 t**，不用一步步加噪——这是 DDPM 训练高效的关键。

### 1.2 反向去噪（推理时）

训练一个神经网络 `ε_θ(x_t, t, condition)` 去**预测当前噪声 ε**。
推理时从纯噪声 `x_T ~ N(0,I)` 出发，反复让网络预测噪声、把噪声减掉，一步步回到干净动作：

```
预测干净动作:  x0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
再往回一步:   x_{t-1} = √ᾱ_{t-1}·x0 + √(1-ᾱ_{t-1})·ε_θ
```

### 1.3 条件从哪来

网络不是无条件去噪：图像 CNN 编码 + 状态 MLP 编码拼成一个条件向量 `c`，
在每一步注入 U-Net，让去噪「看当前场景」生成动作。

```
image ──► CNN ──┐
                ├─► concat ──► MLP ──► condition c ──► 注入 UNet 每一层
state ──► MLP ──┘
```

> 论文里的 1D U-Net 处理的是**动作序列这一维度**，不是图像通道；图像只在最开头被压成向量当条件。

---

## 2. 环境配置

### 2.1 安装依赖

```powershell
pip install numpy mujoco gymnasium opencv-python tqdm torch
```

### 2.2 Windows 必踩坑：限制 MKL/OMP 线程

本机 Python 3.13 + torch 2.15 CPU，**不设线程数时 `import torch` 会卡死 3 分钟以上**。
根因是 MKL 默认按 CPU 核数开线程初始化，在 Windows 上触发争用。每次跑 torch 命令前先设：

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
```

设完后 import torch 约 7.5 秒。

### 2.3 目录约定

```
diffusion_policy_push/
├── data/          # 采集的 demonstrations
├── checkpoints/   # 训练权重
├── videos/        # rollout 视频
└── *.py           # 源码
```

---

## 3. `env.py`：MuJoCo 推方块环境

### 3.1 文件头与观测/动作约定（1–21 行）

```python
from __future__ import annotations   # 类型注解支持前向引用
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
```

- 环境遵循 **Gymnasium API**：`reset()` 返回 `(obs, info)`，`step(action)` 返回 `(obs, reward, terminated, truncated, info)`。
- 观测是字典：`image`（俯视 RGB）+ `state`（8 维本体状态）。
- 动作是二维位移 `(dx, dy) ∈ [-1,1]²`，环境内部再缩放到每步 3cm。

### 3.2 MJCF 模型 XML（23–53 行）

```xml
<mujoco model="push">
  <compiler angle="radian" autolimits="true"/>   <!-- 角度用弧度，关节范围自动生效 -->
  <option timestep="0.01"/>                       <!-- 物理步长 10ms -->
  <default>
    <geom condim="4" friction="0.6 0.1 0.01"/>    <!-- 接触摩擦系数 -->
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="table" type="plane" size="0.6 0.6 0.02" rgba="0.3 0.3 0.35 1"/>
```

- 地面平面尺寸 `0.6×0.6`，灰色。
- 目标圈是一个绿色薄圆柱（可视化标记，不参与碰撞）：

```xml
<geom name="goal" type="cylinder" size="0.08 0.005" pos="0.3 0.3 0.005" rgba="0.2 0.9 0.3 0.6"/>
```

- 末端执行器 EE：挂在两个平移关节 `ee_x / ee_y` 上的小红圆柱：

```xml
<body name="ee" pos="0 0 0.03">
  <joint name="ee_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
  <joint name="ee_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>
  <geom name="ee_g" type="cylinder" size="0.03 0.03" rgba="0.9 0.3 0.3 1"/>
</body>
```

- 被推的方块：用**扁平圆柱**而不是 box（box 一推就翻倒），带 `freejoint` 可自由平移旋转：

```xml
<body name="cube" pos="-0.2 -0.2 0.025">
  <freejoint name="cube_joint"/>
  <geom name="cube_g" type="cylinder" size="0.04 0.025" rgba="0.2 0.5 0.9 1" mass="0.03"/>
</body>
```

- 俯视相机：

```xml
<camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0" fovy="40"/>
```

- 执行器：两个 position 控制器，`kp=500` 位置刚度、`damping=20` 阻尼：

```xml
<actuator>
  <position name="act_x" joint="ee_x" kp="500" damping="20"/>
  <position name="act_y" joint="ee_y" kp="500" damping="20"/>
</actuator>
```

> 工程调参回顾：质量从 0.1 降到 0.03（太重推不动），摩擦 1.0→0.6（太涩），kp 200→500（响应更快）。

### 3.3 环境类初始化（56–80 行）

```python
class PushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, image_size=48, max_steps=150, randomize=True, seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML)   # 编译 XML 为模型
        self.data = mujoco.MjData(self.model)             # 仿真状态容器
```

- `MjModel` 是静态模型（几何、惯量、参数）；`MjData` 是每步变化的状态（qpos、ctrl 等）。

```python
        self.image_size = image_size
        self.max_steps = max_steps
        self.randomize = randomize
        self._rng = np.random.default_rng(seed)
        self._step = 0
```

- 记录步数上限、随机数生成器、当前步数。

```python
        self.ee_x_adrs = self.model.joint("ee_x").qposadr[0]
        self.ee_y_adrs = self.model.joint("ee_y").qposadr[0]
        self.cube_qpos_adrs = self.model.joint("cube_joint").qposadr[0]
```

- 提前查好各关节在 `qpos` 向量里的下标，避免每步反复查表。`freejoint` 的 qpos 占 7 维：`(x, y, z, quat_w, x, y, z)`。

```python
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._renderer = None
        self.goal_xy = np.array([0.3, 0.3])
```

- 声明观测/动作空间，供 Gymnasium 校验；renderer 懒加载；目标位置。

### 3.4 辅助方法（83–108 行）

**渲染图像**：

```python
def _render_image(self) -> np.ndarray:
    if self._renderer is None:
        self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
    self._renderer.update_scene(self.data, camera="top")
    return self._renderer.render()
```
- Renderer 懒创建；`update_scene` 刷新当前帧，`render()` 返回 `(H,W,3)` uint8。

**取状态向量**（8 维）：

```python
def _get_state(self) -> np.ndarray:
    q = self.data.qpos
    ee = np.array([q[self.ee_x_adrs], q[self.ee_y_adrs]])
    cube_xy = q[self.cube_qpos_adrs: self.cube_qpos_adrs + 2]
    quat = q[self.cube_qpos_adrs + 3: self.cube_qpos_adrs + 7]
    w, x, y, z = quat
    yaw = 2 * np.arctan2(z, w)
    return np.concatenate([ee, cube_xy, [np.cos(yaw), np.sin(yaw)], self.goal_xy]).astype(np.float32)
```

- 8 维 = `[ee_x, ee_y, cube_x, cube_y, cos(yaw), sin(yaw), goal_x, goal_y]`。
- 姿态用四元数 → yaw 角，再拆成 `(cos, sin)`。**用 sin/cos 而不是角度**，是为了避免 ±π 跳变造成的不连续。

**奖励与成功判定**：

```python
def _reward(self):
    d = float(np.linalg.norm(self._cube_xy() - self.goal_xy))
    return -d, d < 0.15
```
- 奖励 = 负距离；成功条件 = 方块距目标 < 0.15m。阈值是放宽过的（点末端精度有限）。

### 3.5 reset（111–130 行）

```python
def reset(self, *, seed=None, options=None):
    if seed is not None:
        self._rng = np.random.default_rng(seed)
    mujoco.mj_resetData(self.model, self.data)          # 清空所有 qpos/ctrl
    self.data.qpos[self.ee_x_adrs] = -0.3
    self.data.qpos[self.ee_y_adrs] = -0.3               # EE 固定在左下
```
- EE 固定起点 `(-0.3,-0.3)`。

```python
    if self.randomize:
        self.data.qpos[self.cube_qpos_adrs] = self._rng.uniform(-0.3, 0.0)
        self.data.qpos[self.cube_qpos_adrs + 1] = self._rng.uniform(-0.3, 0.0)
        self.data.qpos[self.cube_qpos_adrs + 2] = 0.03   # 抬到桌面高度
        self.goal_xy = np.array([self._rng.uniform(0.15, 0.4),
                                 self._rng.uniform(0.15, 0.4)])
        self.model.geom("goal").pos[0] = self.goal_xy[0]
        self.model.geom("goal").pos[1] = self.goal_xy[1]
```
- 方块随机在左下区域，目标随机在右上区域，保证「推」这个动作有意义。
- 同步把绿色目标圆柱的位置改到新目标。

```python
    mujoco.mj_forward(self.model, self.data)   # 刷新一次碰撞/运动学
    self._step = 0
    obs = {"image": self._render_image(), "state": self._get_state()}
    return obs, {}
```

### 3.6 step（132–148 行）

```python
def step(self, action: np.ndarray):
    action = np.clip(action, -1.0, 1.0)        # 安全截断
    cur = self._ee_xy()
    target = cur + action * 0.03               # 动作缩放：每步最大 3cm
    target = np.clip(target, -0.5, 0.5)
    self.data.ctrl[0] = target[0]
    self.data.ctrl[1] = target[1]
```

- 动作是**相对位移**：当前位置 + `action×0.03` 作为 position 控制器目标。
- 控制器会自己把 EE 拉向 target（PD 闭环），我们不直接写位置。

```python
    for _ in range(10):                         # 10 个子步 @100Hz = 策略 10Hz
        mujoco.mj_step(self.model, self.data)
```

- 每个策略步内部跑 10 个物理子步，让 PD 控制器收敛、接触稳定。

```python
    self._step += 1
    reward, success = self._reward()
    terminated = success
    truncated = self._step >= self.max_steps
    obs = {"image": self._render_image(), "state": self._get_state()}
    info = {"success": success, "cube_xy": self._cube_xy(), "ee_xy": self._ee_xy()}
    return obs, reward, terminated, truncated, info
```

- `terminated` 是任务成功（提前结束）；`truncated` 是到步数上限（超时截断），二者语义不同。

---

## 4. `collect_data.py`：脚本 demonstration 采集

### 4.1 脚本策略的两阶段思路（17–49 行）

```python
class _ScriptedPolicy:
    def __init__(self):
        self.approach = None   # 预先算好的「绕到方块后方」的点
        self.phase = 0         # 0=绕后, 1=推
```

**为什么要两阶段？** 直接冲方块会从正面把它推跑、自己也追不上。正确推法：先绕到方块**背向目标的一侧**，再从背后朝目标推。

```python
def __call__(self, env: PushEnv) -> np.ndarray:
    cube = env._cube_xy()
    ee = env._ee_xy()
    goal = env.goal_xy
    d = goal - cube
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return np.zeros(2, dtype=np.float32)
    d_hat = d / dist                    # 方块→目标 的单位方向
```

- `d_hat` 是从方块指向目标的方向；方块背后（approach 点）就是方块减去这个方向一段距离。

```python
    rest = 0.07
    if self.phase == 0:
        if self.approach is None:
            self.approach = cube - d_hat * rest   # episode 开始时算一次，固定不变
        if np.linalg.norm(ee - self.approach) < 0.02:
            self.phase = 1                       # 到位，进入推阶段
        target = self.approach
    else:
        target = goal                            # 推阶段：直冲目标
```

> 关键细节：`approach` 在 episode **开始时算一次**，而不是每步跟着移动中的方块重算。
> 否则 approach 点会追着方块跑，形成死循环——这是早期脚本策略 0% 成功的原因。

```python
    delta = target - ee
    delta = np.clip(delta / 0.03, -1.0, 1.0)    # 位移换算回 [-1,1] 动作空间
    return delta.astype(np.float32)
```

- 目标位移 ÷ 0.03（每步最大距离），再 clip 到动作空间。

### 4.2 采集主循环（52–93 行）

```python
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
        images.append(obs["image"])     # 存当前观测
        states.append(obs["state"])
        actions.append(a)               # 存采取的动作
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        dones.append(term or trunc)
        ep_len += 1
        done = term or trunc
```

- 行为克隆的数据格式：**在状态 s 下采取动作 a**，即 `(s, a)` 配对。
- 注意存的是 `step` 之前的 `obs`，动作也是当时决策的动作，时序对齐正确。

```python
    successes += int(info["success"])
    ep_ends.append(len(actions))         # 记录每条 episode 在全局数组里的结束位置
```

- `ep_ends` 用来在切动作 chunk 时判断「不能跨 episode 取序列」。

```python
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
```

- 压缩存成单个 `.npz`；`ep_ends` 是每条 episode 结束的全局索引。

### 4.3 运行采集

```powershell
python collect_data.py --episodes 60 --out data/push_demos.npz
```
产出 60 episodes / 2061 steps，脚本成功率约 87%。

---

## 5. `bc.py`：numpy 手写 MLP 行为克隆（对照组）

这是论文里的 BC baseline。**不用 PyTorch**，纯 numpy 手写前向 + 反向，方便把梯度推导讲透。

### 5.1 网络结构与初始化（16–29 行）

```python
class MLPBC:
    def __init__(self, obs_dim=8, action_dim=2, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2.0 / obs_dim), (obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0, 0.01, (hidden, hidden))
        self.b3 = np.zeros(hidden)
        self.w4 = rng.normal(0, 0.01, (hidden, action_dim))
        self.b4 = np.zeros(action_dim)
```

- 4 层：`8 → 128 → 128 → 128 → 2`。
- 前两层用 **He 初始化** `√(2/风扇入层)` 配合 ReLU；后两层小初始化 `0.01`，让初始输出接近 0。

### 5.2 标准化与前向（31–41 行）

```python
def normalize(self, x):
    if self.obs_mean is None:
        return x
    return (x - self.obs_mean) / (self.obs_std + 1e-6)
```
- 观测按训练集均值/方差标准化，`1e-6` 防止除零。

```python
def forward(self, x):
    x = self.normalize(x)
    h = np.maximum(0.0, x @ self.w1 + self.b1)
    h = np.maximum(0.0, h @ self.w2 + self.b2)
    h = np.maximum(0.0, h @ self.w3 + self.b3)
    return h @ self.w4 + self.b4
```
- 矩阵乘法 `@`；`np.maximum(0,·)` 就是 ReLU。

### 5.3 反向传播（43–72 行，逐行梯度推导）

损失是 MSE：`L = mean((pred - y)²)`。

```python
def train_step(self, x, y, lr):
    x = self.normalize(x)
    # ---- 前向，缓存各层 z（激活前）和 h（激活后） ----
    z1 = x @ self.w1 + self.b1;  h1 = np.maximum(0.0, z1)
    z2 = h1 @ self.w2 + self.b2; h2 = np.maximum(0.0, z2)
    z3 = h2 @ self.w3 + self.b3; h3 = np.maximum(0.0, z3)
    pred = h3 @ self.w4 + self.b4
    diff = pred - y
    loss = float(np.mean(diff ** 2))
```

```python
    B = x.shape[0]
    # 输出层梯度：dL/dpred = 2/B/A · diff （A=动作维度）
    d4 = (2.0 / B / y.shape[1]) * diff
    gw4 = h3.T @ d4; gb4 = d4.sum(0)            # w4 梯度 = 上游 h3^T · d4
```

链式往下传：

```python
    dh3 = d4 @ self.w4.T                        # 传给 h3
    dz3 = dh3 * (z3 > 0)                        # ReLU 导数：z3>0 为 1，否则 0
    gw3 = h2.T @ dz3; gb3 = dz3.sum(0)

    dh2 = dz3 @ self.w3.T
    dz2 = dh2 * (z2 > 0)
    gw2 = h1.T @ dz2; gb2 = dz2.sum(0)

    dh1 = dz2 @ self.w2.T
    dz1 = dh1 * (z1 > 0)
    gw1 = x.T @ dz1; gb1 = dz1.sum(0)
```

- ReLU 导数就是「前向时激活是否 >0」的布尔掩码。
- 权重梯度 = 输入转置 × 上游梯度；偏置梯度 = 上游梯度按样本求和。

```python
    for w, g in [(self.w1, gw1), (self.b1, gb1), ...]:
        w -= lr * g                             # SGD 更新
    return loss
```

### 5.4 推理与保存（74–90 行）

```python
def predict(self, obs):
    a = self.forward(obs[None, :])[0]
    return np.clip(a, -1.0, 1.0)                 # 动作 clip 到合法范围
```
- `obs[None,:]` 加一个 batch 维；输出取第 0 个样本；clip 到 [-1,1]。

```python
def save(self, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(self.__dict__, f)

@classmethod
def load(cls, path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    p = cls(obs_dim=d["w1"].shape[0], action_dim=d["w3"].shape[1],
            hidden=d["w1"].shape[1])
    p.__dict__.update(d)
    return p
```
- pickle 存全部属性；load 时先按维度建空网络，再把权重字典灌进去。

---

## 6. `train_bc.py`：BC 训练脚本（9–40 行）

```python
d = np.load(args.data, allow_pickle=True)
states = d["states"].astype(np.float32)
actions = d["actions"].astype(np.float32)
model = MLPBC(obs_dim=states.shape[1], action_dim=actions.shape[1], hidden=128)
# 用全量训练集统计均值/方差做标准化
model.obs_mean = states.mean(axis=0)
model.obs_std = states.std(axis=0)
```

```python
N = states.shape[0]
rng = np.random.default_rng(0)
for ep in range(args.epochs):
    idx = rng.permutation(N)                    # 每轮打乱
    total = 0.0; nb = 0
    for s in range(0, N - args.batch_size, args.batch_size):
        bi = idx[s: s + args.batch_size]
        loss = model.train_step(states[bi], actions[bi], args.lr)
        total += loss; nb += 1
```
- 标准 minibatch SGD：每 epoch 打乱索引，按 batch 切。

```python
model.save(args.out)
```

运行：

```powershell
python train_bc.py --epochs 500 --lr 1e-3
```
loss 从 0.64 降到 0.044，评估 40% 成功率。

---

## 7. `policy.py`：核心 Diffusion Policy（重点）

### 7.1 图像编码器 ImageEncoder（19–31 行）

```python
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),   # 48→24
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 24→12
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 12→6
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                  # 6×6→1 向量
            nn.Linear(64, out_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)     # x: (B,3,H,W) ∈ [0,1]
```

- 三层 stride-2 卷积把 48×48 压到 6×6，再全局平均池化成 64 维向量。
- 输入要归一化到 `[0,1]`（在 Dataset 里 `/255`）。

### 7.2 1D U-Net UNet1d（34–61 行）

这是论文的核心结构。**处理对象是动作序列**：长度 `H=8`，每步 2 维动作。

```python
class UNet1d(nn.Module):
    def __init__(self, action_dim=2, horizon=8, cond_dim=128, hidden=32):
        super().__init__()
        # 时间步嵌入：标量 t → hidden 维向量
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, hidden))
        # 条件 + 时间嵌入 → hidden 维
        self.cond_proj = nn.Linear(cond_dim + hidden, hidden)
        # 先把动作通道 2 投影到 hidden 32（关键修复，见下）
        self.in_proj = nn.Conv1d(action_dim, hidden, 1)
        self.down1 = self._block(hidden, hidden)
        self.down2 = self._block(hidden, hidden * 2)
        self.up2 = self._block(hidden * 2 + hidden, hidden)
        self.out = nn.Conv1d(hidden, action_dim, 1)
```

```python
    def _block(self, cin, cout):
        return nn.Sequential(
            nn.Conv1d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(),
        )
```
- 每个 block 两层 `1D 卷积(3×3)` + GroupNorm + ReLU。

**前向传播**：

```python
def forward(self, noisy_actions, t, cond):
    # noisy_actions: (B, H, A)
    x = noisy_actions.transpose(1, 2)        # (B, A, H)  把序列长度放最后，符合 Conv1d 约定
    x = self.in_proj(x)                      # (B, hidden, H)
    temb = self.time_mlp(t.view(-1, 1))      # (B, hidden) 时间嵌入
    c = self.cond_proj(torch.cat([cond, temb], dim=-1))  # 条件+时间 → (B, hidden)
    x = x + c.unsqueeze(-1)                  # 广播加到每个时间步
```

> **踩坑修复（重要）**：最初版本直接 `x + c.unsqueeze(-1)`，但 `x` 通道是动作维 2，`c` 是 32 维，广播报错。
> 正确做法是先用 `in_proj` 把动作投影到 hidden=32，再加条件向量。

```python
    h1 = self.down1(x)                                   # 编码器第一层
    h2 = self.down2(F.avg_pool1d(h1, 2))                 # 下采样到 H/2
    u2 = self.up2(torch.cat([
        F.interpolate(h2, scale_factor=2, mode="nearest"), # 上采样回 H
        h1                                                 # 跳连：拼上编码器特征
    ], dim=1))
    return self.out(u2).transpose(1, 2)                  # 回到 (B, H, A)
```
- 标准 U-Net 跳连结构：编码器低层特征直接接到解码器对应层。

### 7.3 DiffusionPolicy 主体（64–83 行）

```python
class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=2, horizon=8, n_train_steps=50):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_train_steps = n_train_steps
        self.img_enc = ImageEncoder(64)
        self.state_enc = nn.Sequential(nn.Linear(8, 32), nn.ReLU(),
                                       nn.Linear(32, 32), nn.ReLU())
        self.cond_net = nn.Sequential(nn.Linear(64 + 32, 128), nn.ReLU())
        self.unet = UNet1d(action_dim, horizon, 128)
```

- 条件拼接：图像 64 维 + 状态 32 维 → MLP 压成 128 维条件。

**扩散系数表**：

```python
    betas = torch.linspace(1e-4, 0.02, n_train_steps)   # 线性 beta 调度
    self.register_buffer("betas", betas)
    self.register_buffer("alphas", 1.0 - betas)
    self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))
```
- `ᾱ_t = ∏(1-β_s)`，累积保留率。`register_buffer` 让这些张量随模型保存但不参与梯度更新。

**条件编码**：

```python
def cond(self, image, state):
    zi = self.img_enc(image)
    zs = self.state_enc(state)
    return self.cond_net(torch.cat([zi, zs], dim=-1))
```

### 7.4 训练损失 loss（85–93 行）—— 对应 1.1 节公式

```python
def loss(self, image, state, action_chunk):
    c = self.cond(image, state)
    B = action_chunk.shape[0]
    t = torch.randint(0, self.n_train_steps, (B,), device=action_chunk.device)
    noise = torch.randn_like(action_chunk)                # 采样噪声 ε
    ab = self.alpha_bars[t].view(B, 1, 1)
    noisy = torch.sqrt(ab) * action_chunk + torch.sqrt(1 - ab) * noise  # 加噪
    pred = self.unet(noisy, t.float() / self.n_train_steps, c)         # 预测噪声
    return F.mse_loss(pred, noise)                        # MSE 回归噪声
```

逐行：
1. 随机采一个时间步 `t`（每个样本独立）。
2. 采标准高斯噪声 `ε`。
3. 按公式 `x_t = √ᾱ·x0 + √(1-ᾱ)·ε` 合成加噪动作。
4. U-Net 预测噪声 `ε_θ`，损失 = 预测噪声与真实噪声的 MSE。

> 训练目标不是「预测干净动作」，而是「预测被加进去的噪声」——这是 DDPM 的标准做法。

### 7.5 推理 predict（95–112 行）—— 对应 1.2 节去噪

```python
@torch.no_grad()
def predict(self, image, state, n_infer_steps=20):
    c = self.cond(image, state)
    B = c.shape[0]
    device = c.device
    x = torch.randn(B, self.horizon, self.action_dim, device=device)  # 纯噪声起步
    ts = torch.linspace(0, self.n_train_steps - 1, n_infer_steps).long()
```

- 从纯噪声 `x_T` 开始；`ts` 是推理用的时间步序列（DDIM 跳步）。

```python
    for i in reversed(range(n_infer_steps)):
        t = ts[i].expand(B).to(device)
        eps = self.unet(x, t.float() / self.n_train_steps, c)
        ab = self.alpha_bars[t].view(B, 1, 1)
        x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)   # 反推干净动作
        if i > 0:
            ab_prev = self.alpha_bars[ts[i - 1]].view(B, 1, 1)
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
        else:
            x = x0
    return x.clamp(-1.0, 1.0)
```

- 反向遍历时间步：预测噪声 → 反推 `x0` → 重建 `x_{t-1}`。
- 最后 `clamp(-1,1)` 把动作压回合法空间（**不加这步会超范围、方向紊乱**）。

> DDIM 推理步数：论文默认 5 步；我们小模型数据少，增到 20 步动作更平滑。

---

## 8. `train.py`：数据集与训练循环

### 8.1 PushDataset（11–35 行）

```python
class PushDataset(Dataset):
    def __init__(self, npz_path, horizon=8, stride=2):
        d = np.load(npz_path, allow_pickle=True)
        self.images = d["images"]
        self.states = d["states"].astype(np.float32)
        self.actions = d["actions"].astype(np.float32)
        self.ep_ends = list(d["ep_ends"])
        self.horizon = horizon
        self.starts = []
        prev = 0
        for end in self.ep_ends:
            for s in range(prev, end - horizon, stride):
                self.starts.append(s)      # 合法起点：后面凑得齐 horizon 步
            prev = end
```

- **切动作 chunk**：从起点 `s` 取连续 `horizon=8` 步动作。
- `stride=2` 隔两个样本取一个起点，增加样本多样性、避免相邻 chunk 高度重复。
- 严格按 `ep_ends` 分段，**绝不跨 episode 拼接**（否则 chunk 里混入下一条演示的动作）。

```python
def __getitem__(self, idx):
    s = self.starts[idx]
    h = self.horizon
    img = torch.from_numpy(self.images[s]).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(self.states[s])
    act = torch.from_numpy(self.actions[s: s + h])
    return img, state, act
```
- 图像 `(H,W,3) → (3,H,W)` 并归一化；状态取当前帧；动作取未来 8 步 chunk。

### 8.2 训练主循环（38–71 行）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = PushDataset(args.data, horizon=args.horizon)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

model = DiffusionPolicy(horizon=args.horizon).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

for ep in range(args.epochs):
    model.train()
    total = 0.0; n = 0
    for img, state, act in dl:
        img = img.to(device); state = state.to(device); act = act.to(device)
        loss = model.loss(img, state, act)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * img.size(0)
        n += img.size(0)
```

- 标准 PyTorch 训练三步：清零梯度 → 反向 → 更新。
- `weight_decay` 轻微正则防过拟合。

```python
torch.save({"model": model.state_dict(), "horizon": args.horizon}, args.out)
```
- 保存权重 + horizon 配置，推理时按 horizon 重建网络。

运行：

```powershell
$env:OMP_NUM_THREADS="4"; $env:MKL_NUM_THREADS="4"
python train.py --epochs 100 --batch-size 32 --horizon 8
```
loss 0.68 → 0.096。

---

## 9. `evaluate.py`：部署 rollout + 录视频

```python
ckpt = torch.load(args.ckpt, map_location=device)
model = DiffusionPolicy(horizon=ckpt["horizon"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
```
- 加载权重，切到 eval 模式（关闭 dropout 等）。

```python
env = PushEnv(randomize=True)
for ep in range(args.episodes):
    obs, _ = env.reset(seed=1000 + ep)
    done = False
    while not done:
        img = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        state = torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
        with torch.no_grad():
            chunk = model.predict(img, state, n_infer_steps=20)[0].cpu().numpy()
        act = chunk[0]                       # 只取 chunk 第一步动作（receding horizon）
        obs, r, term, trunc, info = env.step(act)
```

- **Receding horizon 执行**：每步重新预测整个 chunk，但只执行第 0 步；下一个观测再重新预测。
- `torch.no_grad()` 关梯度省显存/内存。

```python
        if ep == 0:                          # 只录第一条 episode
            frame = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame, f"d={np.linalg.norm(env._cube_xy()-env.goal_xy):.2f}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(frame)
    successes += int(info["success"])
```

- OpenCV 用 BGR，所以 RGB→BGR 转换；画面上叠加当前距离 `d`。

```python
if frames:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out_video, fourcc, 10.0, (256, 256))
    for f in frames: vw.write(f)
    vw.release()
```
- 10 FPS 写出 mp4。

运行：

```powershell
python evaluate.py --episodes 10
```

---

## 10. 完整复现命令清单

```powershell
# 0. 依赖
pip install numpy mujoco gymnasium opencv-python tqdm torch

# 1. 采集数据
python collect_data.py --episodes 60 --out data/push_demos.npz

# 2. 训练 BC baseline
python train_bc.py --epochs 500 --lr 1e-3

# 3. 评估 BC
python evaluate_bc.py --episodes 20

# 4. 训练 Diffusion Policy（Windows 先限线程）
$env:OMP_NUM_THREADS="4"; $env:MKL_NUM_THREADS="4"
python train.py --epochs 100 --batch-size 32 --horizon 8

# 5. 评估 DP + 录视频
python evaluate.py --episodes 10
```

## 11. 常见报错与排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `import torch` 卡死 3 分钟+ | MKL 多线程争用 | 设 `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4` |
| UNet 广播维度报错 | 动作维 2 ≠ 条件维 32 | 先 `in_proj` 投影到 hidden 再加条件 |
| 动作超范围、EE 抖动 | 推理输出未 clip | `predict()` 返回前 `clamp(-1,1)` |
| 动作方向每步乱跳 | DDIM 步数太少 | 推理步数 5 → 20 |
| chunk 跨 episode 串味 | 切序列没按 `ep_ends` 分段 | Dataset 按 episode 边界切起点 |
| 方块被推翻倒 | 用了 box 几何 | 改成扁平 cylinder |

## 12. 结果与对照

| 方法 | 训练 loss | 成功率 | 说明 |
|---|---|---|---|
| Scripted 策略 | — | 87% | 上帝视角 PD，作数据来源 |
| BC (numpy MLP) | 0.64→0.044 | 40% | 单步状态→动作回归 |
| Diffusion Policy | 0.68→0.096 | 20% | CPU 小数据训练；论文在 GPU 大数据下 DP 超过 BC |

> 为什么这里 DP 反而低于 BC？扩散模型数据效率低于直接回归，801 样本 + CPU 100 epochs 不足以发挥多模态建模优势；这正是一个值得记录的工程发现。
