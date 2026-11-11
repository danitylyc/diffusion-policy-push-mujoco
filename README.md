# Diffusion Policy on MuJoCo Push

复现论文 **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
(Zhao, Radosavovic, Kumar, Levine, Malik, RSS 2023)，
在自写的 MuJoCo 平面推方块任务上完整跑通：数据采集 → 扩散策略训练 → 部署 rollout → 视频可视化。

> 论文链接: https://arxiv.org/abs/2303.04137
> 官方代码: https://github.com/real-stanford/diffusion_policy

---

## 1. 论文做了什么

传统视觉运动策略（BC-RNN、ACT、IBC）把策略直接回归到动作或用分类方式建模。
Diffusion Policy 换了一个思路：

- 把策略 `π(a | o)` 建模为 **条件去噪扩散过程**
- 训练时：对一段动作序列 `a_{t:t+H}` 加高斯噪声，让 U-Net 预测噪声（DDPM）
- 推理时：从纯噪声出发，以观测为条件，用 DDIM 迭代去噪出 action chunk
- 优势：天然建模多模态动作分布（同一个观测可能对应多种合理动作），
  不需要离散化动作空间，比 BC-RNN 平均成功率高 46.9%

核心组件：
- **1D U-Net** over action sequence（不是 2D 图像 U-Net）
- **双相机视觉 encoder**（top + wrist）
- **动作分块（action chunking）** + receding horizon 执行

## 2. 本仓库复现了什么

| 论文组件 | 本仓库实现 | 文件 |
|---|---|---|
| 1D U-Net over action chunk | horizon=8, action_dim=2, hidden=32 | `policy.py` UNet1d |
| DDPM 训练（50 步）/ DDIM 推理（20 步） | MSE noise prediction | `policy.py` loss / predict |
| 视觉 encoder | 4 层 CNN, 48×48 RGB → 64 维 | `policy.py` ImageEncoder |
| 状态 encoder | MLP 8→32 维 | `policy.py` state_enc |
| BC baseline | 4 层 MLP, obs→action 直接回归 | `bc.py` |
| MuJoCo 仿真环境 | 平面点末端执行器推圆盘到目标圈 | `env.py` |
| Scripted demonstration 采集 | PD 两阶段策略（approach → push） | `collect_data.py` |
| 部署 + 可视化 | 实时 rollout, OpenCV 写 mp4 | `evaluate.py`, `_record_demo.py` |

## 3. 实验结果

| 方法 | 训练 loss | 评估成功率 | 成功步数 |
|---|---|---|---|
| Scripted 策略 | — | 87% (60 episodes) | 10-20 |
| BC (numpy MLP) | 0.64 → 0.044 | **40%** (20 episodes) | 14-23 |
| Diffusion Policy (PyTorch) | 0.68 → 0.096 | **20%** (10 episodes) | 20-40 |

视频产物：
- `videos/demo_success.mp4` — BC 成功 rollout（14 帧）
- `videos/eval.mp4` — Diffusion Policy rollout（79 帧，含成功和失败 case）

## 4. 自己的思考和工程发现

这部分是复现过程中踩过的坑和得到的 insight，论文里不会写：

### 4.1 Windows + PyTorch CPU 的致命坑：MKL 线程卡死

本机 Python 3.13 + torch 2.15 CPU，首次 `import torch` 卡死超过 3 分钟。
根因不是 torch 本身慢，而是 **MKL 默认按 CPU 核数开线程做初始化**，
在 Windows 上触发线程争用。解决：

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
```

设完之后 import 从 3+ 分钟降到 **7.5 秒**。这个坑差点让整个 PyTorch 版 DP 放弃。

### 4.2 UNet1d 的 condition 维度 bug

论文图里画的是 "condition 注入到 U-Net 的每一层"，但具体怎么注入容易写错。
第一版我直接 `x + c.unsqueeze(-1)`，结果：
- `x` 形状 `(B, action_dim=2, H=8)`
- `c` 形状 `(B, 32)`
- 广播报错

正确做法：在进 U-Net 的 down path 之前，先用 `Conv1d(action_dim, hidden, 1)`
把动作投影到 hidden 维，再加 condition。这个 bug 导致训练 loss 一直降不下去。

### 4.3 DP 推理输出必须 clip

DDIM 从噪声迭代出来的动作范围在 `[-1.5, 2.1]`，超出环境动作空间 `[-1, 1]`。
如果不 clip，环境会硬截断，导致动作方向紊乱、ee 在起始位置抖动、cube 完全不动。
在 `predict()` 返回前 `x.clamp(-1, 1)` 后，成功率从 0% 升到 20%。

### 4.4 DDIM 推理步数不是越少越好

论文默认 DDIM 5 步推理。在我们小数据 + 小模型上，5 步去噪严重不足，
动作方向每步乱跳。增到 20 步后，动作序列平滑多了。
论文用 5 步是因为他们模型大、数据多、收敛好；小模型需要更多推理步。

### 4.5 BC 反而比 DP 好（在我们的设置下）

我们的 DP 20% < BC 40%，这和论文结论相反。原因：
- 论文用 GPU 训了几百 epochs + 上万 demonstrations
- 我们 CPU 训 100 epochs + 801 samples（60 episodes）
- 扩散模型数据效率比 BC 低，小数据下 BC 直接回归反而更稳
- 这本身就是一个有价值的发现：**DP 的优势在多模态、长 horizon 任务上才体现，
  简单单模态推方块任务上 BC 足够**

### 4.6 仿真环境的物理调参

推方块任务的物理参数调了很多轮：
- cube 从 box 改成 cylinder（box 会被推翻）
- 质量从 0.1 降到 0.03（太重推不动）
- friction 从 1.0 降到 0.6（太涩）
- 成功阈值从 0.06 放宽到 0.15（点末端执行器精度有限）
- PD 控制 kp 从 200 提到 500（响应更快）

这些细节论文里一句 "we use MuJoCo" 就带过了，实际工程调参占了一半时间。

## 4.5 逐行复现教程

完整的逐文件、逐行代码精讲（含扩散原理、反向传播推导、踩坑修复）见
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)。

## 5. 环境安装

```powershell
pip install -r requirements.txt
```

依赖：`numpy`, `mujoco`, `gymnasium`, `opencv-python`, `tqdm`, `torch`

> **Windows 用户必读**：训练前设环境变量，否则 torch import 卡死：
> ```powershell
> $env:OMP_NUM_THREADS="4"
> $env:MKL_NUM_THREADS="4"
> ```

## 6. 运行步骤

```powershell
# 1. 采集 60 条 scripted demonstrations
python collect_data.py --episodes 60 --out data/push_demos.npz

# 2. 训练 BC baseline（CPU, ~10 秒）
python train_bc.py --epochs 500 --lr 1e-3

# 3. 评估 BC
python _quick_eval.py

# 4. 录 BC 成功 episode 视频
python _record_demo.py

# 5. 训练 Diffusion Policy（CPU, ~30 分钟）
python train.py --epochs 100 --batch-size 32 --horizon 8

# 6. 评估 DP 并录视频（CPU, ~20 分钟）
python evaluate.py --episodes 10
```

## 7. 文件结构

```
diffusion_policy_push/
├── env.py                  # MuJoCo 推方块环境（MJCF 内联 XML）
├── collect_data.py         # scripted demonstration 采集器
├── bc.py                   # numpy MLP BC policy
├── train_bc.py             # BC 训练脚本
├── evaluate_bc.py          # BC 评估
├── _quick_eval.py          # BC 无头评估
├── _record_demo.py         # 录 BC 成功视频
├── policy.py               # PyTorch Diffusion Policy（UNet1d + DDPM + CNN）
├── train.py                # DP 训练脚本
├── evaluate.py             # DP 评估 + 录视频
├── data/
│   └── push_demos.npz      # 60 episodes demonstration 数据
├── checkpoints/
│   ├── bc_push.pkl         # BC 权重 (40% 成功率)
│   └── dp_push.pt          # DP 权重 (20% 成功率)
├── videos/
│   ├── demo_success.mp4   # BC 成功 rollout
│   └── eval.mp4            # DP rollout
├── requirements.txt
└── README.md
```

## 8. 局限和后续方向

- **数据量太小**：60 episodes / 801 samples 对扩散模型不够，论文用 50-100 demos
  但在更复杂任务上。后续可以加 demonstrations 到 200+
- **CPU 训练**：100 epochs 花了 30 分钟，GPU 上可以轻松训 500+ epochs
- **2D 平面简化版**：不是 6-DoF 机械臂，复现完整论文需要接 robosuite / LIBERO
- **视觉信息冗余**：48×48 简单平面场景里，state 已含全部信息，CNN 可能引入噪声；
  后续可以试 state-only 版本对比
- **action chunk 执行**：当前每步重新 predict + 取 chunk[0]（receding horizon），
  论文还支持 n_action_steps < horizon 的滑动窗口执行，可作为改进

## 9. 引用

```bibtex
@inproceedings{zhao2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Zhao, Chengchi and Radosavovic, Isadora and Kumar, Vikash and Levine, Sergey and Malik, Jitendra},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```
