# alphaRM-JAX

> **alphaRM-JAX** 是一个面向 **RoboMaster 3v3 简化规则**的 JAX/Flax 多智能体强化学习项目。
> 6 台同构机器人共享一套 actor，通过 MAPPO、自博弈和冻结最优代联赛学习移动、占点、
> 锁定、开火、热量管理、规避和团队配合。
>
> 本仓库 fork 自 [JaxMARL](https://github.com/bold-lab-ai/JaxMARL)，保留完整的 JAX 环境、
> 训练、评测、观察窗，以及可直接运行的 **人机大战 / AI 对战**浏览器演示。

- **在线演示：** https://captainkaz.github.io/alphaRM-JAX/
- **规则书：** [`docs/RULES.md`](docs/RULES.md)
- **PyTorch/VMAS 参考实现：** [CaptainKAZ/alphaRM](https://github.com/CaptainKAZ/alphaRM)
- **提交规范：** [`CONTRIBUTING.md`](CONTRIBUTING.md)

[快速开始](#2-快速开始) · [环境与规则](#3-环境规则与命中模型) · [网络与训练](#4-网络训练与-aux-loss) · [评测](#5-评测与当前结果) · [仓库结构](#6-仓库结构)

---

## 1. 项目定位

### 1.1 任务

- **对阵**：红方 3 台 vs 蓝方 3 台，所有机器人同构。
- **单局**：180 s / 1800 步，`dt=0.1 s`。
- **目标**：先到 **90 分**；时间结束则依次比较比分、总伤害、存活总 HP。
- **场地**：12 m × 8 m，含 L 形掩体、短掩体、中心占点区和双方启动区。
- **信息**：140 维局部观测；有 FOV、遮挡、锁定距离和观测噪声。
- **控制**：每台机器人 6 维混合动作；6 台机器人共享 actor。

### 1.2 一句话架构

```text
JAX RM 3v3 环境
  ├─ 规则 / 物理 / FOV / 占点 / 伤害 / 终局
  ├─ actor obs 140
  └─ critic global state 191
          │
          ▼
单组共享 Actor
  11 tokens → Transformer ×2 → GRU 256 → 14 个策略参数
          │
          ▼
单组 Team-Canonical Critic
  red/blue views → Transformer ×3 → 6 个 state values
          │
          ▼
MAPPO + 自博弈 + 75/25 冻结最优代联赛 + 训练期 Aux Loss
          │
          ├─ 8768 训练观察窗
          ├─ JAX 人机大战
          └─ 单文件 AI 观战 / 人机大战
```

### 1.3 当前状态

| 项 | 当前值 |
|---|---|
| 最新 checkpoint | `outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl` |
| 联赛最优 | `league_best.pkl`，iteration **1200** |
| 训练规模 | `B=800`、`T=200`、10 PPO epochs |
| 训练加速 | bf16、`scan-all`、activation remat、rollout lag=1 |
| 对手混合 | 75% 当前自博弈 + 25% 冻结最优代 |
| Aux Loss | 五类世界理解头，`aux_coef=0.02` |
| 当前训练进程 | 无 |

checkpoint、训练视频和 live 数据默认不提交 Git。

### 1.4 浏览器演示

![alphaRM-JAX 浏览器演示](docs/assets/alphaRM_jax_demo.png)

- 在线：https://captainkaz.github.io/alphaRM-JAX/
- 本地：打开 [`docs/index.html`](docs/index.html)
- 默认六台 AI 自战；点击“退出观战”后接管当前机器人。
- 支持战争迷雾、敌人真值、FOV、暂停、重开和左右键同时射击。

### 1.5 主要特性

| 特性 | 说明 |
|---|---|
| 单组共享 Actor | 6 个 agent 使用同一套参数 |
| 实体 token | self / spawn / center / 2 mates / 3 enemies / HUD / cue / obstacle，共 11 个 |
| 时序策略 | 2 层 Transformer（d=128，4 heads）→ GRU 256 |
| 复合动作 | `vx, vy, yaw, spin, target, fire` |
| 双 mask | target mask 约束目标，fire mask 约束开火 |
| 集中式 Critic | 191 维全局状态，红蓝 canonical 双视角 |
| 联赛训练 | 当前自博弈 + 冻结最优代，降低自博弈退化风险 |
| Aux Loss | 物体位置、伤害、敌方热量、规则、自身模型 |
| 完整终局 | 到分、时间、伤害、血量、碰撞、推墙、推人、空局 |
| 开发工具 | 8768 观察窗、统计页、文件浏览、ONNX Web 演示 |

---

## 2. 快速开始

### 2.1 直接玩单文件演示

```text
docs/index.html
```

该文件已经内嵌 actor ONNX、环境 ONNX、ORT Web、WASM 和 UI，可直接用 `file://`
打开，也可以启动静态服务：

```bash
python3 -m http.server 8080
# http://localhost:8080/docs/
```

操作：

- 默认观战，六台全部由 AI 控制
- “退出观战”后控制当前视角机器人
- 鼠标移动控制朝向
- 右键锁定离鼠标方向最近的可见敌人；按住时可移动鼠标切换
- 左键对锁定目标开火
- `WASD` 移动，`Shift` 小陀螺，`R` 重开，`P` 暂停，`T` 显示 FOV，`H` 换视角

### 2.2 JAX 后端人机大战

```bash
PYTHONPATH=. python human_jax_server.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --mode fixed \
  --human 0 \
  --port 8767
```

打开 `http://localhost:8767/`。

### 2.3 训练观察窗

```bash
bash liveview/start_rm_live.sh
```

- 主观察窗：`http://localhost:8768/`
- 终局/得分统计：`http://localhost:8768/stats.html`
- 输出文件浏览：`http://localhost:8768/files`

训练时写入 live 数据：

```bash
python baselines/MAPPO/train_rm_jax.py \
  --live-view \
  --live-dir "$PWD/outputs/live" \
  ...
```

### 2.4 本地安装

Python 3.11+。先按硬件安装合适的 JAX：

```bash
pip install "jax[cuda13]"
# CPU:
# pip install jax
```

安装仓库：

```bash
pip install -e .[algs,dev]
```

验证：

```bash
python - <<'PY'
import jax
from jaxmarl import make

env = make("rm")
obs, state = env.reset(jax.random.PRNGKey(0), mode="fixed")
print(env.num_agents, obs[env.agents[0]].shape, state.pos.shape)
PY
```

### 2.5 Docker

```bash
docker build --build-arg UID="$(id -u)" \
  -t alphaRM-jax:latest \
  -f Dockerfile.alphaRM .

docker run --rm -it --gpus all --shm-size=20g \
  -v "$PWD":/workspace/alphaRM-JAX \
  alphaRM-jax:latest
```

需要 `perf` / `py-spy` 时添加 `--privileged`。完整配方见
[`docs/DOCKER.md`](docs/DOCKER.md)。

### 2.6 训练 smoke 与评测

最小训练 smoke：

```bash
PYTHONPATH=. python baselines/MAPPO/train_rm_jax.py \
  --B 16 --T 16 --updates 1 --epochs 1 \
  --disable-jit --no-log
```

评测：

```bash
PYTHONPATH=. python baselines/MAPPO/eval_rm_jax.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --episodes 8 --seed 0 --mode fixed \
  --out outputs/eval_1200
```

完整 1200 轮恢复配方和评估口径见
[`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md)。

---

## 3. 环境、规则与命中模型

### 3.1 规则入口

- 完整规则：[`docs/RULES.md`](docs/RULES.md)
- 当前 JAX 实现：[`jaxmarl/environments/rm/env.py`](jaxmarl/environments/rm/env.py)
- 当前参数：[`jaxmarl/environments/rm/config.py`](jaxmarl/environments/rm/config.py)

文档和代码冲突时，以当前 JAX 代码和配置为准。

### 3.2 观测与信息不对称

Actor 输入 **140 维**，拆成 **11 个 token**：

| token | 内容 |
|---|---|
| self | 自身位置、速度、朝向、HP、热量、存活、目标、8 向障碍距离 |
| spawn | 己方启动区信息 |
| center | 中心占点区信息 |
| 2 × mate | 队友位置、速度、朝向、HP、热量、存活、复活倒计时 |
| 3 × enemy | 可见性、相对/绝对位置、带噪 HP/朝向、开火、无敌、复活 |
| HUD | 比分、时间、占点、队伍存活和总 HP |
| hit cue | 受击方向提示，带角度噪声 |
| obstacle | 8 向墙/掩体距离，独立 token |

不可见敌人和无效 cue 会被 attention mask 排除。队友信息共享；敌人信息受 FOV、
遮挡和观测噪声限制。

Critic 额外使用 **191 维**全局状态：

```text
6 agents × 30 + global 11 = 191
```

其中每个 agent 是 22 个标量 + 8 向障碍距离。

### 3.3 动作与掩码

```text
[vx, vy, yaw, spin, target, fire]
```

- `vx, vy`：归一化底盘运动
- `yaw`：云台转向
- `spin`：小陀螺
- `target`：`0=none / 1..3=enemy slot`
- `fire`：`0=release / 1=press`

规则：

- `target=0` 始终合法
- 敌方槽位要求存活、可见且距离小于 12 m
- `fire=0` 始终合法
- 开火要求已有目标、射手存活、非无敌、未过热、热量未满且目标非无敌

### 3.4 计分与终局

- 中心占点按红蓝净人数计分，每名净优势每帧 `0.1` 分。
- 击杀 `+10` 分，先到 `90` 分结束。
- 时间到后依次比较比分、总伤害、存活总 HP。
- 10 s 无战斗/占点/回血事件触发空局。

| 终局码 | 含义 |
|---|---|
| 1 / 2 | 红/蓝先到目标分 |
| 3 / 4 | 红/蓝时间胜 |
| 5 / 6 | 红/蓝伤害 tiebreak |
| 7 / 8 | 红/蓝血量 tiebreak |
| 9 / 10 | 红/蓝因对方碰撞犯规获胜 |
| 20–22 | 时间 / 同帧 200 / 同帧犯规平局 |
| 23 | 10 s 无事件空局 |
| 24–26 | 推墙/掩体犯规：红胜 / 蓝胜 / 双方 |
| 27–29 | 推人犯规：红胜 / 蓝胜 / 双方 |

### 3.5 奖励概要

- 正常胜负包含基础胜负、剩余时间和分差项。
- 占点、命中/伤害、接近中心、存活和低血回家有 dense reward。
- 贴人、墙/掩体接触、持续推挤和高速冲击有连续惩罚。
- 最新 1200 轮关闭了**推人终局**，保留 dense 推人惩罚。
- 严格部署评测默认保留推墙/推人终局，因此训练目标和最终规则分布存在已知差异。

奖励系数以 `config.py` 中的 `k_*`、`terminal_*` 为准。

### 3.6 命中率模型

瞄准不等于必中。命中率由距离、自身速度、目标横穿、可见弧、锁定低通和目标小陀螺共同决定：

```text
p_base  = p_dist × p_self × p_motion × p_vis
p_ideal = min(p_base, 1 - 0.1 × s_self)
p_t     ← p_t + α × (p_ideal - p_t)
p_final = p_t × (1 - 0.4 × s_target)
damage/frame = 30 × p_final
```

其中：

```text
p_dist   = max(0, 1 - ((distance - 1) / 11)^2)
p_self   = 1 - 0.25 × min(1, |v_self| / 3)
p_motion = 1 / (1 + (ω / 0.75)^2)
```

![命中率流水线](docs/assets/rm_hitrate_pipeline.svg)

![距离因子](docs/assets/rm_hitrate_distance.svg)

![命中率各因子](docs/assets/rm_hitrate_factors.svg)

![p_base 热力图](docs/assets/rm_hitrate_base_heatmap.svg)

> 曲线假设 `p_self = p_vis = 1`。实际开火还要求角度误差小于 1°、目标可见、
> 距离小于 12 m、未过热且 fire 指令为 1。

### 3.7 可见弧

![可见弧示意](docs/assets/rm_visibility_arc.svg)

可见性不是只测目标圆心，而是在目标圆朝向观察者的一侧采样：

- 观测/锁定：16 个采样点
- 开火/伤害：64 个采样点
- 可见点比例即 `p_vis`
- `p_vis > 0` 即可锁定，但命中率会乘以 `p_vis`

目标只露出一部分时仍可攻击，只是命中概率更低。

---

## 4. 网络、训练与 Aux Loss

### 4.1 Actor 与 Critic

![alphaRM-JAX actor / critic 结构图](docs/assets/rm_jax_actor_critic_arch.svg)

> Aux Loss 是 Actor 训练分支，不接在 Critic 后面；图中辅助头框画在 Actor 面板内。

**Actor**

- 140 维观测 → 11 tokens
- token projection：d=128
- Transformer ×2：4 heads，FFN 256
- 语义池化后编码为 256 维
- GRU 256 保留时序状态
- `concat(encoded, hidden)` → 2×256 MLP → 14 个 raw 参数
- 输出 4 个连续 loc、4 个 scale、4 个 target logits、2 个 fire logits

**Critic**

- 191 维全局 state → global token + 6 agent tokens
- 红蓝两个 canonical 视角
- Transformer ×3：d=192，4 heads，FFN 384
- 共享 value head，输出 6 个 value

### 4.2 Aux Loss：训练期世界理解

辅助头不是部署输出，而是给 Actor 的 encoder/GRU 增加可验证的表示学习信号。

输入：

```text
[GRU hidden 256, encoded 256, executed action 6, local obs 140]
```

五组头：

| Head | 维度 | 学习内容 |
|---|---:|---|
| Tracking | 12 | 3 敌人的下一帧相对位移、不确定度、下一帧可见性 |
| Damage | 4 | 开火、条件命中率、伤害、击杀 |
| Enemy heat | 9 | 3 敌人的热量余量、一步热量变化、过热 |
| Rules | 7 | 占点、中心得分、击杀分、done |
| Self | 14 | 自身移动/转向、伤害、回血、冷却、过热、存活和无敌事件 |

位置预测的是**下一帧相对位移**，不是绝对坐标。被遮挡后的位置权重：

```text
age_weight = 0.5 × (1 + cos(pi × min(age / 15, 1)))
```

- 0 s：权重 1
- 0.75 s：权重 0.5
- 1.5 s：位置权重 0
- 1.5 s 后辅助头继续训练，只是不再追旧位置

总 loss：

```text
aux_total =
    1.00 × track
  + 1.00 × damage
  + 0.75 × heat
  + 0.75 × rules
  + 1.00 × self

actor_loss = PPO_loss + aux_coef × aux_total
```

最新训练使用 `aux_coef=0.02`，从 iteration 500 开始用 10 iter 线性升温。
执行动作使用 `stop_gradient`；league 冻结对手由 actor mask 排除。

特权状态只生成标签，不进入部署 Actor。辅助参数保存在 checkpoint 中，但不会进入
导出的 ONNX/Web Actor。

完整公式、梯度和指标见 [`docs/AUX_LOSS.md`](docs/AUX_LOSS.md)。

### 4.3 当前训练配方

| 参数 | 值 |
|---|---:|
| B / T | 800 / 200 |
| iterations | 900 → 1200 |
| epochs | 10 |
| minibatch envs | 16 |
| learning rate | `5e-5` |
| gamma / lambda | `0.999 / 0.95` |
| clip | `0.2` |
| bf16 / scan-all / remat | 开启 |
| rollout lag | 1 |
| fixed / random | 90% / 10% |
| league self-play | 75% 当前 + 25% frozen best |
| aux coef | 0.02 |

完整恢复命令见 [`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md)。

---

## 5. 评测与当前结果

最后联赛评估：

```text
outputs/rm_jax_league900_1200/league_eval.json
```

1200 对上一冻结最优：

| 指标 | 数值 |
|---|---:|
| 总胜率 | 0.5625 |
| 红方胜率 | 0.3125 |
| 蓝方胜率 | 0.8125 |
| 红方平均分差 | -18.76 |
| 蓝方平均分差 | +19.12 |
| 平均步数 | 802.2 |

对 base500：64/64 全胜，平均分差约 79.05。

**注意：** 1200 对上一代仍有明显红蓝不对称，不能只看总胜率。

辅助预测 1200 checkpoint 的代表指标：

| 指标 | 数值 |
|---|---:|
| 可见敌人位置 MAE | 0.0269 m |
| 隐藏敌人位置 MAE | 0.3344 m |
| 可见性准确率 | 0.9706 |
| 开火准确率 | 0.9985 |
| 条件命中概率 MAE | 0.0295 |
| 过热总体准确率 | 0.9580 |
| 过热正例召回率 | 0.4933 |
| 自身资源 MAE | 0.0335 |

---

## 6. 仓库结构

```text
alphaRM-JAX/
├── jaxmarl/environments/rm/       # config / env / actor / critic / auxiliary
├── baselines/MAPPO/               # MAPPO / league / logger / eval
├── parity/                        # Torch-JAX 对拍、转换和性能分析
├── liveview/                      # 8768 训练观察窗
├── human_jax_server.py            # JAX 后端人机大战
├── demo/
│   ├── human_game.html            # HTTP 人机大战前端
│   └── web/                       # ONNX 导出、Worker 和单文件构建
├── docs/
│   ├── index.html                 # GitHub Pages 单文件演示
│   ├── ARCHITECTURE.md
│   ├── AUX_LOSS.md
│   ├── TRAINING_AND_EVALUATION.md
│   ├── RULES.md
│   └── assets/
├── Dockerfile.alphaRM
├── CONTRIBUTING.md
└── README_UPSTREAM.md
```

---

## 7. 开发、测试与已知限制

### 7.1 快速检查

```bash
python -m compileall jaxmarl/environments/rm baselines/MAPPO liveview human_jax_server.py
python parity/test_ppo_math.py
```

安装 dev 依赖后也可以使用：

```bash
python -m pytest parity/test_ppo_math.py -q
```

提交信息采用 Conventional Commits，详细规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

### 7.2 已知限制

- 固定 reset、普通移动、dense reward、global state、actor/critic 已做 Torch/JAX 对拍。
- 碰撞/推挤后的物理残差、random reset 的 RNG 和观测噪声仍可能有差异。
- 最新训练关闭推人终局，但保留 dense 推人惩罚；严格规则评测必须写清设置。
- 1200 联赛评估仍有红蓝胜率不对称。
- Actor 不接收全局地图；多地图泛化尚未实现。

---

## 8. 文档、上游与许可

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 代码分层、环境、网络和 Web 运行时 |
| [`docs/AUX_LOSS.md`](docs/AUX_LOSS.md) | 五类辅助目标、渐消位置 loss、权重和指标 |
| [`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md) | 1200 轮配方、恢复、评测 |
| [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) | JAX 实现口径和 parity 状态 |
| [`docs/RULES.md`](docs/RULES.md) | 完整规则书 |
| [`docs/DOCKER.md`](docs/DOCKER.md) | GPU/CPU、privileged 和 Dev Container |
| [`docs/GITHUB_PAGES.md`](docs/GITHUB_PAGES.md) | Pages 和单文件发布 |
| [`demo/README.md`](demo/README.md) | 人机大战、AI 观战、ONNX 转换与构建 |
| [`liveview/README.md`](liveview/README.md) | 训练观察窗 |

本项目基于 Apache-2.0 许可的 JaxMARL。fork 关系和上游引用见
[`docs/UPSTREAM.md`](docs/UPSTREAM.md)，许可证保留在 [`LICENSE`](LICENSE)。
