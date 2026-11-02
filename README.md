# alphaRM-JAX

**alphaRM-JAX** 是 [JaxMARL](https://github.com/FLAIROx/JaxMARL) 的 alphaRM 专用 fork，面向 RoboMaster 3v3 简化规则的 JAX/Flax 多智能体强化学习实现与浏览器演示。

当前仓库包含完整的 JAX 环境、MAPPO 训练、冻结最优对手联赛、自博弈、跨代评估、训练观察窗，以及可直接双击打开的人机大战 / AI 对战单文件演示。

> 上游基线：`JaxMARL` commit `3ba06c2`。上游通用说明保留在 [README_UPSTREAM.md](README_UPSTREAM.md)。

## 当前状态

- 最新训练产物：`outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl`
- 当前联赛最优：`outputs/rm_jax_league900_1200/league_best.pkl`，iteration **1200**
- 最后训练配置：B=800、T=200、10 PPO epochs、bf16、`scan-all`、rollout lag=1
- 对手混合：75% 当前自博弈 + 25% 冻结最优代
- 辅助头：五类世界理解辅助预测，`aux_coef=0.02`
- 当前没有训练进程在运行

训练输出和 checkpoint 默认不进入 Git，使用 `.gitignore` 排除。

## 能做什么

| 模块 | 能力 |
|---|---|
| `jaxmarl/environments/rm` | JAX 版 RM 3v3 reset/step/observation/reward/mask/geometry |
| `baselines/MAPPO/train_rm_jax.py` | MAPPO 自博弈、冻结最优代联赛、辅助头、断点恢复 |
| `baselines/MAPPO/eval_rm_jax.py` | deterministic/stochastic 批量评测 |
| `liveview/` | 8768 训练观察窗、终局统计、得分曲线和输出文件浏览 |
| `human_jax_server.py` | JAX 后端的人机大战 HTTP 服务 |
| `demo/rm_single.html` | 完全本地、模型内嵌的人机大战 / AI 观战演示 |
| `demo/web/` | ONNX 导出、ONNX Runtime Web、Worker、单文件构建源码 |

## 环境与网络

### 场地与比赛

- 场地：12 m × 8 m，`dt=0.1 s`
- 一局：180 s = 1800 步
- 每队 3 台同构机器人，先到 90 分
- 中心占点区半径 1.5 m
- 红方在 `-x` 侧，蓝方在 `+x` 侧

### 动作

每台机器人 6 维混合动作：

```text
[vx, vy, yaw, spin, target, fire]
```

- `vx, vy`：归一化底盘运动
- `yaw`：云台转向
- `spin`：小陀螺
- `target`：`0=无目标 / 1..3=敌方槽位`
- `fire`：`0=释放 / 1=开火`

`target` 和 `fire` 分别受 action mask 和 fire mask 约束；没有合法目标时不能开火。

### 观测与网络

- Actor 局部观测：**140 维**
- Actor tokens：**11 个**
- Actor：2 层 Transformer（d=128，4 heads，FFN=256）+ GRU(256) + 2×256 动作头
- Critic 全局状态：**191 维**
- Critic：红蓝两个 canonical 视角，3 层 Transformer（d=192，4 heads，FFN=384），共享 value head，输出 6 个 value
- 无 lidar、无地图 CNN；局部障碍信息是 8 向射线距离

完整结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 安装

Python 3.11+。先根据硬件安装合适的 JAX，再安装本仓库：

```bash
pip install "jax[cuda13]"
# 或 CPU:
# pip install jax

pip install -e .[algs,dev]
```

验证环境：

```bash
python - <<'PY'
import jax
from jaxmarl import make

env = make("rm")
obs, state = env.reset(jax.random.PRNGKey(0), mode="fixed")
print(env.num_agents, obs[env.agents[0]].shape, state.pos.shape)
PY
```

### Docker

仓库提供 `Dockerfile.alphaRM`，不会安装 alphaRM 不需要的 Robotarium submodule：

```bash
docker build --build-arg UID="$(id -u)" \
  -t alphaRM-jax:latest \
  -f Dockerfile.alphaRM .

docker run --rm -it --gpus all --shm-size=20g \
  -v "$PWD":/workspace/alphaRM-JAX \
  alphaRM-jax:latest
```

需要 `perf` / `py-spy` 时给容器增加 `--privileged`。完整配方、CPU smoke、端口和 Dev Container 说明见
[docs/DOCKER.md](docs/DOCKER.md)。

## 快速开始

### 1. 单文件 AI 对战 / 人机大战

直接打开：

```text
demo/rm_single.html
```

单 HTML 默认进入 **观战挂机**：六台机器人全部由 AI 控制。点击“退出观战”后接管当前所选机器人。

也可以本地启动静态服务：

```bash
python3 -m http.server 8080 --directory demo
# http://localhost:8080/rm_single.html
```

详细操作、模型导出和单文件构建见 [demo/README.md](demo/README.md)。

GitHub Pages 发布由 `.github/workflows/pages.yml` 完成；首次在仓库 Settings 中选择
`Pages -> Source -> GitHub Actions`。部署说明见
[docs/GITHUB_PAGES.md](docs/GITHUB_PAGES.md)。

### 2. JAX 后端人机大战

```bash
PYTHONPATH=. python human_jax_server.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --port 8767
```

打开 `http://localhost:8767/`。

### 3. 训练观察窗

```bash
bash liveview/start_rm_live.sh
```

- 主观察窗：`http://localhost:8768/`
- 统计页：`http://localhost:8768/stats.html`
- 输出文件：`http://localhost:8768/files`

训练时需要写入 live 数据：

```bash
python baselines/MAPPO/train_rm_jax.py \
  --live-view \
  --live-dir "$PWD/outputs/live" \
  ...
```

### 4. 训练

最小 smoke 示例：

```bash
PYTHONPATH=. python baselines/MAPPO/train_rm_jax.py \
  --B 16 --T 16 --updates 1 --epochs 1 \
  --disable-jit --no-log
```

当前 1200 轮配方的完整参数、恢复命令和评估口径见
[docs/TRAINING_AND_EVALUATION.md](docs/TRAINING_AND_EVALUATION.md)。

### 5. 评测

```bash
PYTHONPATH=. python baselines/MAPPO/eval_rm_jax.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --episodes 8 --seed 0 --mode fixed \
  --out outputs/eval_1200
```

## 仓库结构

```text
alphaRM-JAX/
├── jaxmarl/environments/rm/       # RM 配置、环境、actor、critic、辅助头
├── baselines/MAPPO/               # JAX MAPPO、联赛、日志、评测
├── parity/                        # Torch/JAX 对拍、转换和性能分析
├── liveview/                      # 8768 训练观察窗
├── demo/
│   ├── rm_single.html             # 内嵌权重/WASM 的单文件演示
│   ├── human_game.html            # JAX HTTP 人机大战前端
│   └── web/                       # ONNX 导出、Worker、构建源码和资产
├── docs/                          # 架构、规则、训练、演示与上游说明
├── human_jax_server.py            # JAX 人机大战服务
├── CONTRIBUTING.md                # 提交规范、测试和评审要求
└── README_UPSTREAM.md             # 上游 JaxMARL README
```

## 当前验证边界

- 固定 reset、普通移动、dense reward、global state、actor/critic 已做 Torch/JAX 对拍。
- 碰撞/推挤后的物理残差、random reset 的 RNG 序列和观测噪声仍可能存在统计或样本级差异。
- 最新训练关闭了推人终局（`--no-body-push-terminal`），保留 dense 推人惩罚；评测时必须写清是否关闭推人/推墙终局。
- `league_eval.json` 中的红蓝胜率仍有明显不对称，不能只用总胜率推断真实强度。
- 当前 actor 不接收全局地图；多地图泛化不在已实现范围内。

详细实现与历史差异见：

- [docs/RULES.md](docs/RULES.md)：权威简化规则
- [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)：JAX 实现口径与已知差异
- [docs/TORCH_IMPLEMENTATION_NOTES.md](docs/TORCH_IMPLEMENTATION_NOTES.md)：从 PyTorch 仓库带入的参考说明

## 测试

```bash
python -m compileall jaxmarl/environments/rm baselines/MAPPO liveview human_jax_server.py
python parity/test_ppo_math.py
# 安装 dev 依赖后也可以使用 pytest
python -m pytest parity/test_ppo_math.py -q
```

提交前要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，提交信息采用 Conventional Commits。

## 上游与许可

本项目基于 Apache-2.0 许可的 JaxMARL。上游归属、引用信息和 fork 基线见
[docs/UPSTREAM.md](docs/UPSTREAM.md)，许可证保留在 [LICENSE](LICENSE)。

如果使用 JaxMARL 的环境或基础设施，请同时引用上游 JaxMARL 论文，引用格式见
[README_UPSTREAM.md](README_UPSTREAM.md)。
