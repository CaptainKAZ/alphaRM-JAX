# alphaRM 环境实现说明（v1.21）

> **移植说明**：本文来自 PyTorch/VMAS 仓库，用于保留规则背景和跨实现对照。
> 文中 BenchMARL、VMAS 文件路径与回调名称不属于当前 JAX 仓库入口。
> 当前 JAX 口径见 [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)，
> 数值冲突时以 `jaxmarl/environments/rm/config.py` 和 `env.py` 为准。

> 规则书：[`RULES.md`](RULES.md)（正文已同步到 v1.21）。
> 本文只记录**当前代码**的实现口径；数值以 `rm_config.py` 和 `rm_rules.py` 为准。
> 旧版文档中的 128 线 lidar、256 维观测、16×24 critic 地图、击杀 +20 等均为历史设计，不再适用。

---

## 1. 入口与文件

| 文件 | 作用 |
|---|---|
| `VectorizedMultiAgentSimulator/vmas/scenarios/rm_config.py` | 规则常量唯一来源（场地、动作、武器、热量、复活、碰撞、奖励、RSI） |
| `VectorizedMultiAgentSimulator/vmas/scenarios/rm_geometry.py` | 2D 几何：点/线段/多边形/圆、射线、可见弧采样 |
| `VectorizedMultiAgentSimulator/vmas/scenarios/rm_rules.py` | 规则引擎：动作、碰撞、视野/锁定、命中/热量、死亡/复活、占点/比分、奖励、观测、state |
| `VectorizedMultiAgentSimulator/vmas/scenarios/rm.py` | VMAS 包装：`process_action`、`post_step`、`observation`、`reward`、`done`、mask、render |
| `BenchMARL/train_rm.py` | MAPPO 训练入口 |
| `BenchMARL/benchmarl/models/rm_encoder.py` | actor（10-token Transformer + GRU） |
| `BenchMARL/benchmarl/models/rm_critic.py` | critic（191 维全局 state、队视角 canonical、共享 value head） |
| `.opencode/rm_selftest.py` | 规则自测（当前 **43/43**，不依赖 pytest） |

运行自测：

```bash
docker exec -w /home/vscode/workspace/VectorizedMultiAgentSimulator robocon2025-marl bash -lc \
  'OMP_NUM_THREADS=1 PYTHONPATH=/home/vscode/workspace/VectorizedMultiAgentSimulator \
   python /home/vscode/workspace/.opencode/rm_selftest.py'
```

---

## 2. 比赛与动作

- 场地 **12 m × 8 m**，`dt=0.1 s`，一局 **180 s（1800 帧）**；中心占点区半径 1.5 m。
- 每队 3 台同构机器人；红方在 `-x` 侧，蓝方在 `+x` 侧；目标分 90。
- 动作是 **6 维混合动作**：
  1. `vx`、`vy`：归一化底盘速度；
  2. `yaw`：归一化云台转向速率；
  3. `spin`：小陀螺目标强度；
  4. `target`：`0=不锁定 / 1..3=敌人 1..3`；
  5. `fire`：`0=松开 / 1=开火`。
- `target` 用 `action_mask`；`fire` 用独立的 `fire_mask`：没有有效目标时 fire 键被 mask，不能“对着空气开火”。
- 锁定目标还要求 `dist < lock_range=12 m`；远距离可以看见，但看不见/超出 12 m 时不能锁定。
- `target` 的可见性用 16 点粗采样；真正开火/伤害用 64 点可见弧。

---

## 3. 观测与网络

### Actor

- 输入 **140 维**，编码为 **10 个 token**：
  - self（自身 24 维 + 8 向障碍距离）
  - spawn / center
  - 2 个 mate
  - 3 个 enemy
  - HUD / cue
- 不读取 lidar，也不读取全局地图；8 向墙/掩体距离已经并入 self token。
- 不可见敌人和无效 cue 用 `src_key_padding_mask` 排除在 attention 与 pooling 之外。
- token 投影到 128 维，2 层 Transformer（4 heads，FFN 256），类型池化
  `self + center + mean(mates) + masked mean(enemies) + mean(world)`，Linear→14 参数；
  GRU 256 后接角色动作头，输出 14 个 raw 参数：
  - 4 个连续 `loc` + 4 个 `scale`（`vx/vy/yaw/spin`，SafeTanhNormal）；
  - target logits 4；
  - fire logits 2。

### Critic

- 输入 **191 维**全局 state：`6 × 30 + 11`（每 agent 22 标量 + 8 障碍距离，外加 11 维全局信息）。
- 红/蓝两个 canonical 视角（蓝方旋转 180°、own/opponent 交换）。
- 每个视角 7 个 token（global + 6 agents），2 层 Transformer，共享一个 MLP value head。
- 输出 `[B, 6, 1]` 的 6 个 value；不输出 advantage，也没有地图 CNN。

---

## 4. 视野与命中

- 普通敌人 FOV 90°；已锁定目标同样要求当前帧可见，离开 FOV/被完全遮挡立即脱锁。
- 可见弧：在目标圆**面向观察者的一侧**采样 64 点；`p_vis = 可见点数 / 64`。
  `p_vis > 0` 就能锁定，但命中率乘 `p_vis`；观测/锁定用 16 点，开火用 64 点。
- 命中率：

```text
p_dist   = max(0, 1 - ((d - 1) / 11)^2)     # 1 m 内 1，12 m 归零
p_self   = 1 - 0.25 * min(1, |v_self| / 3)
p_motion = 1 / (1 + (ω / 0.75)^2)
p_base   = p_dist * p_self * p_motion * p_vis
p_ideal  = min(p_base, 1 - 0.1 * s_self)
p_t      ← p_t + α * (p_ideal - p_t)
p_final  = p_t * (1 - 0.4 * s_target)
伤害/帧 = 30 * p_final
```

- 开火还需要：目标存活、非无敌、未过热、`|bearing| < 1°`、`fire=1`。
- 小陀螺满转需要 **1.0 s**（`spin_follow_rate=1.0`），不是旧文档的 0.5 s。

---

## 5. 碰撞、推挤与复活

- 碰撞判负只罚**实际肇事 agent**：基础 `-10`，再按相对法向速度加 `-3 × severity`；
  机器人碰撞的 severity 在相对速度 **2→6 m/s** 映射 0→1；对手不加分。
- 持续推墙/掩体 **1.0 s** 触发终局：主动方 `-10`，同队非主动方 `-0.5`，对手 `0`。
- 持续推人/尸体 **0.5 s** 触发终局（法向速度 > 0.3 m/s）：同样主动方 `-10`、
  队友 `-0.5`、对手 `0`；双方都推则平局，双方主动方各 `-10`。
- 复活：`respawn_time=5 s`，复活位置为尸体位置，`hp_revive=65`；
  复活后 **3 s 无敌**，无敌结束时枪口进入**过热门锁**（heat=heat_max、overheated=True），
  必须先冷却才能开火。
- 小陀螺在**实际接触**（机器人/尸体/墙/掩体）瞬间清零；0.65 m 贴人范围本身不清零。

---

## 6. 得分、终局与奖励

### 比赛得分

- 圈内按**净人数**计分：`net_occ = 红圈内人数 - 蓝圈内人数`；
  红方 `+0.1 × max(net_occ,0)/帧`，蓝方对称。
- 击杀 `+10`；先到 90 立即结束。
- 时间到：比分 → 累计伤害 → 存活总血量 → 平局。
- 10 s 无开火/伤害/实际回血/占点事件触发**空局**（outcome 23）。

### 终局训练奖励

- 正常赢：`6 + 3 × 剩余时间比例 + 3 × min(分差/90, 1)`，即约 **+6 ~ +12**；
- 正常输：固定 **-6**；
- 碰撞犯规：肇事 agent 基础 **-10**，再减烈度分；对手 0；
- 推墙/推人：主动方 **-10**，同队非主动方 **-0.5**，对手 **0**；
- 双方 200：双方 +2；双方犯规：双方 -10；
- 空局：基础 `-5`，再按障碍距离、离出生点距离、队伍离中心距离和移动速度折扣；
  移动最多减半，离出生点越远扣得越少。

### Dense 奖励（训练）

按 `_finalize_rewards()` 实现：占点、命中/伤害、中心接近与驻留、存活、低血回家、
击杀池（0.3，按累计伤害分配）、首次冲击/持续推挤、贴人、墙/掩体接触等。
具体系数见 `rm_config.py` 的 `k_*` 字段；不要在文档里凭旧表猜数值。

---

## 7. 观测窗与评测

```bash
# RM 训练观察窗（8768）
bash BenchMARL/liveview/start_rm_live.sh

# 浏览器人机/回放（8767）
bash BenchMARL/rm_game/start_game.sh
```

- `http://localhost:8768/`：逐帧回放、终局、训练状态；
- `http://localhost:8768/stats.html`：全 batch 终局占比、逐轨迹得分增量；
- `http://localhost:8768/files`：浏览 `BenchMARL/outputs/` 下的视频/日志；
- strict fixed eval 用 `RM_EVAL_INIT=fixed` + deterministic；需要看“无推挤终局”时
  才加 `--no-push-foul`，它只影响评测，不改训练规则。

---

## 8. 已知限制

- actor 不接收全局地图；泛化到多地图需要额外的局部地图输入/课程，不在当前版本。
- fixed eval 与训练分布（90% fixed + 10% late-game、jitter、stochastic）不同；
  单局 deterministic 结果不能代表训练分布。
- 训练主机内存是主要瓶颈；B=800×200 或 B=1000×180 比 B=1024×200 更安全。
- 规则书正文中的历史网络/奖励段落仅作沿革参考；当前实现以代码为准。
