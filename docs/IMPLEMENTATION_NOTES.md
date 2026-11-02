# JAX 实现说明

本文记录当前 JAX 代码口径，并区分“已对拍”和“仍有差异”的部分。

## 1. 真相来源

- 规则与数值：`jaxmarl/environments/rm/config.py`
- 环境逻辑：`jaxmarl/environments/rm/env.py`
- 保存的训练参数：run 目录下 `logs/hparams.json`
- 当前网络：`actor.py`、`critic.py`、`auxiliary.py`

当文档与代码冲突时以代码为准。完整规则背景见 [RULES.md](RULES.md)。

## 2. 动作与 mask

6 维动作：

```text
[vx, vy, yaw, spin, target, fire]
```

### Target mask

目标槽位只有在以下条件同时成立时合法：

- 敌方存活
- 普通 FOV / 几何可见性成立
- 距离小于 12 m

`target=0` 始终合法。死亡的机器人只能选择 `target=0`。

### Fire mask

`fire=0` 始终合法。`fire=1` 需要：

- 当前有锁定目标
- 射手存活
- 射手非无敌
- 未过热且热量未满
- 目标不在无敌状态

没有目标时不能“对空开火”。

## 3. 视野、锁定和命中

- 普通 FOV：90° 总角宽（±45°）
- 锁定目标后继帧：80° 总角宽（±40°）
- 最大锁定距离：12 m
- 观测/目标 mask 使用 16 个可见弧采样点
- 开火/命中率使用 64 个可见弧采样点
- 完全遮挡会脱锁
- 可见弧部分覆盖时，`p_vis` 作为命中率乘子

命中率的完整公式在规则书中；实现入口是 `RmEnv` 内的命中与伤害计算。

## 4. 热量与开火

| 参数 | 当前值 |
|---|---:|
| 热量上限 | 160 |
| 开火增量 | 15 / 帧 |
| 常规冷却 | 2 / 帧 |
| 回家冷却倍率 | 2 |
| 过热锁 | 是 |

实际开火受到 fire mask、目标合法性和云台角误差共同约束。

## 5. 死亡、复活与回血

- 死亡后 5 s，在尸体位置复活
- 复活 HP 65
- 复活后 3 s 无敌
- 无敌结束进入过热门锁，需要冷却
- 尸体仍有物理和视线效果
- 队友、尸体都参与碰撞/推挤相关规则

## 6. 碰撞与推挤

- 碰撞/推墙/推人规则在 `_check_collision_fouls()`、`_push_fouls()` 等函数中
- 最新训练 run 关闭了 **推人终局**：`--no-body-push-terminal`
- dense 推人惩罚仍存在
- 部署/规则评测默认仍保留推人/推墙终局，除非评测显式关闭

因此训练目标和严格规则部署之间存在一个已知分布差异，评测报告必须写清 run 的 terminal foul 设置。

## 7. 占点、比分和终局

- 中心区半径 1.5 m
- 占点按红蓝净人数计分
- 击杀 +10 分
- 先到 90 分结束
- 时间到后依次比较比分、总伤害、存活总 HP
- 空局、双方同时到分、双方犯规等有独立终局码

终局码列表和奖励细节由代码和规则书共同定义，统计页不得把未知终局码归成“其他”。

## 8. 观测噪声

默认 `obs_noise=True`：

- 敌人 HP 带基础噪声
- 敌方小陀螺会提高 HP 噪声
- 敌方 yaw 有 ±5° 噪声
- 敌方开火帧不额外加 yaw 噪声
- 受击方向 cue 有 ±45° 噪声

单文件演示的环境 ONNX 将环境内部纯逻辑固化后在 JS 层重新注入敌人 HP/yaw 噪声，用于可视化与 Actor 输入。受击 cue 的样本级 RNG 可能与 JAX 运行不完全一致。

## 9. Torch/JAX parity 状态

### 已验证

- 固定 reset 状态布局
- 普通移动和边界裁剪
- action/fire mask
- actor raw 参数
- critic value
- global state 主要字段
- dense reward 的主要组成
- deterministic ONNX 环境单步

### 仍有差异或待持续验证

- 随机 reset 的 PRNG 样本序列
- 碰撞/推挤后的精细物理残差
- 观测噪声和 cue 噪声的样本级一致性
- 极端接触、多起碰撞、同帧多事件的顺序

不要因为单步对拍通过就宣称两个仿真器完全等价。

## 10. 代码核对时发现的重要事项

- Actor 使用 **11 tokens**，不是早期文档常写的 10 tokens；cue 是独立 token。
- Critic 当前是 **3 层**，旧交接文档曾写 2 层。
- `config.py` 是数值权威；历史训练总结中的 reward/foul 数值可能属于旧版本。
- liveview 与 checkpoint 状态叶子顺序耦合；新增 `RmState` 字段会破坏旧 ONNX 输入顺序。
- 最新单文件演示内嵌的是 1200 actor，不是 `league_best.pkl` 的通用别名；更新权重后必须重建 HTML。

## 11. 历史 Torch 文档

[TORCH_IMPLEMENTATION_NOTES.md](TORCH_IMPLEMENTATION_NOTES.md) 从 PyTorch/VMAS 工程带入，保留用于规则背景与对照。其中提到的文件路径、回调名称和部分网络结构不是本 JAX 仓库的当前入口。
