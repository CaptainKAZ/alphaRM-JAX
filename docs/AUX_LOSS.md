# Auxiliary Loss 设计

## 1. 目标

Auxiliary loss 不是为了增加部署时输出，而是给 Actor 的共享表示增加**可验证的物理与规则监督**：

- 被遮挡敌人的短时位置保持
- 开火后的条件命中率、伤害和击杀
- 敌方热量与过热风险
- 占点、得分、终局等规则状态
- 自身动作后果、资源变化和事件

这些标签来自仿真真实状态，但特权标签只进入辅助 loss，不作为 Actor 部署输入。

## 2. 网络位置

辅助头 `RmAuxHeads` 输入：

```text
[GRU hidden 256, encoder 256, executed action 6, current local obs 140]
```

其中：

- `hidden` 和 `encoder` 来自当前 Actor
- `action` 使用 `stop_gradient`
- `obs` 是 Actor 本来就能看到的局部观测

辅助头是两层 256 维 MLP，分叉为五组输出：

```text
track 12 · damage 4 · heat 9 · rules 7 · self 14
= 46
```

辅助参数只存在于训练 checkpoint：

```text
aux_params
aux_opt_state
aux_step
```

导出/部署 Actor 时不会包含这些参数。

## 3. 五组预测目标

| Head | 输出 | 标签 |
|---|---|---|
| Tracking | 3 敌人 × `(dx, dy, log_sigma, visible_next)` | 队伍 canonical 的相对位移；下一帧可见性 |
| Damage | `(fire, conditional hit probability, normalized damage, kill)` | 动作选中的敌人和真实一步结果 |
| Enemy heat | 3 敌人 × `(heat margin, heat delta, overheated)` | 下一帧热量余量和过热事件 |
| Rules | `(occ own/opp, center score own/opp, kill score own/opp, done)` | 当前优势方视角的规则量 |
| Self | 14 维运动/资源/事件 | 自身一步位置、速度、yaw、spin、伤害、回血、热量和状态变化 |

## 4. 位置预测与渐消权重

位置不是预测绝对世界坐标，而是预测：

```text
(下一帧敌人相对位置 - 当前锚点)
```

相对位置的归一化尺度：

```text
2 × v_max × dt = 0.6 m
```

敌人可见时刷新锚点；被遮挡后保留最后一次可见位置。遮挡年龄：

```text
age_frac = min(age / 15, 1)
age_weight = 0.5 × (1 + cos(pi × age_frac))
```

- 刚被遮挡：权重 1
- 0.75 s：权重 0.5
- 1.5 s：权重 0
- 超过 1.5 s：位置 loss 权重保持 0

辅助头本身不会停止训练。可见性 BCE 继续监督“敌人是否重新出现”，只是旧位置锚点不再贡献位置误差。

位置项使用可学习方差：

```text
NLL = 0.5 × (error² × exp(-2 × log_sigma) + 2 × log_sigma)
```

`log_sigma` 截断在 `[-4, 2]`。裁剪范围内 NLL 恒非负。

## 5. 各路 loss

### Tracking

```text
track = position_NLL + 0.25 × visibility_BCE
```

- 可见与隐藏位置分别监控
- 隐藏位置误差使用遮挡年龄权重
- 敌人尸体仍可作为物理对象被追踪

### Damage

```text
damage = BCE(fire)
       + BCE(hit_probability)
       + Huber(normalized_damage)
       + 0.5 × BCE(kill)
```

`hit_probability` 是后一步的条件命中概率，不是采样得到的 Bernoulli 结果。

### Enemy heat

```text
heat = Huber(heat_margin)
     + Huber(heat_delta)
     + 0.5 × BCE(overheated)
```

`heat_margin` 走到热上限的剩余比例，`heat_delta` 是一步热量变化。

### Rules

```text
rules = Huber(occupancy)
      + Huber(center_score)
      + Huber(kill_score)
      + 0.5 × BCE(done)
```

得分目标被缩放，避免 90 分目标量纲压制梯度。

### Self

```text
self = Huber(motion 6)
     + Huber(resource 4)
     + 0.5 × BCE(events 4)
```

资源项排除死亡和复活帧，避免把 HP 的归零/恢复误当成普通回血或伤害。

## 6. 总损失

```text
aux_total =
    1.00 × track
  + 1.00 × damage
  + 0.75 × heat
  + 0.75 × rules
  + 1.00 × self

actor_total = ppo_actor_loss + aux_coef × aux_total
```

最新 1200 轮 run：

```text
aux_coef = 0.02
aux_lr = 1e-4
```

辅助权重不是一开始直接拉满：

```text
ramp = clamp((iter - start_iter + 1) / warmup_iters, 0, 1)
effective_aux_coef = aux_coef × ramp
```

当前配置：

```text
start_iter = 500
warmup_iters = 10
```

## 7. 梯度边界

- 辅助头参数通过独立 Adam optimizer 更新。
- Actor encoder/GRU 会收到辅助 loss 的表示学习梯度。
- 已执行 action 使用 `stop_gradient`，防止辅助头反向改变动作标签本身。
- Critic 不接收辅助 loss。
- League 中冻结对手通过 `actor_train_mask` 排除，避免其特权标签影响学习者。

## 8. 为什么不是信息泄漏

部署 Actor 的输入仍然只有 140 维局部观测，不包含：

- 全部敌人真实位置
- 敌方真实热量
- 全局状态
- 规则终局真值

特权状态只用于生成训练标签。辅助 loss 改变 Actor 表示，但不改变部署接口。

## 9. 监控指标

训练日志和离线 probe 关注：

- `track_visible_pos_mae_m`
- `track_hidden_pos_mae_m`
- `track_hidden_age_mean`
- `track_visibility_accuracy`
- `damage_fire_accuracy`
- `damage_hit_probability_mae`
- `enemy_heat_margin_mae`
- `enemy_overheat_accuracy`
- `rules_*`
- `self_*`

1200 checkpoint 的代表值：

| 指标 | 值 |
|---|---:|
| 可见位置 MAE | 0.0269 m |
| 隐藏位置 MAE | 0.3344 m |
| 可见性准确率 | 0.9706 |
| 开火准确率 | 0.9985 |
| 条件命中概率 MAE | 0.0295 |
| 过热总体准确率 | 0.9580 |
| 过热 balanced accuracy | 0.7384 |
| 过热正例召回率 | 0.4933 |
| 自身资源 MAE | 0.0335 |

隐藏位置误差随遮挡年龄增长；过热分类仍偏向负例。文档和报告不能只引用总体 accuracy。
