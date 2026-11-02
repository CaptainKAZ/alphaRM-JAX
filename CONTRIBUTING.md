# 贡献与提交规范

本仓库使用 **Conventional Commits**。一次提交只解决一个可描述的问题，提交后仓库应可构建、可测试。

## 1. 分支

- `alphaRM-JAX`：主开发分支
- 功能分支：`feat/<short-name>`
- 修复分支：`fix/<short-name>`
- 文档分支：`docs/<short-name>`

不要直接在已有发布 tag 上追加提交。

## 2. 提交信息格式

```text
<type>(<scope>): <summary>

[optional body]

[optional footer]
```

`summary` 使用简短、明确的祈使句。推荐中文摘要，技术名词保留英文。

### type

| type | 用途 |
|---|---|
| `feat` | 新功能、新环境、新训练能力 |
| `fix` | 修复规则、训练、评测或演示错误 |
| `docs` | 只修改文档 |
| `test` | 增加或修正测试 |
| `refactor` | 不改变行为的代码重构 |
| `perf` | 性能优化 |
| `style` | 仅格式、布局或视觉调整 |
| `build` | 构建、依赖、导出流水线 |
| `ci` | GitHub Actions、Pages 和自动检查 |
| `chore` | 仓库维护、忽略规则、清理 |

### scope

推荐 scope：

- `rm`：环境、规则、actor、critic
- `mappo`：训练循环、GAE、PPO
- `league`：自博弈和冻结最优代
- `aux`：辅助预测头
- `eval`：评测和跨代对打
- `liveview`：训练观察窗
- `demo`：人机大战、AI 对战、ONNX/Web
- `docs`：文档结构
- `repo`：仓库级配置

### 示例

```text
feat(rm): add JAX 3v3 environment and action masks
feat(mappo): add frozen-best league training
feat(demo): bundle ORT Web and checkpoint into one HTML
fix(rm): align fire mask with target invulnerability
docs(training): document the 1200-iteration league recipe
test(auth): add actor and critic parity checks
perf(liveview): cache parsed statistics by file mtime
style(demo): switch to stacked arena and information panels
```

不要把多个不相关改动写进同一条提交，例如：

```text
fix stuff + update docs + new model
```

## 3. 提交前检查

### Python

```bash
python -m compileall jaxmarl/environments/rm baselines/MAPPO liveview human_jax_server.py
python parity/test_ppo_math.py
python -m pytest parity/test_ppo_math.py -q
git diff --check
```

规则改动还应运行对应的 Torch/JAX parity 测试。涉及训练入口时，至少运行一个：

```bash
PYTHONPATH=. python baselines/MAPPO/train_rm_jax.py \
  --B 16 --T 16 --updates 1 --epochs 1 \
  --disable-jit --no-log
```

### 观察窗

修改 `liveview/` 后：

```bash
bash liveview/start_rm_live.sh 8768
curl -fsS http://127.0.0.1:8768/status.json
curl -fsS http://127.0.0.1:8768/stats.html >/dev/null
```

### 演示

修改 `demo/web/index.html`、`engine.js`、Worker 或模型后，需要重新生成单文件：

```bash
python demo/web/build_single_file_web.py
```

`demo/rm_single.html` 是发布产物，应与源码一起提交。提交前至少验证：

- 页面可通过 HTTP 打开
- `file://` 直接打开时 Blob Worker 能初始化
- 默认进入观战模式
- 退出观战后右键锁定、左键开火、左右键同时按均正常
- 战争迷雾和敌人真值开关生效

## 4. 不应提交的内容

- `outputs/`、checkpoint、评测视频、JAX cache
- 虚拟环境、`__pycache__`、profiler trace
- 密钥、token、SSH 配置、内部绝对路径
- 未经说明的巨大临时 `.npz` / `.pkl` 文件

例外：`demo/rm_single.html` 及其模型/ORT 资产是演示发布物，允许提交；更新权重时必须同步更新演示说明。

## 5. 文档要求

以下改动必须同步文档：

| 改动 | 至少更新 |
|---|---|
| 观测/动作/奖励/终局 | `docs/IMPLEMENTATION_NOTES.md`、相关规则说明 |
| 训练参数或恢复方式 | `docs/TRAINING_AND_EVALUATION.md` |
| 模型或网络结构 | `docs/ARCHITECTURE.md` |
| 演示操作或权重 | `demo/README.md` |
| 观察窗入口或数据格式 | `liveview/README.md` |

文档中的数值必须能在代码或保存的配置中找到依据。规则书、历史交接文档和当前代码冲突时，以当前代码和配置为准，并在实现说明中记录差异。

## 6. Commit 前自检

- [ ] 提交只包含一个逻辑主题
- [ ] 没有把训练输出或 secret 加进暂存区
- [ ] 测试已运行，或说明了无法运行的原因
- [ ] 行为变化已更新文档
- [ ] `git diff --check` 通过
- [ ] commit message 符合 Conventional Commits
