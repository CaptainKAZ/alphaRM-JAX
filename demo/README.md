# alphaRM 人机大战与 AI 观战

`docs/index.html` 是完全本地的单文件演示，内嵌：

- ORT Web 1.31.1 JavaScript runtime
- WebAssembly runtime
- 1200 轮 actor ONNX
- ORT 优化后的环境单步 ONNX
- 固定开局状态和 UI

页面不依赖 Python、HTTP、API 或外部模型文件，可以直接通过 `file://` 打开。

## 默认模式

打开后默认进入 **观战挂机**：

- 六台机器人全部由 AI 控制
- 当前选择的机器人只作为观察视角
- 键鼠不会覆盖 AI 动作

点击“本地设置”中“控制 R1/R2/.../B3”旁边的 **退出观战**，才能接管当前视角机器人。

## 操作

| 输入 | 行为 |
|---|---|
| `W A S D` | 世界方向移动 |
| `Shift` | 小陀螺 |
| 鼠标移动 | 未锁定时按鼠标方向转动云台 |
| 右键按住 | 选择离鼠标角度最近的可见敌人；按住时移动鼠标可换目标 |
| 左键按住 | 对已锁定目标开火 |
| `R` | 重开本局 |
| `P` | 暂停/继续 |
| `T` | 显示/隐藏 FOV |
| `H` | 切换当前视角机器人 |

左右键可任意先后同时按。没有锁定目标时，左键不会产生开火动作。

## 调试开关

- **战争迷雾**：开启时只显示当前视角观测中可见的敌人；关闭后显示所有敌人位置，但未观测敌人的 HP/热量仍未知。
- **敌人真值**：显示敌人的真实 HP、热量、过热、无敌和内部状态。
- **FOV**：只控制视野锥绘制，不改变 Actor 输入。

## 文件

| 文件 | 作用 |
|---|---|
| `../docs/index.html` | 发布用单文件演示，也是 GitHub Pages 首页 |
| `human_game.html` | JAX HTTP 后端人机大战前端 |
| `web/index.html` | 多文件 Web 源码 |
| `web/engine.js` | 状态、动作解码、mask、快照和调试视图 |
| `web/worker.js` / `web/embedded_worker.js` | 普通 Worker 与单文件 Blob Worker |
| `web/actor.onnx` | 1200 actor |
| `web/env_step.onnx` | ORT-extended 环境图 |
| `web/fixture.json` | 固定 reset 的 62 个状态叶子和初始观测 |
| `web/vendor/` | ORT Web JS、loader 和 WASM |
| `web/build_single_file_web.py` | 双栏/上下布局源码到单 HTML 的打包脚本 |
| `web/export_*.py` | checkpoint、环境图和 fixture 导出 |
| `web/optimize_env_onnx.py` | ORT-extended 环境图优化 |

## JAX 后端人机大战

```bash
PYTHONPATH=. python human_jax_server.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --mode fixed \
  --human 0 \
  --port 8767
```

打开：

```text
http://localhost:8767/
```

这条路线直接运行 JAX 环境和 actor；`docs/index.html` 则运行 ONNX Web 版本。

## 重新导出 ONNX

### 1. actor

```bash
python demo/web/export_actor_onnx_spike.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --out demo/web/actor.onnx
```

导出脚本会与原 JAX actor 对拍 raw 和 hidden_next。

### 2. fixture

```bash
python demo/web/export_web_fixture.py \
  --out demo/web/fixture.json \
  --seed 0
```

### 3. 环境图

```bash
python demo/web/export_env_onnx_spike.py \
  --out demo/web/env_step.web.onnx

python demo/web/optimize_env_onnx.py \
  --input demo/web/env_step.web.onnx \
  --output demo/web/env_step.onnx
```

`export_env_onnx_spike.py` 会：

- 导出一个环境 step
- 修补 ORT CPU/WASM 不支持的 bool `Where` / `EyeLike`
- 保证 64 个输出名称唯一
- 与 JAX 单步结果对拍

优化后环境图约 15.6K 节点，浏览器首次加载约 3–5 秒。

### 4. 构建单文件

```bash
python demo/web/build_single_file_web.py
```

输出：

```text
docs/index.html
```

当前文件约 30 MB。

## 为什么使用经典 Blob Worker

`file://` 页面是 opaque origin。Chromium 对 Blob **module Worker** 的导入限制更严格；当前方案使用：

1. 主页面创建经典 Blob Worker
2. Worker 通过动态 import 加载 Blob ORT/engine
3. ORT 的 loader 和 WASM 也通过 Blob URL 提供

这样单文件可直接双击打开，不依赖 HTTP。

## 当前性能参考

测试环境：Chromium 1228、单线程 ORT WASM。

| 项目 | 约值 |
|---|---:|
| actor 创建 | 0.3–0.5 s |
| 环境创建 | 3–5 s |
| actor 6 agents/step | 2.5–3 ms |
| env step | 12–20 ms |
| 页面控制周期 | 100 ms |

环境图包含 63 个输入和 64 个输出。输出对象必须使用唯一名称，否则 JavaScript 以 name 为 key 时会丢失重复输出。

## 已知限制

- 单文件固定使用内嵌权重；换 checkpoint 必须重新导出并构建。
- 环境 ONNX 是固定开局 fixture，不包含 reset RNG 和所有训练随机化。
- JS 层重新注入敌人 HP/yaw 观测噪声，受击 cue 的样本级 RNG 与 JAX 不完全一致。
- 浏览器端演示不用于训练。
