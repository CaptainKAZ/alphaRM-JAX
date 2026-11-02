# alphaRM-JAX 训练观察窗

标准库 HTTP 服务，默认端口 **8768**。

```bash
bash liveview/start_rm_live.sh
```

页面：

- `http://localhost:8768/`：训练片段逐帧回放、比分、终局、HP/热量/动作
- `http://localhost:8768/stats.html`：逐 iter 终局占比与得分统计
- `http://localhost:8768/files`：只读浏览仓库 `outputs/`

## 数据流

JAX 训练入口在以下参数开启后写 live JSON：

```bash
python baselines/MAPPO/train_rm_jax.py \
  --live-view \
  --live-dir "$PWD/outputs/live" \
  --live-every 1 \
  ...
```

文件：

```text
outputs/live/rm_live.json
outputs/live/rm_stats_history.json
```

`serve_rm_live.py` 读取 `rm_live.json`，按 iteration 累积统计历史。服务本身只读训练输出，不参与训练。

## 文件

| 文件 | 作用 |
|---|---|
| `serve_rm_live.py` | `/`、`/stats.html`、`/rm_live.json`、`/rm_stats.json`、`/files`、`/raw` |
| `rm_live.html` | 主观察窗 |
| `rm_stats.html` | 终局码和得分统计 |
| `chart.umd.min.js` | 本地 Chart.js |
| `start_rm_live.sh` | 启动/重启服务 |

## 入口参数

```bash
bash liveview/start_rm_live.sh 8768
# 或
python liveview/serve_rm_live.py --port 8768 --bind 0.0.0.0
```

日志默认写入：

```text
/tmp/alphaRM_JAX_liveview.log
```

可通过 `RM_LIVE_LOG` 覆盖。

## 注意

- 修改 HTML 后需强制刷新浏览器。
- 修改 `serve_rm_live.py` 后重启服务。
- 不要热改正在写 `rm_live.json` 的训练进程。
- live JSON 可能较大，统计服务按文件 mtime/size 缓存解析结果。
