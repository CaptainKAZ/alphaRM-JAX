# Docker 配方

仓库提供新的 alphaRM 专用镜像：

```text
Dockerfile.alphaRM
```

上游 JaxMARL 的 `Dockerfile` 会安装 Robotarium 等 submodule 环境；alphaRM-JAX 不需要这些额外组件，因此使用专用镜像可以显著减少构建失败面。

## 1. 构建

```bash
docker build \
  --build-arg UID="$(id -u)" \
  -t alphaRM-jax:latest \
  -f Dockerfile.alphaRM .
```

默认基础镜像：

```text
nvcr.io/nvidia/jax:26.04-py3
```

需要其他已安装合适 JAX/CUDA 的镜像时：

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/jax:26.04-py3 \
  --build-arg UID="$(id -u)" \
  -t alphaRM-jax:latest \
  -f Dockerfile.alphaRM .
```

## 2. 启动 GPU 训练容器

```bash
docker run --rm -it \
  --name alphaRM-jax \
  --gpus all \
  --ipc=host \
  --shm-size=20g \
  -v "$PWD":/workspace/alphaRM-JAX \
  alphaRM-jax:latest
```

首次安装如果绑定挂载覆盖了镜像内的 `src` 文件，在容器中执行：

```bash
cd /workspace/alphaRM-JAX
pip install -e .[algs,dev,demo]
```

## 3. 特权模式

普通训练和推理不需要特权模式。只有需要 `perf`、`py-spy` 或 profiler 访问内核/进程性能事件时：

```bash
docker run --rm -it \
  --name alphaRM-jax \
  --gpus all \
  --privileged \
  --ipc=host \
  --shm-size=20g \
  -v "$PWD":/workspace/alphaRM-JAX \
  alphaRM-jax:latest
```

## 4. CPU 冒烟

```bash
docker run --rm -it \
  -e JAX_PLATFORMS=cpu \
  -v "$PWD":/workspace/alphaRM-JAX \
  alphaRM-jax:latest \
  bash -lc '
    cd /workspace/alphaRM-JAX
    PYTHONPATH=. python baselines/MAPPO/train_rm_jax.py \
      --B 16 --T 16 --updates 1 --epochs 1 \
      --disable-jit --no-log
  '
```

## 5. 端口

需要从宿主机访问观察窗/人机大战时：

```bash
docker run --rm -it \
  --gpus all \
  -p 8767:8767 \
  -p 8768:8768 \
  -v "$PWD":/workspace/alphaRM-JAX \
  alphaRM-jax:latest
```

- 8767：`human_jax_server.py`
- 8768：`liveview/start_rm_live.sh`

单文件 `demo/rm_single.html` 不需要容器端口。

## 6. 数据与 checkpoint

- 训练输出在 `outputs/`，默认被 Git 忽略。
- 宿主目录应使用 bind mount 持久化，不要把重要 checkpoint 只放在容器内。
- 大 batch 训练建议 `--shm-size=20g`。

## 7. Dev Container

仓库的 `.devcontainer/devcontainer.json` 使用：

- `Dockerfile.alphaRM`
- `--gpus all`
- `--privileged`
- `--shm-size=20g`

它适合需要 profiler 的开发环境。纯训练环境可以移除 `--privileged`。
