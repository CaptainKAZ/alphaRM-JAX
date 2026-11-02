# 上游来源与许可

## Fork 基线

本仓库从工作区 `JaxMARL` 创建，基线 commit：

```text
3ba06c2 Merge pull request #205 from prakharsahup/fix/overcooked-v2-walkable-border-interact
```

原始项目：

- 上游：https://github.com/FLAIROx/JaxMARL
- 工作区 fork 来源：https://github.com/bold-lab-ai/JaxMARL

本地 Git remote：

```text
origin   https://github.com/CaptainKAZ/alphaRM-JAX.git
upstream https://github.com/FLAIROx/JaxMARL.git
```

- `origin`：alphaRM-JAX 发布仓库
- `upstream`：官方 JaxMARL
- GitHub 远端长期分支只有 `main`

## 保留内容

- 上游 JaxMARL 历史
- Apache-2.0 `LICENSE`
- 其他环境、算法、测试和文档
- 上游 README：`README_UPSTREAM.md`

## alphaRM 新增内容

- `jaxmarl/environments/rm/`
- RM MAPPO 训练、联赛、日志和评测
- Torch/JAX parity 工具
- 训练观察窗
- 人机大战与 AI 观战演示
- ONNX Runtime Web 单文件构建流水线

## 许可

本仓库继续使用 Apache License 2.0。再分发修改版本时应：

- 保留 `LICENSE`
- 保留上游归属与论文引用
- 在新增或修改文件中保留必要的版权/许可说明

## 上游引用

使用 JaxMARL 基础设施时，请引用上游 README 中的 BibTeX：

```bibtex
@inproceedings{
    flair2024jaxmarl,
    title={JaxMARL: Multi-Agent RL Environments and Algorithms in JAX},
    author={Alexander Rutherford and Benjamin Ellis and Matteo Gallici and Jonathan Cook and Andrei Lupu and Gar{\dh}ar Ingvarsson and Timon Willi and Ravi Hammond and Akbir Khan and Christian Schroeder de Witt and Alexandra Souly and Saptarashmi Bandyopadhyay and Mikayel Samvelyan and Minqi Jiang and Robert Tjarko Lange and Shimon Whiteson and Bruno Lacerda and Nick Hawes and Tim Rockt{\"a}schel and Chris Lu and Jakob Nicolaus Foerster},
    booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2024},
}
```
