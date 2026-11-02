# GitHub Pages 部署

`.github/workflows/pages.yml` 将单文件演示发布为 Pages 首页：

```text
demo/rm_single.html -> _site/index.html
```

工作流：

- 在 push 到 `main` 或 `alphaRM-JAX` 时触发
- 修改单文件或 workflow 时触发
- 支持手动 `workflow_dispatch`
- 使用 GitHub Actions Pages artifact，不运行 Jekyll

## 首次启用

远程仓库创建后：

1. 打开 GitHub 仓库 `Settings -> Pages`
2. Source 选择 **GitHub Actions**
3. 运行 `Deploy alphaRM demo to GitHub Pages`

预期地址：

```text
https://<owner>.github.io/alphaRM-JAX/
```

## 注意

- 只发布 `demo/rm_single.html`，不会把训练输出、checkpoint 或源码目录暴露到 Pages。
- 单文件约 30 MB，首次加载明显慢于普通页面。
- 页面运行时不需要外部网络；由 GitHub Pages 仅负责交付 HTML。
- 新 checkpoint 需要先重新导出 ONNX、重建单文件并提交，Pages 才会更新。
