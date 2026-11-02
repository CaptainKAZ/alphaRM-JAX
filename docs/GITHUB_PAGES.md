# GitHub Pages 部署

仓库只保留一个长期分支：`main`。

GitHub Pages 使用：

```text
Branch: main
Folder: /docs
Entry:  docs/index.html
```

`docs/index.html` 是完全本地的单文件演示，已经内嵌：

- actor ONNX
- ORT 优化环境 ONNX
- ORT Web
- WASM runtime
- Worker、fixture 和 UI

`docs/.nojekyll` 关闭 Jekyll 处理，避免大文件和内嵌脚本被额外转换。

## 发布地址

```text
https://captainkaz.github.io/alphaRM-JAX/
```

## 更新流程

```bash
python demo/web/build_single_file_web.py
git add docs/index.html
git commit -m "build(demo): refresh embedded alphaRM single-file game"
git push origin main
```

GitHub Pages 会直接从 `main/docs` 发布新文件。

## 分支规范

- 长期分支只有 `main`
- 不创建 `gh-pages` 并行镜像
- 不使用 `alphaRM-JAX` 作为第二长期开发分支
- 临时实验分支在合并后删除

## 注意

- 单文件约 30 MB，首次加载明显慢于普通页面。
- 页面运行时不需要外部网络；GitHub Pages 只负责交付 HTML。
- 新 checkpoint 需要先重新导出 ONNX、重建 `docs/index.html` 并提交。
