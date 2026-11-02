#!/usr/bin/env python3
"""Bundle the ORT runtime, models, fixture, engine, and worker into one HTML file."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rm_single.html",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    assets = {
        "ortBundle": b64(root / "vendor/ort.wasm.bundle.min.mjs"),
        "ortLoader": b64(root / "vendor/ort-wasm-simd-threaded.mjs"),
        "ortWasm": b64(root / "vendor/ort-wasm-simd-threaded.wasm"),
        "actor": b64(root / "actor.onnx"),
        "env": b64(root / "env_step.onnx"),
        "fixture": b64(root / "fixture.json"),
        "engine": b64(root / "engine.js"),
    }
    worker_source = (
        f"globalThis.RM_EMBEDDED_ASSETS = {json.dumps(assets, separators=(',', ':'))};\n"
        + (root / "embedded_worker.js").read_text(encoding="utf-8")
    )
    if "</script" in worker_source.lower():
        raise RuntimeError("embedded worker source contains a closing script tag")

    html = (root / "index.html").read_text(encoding="utf-8")
    old_worker = "const localWorker = new Worker('./worker.js', {type:'module'});"
    if old_worker not in html:
        raise RuntimeError("worker constructor marker not found")
    new_worker = """const __rmWorkerSource = document.getElementById('rm-worker-source').textContent;
const __rmWorkerUrl = URL.createObjectURL(new Blob([__rmWorkerSource], {type:'text/javascript'}));
const localWorker = new Worker(__rmWorkerUrl);"""
    html = html.replace(old_worker, new_worker, 1)

    app_marker = "<script>\nconst CV = document.getElementById('cv');"
    if app_marker not in html:
        raise RuntimeError("application script marker not found")
    worker_block = (
        '<script id="rm-worker-source" type="text/plain">\n'
        + worker_source
        + "\n</script>\n"
    )
    html = html.replace(app_marker, worker_block + app_marker, 1)
    html = html.replace(
        '<meta charset="utf-8">',
        '<meta charset="utf-8">\n<link rel="icon" href="data:,">',
        1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(
        f"wrote {args.out} size={args.out.stat().st_size} "
        f"assets={sum(len(v) for v in assets.values())} chars"
    )


if __name__ == "__main__":
    main()
