#!/usr/bin/env python3
"""Create the portable ORT-extended environment graph used by the web demo."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.optimized_model_filepath = str(output)
    ort.InferenceSession(
        str(source),
        options,
        providers=["CPUExecutionProvider"],
    )

    model = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(model)
    print(
        f"input={source} output={output} "
        f"nodes={len(model.graph.node)} bytes={output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
