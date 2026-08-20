#!/usr/bin/env python3
"""Generate the ONNX models used by the ONNX and plugin examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def save_static(path: Path) -> None:
    node = helper.make_node(
        "MaxPool", ["input"], ["output"], kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = helper.make_graph(
        [node],
        "static_pool",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def save_dynamic(path: Path) -> None:
    node = helper.make_node(
        "MaxPool", ["input"], ["output"], kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = helper.make_graph(
        [node],
        "dynamic_pool",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, "height", "width"]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, 3, "out_height", "out_width"]
            )
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def save_custom(path: Path) -> None:
    shape = (1, 2, 3, 4)
    offset = numpy_helper.from_array(np.ones(shape, dtype=np.float32), "offset")
    multiplier = numpy_helper.from_array(
        np.square(np.arange(np.prod(shape), dtype=np.float32)).reshape(shape), "mul"
    )
    node = helper.make_node(
        "CustomLayer",
        ["input", "offset", "mul"],
        ["output"],
        name="custom_layer",
        domain="trt.plugins",
        alpha=0.1,
        plugin_version="1",
        plugin_namespace="",
    )
    graph = helper.make_graph(
        [node],
        "custom_layer_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [offset, multiplier],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("trt.plugins", 1)],
    )
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="generated_models", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_static(args.output_dir / "static_pool.onnx")
    save_dynamic(args.output_dir / "dynamic_pool.onnx")
    save_custom(args.output_dir / "custom_layer.onnx")
    print(f"Models written to {args.output_dir}")


if __name__ == "__main__":
    main()

