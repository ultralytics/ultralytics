# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Measure ONNX parameters and formula-complete inference FLOPs from a PT or ONNX model.

Examples:
    python examples/onnx_flops.py yolo27x-detr-flat.pt --imgsz 640
    python examples/onnx_flops.py yolo27x-detr-flat.onnx
    python examples/onnx_flops.py model.onnx --input-shape 1 3 640 640 --json

The counter executes an ONNX Runtime CPU forward pass with graph optimization disabled, uses the runtime tensor shapes
for every executed node, and fails when it encounters an unsupported arithmetic operator. Multiply-accumulate is
counted as two FLOPs. Comparisons, selection, indexing, and memory movement are reported separately from FLOPs.
Parameters are the floating-point initializer elements retained in the exported ONNX inference graph.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, shape_inference

METADATA_OPS = {
    "Cast",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Expand",
    "Flatten",
    "Gather",
    "GatherElements",
    "Identity",
    "If",
    "Range",
    "Reshape",
    "Shape",
    "Size",
    "Slice",
    "Split",
    "Squeeze",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "Where",
}
LOGICAL_OPS = {"And", "Equal", "Greater", "Less", "Not", "Xor"}
UNARY_ARITHMETIC_OPS = {"Abs", "Neg", "Reciprocal"}
BINARY_ARITHMETIC_OPS = {"Add", "Div", "Mod", "Mul", "Pow", "Sub"}
SPECIAL_OPS = {"Cos", "Erf", "Exp", "Log", "Sin", "Sqrt"}
FLOAT_TYPES = {"bfloat16", "double", "float", "float16"}
FLOAT_INITIALIZER_TYPES = {TensorProto.BFLOAT16, TensorProto.DOUBLE, TensorProto.FLOAT, TensorProto.FLOAT16}
NUMPY_DTYPES = {
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}
FLOP_CATEGORIES = ("dense", "elementwise", "reduction_norm", "interpolation", "special")


def product(shape):
    """Return the number of elements in a concrete tensor shape."""
    return math.prod(shape)


def graph_nodes(graph):
    """Yield nodes recursively, including control-flow subgraphs."""
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from graph_nodes(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from graph_nodes(subgraph)


def profile_shapes(items):
    """Convert an ONNX Runtime profile shape list into (dtype, shape) tuples."""
    result = []
    for item in items:
        if not item:
            result.append((None, None))
            continue
        dtype, shape = next(iter(item.items()))
        result.append((dtype, tuple(shape)))
    return result


def export_onnx(path, imgsz, batch, opset):
    """Export a PT checkpoint to fixed-shape ONNX without graph simplification."""
    from ultralytics import YOLO

    model = YOLO(path)
    output = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        dynamic=False,
        simplify=False,
        opset=opset,
        device="cpu",
        nms=False,
        verbose=False,
    )
    return Path(output)


def onnx_parameter_count(model_path):
    """Count floating-point initializer elements retained in an ONNX inference graph."""
    model = onnx.load(model_path, load_external_data=False)
    return sum(
        product(initializer.dims)
        for initializer in model.graph.initializer
        if initializer.data_type in FLOAT_INITIALIZER_TYPES
    )


def runtime_profile(model_path, input_shape=None):
    """Execute ONNX once and return ONNX Runtime's per-node profile events."""
    with tempfile.TemporaryDirectory(prefix="onnx-flops-") as directory:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "profile")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1:
            raise ValueError(f"Expected one model input, found {len(session.get_inputs())}")

        model_input = session.get_inputs()[0]
        shape = tuple(input_shape or model_input.shape)
        if any(not isinstance(dimension, int) for dimension in shape):
            raise ValueError(f"Input has dynamic shape {shape}; supply --input-shape with concrete dimensions")
        dtype = NUMPY_DTYPES.get(model_input.type)
        if dtype is None:
            raise ValueError(f"Unsupported input type {model_input.type}")

        session.run(None, {model_input.name: np.zeros(shape, dtype=dtype)})
        profile_path = Path(session.end_profiling())
        return [event for event in json.loads(profile_path.read_text()) if event.get("cat") == "Node"]


class OnnxFlopCounter:
    """Count executed ONNX arithmetic with explicit, fail-closed formulas."""

    def __init__(self, model_path, events):
        """Initialize graph metadata and runtime events."""
        model = onnx.load(model_path)
        self.nodes = {node.name: node for node in graph_nodes(model.graph)}
        self.initializers = {tensor.name: tuple(tensor.dims) for tensor in model.graph.initializer}
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        self.static_shapes = {}
        for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]:
            dimensions = value.type.tensor_type.shape.dim
            if dimensions and all(dimension.HasField("dim_value") for dimension in dimensions):
                self.static_shapes[value.name] = tuple(dimension.dim_value for dimension in dimensions)

        self.events = events
        self.counts = defaultdict(int)
        self.by_operator = defaultdict(lambda: defaultdict(int))
        self.calls = Counter()
        self.unsupported = []

    def add(self, operator, category, amount):
        """Add an integer operation count to a category."""
        amount = int(amount)
        self.counts[category] += amount
        self.by_operator[operator][category] += amount

    def node(self, event):
        """Return the ONNX node corresponding to a runtime profile event."""
        return self.nodes.get(event["name"].removesuffix("_kernel_time"))

    def attributes(self, event):
        """Return decoded attributes for a runtime profile event."""
        node = self.node(event)
        return (
            {}
            if node is None
            else {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        )

    def count_event(self, event):
        """Classify and count one executed ONNX Runtime node."""
        args = event["args"]
        operator = args["op_name"]
        inputs = profile_shapes(args["input_type_shape"])
        outputs = profile_shapes(args["output_type_shape"])
        node = self.node(event)
        attributes = self.attributes(event)
        self.calls[operator] += 1

        if operator in METADATA_OPS:
            return

        output_dtype, output_shape = outputs[0]
        output_elements = product(output_shape)
        output_is_float = output_dtype in FLOAT_TYPES

        if operator == "Conv":
            weight_shape = inputs[1][1]
            self.add(operator, "dense", 2 * output_elements * product(weight_shape[1:]))
            if len(inputs) > 2:
                self.add(operator, "elementwise", output_elements)
            return

        if operator in {"Gemm", "MatMul"}:
            input_shape = inputs[0][1]
            if operator == "Gemm":
                reduction = input_shape[-2] if attributes.get("transA", 0) else input_shape[-1]
            else:
                if len(inputs) > 1:
                    weight_shape = inputs[1][1]
                elif node is not None and node.input[1] in self.initializers:
                    weight_shape = self.initializers[node.input[1]]
                elif node is not None and node.input[1] in self.static_shapes:
                    weight_shape = self.static_shapes[node.input[1]]
                else:
                    raise ValueError(f"Missing MatMul input shape for {event['name']}")
                reduction = input_shape[-1]
                if reduction not in weight_shape[-2:]:
                    raise ValueError(f"Incompatible MatMul shapes {input_shape} and {weight_shape}")
            self.add(operator, "dense", 2 * output_elements * reduction)
            if operator == "Gemm" and node is not None and len(node.input) > 2 and node.input[2]:
                self.add(operator, "elementwise", output_elements)
                if attributes.get("beta", 1.0) != 1.0:
                    self.add(operator, "elementwise", output_elements)
            if operator == "Gemm" and attributes.get("alpha", 1.0) != 1.0:
                self.add(operator, "elementwise", output_elements)
            return

        if operator in BINARY_ARITHMETIC_OPS:
            if output_is_float:
                self.add(operator, "elementwise", output_elements)
            return

        if operator in UNARY_ARITHMETIC_OPS:
            if output_is_float:
                self.add(operator, "elementwise", output_elements)
            return

        if operator == "Sigmoid":
            self.add(operator, "elementwise", 3 * output_elements)
            self.add(operator, "special", output_elements)
            return

        if operator in SPECIAL_OPS:
            if output_is_float:
                self.add(operator, "special", output_elements)
            return

        if operator == "Softmax":
            axis = attributes.get("axis", -1)
            if axis < 0:
                axis += len(output_shape)
            vectors = output_elements // output_shape[axis]
            self.add(operator, "reduction_norm", 3 * output_elements - vectors)
            self.add(operator, "special", output_elements)
            self.add(operator, "comparison", output_elements - vectors)
            return

        if operator == "LayerNormalization":
            groups = output_elements // product(inputs[1][1])
            self.add(operator, "reduction_norm", 7 * output_elements + groups)
            self.add(operator, "special", groups)
            return

        if operator == "ReduceMean":
            self.add(operator, "reduction_norm", product(inputs[0][1]))
            return

        if operator == "ReduceSum":
            self.add(operator, "reduction_norm", product(inputs[0][1]) - output_elements)
            return

        if operator == "ReduceMax":
            self.add(operator, "comparison", product(inputs[0][1]) - output_elements)
            return

        if operator == "GridSample":
            mode = attributes.get("mode", b"bilinear")
            if mode == b"nearest":
                self.add(operator, "selection", output_elements)
                return
            if mode not in {b"bilinear", b"linear"}:
                raise ValueError(f"Unsupported GridSample mode {mode!r}")
            grid_points = product(inputs[1][1][:-1])
            self.add(operator, "interpolation", 7 * output_elements + 20 * grid_points)
            self.add(operator, "comparison", 8 * grid_points)
            return

        if operator == "Resize":
            mode = attributes.get("mode")
            if mode == b"nearest":
                self.add(operator, "selection", output_elements)
                return
            if mode == b"linear":
                spatial_points = output_shape[0] * product(output_shape[2:])
                self.add(operator, "interpolation", 7 * output_elements + 12 * spatial_points)
                return
            raise ValueError(f"Unsupported Resize mode {mode!r}")

        if operator == "MaxPool":
            kernel = attributes.get("kernel_shape")
            if kernel is None:
                raise ValueError(f"Missing MaxPool kernel_shape for {event['name']}")
            self.add(operator, "comparison", output_elements * (product(kernel) - 1))
            return

        if operator == "Clip":
            self.add(operator, "comparison", 2 * output_elements)
            return

        if operator == "Relu":
            self.add(operator, "comparison", output_elements)
            return

        if operator == "TopK":
            self.add(operator, "selection", product(inputs[0][1]))
            return

        if operator in LOGICAL_OPS:
            self.add(operator, "comparison", output_elements)
            return

        self.unsupported.append((operator, event["name"], inputs, outputs))

    def run(self):
        """Count all executed events and fail if any operator is unsupported."""
        for event in self.events:
            self.count_event(event)
        if self.unsupported:
            operators = ", ".join(sorted({item[0] for item in self.unsupported}))
            raise RuntimeError(f"Unsupported executed arithmetic operators: {operators}")
        return self

    def result(self, model_path, parameters=None):
        """Return a serializable measurement result."""
        ordinary = sum(self.counts[category] for category in FLOP_CATEGORIES if category != "special")
        total = ordinary + self.counts["special"]
        return {
            "model": str(model_path),
            "parameters": parameters,
            "convention": "2 FLOPs per MAC; 1 operation per special-function evaluation",
            "gflops": {
                category: self.counts[category] / 1e9 for category in (*FLOP_CATEGORIES, "comparison", "selection")
            },
            "ordinary_gflops": ordinary / 1e9,
            "total_gflops": total / 1e9,
            "executed_nodes": sum(self.calls.values()),
            "operator_types": len(self.calls),
            "unsupported_operators": len(self.unsupported),
            "by_operator": {
                operator: {category: count / 1e9 for category, count in categories.items()}
                for operator, categories in sorted(self.by_operator.items())
            },
        }


def markdown(result, details=False):
    """Format a measurement result as Markdown."""
    rows = [
        ("Dense", result["gflops"]["dense"]),
        ("Elementwise", result["gflops"]["elementwise"]),
        ("Reduction and normalization", result["gflops"]["reduction_norm"]),
        ("Interpolation", result["gflops"]["interpolation"]),
        ("Special functions", result["gflops"]["special"]),
        ("Exhaustive total", result["total_gflops"]),
        ("Comparisons (excluded)", result["gflops"]["comparison"]),
        ("Selection (excluded)", result["gflops"]["selection"]),
    ]
    lines = [
        f"### {Path(result['model']).name}",
        "",
        "| Category | Giga-operations |",
        "|---|---:|",
        *(f"| {name} | {value:.9f} |" for name, value in rows),
        "",
        (
            f"Executed nodes: {result['executed_nodes']}; operator types: {result['operator_types']}; "
            f"unsupported operators: {result['unsupported_operators']}."
        ),
    ]
    if result["parameters"] is not None:
        lines.append(f"ONNX parameters: {result['parameters']:,} ({result['parameters'] / 1e6:.6f} M).")
    if details:
        lines.extend(("", "| Operator | Category | Giga-operations |", "|---|---|---:|"))
        for operator, categories in result["by_operator"].items():
            lines.extend(f"| {operator} | {category} | {value:.9f} |" for category, value in categories.items())
    return "\n".join(lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Fixed-shape ONNX model or a PT checkpoint to export")
    parser.add_argument("--imgsz", type=int, default=640, help="PT export image size")
    parser.add_argument("--batch", type=int, default=1, help="PT export batch size")
    parser.add_argument("--opset", type=int, default=19, help="PT export ONNX opset")
    parser.add_argument("--input-shape", type=int, nargs="+", help="Concrete ONNX input shape for a dynamic model")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    parser.add_argument("--details", action="store_true", help="Include per-operator counts")
    return parser.parse_args()


def main():
    """Export if necessary, then print the ONNX graph's parameter and operation counts."""
    args = parse_args()
    model_path = args.model.resolve()
    if model_path.suffix.lower() == ".pt":
        model_path = export_onnx(model_path, args.imgsz, args.batch, args.opset)
    elif model_path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected a .pt or .onnx model, received {model_path}")

    parameters = onnx_parameter_count(model_path)
    events = runtime_profile(model_path, args.input_shape)
    result = OnnxFlopCounter(model_path, events).run().result(model_path, parameters)
    print(json.dumps(result, indent=2) if args.json else markdown(result, args.details))


if __name__ == "__main__":
    main()
