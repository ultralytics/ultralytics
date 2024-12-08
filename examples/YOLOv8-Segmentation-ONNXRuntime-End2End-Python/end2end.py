import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnx.parser
from onnx import TensorProto, helper, numpy_helper
from onnxruntime_extensions.tools.pre_post_processing import (
    ChannelsLastToChannelsFirst,
    Identity,
    ImageBytesToFloat,
    LetterBox,
    PrePostProcessor,
    Resize,
    SelectBestBoundingBoxesByNMS,
    Split,
    Squeeze,
    Step,
    Transpose,
    Unsqueeze,
    create_named_value,
    utils,
)


class ScaleNMSBoundingBoxesAndKeyPoints(Step):
    """
    Scale bounding box and mask coordinates back to the original image size.

    This step takes the output of the NMS step and the original, resized, and letter-boxed images, and then adjusts the
    bounding box and mask coordinates from the processed image dimensions back to the original image dimensions.
    """

    def __init__(self, layout: Optional[str] = "HWC", name: Optional[str] = None):
        """
        Initialize the ScaleNMSBoundingBoxesAndKeyPoints step.

        Args:
            layout (str, optional): The layout of the image. Can be "HWC" or "CHW".
            name (str, optional): An optional name for the step.
        """
        super().__init__(
            ["nms_step_output", "original_image", "resized_image", "letter_boxed_image"],
            ["nms_output_with_scaled_boxes_and_masks"],
            name,
        )
        self.layout_ = layout
        if layout not in ["HWC", "CHW"]:
            raise ValueError("Invalid layout. Only HWC and CHW are supported")

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_params = []
        for idx, input_name in enumerate(self.input_names):
            input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, idx)
            graph_input_params.append(f"{input_type_str}[{input_shape_str}] {input_name}")
        graph_input_params = ", ".join(graph_input_params)

        if self.layout_ == "HWC":
            orig_image_h_w_c = "oh, ow, oc"
            scaled_image_h_w_c = "sh, sw, sc"
            letterboxed_image_h_w_c = "lh, lw, lc"
        else:
            orig_image_h_w_c = "oc, oh, ow"
            scaled_image_h_w_c = "sc, sh, sw"
            letterboxed_image_h_w_c = "lc, lh, lw"

        nms_output_type_str, nms_output_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        nms_output_shape = nms_output_shape_str.split(",")
        data_size_per_result = int(nms_output_shape[-1])
        if not isinstance(data_size_per_result, int):
            raise ValueError("Shape of input must have a numeric value for the mask data size")

        data_num_splits = 4
        data_split_sizes = f"2, 2, 2, {data_size_per_result - 6}"

        def split_num_outputs(num_outputs: int):
            split_input_shape_attr = ""
            if onnx_opset >= 18:
                split_input_shape_attr = f", num_outputs = {num_outputs}"
            return split_input_shape_attr

        graph_text = f"""\
            ScaleNMSBoundingBoxesAndKeyPoints 
            ({graph_input_params}) => ({nms_output_type_str}[{nms_output_shape_str}] {self.output_names[0]})
            {{
                i64_2 = Constant <value = int64[1] {{2}}>()
                data_split_sizes = Constant <value = int64[{data_num_splits}] {{{data_split_sizes}}}>()

                boxes_xy, boxes_wh_or_xy, score_class, masks = Split <axis=-1>({self.input_names[0]}, data_split_sizes)

                ori_shape = Shape ({self.input_names[1]})
                scaled_shape = Shape ({self.input_names[2]})
                lettered_shape = Shape ({self.input_names[3]})
                {orig_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (ori_shape)
                {scaled_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (scaled_shape)
                {letterboxed_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (lettered_shape)
                swh = Concat <axis = -1> (sw, sh)
                lwh = Concat <axis = -1> (lw, lh)

                f_oh = Cast <to = 1> (oh)
                f_sh = Cast <to = 1> (sh)
                ratios = Div (f_oh, f_sh)

                pad_wh = Sub (lwh, swh)
                half_pad_wh = Div (pad_wh, i64_2)
                f_half_pad_wh = Cast <to = 1> (half_pad_wh)

                offset_boxes_xy = Sub (boxes_xy, f_half_pad_wh)
                restored_boxes = Concat <axis=-1> (offset_boxes_xy, boxes_wh_or_xy)
                scaled_boxes = Mul (restored_boxes, ratios)

                scaled_masks = Identity(masks)
                {self.output_names[0]} = Concat <axis=-1> (scaled_boxes, score_class, scaled_masks)
            }}
            """
        converter_graph = onnx.parser.parse_graph(graph_text)
        return converter_graph


def export_model_to_onnx(yolo_version: str, onnx_model_name: str, model_input_height=640, model_input_width=640):
    """
    Exports a local or downloaded YOLOv8 segmentation model to ONNX.

    If the corresponding <yolo_version>-seg.pt file exists locally, it uses that. If not, and download is required,
    ultralytics will download it automatically.
    """
    from pip._internal import main as pipmain

    try:
        import ultralytics
    except ImportError:
        pipmain(["install", "ultralytics"])
        import ultralytics

    pt_model = Path(f"{yolo_version}-seg.pt")
    model = ultralytics.YOLO(str(pt_model))
    exported_filename = model.export(format="onnx", opset=15, imgsz=(model_input_height, model_input_width))
    assert exported_filename, f"Failed to export {pt_model} to onnx"
    # Move the exported ONNX to the user-specified name
    shutil.move(exported_filename, onnx_model_name)


def yolo_detection_in_memory(model: onnx.ModelProto, onnx_opset: int = 16, num_classes: int = 80) -> onnx.ModelProto:
    """
    Integrate pre- and post-processing steps into the YOLOv8 ONNX model in- memory.

    This function takes an ONNX model and adds Resize, LetterBox, and other steps
    directly into the model graph. It also sets up post-processing steps, including NMS,
    to produce an end-to-end ONNX model that can handle raw inputs and produce final outputs.

    Args:
        model (onnx.ModelProto): The initial ONNX model of YOLOv8.
        onnx_opset (int): The ONNX opset version to use.
        num_classes (int): The number of classes for the model.

    Returns:
        onnx.ModelProto: The updated ONNX model with integrated pre/post-processing.
    """
    model_with_shape_info = onnx.shape_inference.infer_shapes(model)
    model_input_shape = model_with_shape_info.graph.input[0].type.tensor_type.shape
    h_in = model_input_shape.dim[-2].dim_value
    w_in = model_input_shape.dim[-1].dim_value

    inputs = [create_named_value("rgb_data", onnx.TensorProto.UINT8, [h_in, w_in, 3])]

    pipeline = PrePostProcessor(inputs, onnx_opset)
    pipeline.add_pre_processing(
        [
            Identity(name="RGBInput"),
            Resize((h_in, w_in), policy="not_larger"),
            LetterBox(target_shape=(h_in, w_in)),
            ChannelsLastToChannelsFirst(),
            ImageBytesToFloat(),
            Unsqueeze([0]),
        ]
    )

    post_processing_steps = [
        Squeeze([0]),
        Transpose([1, 0]),
        Split(num_outputs=3, axis=-1, splits=[4, num_classes, 32]),
        SelectBestBoundingBoxesByNMS(has_mask_data=True, iou_threshold=0.5, score_threshold=0.67),
        (
            ScaleNMSBoundingBoxesAndKeyPoints(name="ScaleBoundingBoxes"),
            [
                utils.IoMapEntry("RGBInput", producer_idx=0, consumer_idx=1),
                utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
                utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
            ],
        ),
    ]

    pipeline.add_post_processing(post_processing_steps)

    new_model = pipeline.run(model)
    new_model = onnx.shape_inference.infer_shapes(new_model, strict_mode=True)
    return new_model


def add_resize_node_to_mask_protos_in_memory(model: onnx.ModelProto, model_size: int) -> onnx.ModelProto:
    """
    Add a Resize node to the ONNX model to adjust mask prototypes to the desired model size.

    This function inserts a Resize node into the model graph to ensure mask prototypes
    are correctly shaped for the given input dimensions.

    Args:
        model (onnx.ModelProto): The ONNX model to update.
        model_size (int): The desired size of the model's input (both width and height).

    Returns:
        onnx.ModelProto: The updated ONNX model with the Resize node integrated.
    """
    graph = model.graph
    target_size = helper.make_tensor("target_size", TensorProto.INT64, [4], [1, 32, model_size, model_size])
    graph.initializer.append(target_size)

    resize_node = helper.make_node(
        "Resize", inputs=["output1", "", "", "target_size"], outputs=["mask_protos"], mode="linear"
    )
    graph.node.append(resize_node)

    new_output = helper.make_tensor_value_info("mask_protos", TensorProto.FLOAT, [1, 32, model_size, model_size])
    graph.output.append(new_output)

    output1_index = None
    for i, output in enumerate(graph.output):
        if output.name == "output1":
            output1_index = i
            break
    if output1_index is not None:
        graph.output.remove(graph.output[output1_index])

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def finalize_mask_processing_in_memory(model: onnx.ModelProto, model_size: int) -> onnx.ModelProto:
    """
    Finalize mask processing steps in the ONNX model.

    This function adds nodes for slicing, reshaping, thresholding, and reducing mask data
    to produce final segmentation masks. It ensures masks are properly scaled, thresholded,
    and integrated into the model's output.

    Args:
        model (onnx.ModelProto): The ONNX model to finalize.
        model_size (int): The model input size, used for reshaping masks.

    Returns:
        onnx.ModelProto: The updated ONNX model with final mask processing steps integrated.
    """
    graph = model.graph

    reshape_mask_shape = numpy_helper.from_array(
        np.array([32, model_size * model_size], dtype=np.int64), name="reshape_mask_shape"
    )
    final_reshape_shape = numpy_helper.from_array(
        np.array([-1, model_size, model_size], dtype=np.int64), name="final_reshape_shape"
    )
    start = numpy_helper.from_array(np.array([6], dtype=np.int64), name="slice_start")
    end = numpy_helper.from_array(np.array([38], dtype=np.int64), name="slice_end")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_steps")

    graph.initializer.extend([start, end, axes, steps, final_reshape_shape, reshape_mask_shape])

    threshold = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["threshold_value"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[0.5],
        ),
    )
    graph.node.append(threshold)

    slice_node = helper.make_node(
        "Slice",
        inputs=["nms_output_with_scaled_boxes_and_masks", "slice_start", "slice_end", "slice_axes", "slice_steps"],
        outputs=["sliced_nms_output"],
        name="SliceNMSOutput",
    )
    graph.node.append(slice_node)

    reshape_mask_protos_node = helper.make_node(
        "Reshape",
        inputs=["mask_protos", "reshape_mask_shape"],
        outputs=["reshaped_mask_protos"],
        name="ReshapeMaskProtos",
    )
    graph.node.append(reshape_mask_protos_node)

    matmul_node = helper.make_node(
        "MatMul", inputs=["sliced_nms_output", "reshaped_mask_protos"], outputs=["matmul_output"], name="MatMulMasks"
    )
    graph.node.append(matmul_node)

    reshape_final_output_node = helper.make_node(
        "Reshape", inputs=["matmul_output", "final_reshape_shape"], outputs=["final_masks"], name="ReshapeFinalOutput"
    )
    graph.node.append(reshape_final_output_node)

    binary_masks = helper.make_node("Greater", inputs=["final_masks", "threshold_value"], outputs=["binary_masks"])
    graph.node.append(binary_masks)

    cast_node = helper.make_node(
        "Cast", inputs=["binary_masks"], outputs=["cast_binary_masks"], to=TensorProto.FLOAT, name="CastToInt"
    )
    graph.node.append(cast_node)

    reduced_mask = helper.make_node(
        "ReduceMax", inputs=["cast_binary_masks"], outputs=["input_image_mask"], axes=[0], keepdims=1
    )
    graph.node.append(reduced_mask)

    final_masks_output = helper.make_tensor_value_info("final_masks", TensorProto.FLOAT, [None, model_size, model_size])
    input_image_mask_output = helper.make_tensor_value_info(
        "input_image_mask", TensorProto.FLOAT, [1, model_size, model_size]
    )
    graph.output.append(final_masks_output)
    graph.output.append(input_image_mask_output)

    for i, output in enumerate(graph.output):
        if output.name == "mask_protos":
            del graph.output[i]
            break

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert and finalize YOLOv8 ONNX model with integrated pre/post-processing."
    )
    parser.add_argument("--model-size", type=int, default=640, help="Model input size (both width and height).")
    parser.add_argument(
        "--yolo-version", type=str, default="yolov8n", help="YOLOv8 segmentation model version (e.g. yolov8n, yolov8s)."
    )
    parser.add_argument(
        "--model", type=str, help="Base YOLO ONNX model file. If not specified, uses <yolo-version>-seg.onnx"
    )
    parser.add_argument("--download-model", action="store_true", help="Download YOLO model if not present.")
    parser.add_argument(
        "--final-model",
        type=str,
        default="",
        help="Output final model name. If empty, it uses <version>-seg_end2end.onnx",
    )

    args = parser.parse_args()

    model_size = args.model_size
    yolo_version = args.yolo_version
    # If user did not specify --model, default it based on yolo_version
    if not args.model:
        args.model = f"{yolo_version}-seg.onnx"

    base_model_path = Path(args.model)
    pt_model = Path(f"{yolo_version}-seg.pt")  # e.g. yolov8s-seg.pt if yolo_version='yolov8s'

    # Determine final model name
    if not args.final_model:
        default_final_model_name = pt_model.stem + "_end2end.onnx"
    else:
        default_final_model_name = args.final_model

    # If ONNX model does not exist
    if not base_model_path.exists():
        # If it does not exist, we must create it from either a local pt or download
        if pt_model.exists():
            # We have a local .pt file, use it to export to ONNX
            print(f"Found local {pt_model}, exporting to {base_model_path}...")
            export_model_to_onnx(yolo_version, str(base_model_path), model_size, model_size)
        else:
            # .pt does not exist locally
            if args.download_model:
                print(f"No {pt_model} or ONNX model found. Downloading and exporting now...")
                export_model_to_onnx(yolo_version, str(base_model_path), model_size, model_size)
            else:
                raise FileNotFoundError(
                    f"{base_model_path} and {pt_model} do not exist. Please use --download-model or provide them."
                )
    else:
        print(f"Found existing ONNX model: {base_model_path}, skipping model export.")

    # Now we have base_model_path available
    base_model = onnx.load(str(base_model_path))

    print("Adding pre/post processing steps to the model in memory...")
    processed_model = yolo_detection_in_memory(base_model, onnx_opset=16)

    print("Adding mask protos resize node in memory...")
    processed_model = add_resize_node_to_mask_protos_in_memory(processed_model, model_size)

    print("Finalizing mask processing in memory...")
    final_model = finalize_mask_processing_in_memory(processed_model, model_size)

    # Save only the final end-to-end model
    onnx.save(final_model, default_final_model_name)
    print("Model processing complete. Final model:", default_final_model_name)
