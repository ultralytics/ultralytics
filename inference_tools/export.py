from copy import deepcopy

from ultralytics import YOLO
import torch
from torch import Tensor
import json
import onnx
from onnxsim import simplify
from onnxoptimizer import optimize

def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: Tensor,
    output_file: str,
    opset_version: int = 11,
    input_names: list = ["images"],
    output_names: list = ["output0"],
    do_constant_folding: bool = True,
    verbose: bool = False,
    dynamic_axes: dict = None,
) -> None:
    """
    Exports a PyTorch model to ONNX format.

    Parameters:
        model (torch.nn.Module): The PyTorch model to export.
        dummy_input (Tensor): A dummy input for the model, used for tracing.
        output_file (str): The path where the output ONNX file should be saved.
        opset_version (int, optional): The ONNX opset version. Defaults to 9.
        input_names (list, optional): Names of the model's input layers. Defaults to ['input'].
        output_names (list, optional): Names of the model's output layers. Defaults to ['output'].
        verbose (bool, optional): Whether to print detailed logs during export. Defaults to False.

    Returns:
        None: The function saves the model to an ONNX file.
    """

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_file,
        verbose=verbose,
        input_names=input_names,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    # Load config.json
    with open("./inference_config.json", "r") as f:
        config = json.load(f)
    print("Loaded config: ", config)
    img_height_size = config["img_size"]
    img_width_size = config["img_size"]

    print("🚀 Initializing model...")
    # Initialize and set up model
    # Load model
    model = YOLO(model=config["model_path"], task="detect").model

    print("🔌 Setting up device...")
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepcopy(model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()

    print("📝 Preparing dummy input...")
    # Create a dummy input using img_height_size and img_width_size from config.json
    dummy_input = torch.randn(1, 3, img_height_size, img_width_size, device=device)

    print("📦 Exporting to ONNX format...")
    # Export to ONNX
    onnx_output_file = "./models/"
    # create folder if it doesn't exist, if exists, do nothing
    onnx_output_file += config["model_path"].split('/')[-1].split('.')[0]
    export_to_onnx(model, dummy_input, onnx_output_file + ".onnx", opset_version=config["opset_version"])

    if config["simplify_onnx"]:
        print("🔧 Simplifying ONNX file...")
        # Simplify ONNX file
        onnx_model = onnx.load(onnx_output_file + ".onnx")
        onnx.checker.check_model(onnx_model)
        onnx_model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_model_simp, onnx_output_file + "_simplified.onnx")

    if config["optimize_onnx"] and config["simplify_onnx"]:
        print("🔧 Optimizing ONNX file...")
        # Optimize ONNX file
        onnx_model = onnx.load(onnx_output_file + "_simplified.onnx")
        passes = ["fuse_bn_into_conv", "fuse_add_bias_into_conv", ]
        onnx_model_opt = optimize(onnx_model, passes)
        onnx.save(onnx_model_opt, onnx_output_file + "_simplified" + "_optimized.onnx")

    print("✅ Model exported successfully! 🎉")