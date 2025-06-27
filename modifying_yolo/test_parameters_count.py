import yaml
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel

def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model_for_scale(yaml_path, scale_key):
    """
    Loads a scalable YAML, sets a specific scale, and builds the corresponding model.

    Args:
        yaml_path (str): Path to the scalable yolov8-enhanced.yaml file.
        scale_key (str): The scale to use ('n', 's', 'm', 'l', 'x').

    Returns:
        (torch.nn.Module): The constructed PyTorch model.
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # This is the key step: we are telling the parser which scale to use.
    # The parser will then use the corresponding multipliers from the 'scales' dict.
    cfg['scale'] = scale_key
    
    # The DetectionModel class from tasks.py is responsible for parsing the
    # dictionary and building the model.
    # We pass ch=3 for standard RGB input channels.
    # We pass verbose=False to keep the output clean.
    model = DetectionModel(cfg, ch=3, verbose=False)
    
    return model

def main():
    """
    Main function to build the 'n' and 'x' models and compare their parameters.
    """
    yaml_file = Path('../ultralytics/cfg/models/v8/yolov8-enhanced.yaml') # Assumes the scalable YAML is in the same directory

    if not yaml_file.exists():
        print(f"Error: Could not find '{yaml_file}'. Please make sure the scalable YAML file is present.")
        return

    print("Building and counting parameters for yolov8n-enhanced...")
    try:
        model_n = build_model_for_scale(yaml_file, 'n')
        params_n = count_parameters(model_n)
        print(f"✅ Successfully built yolov8n-enhanced.")
        print(f"   Total Parameters: {params_n:,}\n")
    except Exception as e:
        print(f"❌ Failed to build 'n' scale model. Error: {e}")
        return

    print("Building and counting parameters for yolov8x-enhanced...")
    try:
        model_x = build_model_for_scale(yaml_file, 'x')
        params_x = count_parameters(model_x)
        print(f"✅ Successfully built yolov8x-enhanced.")
        print(f"   Total Parameters: {params_x:,}\n")
    except Exception as e:
        print(f"❌ Failed to build 'x' scale model. Error: {e}")
        return
        
    print("-" * 40)
    print("      Parameter Count Comparison")
    print("-" * 40)
    print(f"yolov8n-enhanced: {params_n:>15,}")
    print(f"yolov8x-enhanced: {params_x:>15,}")
    print(f"Difference:       {params_x - params_n:>15,}")
    print(f"Scale Factor:     {params_x / params_n:15.2f}x")
    print("-" * 40)


if __name__ == "__main__":
    main()
