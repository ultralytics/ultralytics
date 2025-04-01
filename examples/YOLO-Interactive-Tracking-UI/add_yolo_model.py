#!/usr/bin/env python3
"""
add_yolo_model.py

This script manages YOLO model files by allowing the user to specify the name of a YOLO model.
It checks if the specified model file exists in the 'yolo' subfolder (located in the main directory):
  - If it exists, the script exports the model to NCNN format using YOLO's built-in export().
  - If it does not exist, the script downloads the model using an appropriate method:
       - For models starting with "yolov5", it uses YOLO's built-in downloader via torch.hub.
       - For other versions (e.g., "yolo11n"), it asks the user to provide a download URL.
    Then it exports the model.

IMPORTANT SECURITY NOTE:
-------------------------
Due to a change in PyTorch 2.6, torch.load now defaults to weights_only=True in order to improve safety by
restricting the types that can be unpickled. However, this can cause errors when loading checkpoints
that contain custom or standard classes (e.g., torch.nn modules) that are not allowlisted.

To work around this, this script monkey-patches torch.load to force weights_only=False globally.
This bypasses the safety mechanism, so it should ONLY be used if you fully trust the source of your model file.

Usage examples:
    python add_yolo_model.py --model_name yolov5s.pt
    python add_yolo_model.py --model_name yolo11n.pt

Dependencies:
    - Python 3.x
    - torch (pip install torch)
    - requests (pip install requests)
    - ultralytics (pip install ultralytics)

Author:
-------
Alireza Ghaderi  <p30planets@gmail.com>
📅 March 2025
🔗 LinkedIn: https://www.linkedin.com/in/alireza787b/

License & Disclaimer:
---------------------
This project is provided for educational and demonstration purposes only.
The author takes no responsibility for improper use or deployment in production systems.
Use at your own discretion. Contributions are welcome!
"""

import argparse
import os
import sys
import shutil
import torch
import requests

# -----------------------------------------------------------------------------
# Monkey-patch torch.load globally to force weights_only=False.
#
# Explanation:
# PyTorch 2.6+ now defaults to weights_only=True when calling torch.load, which restricts the
# unpickling process and prevents arbitrary code execution. However, this safety mechanism may
# cause errors when loading checkpoints that contain standard or custom classes.
#
# This monkey-patch forces torch.load to always use weights_only=False, bypassing these restrictions.
#
# WARNING: This bypasses PyTorch's security mechanism. Only use this patch if you trust the
# source of your checkpoint file.
# -----------------------------------------------------------------------------
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
# -----------------------------------------------------------------------------

def download_model(model_url: str, destination: str) -> bool:
    """
    Downloads a YOLO model from the given URL to the specified destination.
    
    Args:
        model_url (str): URL to download the model from.
        destination (str): Local file path to save the model.
        
    Returns:
        bool: True if download succeeds, False otherwise.
    """
    try:
        print(f"\n[INFO] Downloading model from {model_url} ...")
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"[INFO] Model successfully downloaded and saved to:\n"
                  f"       File Name: {os.path.basename(destination)}\n"
                  f"       Full Path: {os.path.abspath(destination)}")
            return True
        else:
            print(f"[ERROR] Failed to download model. HTTP status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception during download: {e}")
        return False

def download_model_via_yolo(model_name: str, destination: str) -> bool:
    """
    Downloads a YOLO model using YOLO's built-in downloader via torch.hub from ultralytics/yolov5.
    This method loads the model (triggering the download) and copies the cached file to the target destination.
    
    Args:
        model_name (str): Name of the model file (e.g., "yolov5s.pt").
        destination (str): Local file path where the model will be saved.
        
    Returns:
        bool: True if download and copy succeed, False otherwise.
    """
    try:
        print(f"\n[INFO] Downloading model '{model_name}' using YOLO's built-in downloader...")
        model_type = os.path.splitext(model_name)[0]  # e.g., "yolov5s" from "yolov5s.pt"
        _ = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
        
        # Locate the cached model file in torch hub's directory.
        hub_dir = torch.hub.get_dir()
        cached_model_path = os.path.join(hub_dir, 'ultralytics_yolov5_master', model_name)
        
        if os.path.exists(cached_model_path):
            shutil.copy(cached_model_path, destination)
            print(f"[INFO] Model file copied from cache to:\n"
                  f"       File Name: {os.path.basename(destination)}\n"
                  f"       Full Path: {os.path.abspath(destination)}")
            return True
        else:
            print(f"[ERROR] Could not locate the downloaded model in cache at '{cached_model_path}'.")
            return False
    except Exception as e:
        print(f"[ERROR] Exception during model download via YOLO's built-in downloader: {e}")
        return False

def download_model_generic(model_name: str, destination: str) -> bool:
    """
    Determines the appropriate download method based on the model name.
    Uses the built-in downloader for "yolov5" models; otherwise, prompts the user for a URL.
    
    Args:
        model_name (str): Name of the model file.
        destination (str): Local file path to save the model.
        
    Returns:
        bool: True if the model is downloaded successfully, False otherwise.
    """
    if model_name.lower().startswith("yolov5"):
        return download_model_via_yolo(model_name, destination)
    else:
        print(f"\n[INFO] For model '{model_name}', please provide the download URL.")
        print("       (Typically, available from https://github.com/ultralytics/assets/releases/)")
        model_url = input("Enter the download URL: ").strip()
        if not model_url:
            print("[ERROR] No URL provided. Cannot download the model.")
            return False
        return download_model(model_url, destination)

def export_model_to_ncnn(model_path: str) -> bool:
    """
    Exports the YOLO model to NCNN format using the built-in export() method of the Ultralytics YOLO class.
    The exported files are saved in a folder adjacent to the model file.
    
    Args:
        model_path (str): Path to the YOLO model file (.pt file).
        
    Returns:
        bool: True if export succeeds, False otherwise.
    """
    try:
        from ultralytics import YOLO
        print("\n[INFO] Exporting model to NCNN format using model.export(format='ncnn')...")
        # Instantiate the YOLO model; the monkey-patched torch.load ensures weights_only=False.
        model = YOLO(model_path)
        export_results = model.export(format="ncnn")
        print(f"[INFO] Export successful. Export details:\n{export_results}")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
    except Exception as e:
        print(f"[ERROR] Exception during model export: {e}")
        if "not found" in str(e):
            print("[INFO] Ensure the model file exists and the path is correct.")
        elif "export" in str(e):
            print("[INFO] Check if the model supports export to NCNN.")
    return False

def main():
    """
    Main function to parse command-line arguments, check for model file existence in the 'yolo' folder,
    download the model if necessary using the appropriate method, export it to NCNN format,
    and report results to the user.
    """
    parser = argparse.ArgumentParser(
        description="Add and export a YOLO model to NCNN format, saving all files in the 'yolo' folder."
    )
    parser.add_argument("--model_name", type=str, help="Name of the YOLO model file (e.g., yolov5s.pt or yolo11n.pt).")
    args = parser.parse_args()

    model_name = args.model_name.strip() if args.model_name else input(
        "\nPlease enter the YOLO model file name (e.g., yolov5s.pt or yolo11n.pt): "
    ).strip()

    # Determine the main directory (the directory where this script is located)
    main_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the 'yolo' subfolder for storing model files.
    yolo_folder = os.path.join(main_dir, "yolo")
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)
    print(f"\n[INFO] Yolo folder is set to: '{yolo_folder}'")
    
    # Full path for the model file within the yolo folder.
    model_path = os.path.join(yolo_folder, model_name)
    
    print(f"\n[INFO] Checking for model file '{model_name}' in '{yolo_folder}' ...")
    if os.path.exists(model_path):
        print(f"[INFO] Model file '{model_name}' found at:\n       {os.path.abspath(model_path)}")
    else:
        print(f"[WARNING] Model file '{model_name}' not found in the yolo folder.")
        user_input = input("Do you want to download the model? (y/n): ").strip().lower()
        if user_input != 'y':
            print("[ERROR] Model download aborted by user. Exiting.")
            sys.exit(1)
        if not download_model_generic(model_name, model_path):
            print("[ERROR] Model download failed. Exiting.")
            sys.exit(1)

    # Report final model file details.
    print(f"\n[INFO] Model file ready:\n       File Name: {os.path.basename(model_path)}\n       Full Path: {os.path.abspath(model_path)}")
    
    # Export the model to NCNN format using YOLO's export method.
    if export_model_to_ncnn(model_path):
        print(f"\n[INFO] The model has been exported to NCNN format. Check the exported files in the 'yolo' folder.")
    else:
        print("[ERROR] Model export failed. Exiting.")
        sys.exit(1)

    print("\n[INFO] All steps completed successfully.")

if __name__ == "__main__":
    main()
