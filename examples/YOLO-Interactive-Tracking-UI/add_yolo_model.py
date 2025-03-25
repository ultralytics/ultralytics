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

The export is performed using the built-in export() method of the Ultralytics YOLO class.
All downloaded and exported files are stored in the 'yolo' folder (which you can ignore in Git).

Usage examples:
    python add_yolo_model.py --model_name yolov5s.pt
    python add_yolo_model.py --model_name yolo11n.pt

Dependencies:
    - Python 3.x
    - torch (install via: pip install torch) – used to leverage YOLO’s built-in downloader for supported versions
    - requests (install via: pip install requests) – used for URL-based downloads
    - ultralytics (install via: pip install ultralytics) – provides the YOLO class with the export() method for NCNN conversion

Author:
-------
Alireza Ghaderi  <p30planets@gmail.com>
📅 March 2025  
🔗 LinkedIn: https://www.linkedin.com/in/alireza787b/

License & Disclaimer:
---------------------
This project is provided for **educational and demonstration purposes** only.
The author takes **no responsibility for improper use** or deployment in production systems.
Use at your own discretion. Contributions are welcome!

"""

import argparse
import os
import sys
import shutil
import torch
import requests

def download_model(model_url: str, destination: str) -> bool:
    """
    Downloads a YOLO model from the given URL to the specified destination.
    
    Args:
        model_url (str): The URL from which to download the model.
        destination (str): The local file path where the model will be saved.
        
    Returns:
        bool: True if the download was successful, False otherwise.
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
    Downloads a YOLO model using the built-in downloader via torch.hub from ultralytics/yolov5.
    Loads the model (which triggers the download if necessary) and then copies the cached file to the 'yolo' folder.
    
    Args:
        model_name (str): The name of the YOLO model file (e.g., yolov5s.pt).
        destination (str): The local file path where the model will be saved.
        
    Returns:
        bool: True if the download and copy were successful, False otherwise.
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
    Determines the download method based on the model name.
    For models starting with 'yolov5', uses the built-in downloader.
    For other versions (e.g., 'yolo11n'), prompts the user for a download URL.
    
    Args:
        model_name (str): The name of the YOLO model file.
        destination (str): The local file path where the model will be saved.
        
    Returns:
        bool: True if the download was successful, False otherwise.
    """
    if model_name.lower().startswith("yolov5"):
        return download_model_via_yolo(model_name, destination)
    else:
        print(f"\n[INFO] For model '{model_name}', please provide the download URL.")
        print("       (Typically, this can be obtained from https://github.com/ultralytics/assets/releases/)")
        model_url = input("Enter the download URL: ").strip()
        if not model_url:
            print("[ERROR] No URL provided. Cannot download the model.")
            return False
        return download_model(model_url, destination)

def export_model_to_ncnn(model_path: str) -> bool:
    """
    Exports the YOLO model to NCNN format using the built-in export() method provided by the Ultralytics YOLO class.
    The exported files are saved in a folder adjacent to the model file (e.g., 'yolo11n_ncnn_model').
    
    Args:
        model_path (str): Path to the YOLO model file (e.g., .pt file).
        
    Returns:
        bool: True if the export succeeds, False otherwise.
    """
    try:
        from ultralytics import YOLO
        import ultralytics.nn.modules.block as block
        # Workaround: if 'C3k2' is missing, alias it to 'C3'
        if not hasattr(block, "C3k2"):
            print("[INFO] Registering missing attribute 'C3k2' as an alias for 'C3'.")
            block.C3k2 = block.C3
        print("\n[INFO] Exporting model to NCNN format using model.export(format='ncnn')...")
        model = YOLO(model_path)
        export_results = model.export(format="ncnn")
        print(f"[INFO] Export successful. Export details:\n{export_results}")
        return True
    except Exception as e:
        print(f"[ERROR] Exception during model export: {e}")
        return False

def main():
    """
    Main function to parse command-line arguments, check for model file existence in the 'yolo' folder,
    download the model if necessary using the appropriate method, export it to NCNN format,
    and report results to the user.
    """
    parser = argparse.ArgumentParser(description="Add and export a YOLO model to NCNN format, saving all files in the 'yolo' folder.")
    parser.add_argument("--model_name", type=str, help="Name of the YOLO model file (e.g., yolov5s.pt or yolo11n.pt).")
    args = parser.parse_args()

    if not args.model_name:
        model_name = input("\nPlease enter the YOLO model file name (e.g., yolov5s.pt or yolo11n.pt): ").strip()
    else:
        model_name = args.model_name.strip()

    # Use the directory of the script as the main directory.
    main_dir = os.path.dirname(os.path.abspath(__file__))
    # The 'yolo' folder is a subfolder in the main directory.
    yolo_folder = os.path.join(main_dir, "yolo")
    # Ensure the yolo folder exists
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)
    print(f"\n[INFO] Yolo folder is set to: '{yolo_folder}'")
    
    # Full path for the model file in the yolo folder
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
    
    # Export the model to NCNN format using model.export()
    if export_model_to_ncnn(model_path):
        print(f"\n[INFO] The model has been exported to NCNN format. Check the exported files in the 'yolo' folder.")
    else:
        print("[ERROR] Model export failed. Exiting.")
        sys.exit(1)

    print("\n[INFO] All steps completed successfully.")

if __name__ == "__main__":
    main()
