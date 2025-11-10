#!/usr/bin/env python3
"""
Test script for RKNN export with existing parameters.
"""

import os
import sys
from pathlib import Path

# Add the ultralytics_rknn directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def test_rknn_export():
    """Test RKNN export with existing parameters."""
    # Create a simple YOLO model
    model = YOLO("yolov8n.pt")
    
    # Test 1: Basic RKNN export
    print("Test 1: Basic RKNN export")
    try:
        model.export(format="rknn", name="rk3588")
        print("✓ Basic RKNN export successful")
    except Exception as e:
        print(f"✗ Basic RKNN export failed: {e}")
    
    # Test 2: RKNN export with INT8 quantization
    print("\nTest 2: RKNN export with INT8 quantization")
    try:
        model.export(format="rknn", name="rk3588", int8=True, data="coco8.yaml")
        print("✓ RKNN export with INT8 quantization successful")
    except Exception as e:
        print(f"✗ RKNN export with INT8 quantization failed: {e}")

if __name__ == "__main__":
    test_rknn_export()