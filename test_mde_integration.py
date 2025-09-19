#!/usr/bin/env python3
"""
Test script to verify YOLO11 MDE integration without circular imports.
"""

import sys
import os
import torch

def test_model_loading():
    """Test loading the YOLO11 MDE model configuration."""
    print("🧪 Testing YOLO11 MDE model loading...")
    
    try:
        # Test direct model building without full YOLO import
        from ultralytics.nn.tasks import parse_model
        from ultralytics.utils.torch_utils import model_info
        
        # Load model configuration
        model_cfg = '/root/ultralytics/ultralytics/cfg/models/11/yolo11-mde.yaml'
        
        # Parse model
        model, yaml_info = parse_model(model_cfg, ch=3)
        
        print(f"   ✅ Model parsed successfully!")
        print(f"   📊 Model info:")
        model_info(model, verbose=False)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"   🔄 Forward pass successful!")
        print(f"   📐 Input shape: {dummy_input.shape}")
        print(f"   📤 Output type: {type(outputs)}")
        
        if isinstance(outputs, list):
            print(f"   📊 Number of output layers: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"      Layer {i}: {out.shape}")
        
        # Check if Detect_MDE head is present
        last_layer = model[-1]
        print(f"   🎯 Last layer: {last_layer.__class__.__name__}")
        
        if hasattr(last_layer, 'cv_depth'):
            print("   ✅ Detect_MDE head detected with depth branch!")
        else:
            print("   ❌ Detect_MDE head not found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_config():
    """Test the dataset configuration."""
    print("\n🧪 Testing dataset configuration...")
    
    try:
        import yaml
        
        # Load dataset configuration
        dataset_cfg = '/root/ultralytics/ultralytics/cfg/datasets/kitti_mde.yaml'
        
        with open(dataset_cfg, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"   ✅ Dataset config loaded successfully!")
        print(f"   📁 Dataset path: {config['path']}")
        print(f"   🏷️  Number of classes: {config['nc']}")
        print(f"   📝 Class names: {config['names']}")
        
        # Check if dataset exists
        dataset_path = config['path']
        if os.path.exists(dataset_path):
            print(f"   ✅ Dataset directory exists!")
            
            # Check for images and labels
            images_path = os.path.join(dataset_path, 'images')
            labels_path = os.path.join(dataset_path, 'labels')
            
            if os.path.exists(images_path) and os.path.exists(labels_path):
                import glob
                num_images = len(glob.glob(os.path.join(images_path, '*.png')))
                num_labels = len(glob.glob(os.path.join(labels_path, '*.txt')))
                print(f"   📊 Images: {num_images}, Labels: {num_labels}")
            else:
                print(f"   ❌ Images or labels directory not found!")
                return False
        else:
            print(f"   ❌ Dataset directory not found: {dataset_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script():
    """Test the training script syntax."""
    print("\n🧪 Testing training script...")
    
    try:
        training_script = '/root/ultralytics/examples/train_yolo11_mde.py'
        
        # Check if file exists
        if not os.path.exists(training_script):
            print(f"   ❌ Training script not found: {training_script}")
            return False
        
        # Try to compile the script
        with open(training_script, 'r') as f:
            script_content = f.read()
        
        compile(script_content, training_script, 'exec')
        print(f"   ✅ Training script syntax is valid!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🎯 YOLO11 MDE Integration Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_model_loading()
    success &= test_dataset_config()
    success &= test_training_script()
    
    print("\n" + "=" * 60)
    if success:
        print("🏁 All integration tests passed!")
        print("=" * 60)
        print("\n📋 Ready for training!")
        print("\n🚀 To start training, run:")
        print("   cd /root/ultralytics")
        print("   python examples/train_yolo11_mde.py train")
        print("\n🔍 To validate a trained model:")
        print("   python examples/train_yolo11_mde.py val")
        print("\n🔮 To run inference:")
        print("   python examples/train_yolo11_mde.py predict")
    else:
        print("❌ Some integration tests failed!")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
