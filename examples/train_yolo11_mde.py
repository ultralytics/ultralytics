#!/usr/bin/env python3
"""
YOLO11 MDE (Monocular Depth Estimation) Training Script

This script demonstrates how to train a YOLO11 model with depth estimation
on the KITTI dataset.

Usage:
    python examples/train_yolo11_mde.py
"""

from ultralytics import YOLO
import torch
from ultralytics.utils.metrics import MDEMetrics


def train_yolo_mde():
    """Train YOLO11 MDE model on KITTI dataset."""
    
    # Load model with custom MDE head
    model = YOLO('yolo11-mde.yaml')
    
    # Training arguments
    args = {
        'data': 'kitti_mde.yaml',  # Dataset config
        'epochs': 60,
        'imgsz': [384, 1248],  # As per paper
        'batch': 4,
        'device': 0,
        'lr0': 0.0001,
        'optimizer': 'Adam',
        'project': 'yolo11_mde',
        'name': 'kitti_depth',
        'patience': 20,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'val': True,  # Enable validation
        'plots': True,  # Generate training plots
        'verbose': True,  # Verbose output
    }
    
    print("🚀 Starting YOLO11 MDE training...")
    print(f"📊 Dataset: {args['data']}")
    print(f"🔄 Epochs: {args['epochs']}")
    print(f"📐 Image size: {args['imgsz']}")
    print(f"📦 Batch size: {args['batch']}")
    print(f"🎯 Project: {args['project']}/{args['name']}")
    
    # Train the model
    results = model.train(**args)
    
    print("✅ Training completed!")
    print(f"📁 Results saved to: {args['project']}/{args['name']}")
    
    return model, results


def validate_model(model_path=None):
    """Validate the trained MDE model."""
    
    if model_path is None:
        model_path = 'yolo11_mde/kitti_depth/weights/best.pt'
    
    print(f"🔍 Validating model: {model_path}")
    
    # Load trained model
    model = YOLO(model_path)
    
    # Validation arguments
    val_args = {
        'data': 'kitti_mde.yaml',
        'imgsz': [384, 1248],
        'batch': 8,
        'device': 0,
        'plots': True,
        'save_json': True,
        'verbose': True,
    }
    
    # Run validation
    results = model.val(**val_args)
    
    print("✅ Validation completed!")
    return results


def evaluate_mde_model(model_path=None, data_path=None):
    """
    Comprehensive evaluation of MDE model with depth metrics.
    
    Args:
        model_path: Path to trained model
        data_path: Path to validation dataset
    """
    if model_path is None:
        model_path = 'yolo11_mde/kitti_depth/weights/best.pt'
    
    if data_path is None:
        data_path = '/root/autodl-tmp/kitti_yolo_depth'
    
    print(f"🔍 Comprehensive MDE evaluation...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Data: {data_path}")
    
    # Load trained model
    model = YOLO(model_path)
    
    # Create a simple dataloader for evaluation
    # Note: This is a simplified version - in practice you'd use the full YOLO dataloader
    try:
        # Try to use YOLO's built-in validation
        val_results = model.val(
            data='kitti_mde.yaml',
            imgsz=[384, 1248],
            batch=4,
            device=0,
            plots=True,
            save_json=True,
            verbose=True
        )
        
        print("✅ Standard validation completed!")
        
        # Print depth statistics if available
        if hasattr(val_results, 'depth_stats'):
            # Create MDEMetrics instance for depth statistics
            mde_metrics = MDEMetrics()
            mde_metrics.print_depth_statistics(val_results.depth_stats, "Validation Depth")
        
        return val_results
        
    except Exception as e:
        print(f"⚠️ Standard validation failed: {e}")
        print("🔄 Falling back to simple inference evaluation...")
        
        # Fallback: simple inference on a few images
        import os
        import glob
        
        image_files = glob.glob(os.path.join(data_path, 'images', '*.png'))[:10]  # Test on 10 images
        
        depth_predictions = []
        for img_path in image_files:
            try:
                results = predict_with_depth(model, img_path)
                # Extract depth predictions
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        depths = r.boxes.data[..., -1]  # Last channel is depth
                        depth_predictions.extend(depths.cpu().numpy())
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
        
        if depth_predictions:
            depth_tensor = torch.tensor(depth_predictions)
            # Create MDEMetrics instance for depth statistics
            mde_metrics = MDEMetrics()
            mde_metrics.print_depth_statistics(depth_tensor, "Inference Depth")
        
        return None


def predict_with_depth(model, image_path):
    """
    Run inference and get detections with depth.
    
    Args:
        model: Trained YOLO MDE model
        image_path: Path to input image
        
    Returns:
        results: YOLO results with depth predictions
    """
    results = model(image_path)
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            # Extract depth predictions (last channel)
            depths = boxes.data[..., -1]
            
            # Process each detection
            for i, (box, depth) in enumerate(zip(boxes.xyxy, depths)):
                x1, y1, x2, y2 = box
                cls = boxes.cls[i]
                conf = boxes.conf[i]
                
                print(f"Class: {model.names[int(cls)]}, "
                      f"Confidence: {conf:.2f}, "
                      f"Depth: {depth:.2f}m")
    
    return results


def predict_with_mde(model_path=None, source=None):
    """Run inference with the trained MDE model."""
    
    if model_path is None:
        model_path = 'yolo11_mde/kitti_depth/weights/best.pt'
    
    if source is None:
        source = '/root/autodl-tmp/kitti_yolo_depth/images'
    
    print(f"🔮 Running inference with model: {model_path}")
    print(f"📸 Source: {source}")
    
    # Load trained model
    model = YOLO(model_path)
    
    # Prediction arguments
    pred_args = {
        'source': source,
        'imgsz': [384, 1248],
        'conf': 0.25,
        'iou': 0.45,
        'device': 0,
        'save': True,
        'save_txt': True,
        'save_conf': True,
        'verbose': True,
    }
    
    # Run prediction
    results = model.predict(**pred_args)
    
    print("✅ Inference completed!")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'train':
            # Train the model
            model, results = train_yolo_mde()
            
        elif mode == 'val':
            # Validate the model
            model_path = sys.argv[2] if len(sys.argv) > 2 else None
            results = validate_model(model_path)
            
        elif mode == 'eval':
            # Comprehensive MDE evaluation
            model_path = sys.argv[2] if len(sys.argv) > 2 else None
            data_path = sys.argv[3] if len(sys.argv) > 3 else None
            results = evaluate_mde_model(model_path, data_path)
            
        elif mode == 'predict':
            # Run inference
            model_path = sys.argv[2] if len(sys.argv) > 2 else None
            source = sys.argv[3] if len(sys.argv) > 3 else None
            results = predict_with_mde(model_path, source)
            
        else:
            print("❌ Unknown mode. Use: train, val, eval, or predict")
            sys.exit(1)
    else:
        # Default: train the model
        print("🎯 Starting YOLO11 MDE training (default mode)")
        print("💡 Usage: python train_yolo11_mde.py [train|val|eval|predict] [model_path] [source]")
        print()
        print("📋 Available modes:")
        print("   train  - Train the YOLO11 MDE model")
        print("   val    - Validate the trained model")
        print("   eval   - Comprehensive MDE evaluation with depth metrics")
        print("   predict - Run inference on images")
        print()
        
        model, results = train_yolo_mde()
