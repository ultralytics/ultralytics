#!/usr/bin/env python3
"""
Simple Tennis Frame Visualizer

A standalone script to visualize tennis ball detection on the frame sequence
using the trained model from train_tennis_ball.py configuration.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

# Try to import ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.nn.modules.motion_utils import FlexibleMotionMaskGenerator, MotionConfig
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("❌ Ultralytics not available. Please install: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False


def find_best_model():
    """Find the best available model."""
    model_candidates = [
        "/root/ultralytics/tennis_ball_training/yolo11_tennis_pose_4channel_multi_gpu/weights/best.pt",
        "/root/ultralytics/tennis_ball_training/yolo11_tennis_pose_4channel_multi_gpu/weights/last.pt",
        "/root/autodl-tmp/runs/pose/train15/weights/best.pt",
        "/root/autodl-tmp/runs/pose/train14/weights/best.pt", 
        "/root/ultralytics/yolo11n-pose.pt"
    ]
    
    for model_path in model_candidates:
        if Path(model_path).exists():
            return model_path
    return None


def load_and_prepare_model(model_path: str):
    """Load model with appropriate configuration."""
    if not ULTRALYTICS_AVAILABLE:
        return None
    
    try:
        print(f"🤖 Loading model: {model_path}")
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"❌ Failed to load model {model_path}: {e}")
        return None


def create_motion_generator(window_size=5):
    """Create motion mask generator with proper configuration."""
    config = MotionConfig(
        pixel_threshold=15,
        delta=1,
        window_size=window_size,
        adaptive_threshold=True,
        min_motion_pixels=100,
        max_motion_pixels=50000,
        morphological_ops=True
    )
    return FlexibleMotionMaskGenerator(config)


def annotate_frame(frame, results, frame_name, model):
    """Add detection annotations to frame."""
    annotated = frame.copy()
    
    # Add frame name
    cv2.putText(annotated, f"Frame: {frame_name}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if results and len(results) > 0:
        result = results[0]  # First result
        
        # Draw bounding boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Ball {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 5), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw keypoints if available
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            for kpts in keypoints:
                for i, (x, y, conf) in enumerate(kpts):
                    if conf > 0.5:
                        cv2.circle(annotated, (int(x), int(y)), 4, (255, 0, 0), -1)
                        cv2.putText(annotated, str(i), (int(x) + 5, int(y) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    return annotated


def visualize_tennis_frames(
    frame_dir="/root/ultralytics/frame_0000024624_zqw",
    output_dir="/root/ultralytics/simple_tennis_results",
    max_frames=50,
    show_motion=True,
    window_size=5
):
    """Main visualization function."""
    
    print("🎾 Simple Tennis Frame Visualizer")
    print("=" * 40)
    
    # Setup paths
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get frame files
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    print(f"📁 Frame directory: {frame_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🖼️  Processing {len(frame_files)} frames")
    print(f"🪟 Temporal window size: {window_size}")
    
    # Find and load model
    model_path = find_best_model()
    if not model_path:
        print("❌ No model found!")
        return
    
    model = load_and_prepare_model(model_path)
    if not model:
        print("❌ Failed to load model!")
        return
    
    print(f"✅ Model loaded: {Path(model_path).name}")
    
    # Create motion mask generator
    if ULTRALYTICS_AVAILABLE:
        motion_generator = create_motion_generator(window_size)
        print(f"✅ Motion generator created with window size {window_size}")
    else:
        motion_generator = None
        print("⚠️  Motion generator not available")
    
    # Process frames with temporal window
    detection_count = 0
    frame_window = []  # Temporal window of frames
    
    print(f"\n🚀 Processing frames...")
    
    for i, frame_path in enumerate(tqdm(frame_files, desc="Processing")):
        # Load frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Maintain frame window
        frame_window.append(frame.copy())
        if len(frame_window) > window_size:
            frame_window.pop(0)  # Remove oldest frame
        
        # Prepare input for model (handle 4-channel requirement)
        try:
            # Try with 3-channel input first
            results = model(frame, conf=0.25, verbose=False)
        except Exception as e:
            if "expected input" in str(e) and "4 channels" in str(e):
                # Model expects 4 channels - use enhanced motion mask generation
                if motion_generator and len(frame_window) >= 2:
                    # Generate cache key for this frame
                    cache_key = f"{frame_path.stem}_w{window_size}"
                    
                    # Use FlexibleMotionMaskGenerator for enhanced motion detection
                    motion_mask = motion_generator.generate_enhanced(frame_window, cache_key)
                else:
                    # Fallback to simple zero motion mask
                    motion_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                
                # Add motion mask as 4th channel
                motion_channel = motion_mask[..., np.newaxis]
                model_input = np.concatenate([frame, motion_channel], axis=-1)
                
                try:
                    # Try inference again with 4-channel input
                    results = model(model_input, conf=0.25, verbose=False)
                except Exception as e2:
                    print(f"❌ Error processing {frame_path.name}: {e2}")
                    continue
            else:
                print(f"❌ Error processing {frame_path.name}: {e}")
                continue
        
        # Count detections
        if results and results[0].boxes is not None:
            detection_count += len(results[0].boxes)
        
        # Annotate frame
        annotated = annotate_frame(frame, results, frame_path.stem, model)
        
        # Add motion visualization if requested
        if show_motion and motion_generator and len(frame_window) > 1:
            cache_key = f"{frame_path.stem}_w{window_size}_viz"
            motion_mask = motion_generator.generate_enhanced(frame_window, cache_key)
            # Create motion overlay (red channel)
            motion_overlay = np.zeros_like(frame)
            motion_overlay[:, :, 2] = motion_mask  # Red channel
            # Blend with original
            annotated = cv2.addWeighted(annotated, 0.8, motion_overlay, 0.2, 0)
        
        # Save annotated frame
        output_path = output_dir / f"result_{frame_path.name}"
        cv2.imwrite(str(output_path), annotated)
    
    print(f"\n✅ Processing completed!")
    print(f"🎯 Total detections: {detection_count}")
    print(f"📂 Results saved to: {output_dir}")
    print(f"   📝 Annotated frames: result_*.jpg")
    
    # Create a simple summary
    print(f"\n📊 Summary:")
    print(f"   Frames processed: {len(frame_files)}")
    print(f"   Average detections per frame: {detection_count/len(frame_files):.2f}")
    print(f"   Temporal window size: {window_size} frames")
    if motion_generator:
        print(f"   Motion detection: Enhanced with adaptive thresholding")
    else:
        print(f"   Motion detection: Basic fallback")
    
    return output_dir


def create_simple_video(image_dir, output_path=None, fps=15):
    """Create video from processed images."""
    if output_path is None:
        output_path = Path(image_dir) / "tennis_summary.mp4"
    
    image_files = sorted(Path(image_dir).glob("result_*.jpg"))
    if not image_files:
        print("❌ No result images found")
        return None
    
    # Get video dimensions from first image
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print("❌ Could not read first image")
        return None
    h, w = first_img.shape[:2]
    
    # Create video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    print(f"🎬 Creating video: {output_path}")
    
    for img_path in tqdm(image_files, desc="Creating video"):
        img = cv2.imread(str(img_path))
        if img is not None:
            video_writer.write(img)
    
    video_writer.release()
    print(f"✅ Video created: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple tennis frame visualizer")
    parser.add_argument("--frames", default="/root/autodl-tmp/test_clip",
                       help="Frame directory")
    parser.add_argument("--output", default="/root/auto-tmp/resutls",
                       help="Output directory")
    parser.add_argument("--max-frames", type=int, default=0,
                       help="Maximum frames to process (0 for all)")
    parser.add_argument("--no-motion", action="store_true",
                       help="Disable motion visualization")
    parser.add_argument("--video", action="store_true",
                       help="Create summary video")
    
    args = parser.parse_args()
    
    if not ULTRALYTICS_AVAILABLE:
        exit(1)
    
    # Run visualization
    max_frames = args.max_frames if args.max_frames > 0 else None
    output_dir = visualize_tennis_frames(
        frame_dir=args.frames,
        output_dir=args.output, 
        max_frames=max_frames or 1000,  # Use 1000 as a large default
        show_motion=not args.no_motion
    )
    
    # Create video if requested
    if args.video and output_dir:
        create_simple_video(output_dir)