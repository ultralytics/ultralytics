#!/usr/bin/env python3
"""
Visualize all grounding bboxes and labels for specified image
Support Mixed Grounding and Flickr datasets
Usage: python visual_grounding_img.py --image_path path/to/image.jpg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
import os
from pathlib import Path
import re

def load_cache_data(cache_path):
    """Load cache data"""
    try:
        data = np.load(cache_path, allow_pickle=True)
        print(f"✅ Successfully loaded cache file: {cache_path}")
        print(f"📊 Total samples: {data.size}")
        return data
    except Exception as e:
        print(f"❌ Cannot load cache file {cache_path}: {e}")
        return None

def find_image_data(data, target_image_name):
    """Find data of specified image in cache data"""
    found_items = []

    print(f"🔍 Searching for image: {target_image_name}")

    for i, item in enumerate(data):
        if isinstance(item, dict):
            im_file = item.get('im_file', '')
            if im_file:
                # Extract filename
                image_name = str(im_file).split('/')[-1]
                if image_name == target_image_name:
                    found_items.append(item)
                    print(f"✅ Found match: index {i}")
                    # Show basic info of this item
                    texts = item.get('texts', [])
                    bboxes = item.get('bboxes', np.array([]))
                    print(f"   📝 Text count: {len(texts)}")
                    print(f"   📦 Bbox count: {bboxes.shape[0] if hasattr(bboxes, 'shape') else 0}")
                    if len(texts) > 0:
                        print(f"   📋 First 5 texts: {[t[0] if isinstance(t, list) else str(t) for t in texts[:5]]}")
    
    return found_items

def determine_dataset_type(image_path):
    """Determine dataset type based on image path"""
    image_path_lower = image_path.lower()
    
    if 'flickr' in image_path_lower:
        return 'flickr'
    elif 'gqa' in image_path_lower or 'mixed' in image_path_lower:
        return 'mixed'
    else:
        # Default try both datasets
        return 'auto'

def get_cache_paths(dataset_type):
    """Get cache paths for corresponding dataset"""
    cache_paths = []
    
    if dataset_type == 'mixed' or dataset_type == 'auto':
        cache_paths.append("../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache")
    if dataset_type == 'flickr' or dataset_type == 'auto':
        cache_paths.append("/Users/louis/workspace/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache")
    
    return cache_paths

def normalize_coordinates(x_center, y_center, width, height, img_width, img_height):
    """Convert normalized coordinates to pixel coordinates"""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    return x1, y1, x2 - x1, y2 - y1  # Return (x, y, width, height) format

def generate_colors(n):
    """Generate n different colors"""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    if n <= 10:
        # Use predefined colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 
                 'brown', 'pink', 'gray', 'olive', 'cyan']
        return colors[:n]
    else:
        # Use colormap to generate more colors
        cmap = cm.get_cmap('tab20')
        return [cmap(i / n) for i in range(n)]

def visualize_grounding_image(image_path):
    """Visualize grounding information for specified image"""
    print("🎨 Visualize Grounding Image")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file does not exist: {image_path}")
        return
    
    # Get image name
    image_name = os.path.basename(image_path)
    print(f"📷 Target image: {image_name}")
    
    # Determine dataset type
    dataset_type = determine_dataset_type(image_path)
    print(f"🔍 Dataset type: {dataset_type}")
    
    # Get cache paths
    cache_paths = get_cache_paths(dataset_type)
    assert len(cache_paths) ==1, " number of cache paths should be 1"
    print(f"📁 Search cache files: {cache_paths}")
    
    # Search image data
    found_data = []
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            data = load_cache_data(cache_path)
            if data is not None:
                items = find_image_data(data, image_name)
                if items:
                    found_data.extend(items)
                    print(f"📊 Found {len(items)} annotations in {cache_path}")
    
    if not found_data:
        print(f"❌ No annotation data found for image {image_name} in cache files")
        print("💡 Please check if image name is correct or if image is in training set")
        return
    
    print(f"✅ Total found {len(found_data)} annotation items")
    
    # Load image
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        print(f"📐 Image size: {img.size}")
    except Exception as e:
        print(f"❌ Cannot open image: {e}")
        return
    
    img_width, img_height = img.size
    
    print(f"✅ Total found {len(found_data)} annotation items")
    
    if len(found_data) == 0:
        print("❌ No annotation data found")
        return
    
    # Load image
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        print(f"📐 Image size: {img.size}")
    except Exception as e:
        print(f"❌ Cannot open image: {e}")
        return
    
    img_width, img_height = img.size
    
    # Create visualization for each annotation item
    print(f"\n🎯 Creating visualizations for {len(found_data)} annotation items separately...")
    
    # Calculate tile layout (as close to square as possible)
    num_annotations = len(found_data)
    cols = int(np.ceil(np.sqrt(num_annotations)))
    rows = int(np.ceil(num_annotations / cols))
    
    print(f"📐 Tile layout: {rows} rows x {cols} columns")
    
    # Create large figure
    fig_width = cols * 6  # Each subplot 6 inches wide
    fig_height = rows * 5  # Each subplot 5 inches high
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create subplot for each annotation item
    for idx, item in enumerate(found_data):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        texts = item.get('texts', [])
        bboxes = item.get('bboxes', np.array([]))
        
        if len(texts) == 0 or bboxes.size == 0:
            # Empty subplot
            ax.text(0.5, 0.5, f'Annotation {idx+1}\nNo valid data', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Show original image
        ax.imshow(img)
        
        # Get colors
        num_boxes = min(len(texts), bboxes.shape[0] if hasattr(bboxes, 'shape') else 0)
        colors = generate_colors(num_boxes)
        
        print(f"\nAnnotation {idx+1}: {num_boxes} bboxes")
        
        # Draw each bbox
        for i in range(num_boxes):
            bbox = bboxes[i]
            text_group = texts[i]
            
            # Get text label
            if isinstance(text_group, list) and len(text_group) > 0:
                label = text_group[0]
            else:
                label = str(text_group)
            
            # Convert coordinates
            x, y, w, h = normalize_coordinates(
                bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height
            )
            
            # Draw bbox
            color = colors[i % len(colors)]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add text label (smaller font)
            ax.text(x, y - 3, f"{i+1}: {label[:20]}", 
                   color=color, fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            print(f"   {i+1:2d}. '{label[:30]}'")
        
        # Set subplot title
        ax.set_title(f"Annotation {idx+1} ({num_boxes} bbox)", fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide extra subplots
    for idx in range(num_annotations, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    # Set main title
    fig.suptitle(f"Grounding Annotations: {image_name}\n"
                f"Total {len(found_data)} different annotation items", 
                fontsize=16, y=0.98)
    
    # Save tiled image
    output_name = f"grounding_visual_{image_name.split('.')[0]}_all_{len(found_data)}_annotations.png"
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Leave space for title
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n💾 Tiled visualization result saved: {output_name}")
    
    # Show image
    plt.show()
    
    plt.close()  # Release memory
    
    # Output additional statistics
    print(f"\n📊 Annotation statistics:")
    for idx, item in enumerate(found_data):
        texts = item.get('texts', [])
        bboxes = item.get('bboxes', np.array([]))
        num_boxes = min(len(texts), bboxes.shape[0] if hasattr(bboxes, 'shape') else 0)
        all_texts = [t[0] if isinstance(t, list) else str(t) for t in texts[:num_boxes]]
        print(f"  Annotation {idx+1}: {num_boxes} bbox - {all_texts}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize grounding image bboxes and labels')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Image file path')
    
    args = parser.parse_args()
    
    # Visualize image
    visualize_grounding_image(args.image_path)

if __name__ == "__main__":
    main()
