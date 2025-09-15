#!/usr/bin/env python3
"""
Batch visualize Mixed Grounding bbox overlap analysis results
Organize and save by category to runs/visual/{class_name} directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from pathlib import Path
import re



# data_name="mixed_grounding"
data_name="flickr"



def sanitize_filename(filename):
    """Clean filename, remove unsafe characters"""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.replace(' ', '_')
    filename = filename.strip('.')
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def create_output_dirs():
    """Create output directory structure"""
    if data_name=="mixed_grounding":
        base_dir = Path("../runs/mixed_grounding_visual")
    else:
        base_dir = Path("../runs/flickr_visual")

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def visualize_all_overlaps():
    """Batch visualize all bbox overlap cases"""

    if data_name=="mixed_grounding":
        csv_path = "../runs/mixed_grounding_bbox_overlaps_iou09.csv"
    else:
        csv_path = "../runs/flickr_bbox_overlaps_iou09.csv"

    print("🎨 Batch visualize Mixed Grounding Bbox overlap cases")
    print("=" * 60)
    
    try:
        # Read CSV file
        if not os.path.exists(csv_path):
            print(f"❌ Error: CSV file does not exist: {csv_path}")
            return
        
        print(f"📁 Reading file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print("❌ CSV file is empty")
            return
        
        print(f"✅ Successfully read {len(df)} rows of data")
        
        # Pre-count human-related samples
        human_keywords = [
            'man', 'woman', 'person', 'people', 'guy', 'girl', 'boy', 
            'child', 'baby', 'adult', 'human', 'individual', 'someone',
            'gentleman', 'lady', 'kid', 'teenager', 'elder', 'senior'
        ]
        
        human_related_count = 0
        for _, row in df.iterrows():
            bbox1_label = sanitize_filename(row['bbox1_label'])
            bbox2_label = sanitize_filename(row['bbox2_label'])
            category_name = f"{bbox1_label}_vs_{bbox2_label}".lower()
            
            if any(keyword in category_name for keyword in human_keywords):
                human_related_count += 1
        
        print(f"📊 Human-related samples: {human_related_count}/{len(df)} ({human_related_count/len(df)*100:.1f}%)")
        print(f"🔍 Filter keywords: {', '.join(human_keywords)}")
        
        # Create output directory
        base_dir = create_output_dirs()
        print(f"📁 Output directory: {base_dir}")
        
        # Statistics
        stats = {
            'total_processed': 0,
            'successful_visualizations': 0,
            'failed_visualizations': 0,
            'categories_created': set(),
            'images_with_actual_files': 0,
            'images_with_mock_data': 0
        }
        
        # Organize data by category
        print(f"\n🔄 Starting batch processing...")
        
        # Limit processing quantity to avoid too many files (adjustable)
        max_samples_per_category = 10
        sample_count_per_category = {}
        
        for idx, row in df.iterrows():
            # Create category directory for each bbox
            bbox1_label = sanitize_filename(row['bbox1_label'])
            bbox2_label = sanitize_filename(row['bbox2_label'])
            
            # Create category directory (using combination of two labels)
            category_name = f"{bbox1_label}_vs_{bbox2_label}"
            if len(category_name) > 150:  # Limit directory name length
                category_name = f"{bbox1_label[:50]}_vs_{bbox2_label[:50]}"
            
            # Define human-related keywords
            human_keywords = [
                'man', 'woman', 'person', 'people', 'guy', 'girl', 'boy', 
                'child', 'baby', 'adult', 'human', 'individual', 'someone',
                'gentleman', 'lady', 'kid', 'teenager', 'elder', 'senior'
            ]
            
            # Check if category name contains human-related keywords
            category_lower = category_name.lower()
            contains_human = any(keyword in category_lower for keyword in human_keywords)
            
            if not contains_human:
                continue  # Skip non-human-related categories

            # Limit sample count per category
            if category_name not in sample_count_per_category:
                sample_count_per_category[category_name] = 0
            
            if sample_count_per_category[category_name] >= max_samples_per_category:
                continue  # Skip samples exceeding limit
                
            sample_count_per_category[category_name] += 1
            
            category_dir = base_dir #/ category_name
            category_dir.mkdir(parents=True, exist_ok=True)
            stats['categories_created'].add(category_name)
            
            try:
                # Visualize current row
                success = visualize_single_overlap(row, idx, category_dir, stats)
                if success:
                    stats['successful_visualizations'] += 1
                else:
                    stats['failed_visualizations'] += 1
                    
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                stats['failed_visualizations'] += 1
            
            stats['total_processed'] += 1
            
            # Progress display
            if stats['total_processed'] % 100 == 0:
                print(f"  Progress: {stats['total_processed']}/{len(df)} "
                      f"(Success: {stats['successful_visualizations']}, "
                      f"Failed: {stats['failed_visualizations']})")
        
        # Show final statistics
        print(f"\n📊 Processing completion statistics:")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful visualizations: {stats['successful_visualizations']}")
        print(f"Failed count: {stats['failed_visualizations']}")
        print(f"Created category directories: {len(stats['categories_created'])}")
        print(f"Used real images: {stats['images_with_actual_files']}")
        print(f"Used mock images: {stats['images_with_mock_data']}")
        
        # Show created category directories
        print(f"\n📁 Created category directories (first 20):")
        for i, category in enumerate(sorted(stats['categories_created'])[:20], 1):
            print(f"  {i:2d}. {category}")
        
        if len(stats['categories_created']) > 20:
            print(f"  ... {len(stats['categories_created']) - 20} more directories")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def visualize_single_overlap(row, idx, output_dir, stats):
    """Visualize single overlap case"""



    try:
        # Check if image exists and try multiple possible paths
        image_path = row['image_path']
        if data_name == "mixed_grounding":
            possible_paths = [
                image_path,
                image_path.replace('../', ''),  # Remove ../
                f"datasets/mixed_grounding/gqa/images/{row['image']}",  # Direct path
                f"../datasets/mixed_grounding/gqa/images/{row['image']}",  # Relative path
            ]
        else:  # flickr
            possible_paths = [
                image_path,
                image_path.replace('../', ''),  # Remove ../
                f"datasets/flickr/images/{row['image']}",  # Direct path
                f"../datasets/flickr/images/{row['image']}",  # Relative path
            ]

        
        img = None
        actual_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    actual_path = path
                    stats['images_with_actual_files'] += 1
                    break
                except Exception:
                    continue
        
        if img is None:
            # Create white background instead of noise
            img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255
            img = Image.fromarray(img_array)
            stats['images_with_mock_data'] += 1
        else:
            # Ensure image is in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)
        
        # Convert normalized coordinates to pixel coordinates
        def norm_to_pixel(x_center, y_center, width, height, img_w, img_h):
            x_center_px = x_center * img_w
            y_center_px = y_center * img_h
            width_px = width * img_w
            height_px = height * img_h
            
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            x2 = x_center_px + width_px / 2
            y2 = y_center_px + height_px / 2
            
            return x1, y1, x2, y2
        
        # Draw first bbox (red)
        x1_1, y1_1, x2_1, y2_1 = norm_to_pixel(
            row['bbox1_x'], row['bbox1_y'], row['bbox1_w'], row['bbox1_h'],
            img_width, img_height
        )
        
        bbox1_rect = patches.Rectangle(
            (x1_1, y1_1), x2_1 - x1_1, y2_1 - y1_1,
            linewidth=3, edgecolor='red', facecolor='none', alpha=0.8
        )
        ax.add_patch(bbox1_rect)
        
        # Draw second bbox (blue)
        x1_2, y1_2, x2_2, y2_2 = norm_to_pixel(
            row['bbox2_x'], row['bbox2_y'], row['bbox2_w'], row['bbox2_h'],
            img_width, img_height
        )
        
        bbox2_rect = patches.Rectangle(
            (x1_2, y1_2), x2_2 - x1_2, y2_2 - y1_2,
            linewidth=3, edgecolor='blue', facecolor='none', alpha=0.8
        )
        ax.add_patch(bbox2_rect)
        
        # Add label text
        ax.text(x1_1, y1_1 - 10, f"Red: {row['bbox1_label'][:20]}", 
                color='red', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.text(x1_2, y1_2 - 30, f"Blue: {row['bbox2_label'][:20]}", 
                color='blue', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Calculate and draw intersection area (green semi-transparent)
        intersect_x1 = max(x1_1, x1_2)
        intersect_y1 = max(y1_1, y1_2)
        intersect_x2 = min(x2_1, x2_2)
        intersect_y2 = min(y2_1, y2_2)
        
        if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
            intersect_rect = patches.Rectangle(
                (intersect_x1, intersect_y1), 
                intersect_x2 - intersect_x1, intersect_y2 - intersect_y1,
                linewidth=2, edgecolor='green', facecolor='green', alpha=0.3
            )
            ax.add_patch(intersect_rect)
        
        # Set title and labels
        ax.set_title(f"Bbox Overlap Visualization\n"
                    f"Image: {row['image']} | IoU: {row['iou']:.3f}\n"
                    f"Red: '{row['bbox1_label'][:30]}' | Blue: '{row['bbox2_label'][:30]}'",
                    fontsize=11, pad=15)
        
        ax.set_xlabel("Width (pixels)", fontsize=10)
        ax.set_ylabel("Height (pixels)", fontsize=10)
        
        # Save image
        output_name = f"overlap_{idx:06d}_iou_{row['iou']:.3f}_{sanitize_filename(row['image'])}.png"
        output_path = output_dir / output_name
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()  # Release memory
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to visualize row {idx}: {e}")
        return False

def create_summary_report():
    """Create summary report of visualization results"""

    if data_name=="mixed_grounding":
        base_dir = Path("../runs/mixed_grounding_visual")
    else:
        base_dir = Path("../runs/flickr_visual")

    if not base_dir.exists():
        print("❌ Visualization directory does not exist")
        return
    
    print(f"\n📋 Generating summary report...")
    
    report_path = base_dir / "visualization_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Mixed Grounding Bbox Overlap Visualization Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Count image number in each category directory
        categories = []
        for category_dir in base_dir.iterdir():
            if category_dir.is_dir():
                image_count = len(list(category_dir.glob("*.png")))
                categories.append((category_dir.name, image_count))
        
        # Sort by image count
        categories.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"Total categories: {len(categories)}\n")
        f.write(f"Total images: {sum(count for _, count in categories)}\n\n")
        
        f.write("Category details:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<4} {'Category Name':<50} {'Image Count':<8}\n")
        f.write("-" * 80 + "\n")
        
        for i, (category_name, count) in enumerate(categories, 1):
            f.write(f"{i:<4} {category_name:<50} {count:<8}\n")
    
    print(f"📄 Summary report saved: {report_path}")

if __name__ == "__main__":
    # Batch visualization
    visualize_all_overlaps()
    
    # Generate summary report
    create_summary_report()
    
    print(f"\n✅ All tasks completed!")
    print(f"Visualization results saved in: runs/visual/")
    print(f"You can browse overlap case visualizations by category")
