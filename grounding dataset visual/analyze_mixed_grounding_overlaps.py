#!/usr/bin/env python3
"""
Analyze bbox overlaps within the same image in Mixed Grounding dataset
Extract bbox pairs with IoU > 0.8 and save to CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv
from typing import List, Tuple

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU of two bboxes
    Input format: [x_center, y_center, width, height] (normalized coordinates)
    """
    # Convert to [x1, y1, x2, y2] format
    def xywh_to_xyxy(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return np.array([x1, y1, x2, y2])
    
    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)
    
    # Calculate intersection
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def analyze_bbox_overlaps():
    """Analyze bbox overlap situations in Mixed Grounding dataset"""
    # cache_path = "../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache"
    # output_csv = "mixed_grounding_bbox_overlaps_iou09.csv"
    # iou_threshold = 0.9


    cache_path = "./datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache"
    output_csv = "mixed_flickr_bbox_overlaps_iou09.csv"
    iou_threshold = 0.9

    print("🔍 Analyzing Mixed Flickr Bbox overlaps")
    print("=" * 60)
    print(f"📊 IoU threshold: {iou_threshold}")
    
    try:
        # Load cache file
        print(f"📁 Loading file: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        
        print(f"✅ Successfully loaded cache file")
        print(f"📊 Total samples: {data.size}")
        
        # Store overlap results
        overlap_results = []
        total_images_processed = 0
        total_bbox_pairs = 0
        total_high_iou_pairs = 0
        
        print(f"\n🔄 Analyzing bbox overlaps...")
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                texts = item.get('texts', [])
                classes = item.get('cls', np.array([]))
                bboxes = item.get('bboxes', np.array([]))
                im_file = item.get('im_file', 'unknown')
                
                # Check data validity
                if (len(texts) > 0 and 
                    hasattr(bboxes, 'shape') and bboxes.shape[0] > 1 and
                    hasattr(classes, 'shape') and classes.shape[0] > 1):
                    
                    total_images_processed += 1
                    image_name = str(im_file).split('/')[-1] if im_file else f'image_{i}'
                    
                    # Get all bboxes and corresponding labels for current image
                    num_boxes = bboxes.shape[0]
                    
                    # Ensure texts count matches
                    valid_boxes = min(num_boxes, len(texts))
                    
                    # Calculate IoU for all bbox pairs
                    for idx1 in range(valid_boxes):
                        for idx2 in range(idx1 + 1, valid_boxes):
                            bbox1 = bboxes[idx1]
                            bbox2 = bboxes[idx2]
                            
                            # Get corresponding labels
                            label1 = texts[idx1][0] if isinstance(texts[idx1], list) else str(texts[idx1])
                            label2 = texts[idx2][0] if isinstance(texts[idx2], list) else str(texts[idx2])
                            
                            # Calculate IoU
                            iou = calculate_iou(bbox1, bbox2)
                            total_bbox_pairs += 1
                            
                            # If IoU is greater than threshold, save result
                            if iou > iou_threshold:
                                total_high_iou_pairs += 1
                                
                                overlap_results.append({
                                    'image_path': str(im_file) if im_file else f'unknown_image_{i}',
                                    'image': image_name,
                                    'bbox1_idx': idx1,
                                    'bbox1_label': label1,
                                    'bbox1_x': float(bbox1[0]),
                                    'bbox1_y': float(bbox1[1]),
                                    'bbox1_w': float(bbox1[2]),
                                    'bbox1_h': float(bbox1[3]),
                                    'bbox2_idx': idx2,
                                    'bbox2_label': label2,
                                    'bbox2_x': float(bbox2[0]),
                                    'bbox2_y': float(bbox2[1]),
                                    'bbox2_w': float(bbox2[2]),
                                    'bbox2_h': float(bbox2[3]),
                                    'iou': float(iou)
                                })
            
            # Progress display
            if (i + 1) % 50000 == 0:
                print(f"  Processing progress: {i+1:,}/{data.size:,} samples, "
                      f"found {total_high_iou_pairs:,} high overlap pairs")
        
        print(f"\n📈 Analysis results:")
        print(f"📊 Processed images: {total_images_processed:,}")
        print(f"📊 Total bbox pairs: {total_bbox_pairs:,}")
        print(f"📊 High IoU pairs (>{iou_threshold}): {total_high_iou_pairs:,}")
        print(f"📊 High overlap ratio: {total_high_iou_pairs/total_bbox_pairs*100:.2f}%")
        
        # Save to CSV
        if overlap_results:
            df = pd.DataFrame(overlap_results)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Results saved to: {output_csv}")
            
            # Show some statistics
            print(f"\n📊 Overlap statistics:")
            print(f"Average IoU: {df['iou'].mean():.3f}")
            print(f"Max IoU: {df['iou'].max():.3f}")
            print(f"IoU std: {df['iou'].std():.3f}")
            
            # Show top 10 highest overlap examples
            print(f"\n🏆 TOP 10 highest overlap bbox pairs:")
            print("-" * 120)
            top_overlaps = df.nlargest(10, 'iou')
            for idx, row in top_overlaps.iterrows():
                print(f"IoU: {row['iou']:.3f} | {row['image']} | "
                      f"'{row['bbox1_label']}' vs '{row['bbox2_label']}' | Path: {row['image_path']}")
            
            # Analyze overlapping label types
            print(f"\n🔍 Overlapping label analysis:")
            print("-" * 60)
            
            # Count same category overlaps
            same_label_count = 0
            for _, row in df.iterrows():
                if row['bbox1_label'] == row['bbox2_label']:
                    same_label_count += 1
            
            print(f"Same label overlaps: {same_label_count:,} ({same_label_count/len(df)*100:.1f}%)")
            print(f"Different label overlaps: {len(df)-same_label_count:,} ({(len(df)-same_label_count)/len(df)*100:.1f}%)")
            
            # Most common overlapping label pairs
            label_pairs = []
            for _, row in df.iterrows():
                pair = tuple(sorted([row['bbox1_label'], row['bbox2_label']]))
                label_pairs.append(pair)
            
            from collections import Counter
            pair_counts = Counter(label_pairs)
            
            print(f"\n🔄 TOP 10 most common overlapping label pairs:")
            print("-" * 80)
            for i, (pair, count) in enumerate(pair_counts.most_common(10), 1):
                print(f"{i:2d}. '{pair[0]}' ↔ '{pair[1]}': {count:,} times")
        
        else:
            print(f"\n⚠️  No bbox pairs found with IoU > {iou_threshold}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_bbox_overlaps()
