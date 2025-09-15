#!/usr/bin/env python3
"""
Directly check cache file content to verify annotation situation for each image
"""

import numpy as np
import os

def check_cache_structure():
    """Check cache file structure"""
    cache_files = [
        "../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache",
        "/Users/louis/workspace/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache"
    ]
    
    for cache_path in cache_files:
        if not os.path.exists(cache_path):
            print(f"❌ Cache file does not exist: {cache_path}")
            continue
            
        print(f"\n🔍 Checking cache file: {cache_path}")
        print("=" * 80)
        
        try:
            # Load cache
            data = np.load(cache_path, allow_pickle=True)
            print(f"📊 Total cache entries: {data.size}")
            
            # Check structure of first few entries
            print(f"\n📋 Detailed info of first 5 entries:")
            print("-" * 80)
            
            image_count = {}  # Count occurrences of each image
            
            for i in range(min(5, data.size)):
                item = data[i]
                if isinstance(item, dict):
                    im_file = item.get('im_file', 'unknown')
                    texts = item.get('texts', [])
                    bboxes = item.get('bboxes', np.array([]))
                    cls = item.get('cls', np.array([]))
                    
                    image_name = str(im_file).split('/')[-1] if im_file else 'unknown'
                    
                    print(f"\nEntry {i+1}:")
                    print(f"  Image: {image_name}")
                    print(f"  Path: {im_file}")
                    print(f"  Texts count: {len(texts)}")
                    print(f"  Bboxes shape: {bboxes.shape if hasattr(bboxes, 'shape') else 'N/A'}")
                    print(f"  Cls shape: {cls.shape if hasattr(cls, 'shape') else 'N/A'}")
                    
                    if len(texts) > 0:
                        print(f"  First 3 texts: {[t[0] if isinstance(t, list) else str(t) for t in texts[:3]]}")
                    
                    # Count image occurrences
                    if image_name not in image_count:
                        image_count[image_name] = 0
                    image_count[image_name] += 1
            
            # Count occurrences of all images
            print(f"\n📈 Counting occurrences of all images...")
            all_image_count = {}
            
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    im_file = item.get('im_file', 'unknown')
                    image_name = str(im_file).split('/')[-1] if im_file else 'unknown'
                    
                    if image_name not in all_image_count:
                        all_image_count[image_name] = 0
                    all_image_count[image_name] += 1
                
                # Progress display
                if (i + 1) % 50000 == 0:
                    print(f"  Processing progress: {i+1}/{data.size}")
            
            # Analyze results
            duplicate_images = {img: count for img, count in all_image_count.items() if count > 1}
            
            print(f"\n📊 Statistics:")
            print(f"  Total images: {len(all_image_count)}")
            print(f"  Total entries: {data.size}")
            print(f"  Images appearing multiple times: {len(duplicate_images)}")
            
            if duplicate_images:
                print(f"\n⚠️  Images appearing multiple times (first 10):")
                for i, (img, count) in enumerate(list(duplicate_images.items())[:10], 1):
                    print(f"    {i:2d}. {img}: {count} times")
                    
                if len(duplicate_images) > 10:
                    print(f"    ... {len(duplicate_images) - 10} more images appear multiple times")
            else:
                print(f"✅ Each image appears only once (as expected)")
            
            # Test specific image
            print(f"\n🔍 Test specific image: 2323167250.jpg")
            test_image = "2323167250.jpg"
            found_items = []
            
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    im_file = item.get('im_file', 'unknown')
                    image_name = str(im_file).split('/')[-1] if im_file else 'unknown'
                    
                    if image_name == test_image:
                        found_items.append((i, item))
            
            print(f"  Found {len(found_items)} matching items:")
            for idx, (i, item) in enumerate(found_items):
                texts = item.get('texts', [])
                bboxes = item.get('bboxes', np.array([]))
                print(f"    Item {idx+1} (index{i}): {len(texts)} texts, {bboxes.shape[0] if hasattr(bboxes, 'shape') else 0} bboxes")
                if len(texts) > 0:
                    print(f"      First 5 texts: {[t[0] if isinstance(t, list) else str(t) for t in texts[:5]]}")
                        
        except Exception as e:
            print(f"❌ Error processing cache file: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_cache_structure()
