"""
visualize_contact_points.py

可视化接触点碰撞分析结果
============================================

功能：
1. 加载 near_misses.json 和 tracks.json
2. 可视化最危险的碰撞事件
3. 在原视频上绘制接触点和距离信息
4. 生成统计图表

使用：
python visualize_contact_points.py \
  --near-misses runs/trajectory_demo/xxx/near_misses.json \
  --tracks runs/trajectory_demo/xxx/tracks.json \
  --video videos/test.mp4 \
  --output visualization/

或简单模式（只分析数据，不画视频）：
python visualize_contact_points.py \
  --near-misses runs/trajectory_demo/xxx/near_misses.json \
  --tracks runs/trajectory_demo/xxx/tracks.json \
  --analyze-only
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import cv2
from pathlib import Path


def analyze_contact_points(near_misses_path, tracks_path):
    """分析接触点碰撞数据"""
    
    print("=" * 60)
    print("Contact Point Analysis")
    print("=" * 60)
    
    # 加载数据
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)
    
    print(f"\n✓ Loaded {len(near_misses)} near-miss events")
    print(f"✓ Loaded {len(tracks)} tracked objects\n")
    
    if not near_misses:
        print("⚠ No near-miss events found")
        return
    
    # 【新增分析】按接触点类型统计
    print("【Contact Point Statistics】")
    print("-" * 60)
    
    point_type_stats = defaultdict(int)
    for event in near_misses:
        if 'closest_point_pair' in event:
            point_pair = event['closest_point_pair']
            key = f"{point_pair['obj1_point_type']}-{point_pair['obj2_point_type']}"
            point_type_stats[key] += 1
    
    if point_type_stats:
        print("Collision point type distribution:")
        for point_pair, count in sorted(point_type_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {point_pair}: {count} events")
    else:
        print("  No contact point data (using fallback center distance)")
    
    # 分析危险碰撞
    print("\n【High-Risk Events (TTC < 3s)】")
    print("-" * 60)
    
    collision_risks = [nm for nm in near_misses if nm.get('is_collision_risk', False)]
    print(f"Total high-risk events: {len(collision_risks)}")
    
    if collision_risks:
        # 按距离排序
        sorted_by_distance = sorted(collision_risks, key=lambda x: x['distance'])
        print("\nTop 5 closest contact points:")
        for i, event in enumerate(sorted_by_distance[:5], 1):
            obj1, obj2 = event['id1'], event['id2']
            dist = event['distance']
            ttc = event['ttc']
            
            if 'closest_point_pair' in event:
                point_info = event['closest_point_pair']
                pt_type = f"{point_info['obj1_point_type']}-{point_info['obj2_point_type']}"
                print(f"\n  {i}. Object {obj1} vs Object {obj2}")
                print(f"     Contact: {pt_type} ({dist:.2f} units)")
                print(f"     TTC: {ttc:.2f}s (if TTC < 0.5s, very critical!)")
            else:
                print(f"\n  {i}. Object {obj1} vs Object {obj2}")
                print(f"     Distance: {dist:.2f} units")
                print(f"     TTC: {ttc:.2f}s")
    
    # 按物体对统计
    print("\n【Object Pair Statistics】")
    print("-" * 60)
    
    pair_stats = defaultdict(list)
    for event in near_misses:
        key = (min(event['id1'], event['id2']), max(event['id1'], event['id2']))
        pair_stats[key].append(event)
    
    print(f"Total unique object pairs: {len(pair_stats)}")
    print("\nPairs with most near-miss events:")
    for pair, events in sorted(pair_stats.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        obj1, obj2 = pair
        min_dist = min(e['distance'] for e in events)
        min_ttc = min((e['ttc'] for e in events if e['ttc'] is not None), default=None)
        
        print(f"  Object {obj1} ↔ Object {obj2}: {len(events)} events, min_distance={min_dist:.2f}, min_ttc={min_ttc}")


def create_summary_plot(near_misses_path, output_dir):
    """生成统计图表"""
    
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    
    if not near_misses:
        print("No data to plot")
        return
    
    # 准备数据
    distances = [nm['distance'] for nm in near_misses]
    ttcs = [nm['ttc'] for nm in near_misses if nm['ttc'] is not None]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 距离分布
    axes[0, 0].hist(distances, bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distance Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Distance (units)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(sum(distances)/len(distances), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # TTC分布
    if ttcs:
        axes[0, 1].hist(ttcs, bins=30, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('TTC Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time to Collision (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(2.0, color='red', linestyle='--', label='Warning threshold')
        axes[0, 1].legend()
    
    # 接触点类型分布
    point_type_stats = defaultdict(int)
    for event in near_misses:
        if 'closest_point_pair' in event:
            point_pair = event['closest_point_pair']
            key = f"{point_pair['obj1_point_type'][0].upper()}-{point_pair['obj2_point_type'][0].upper()}"
            point_type_stats[key] += 1
    
    if point_type_stats:
        axes[1, 0].bar(point_type_stats.keys(), point_type_stats.values(), color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Contact Point Types', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Contact Type (F=Front, C=Center, B=Back)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 时间序列
    timestamps = [nm['timestamp'] for nm in near_misses]
    axes[1, 1].scatter(timestamps, distances, alpha=0.6, s=30, color='purple')
    axes[1, 1].set_title('Near-miss Events Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('Distance (units)')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'contact_points_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Analysis plot saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize contact point collision analysis')
    parser.add_argument('--near-misses', required=True, help='Path to near_misses.json')
    parser.add_argument('--tracks', required=True, help='Path to tracks.json')
    parser.add_argument('--output', default='visualization', help='Output directory')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze data, do not create video')
    
    args = parser.parse_args()
    
    # 数据分析
    analyze_contact_points(args.near_misses, args.tracks)
    
    # 生成图表
    print("\n【Generating Summary Plot】")
    print("-" * 60)
    create_summary_plot(args.near_misses, args.output)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output}")
