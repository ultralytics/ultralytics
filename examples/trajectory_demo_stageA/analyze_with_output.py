"""
analyze_with_output.py

分析轨迹和准碰撞数据，并自动生成报告

用法：
python examples/trajectory_demo/analyze_with_output.py runs/trajectory_demo/NewYorkSample
"""
import json
import statistics
from collections import defaultdict
import sys
import os


def analyze_tracks(tracks_path):
    """Analyze trajectory data for all tracked objects."""
    print("=" * 60)
    print("Trajectory Data Analysis")
    print("=" * 60)
    
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)
    
    print(f"\nTotal tracked objects: {len(tracks)}")
    
    # Calculate trajectory statistics
    track_lengths = [len(samples) for samples in tracks.values()]
    
    print(f"   Average trajectory length: {statistics.mean(track_lengths):.1f} frames")
    print(f"   Maximum trajectory length: {max(track_lengths)} frames")
    print(f"   Minimum trajectory length: {min(track_lengths)} frames")
    
    # Categorize objects by class
    class_counts = defaultdict(int)
    for samples in tracks.values():
        if samples:
            cls = samples[0].get('cls')
            if cls is not None:
                class_counts[cls] += 1
    
    print(f"\nDetected object classes:")
    class_names = {0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    for cls_id, count in sorted(class_counts.items()):
        cls_name = class_names.get(cls_id, f'Unknown({cls_id})')
        print(f"   {cls_name}: {count} objects")
    
    return dict(class_counts)


def analyze_near_misses(near_misses_path):
    """Analyze collision proximity events."""
    print("\n" + "=" * 60)
    print("Collision Proximity Analysis")
    print("=" * 60)
    
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    
    print(f"\nTotal proximity events: {len(near_misses)}")
    
    # Filter high-risk collision events
    collision_risks = [nm for nm in near_misses if nm.get('is_collision_risk', False)]
    print(f"   High-risk collision events: {len(collision_risks)} events")
    
    # Distance statistics
    distances = [nm['distance'] for nm in near_misses if nm['distance'] is not None]
    if distances:
        print(f"\nDistance Statistics (pixels):")
        print(f"   Average distance: {statistics.mean(distances):.2f}")
        print(f"   Minimum distance: {min(distances):.2f}")
        print(f"   Maximum distance: {max(distances):.2f}")
    
    # Time-to-collision (TTC) statistics
    ttcs = [nm['ttc'] for nm in near_misses if nm['ttc'] is not None]
    if ttcs:
        print(f"\nTime-to-Collision (TTC) Statistics (seconds):")
        print(f"   Average TTC: {statistics.mean(ttcs):.2f}")
        print(f"   Minimum TTC: {min(ttcs):.2f} (Most critical)")
        print(f"   Maximum TTC: {max(ttcs):.2f}")
    
    # Rank most dangerous object pairs
    if collision_risks:
        print(f"\nMost Critical Object Pairs (TTC < 3 seconds):")
        sorted_risks = sorted(collision_risks, key=lambda x: x['ttc'] if x['ttc'] else float('inf'))
        for i, nm in enumerate(sorted_risks[:10], 1):
            ttc_val = nm['ttc'] if nm['ttc'] else 'N/A'
            print(f"   {i}. Object {nm['id1']} - Object {nm['id2']}: "
                  f"Distance={nm['distance']:.2f}px, TTC={ttc_val}s")
    
    # Count proximity events per object
    object_collision_count = defaultdict(int)
    for nm in near_misses:
        object_collision_count[nm['id1']] += 1
        object_collision_count[nm['id2']] += 1
    
    print(f"\nMost Involved Objects (Top 5):")
    sorted_objects = sorted(object_collision_count.items(), key=lambda x: x[1], reverse=True)
    for i, (obj_id, count) in enumerate(sorted_objects[:5], 1):
        print(f"   {i}. Object {obj_id}: {count} proximity events")
    
    return sorted_risks[:10] if collision_risks else []


def save_report(output_dir, top_risks):
    """Save analysis report to text file."""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("YOLO Object Tracking and Collision Risk Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Critical Object Pairs (requiring attention):\n")
        for i, nm in enumerate(top_risks[:5], 1):
            f.write(f"{i}. Object {nm['id1']} - Object {nm['id2']}: "
                   f"Distance={nm['distance']:.2f}px, TTC={nm['ttc']:.2f}s\n")
        
        f.write("\nFor detailed analysis, refer to tracks.json and near_misses.json\n")
    
    print(f"\nAnalysis report saved: {report_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Use default path
        output_dir = '/workspace/ultralytics/runs/trajectory_demo/NewYorkSample'
    else:
        output_dir = sys.argv[1]
    
    tracks_path = os.path.join(output_dir, 'tracks.json')
    near_misses_path = os.path.join(output_dir, 'near_misses.json')
    
    if not os.path.exists(tracks_path) or not os.path.exists(near_misses_path):
        print(f"Error: Output files not found")
        print(f"   Expected path: {output_dir}")
        sys.exit(1)
    
    try:
        analyze_tracks(tracks_path)
        top_risks = analyze_near_misses(near_misses_path)
        save_report(output_dir, top_risks)
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
