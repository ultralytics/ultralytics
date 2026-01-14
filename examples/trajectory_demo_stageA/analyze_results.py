"""
analyze_results.py.

简单脚本：分析 tracks.json 和 near_misses.json 的结果
"""

import json
import statistics
from collections import defaultdict


def analyze_tracks(tracks_path):
    """分析轨迹数据."""
    print("=" * 60)
    print("📊 轨迹数据分析")
    print("=" * 60)

    with open(tracks_path) as f:
        tracks = json.load(f)

    print(f"\n✅ 追踪到的对象总数: {len(tracks)}")

    # 统计每个对象的轨迹长度
    track_lengths = [len(samples) for samples in tracks.values()]

    print(f"   平均轨迹长度: {statistics.mean(track_lengths):.1f} 帧")
    print(f"   最长轨迹: {max(track_lengths)} 帧")
    print(f"   最短轨迹: {min(track_lengths)} 帧")

    # 统计类别
    class_counts = defaultdict(int)
    for samples in tracks.values():
        if samples:
            cls = samples[0].get("cls")
            if cls is not None:
                class_counts[cls] += 1

    print("\n📦 检测到的类别:")
    class_names = {0: "人", 1: "自行车", 2: "汽车", 3: "摩托车", 5: "公交车", 7: "卡车"}
    for cls_id, count in sorted(class_counts.items()):
        cls_name = class_names.get(cls_id, f"未知({cls_id})")
        print(f"   - {cls_name}: {count} 个")


def analyze_near_misses(near_misses_path):
    """分析准碰撞事件."""
    print("\n" + "=" * 60)
    print("⚠️  准碰撞事件分析")
    print("=" * 60)

    with open(near_misses_path) as f:
        near_misses = json.load(f)

    print(f"\n✅ 总准碰撞事件数: {len(near_misses)}")

    # 统计高风险事件
    collision_risks = [nm for nm in near_misses if nm.get("is_collision_risk", False)]
    print(f"   其中碰撞风险事件: {len(collision_risks)} 个")

    # 距离统计
    distances = [nm["distance"] for nm in near_misses if nm["distance"] is not None]
    if distances:
        print("\n📏 距离统计:")
        print(f"   平均距离: {statistics.mean(distances):.2f} 像素")
        print(f"   最小距离: {min(distances):.2f} 像素")
        print(f"   最大距离: {max(distances):.2f} 像素")

    # TTC 统计（只统计有值的）
    ttcs = [nm["ttc"] for nm in near_misses if nm["ttc"] is not None]
    if ttcs:
        print("\n⏱️  TTC (碰撞预计时间) 统计:")
        print(f"   平均 TTC: {statistics.mean(ttcs):.2f} 秒")
        print(f"   最小 TTC: {min(ttcs):.2f} 秒（最危险）")
        print(f"   最大 TTC: {max(ttcs):.2f} 秒")

    # 最危险的对
    if collision_risks:
        print("\n🚨 最危险的对象对（TTC < 3 秒）:")
        sorted_risks = sorted(collision_risks, key=lambda x: x["ttc"] if x["ttc"] else float("inf"))
        for i, nm in enumerate(sorted_risks[:5], 1):
            print(f"   {i}. 对象 {nm['id1']} 和 {nm['id2']}: 距离={nm['distance']:.2f}px, TTC={nm['ttc']:.2f}s")


if __name__ == "__main__":
    tracks_path = "/workspace/ultralytics/runs/trajectory_demo/tracks.json"
    near_misses_path = "/workspace/ultralytics/runs/trajectory_demo/near_misses.json"

    try:
        analyze_tracks(tracks_path)
        analyze_near_misses(near_misses_path)
        print("\n" + "=" * 60)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请确保已运行 yolo_runner.py 生成了输出文件")
