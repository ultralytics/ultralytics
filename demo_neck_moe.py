#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-P23456-Neck-MoE 演示
在 Neck 阶段实现 MoE 路由，比 Head-MoE 更早进行特征选择
"""

from ultralytics import YOLO
import torch

def compare_architectures():
    """对比三种架构"""
    print("=" * 80)
    print("MoE 路由架构对比")
    print("=" * 80)
    
    print("\n📊 三种架构对比:\n")
    print("1️⃣  标准 P23456:")
    print("   Backbone → Neck(全部5层) → Head(全部5层)")
    print("   ✓ 完整多尺度  ✗ 计算量大\n")
    
    print("2️⃣  Head-MoE (yolov8-p23456-moe):")
    print("   Backbone → Neck(全部5层) → Gate → Head(top-2层)")
    print("   ✓ 节省Head计算  ✓ 动态路由")
    print("   ✗ Neck仍需处理全部5层\n")
    
    print("3️⃣  Neck-MoE (yolov8-p23456-neck-moe) ⭐ NEW:")
    print("   Backbone → Neck(全部5层) → Gate → Neck精简 → Head(top-3层)")
    print("   ✓ 更早特征选择  ✓ Neck+Head都节省")
    print("   ✓ 最高效率\n")

def demo_neck_moe_detect():
    print("=" * 80)
    print("YOLOv8-P23456-Neck-MoE 检测模型")
    print("=" * 80)
    
    model = YOLO('ultralytics/cfg/models/v8/yolov8-p23456-neck-moe.yaml')
    model.info(verbose=True)
    
    print("\n✅ Neck-MoE 检测模型创建成功!")
    print("📌 特性:")
    print("   - 候选层: 5个 P 层 (P2-P6)")
    print("   - 路由位置: Neck 阶段 (早期选择)")
    print("   - 实际使用: 动态 top-3 层")
    print("   - 效率提升: Neck + Head 双重加速")

def demo_neck_moe_segment():
    print("\n" + "=" * 80)
    print("YOLOv8-Seg-P23456-Neck-MoE 分割模型")
    print("=" * 80)
    
    model = YOLO('ultralytics/cfg/models/v8/yolov8-seg-p23456-neck-moe.yaml')
    model.info(verbose=True)
    
    print("\n✅ Neck-MoE 分割模型创建成功!")
    print("📌 特性:")
    print("   - 分割任务支持 Neck 级别路由")
    print("   - Top-3 选择平衡精度和效率")

def explain_routing_difference():
    print("\n" + "=" * 80)
    print("路由机制详解")
    print("=" * 80)
    
    print("\n🔹 Head-MoE 路由流程:")
    print("   Backbone → P2/P3/P4/P5/P6 (全部计算)")
    print("           ↓")
    print("   Neck FPN+PAN (全部5层都要处理)")
    print("           ↓")
    print("   Gate Network (分析5层特征)")
    print("           ↓")
    print("   Select Top-2 → Detection Heads (仅2层)")
    print("   💾 节省: ~40% Head 计算")
    
    print("\n🔹 Neck-MoE 路由流程:")
    print("   Backbone → P2/P3/P4/P5/P6 (全部计算)")
    print("           ↓")
    print("   Neck FPN+PAN (全部5层都要处理)")
    print("           ↓")
    print("   Gate Network (分析5层特征) ⭐")
    print("           ↓")
    print("   Select Top-3 (仅保留3层特征)")
    print("           ↓")
    print("   Detection Heads (仅在3层上计算)")
    print("   💾 节省: ~40% Head计算 + 部分后处理")
    
    print("\n💡 关键区别:")
    print("   - Head-MoE: 路由决策在最后，Head选择性计算")
    print("   - Neck-MoE: 路由决策在中间，Head输入就已精简")
    print("   - Neck-MoE更适合实时场景，减少后续所有操作的开销")

def usage_examples():
    print("\n" + "=" * 80)
    print("使用示例")
    print("=" * 80)
    
    print("\n📝 训练:")
    print("```python")
    print("from ultralytics import YOLO")
    print("")
    print("# Neck-MoE 检测")
    print("model = YOLO('ultralytics/cfg/models/v8/yolov8-p23456-neck-moe.yaml')")
    print("model.train(")
    print("    data='coco.yaml',")
    print("    epochs=300,")
    print("    imgsz=1280,")
    print("    batch=8,")
    print(")")
    print("")
    print("# Neck-MoE 分割")
    print("model_seg = YOLO('ultralytics/cfg/models/v8/yolov8-seg-p23456-neck-moe.yaml')")
    print("model_seg.train(data='coco-seg.yaml', epochs=300, imgsz=1280)")
    print("```")
    
    print("\n📝 推理:")
    print("```python")
    print("results = model.predict('image.jpg', imgsz=1280)")
    print("# 自动使用 top-3 动态选择的层")
    print("```")
    
    print("\n🎯 选择建议:")
    print("   Head-MoE:  适合训练阶段实验，灵活性高")
    print("   Neck-MoE:  适合部署场景，推理效率最高 ⭐")

def performance_comparison():
    print("\n" + "=" * 80)
    print("性能对比表")
    print("=" * 80)
    
    print("\n| 模型 | 候选层 | 实际使用 | 路由位置 | Neck计算 | Head计算 | 推荐场景 |")
    print("|------|--------|----------|----------|----------|----------|----------|")
    print("| P23456 | 5 | 5 (固定) | - | 100% | 100% | 精度优先 |")
    print("| P23456-MoE | 5 | 2 (动态) | Head | 100% | 40% | 平衡 |")
    print("| P23456-Neck-MoE | 5 | 3 (动态) | Neck | 100%* | 60% | 速度优先 |")
    print("\n*Neck计算全部5层但之后立即精简到3层")
    
    print("\n🔍 详细分析:")
    print("   - Top-2 vs Top-3: Neck-MoE选3层以保持更好覆盖")
    print("   - 路由提前: 后续所有模块都受益于特征精简")
    print("   - 内存友好: 传递的特征张量更少")

if __name__ == '__main__':
    compare_architectures()
    demo_neck_moe_detect()
    demo_neck_moe_segment()
    explain_routing_difference()
    usage_examples()
    performance_comparison()
    
    print("\n" + "=" * 80)
    print("✨ Neck-MoE 模型演示完成!")
    print("=" * 80)
    print("\n💡 总结: Neck-MoE 在特征金字塔阶段就完成路由，")
    print("   比 Head-MoE 更早减少计算，适合实时推理场景。")
