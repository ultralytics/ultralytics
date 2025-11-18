#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-Seg-P23456 演示
结合 seg-p2 和 seg-p6 生成 P2-P6 五层输出的实例分割模型
"""

from ultralytics import YOLO

def demo_detect():
    print("=" * 70)
    print("YOLOv8-P23456 检测模型 (5层输出: P2-P6)")
    print("=" * 70)
    
    model_det = YOLO('ultralytics/cfg/models/v8/yolov8-p23456.yaml')
    model_det.info(verbose=True)
    
    print("\n✅ 检测模型创建成功!")

def demo_segment():
    print("\n" + "=" * 70)
    print("YOLOv8-Seg-P23456 分割模型 (5层输出: P2-P6)")
    print("=" * 70)
    
    model_seg = YOLO('ultralytics/cfg/models/v8/yolov8-seg-p23456.yaml')
    model_seg.info(verbose=True)
    
    print("\n✅ 分割模型创建成功!")
    print("📌 输出层级: P2(1/4), P3(1/8), P4(1/16), P5(1/32), P6(1/64)")
    print("📌 使用方法:")
    print("   检测: model = YOLO('yolov8-p23456.yaml')")
    print("   分割: model = YOLO('yolov8-seg-p23456.yaml')")
    print("\n   训练: model.train(data='coco.yaml', epochs=100, imgsz=1280)")
    print("   推理: results = model.predict('image.jpg', imgsz=1280)")
    print("   导出: model.export(format='onnx', imgsz=1280)")

if __name__ == '__main__':
    demo_detect()
    demo_segment()
