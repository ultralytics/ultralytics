from ultralytics import YOLO

if __name__ == "__main__":
    YOLO.new("yolov8n.yaml")
    YOLO.new("yolov8n-seg.yaml")
    YOLO.new("yolov8s.yaml")
    YOLO.new("yolov8s-seg.yaml")
    YOLO.new("yolov8m.yaml")
    YOLO.new("yolov8m-seg.yaml")
    YOLO.new("yolov8l.yaml")
    YOLO.new("yolov8l-seg.yaml")
    YOLO.new("yolov8x.yaml")
    YOLO.new("yolov8x-seg.yaml")

    # n vs n-seg: 8.9GFLOPs vs 12.8GFLOPs, 3.16M vs 3.6M. ch[0] // 4 (11.9GFLOPs, 3.39M)
    # s vs s-seg: 28.8GFLOPs vs 44.4GFLOPs, 11.1M vs 12.9M. ch[0] // 4 (39.5GFLOPs, 11.7M)
    # m vs m-seg: 79.3GFLOPs vs 113.8GFLOPs, 25.9M vs 29.5M. ch[0] // 4 (103.GFLOPs, 27.1M)
    # l vs l-seg: 165.7GFLOPs vs 226.3GFLOPs, 43.7M vs 49.6M. ch[0] // 4 (207GFLOPs, 45.7M)
    # x vs x-seg: 258.5GFLOPs vs 353.0GFLOPs, 68.3M vs 77.5M. ch[0] // 4 (324GFLOPs, 71.4M)
