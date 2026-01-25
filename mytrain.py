from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolov10n.pt")  # 加载预训练的YOLOv10n模型
    model.train(
        data=r"coco8.yaml",  # 数据集配置文件路径
        epochs=30,  # 训练轮数
        imgsz=640,  # 输入图像尺寸
        batch=2,  # 批量大小
        cache=False,  # 是否缓存数据集
        workers=0,  # 数据加载线程数
    )
