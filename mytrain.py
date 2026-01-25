from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolov10n.pt")  # 加载预训练的YOLOv10n模型
    model.train(
        data=r"xz_dataset.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图像尺寸
        batch=-1,  # 批量大小
        cache="ram",  # 是否缓存数据集
        workers=1,  # 数据加载线程数
    )