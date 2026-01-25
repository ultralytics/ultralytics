from ultralytics import YOLO

model = YOLO(r"yolov10n.pt")  # 加载预训练的YOLOv10n模型
model.predict(
    source=r"/home/nanyuanchaliang/codespace/YOLO/make_dataset/videos/001.mp4",  # 输入图像路径或URL
    save=True,  # 保存检测结果图像
    show=False,  # 不显示图像
)
