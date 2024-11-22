from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/train23/weights/best.pt")

# 运行测试集评估
if __name__ == '__main__':

    # 对图片进行目标检测
    results = model("tb_mask_2.jpg", save=True, save_txt=False)  # 替换为你的图片路径
    results[0].show()




