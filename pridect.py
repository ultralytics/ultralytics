from ultralytics import YOLO
import os

# 加载训练好的模型
model = YOLO("runs/detect/train23/weights/best.pt")

# 图片所在的目录
image_dir = r"C:\Users\42919\PycharmProjects\yolov11_mask\datasets\mask\mask"

# 获取目录中的所有图片文件
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 运行测试集评估并保存结果
def detect_and_save(model, image_files):
    for image_path in image_files:
        # 对图片进行目标检测，保存标注后的图片和txt文件
        results = model(image_path, save=True, save_txt=True)

        # 打印保存的目录路径
        save_dir = results[0].path if results else None
        if save_dir:
            print(f"标注后的图片和标签文件已保存至: {os.path.dirname(save_dir)}")
        else:
            print(f"图片 {image_path} 没有检测结果保存。")

# 直接运行目标检测函数
detect_and_save(model, image_files)
