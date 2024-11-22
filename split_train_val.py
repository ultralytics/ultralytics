import os
import random
import shutil

# 设置随机数种子以获得可重复的分割
random.seed(42)

# 数据集路径
images_path = "dataset/face_mask/images/"
labels_path = "dataset/face_mask/labels/"
output_dir = "dataset/face_mask_yolo"

# 训练集、验证集、测试集比例
train_ratio = 0.7
val_ratio = 0.2
# 测试集比例为剩下的部分
test_ratio = 1 - train_ratio - val_ratio

# 创建输出目录结构
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

# 获取所有图片文件名（假设标签文件名和图片文件名一致）
image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

# 打乱文件列表
random.shuffle(image_files)

# 计算每个数据集的数量
num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)

# 分割数据集
train_files = image_files[:num_train]
val_files = image_files[num_train : num_train + num_val]
test_files = image_files[num_train + num_val :]


# 复制文件到相应目录
def copy_files(file_list, split):
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]
        # 复制图片
        shutil.copy(os.path.join(images_path, file_name), os.path.join(output_dir, split, "images", file_name))
        # 复制标签（如果存在）
        label_file = base_name + ".txt"
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.copy(os.path.join(labels_path, label_file), os.path.join(output_dir, split, "labels", label_file))


# 复制训练集、验证集和测试集文件
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("数据集已成功分割！")
