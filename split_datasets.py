import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# 定义文件夹路径
dataset_dir = 'C:\\Users\\Administrator\\Desktop\\yolo\\ultralytics\\datasets\\LROC'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 创建目标文件夹结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# 获取所有的图片文件路径
image_files = glob(os.path.join(images_dir, '*.jpg'))

# 将数据集划分为训练集、验证集和测试集 按照【6，2，2】划分
train_files, temp_files = train_test_split(image_files, test_size=0.4, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# 定义一个函数，用来将文件路径写入到txt文件中
def write_txt(file_list, txt_file):
    with open(txt_file, 'w') as f:
        for file_path in file_list:
            file_path.replace(r"images", r"images\val")
            f.write(f'{file_path}\n')

# 将文件路径写入train.txt, val.txt, test.txt
write_txt(train_files, os.path.join(dataset_dir, 'train.txt'))
write_txt(val_files, os.path.join(dataset_dir, 'val.txt'))
write_txt(test_files, os.path.join(dataset_dir, 'test.txt'))

# 定义一个函数，用来移动图片和对应的标签到指定文件夹
def move_files(file_list, split):
    for file_path in file_list:
        filename = os.path.basename(file_path)
        label_file = filename.replace('.jpg', '.txt')

        # 移动图片文件
        shutil.move(file_path, os.path.join(images_dir, split, filename))

        # 移动标签文件
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(labels_dir, split, label_file))

# 移动文件到相应的文件夹
move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

print("YOLO 数据集已生成完成")
