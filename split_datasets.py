import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# 定义文件夹路径
dataset_dir = 'C:\\Users\\Administrator\\Desktop\\yolo\\ultralytics\\datasets\\LROC-1'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
dtm_dir = os.path.join(dataset_dir, 'dtm')

# 创建目标文件夹结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    os.makedirs(os.path.join(dtm_dir, split), exist_ok=True) # 创建DTM文件夹下的train, val, test

# 获取images文件夹所有的图片文件路径
image_files = glob(os.path.join(images_dir, '*.jpg'))

# 将数据集划分为训练集、验证集和测试集 按照【6，2，2】划分
train_files, temp_files = train_test_split(image_files, test_size=0.4, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

dtm_train = [line.replace("images","dtm") for line in train_files]
dtm_val = [line.replace("images","dtm") for line in val_files]
dtm_test = [line.replace("images","dtm") for line in test_files]

# 定义一个函数，用来将文件路径写入到txt文件中
def write_txt(file_list, txt_file):
    with open(txt_file, 'w') as f:
        for file_path in file_list:
            f.write(f'{file_path}\n')

def zengjia(file_list, str):
    return_list = []
    for line in file_list:
        base = os.path.basename(line)
        gai = os.path.join(str, base)
        return_list.append(line.replace(base, gai))
    return return_list

# 将文件路径写入train.txt, val.txt, test.txt
write_txt(zengjia(train_files,"train"), os.path.join(dataset_dir, 'train.txt'))
print("train.txt写入成功")
write_txt(zengjia(val_files,"val"), os.path.join(dataset_dir, 'val.txt'))
print("val.txt写入成功")
write_txt(zengjia(test_files,"test"), os.path.join(dataset_dir, 'test.txt'))
print("test.txt写入成功")


# 写入dtm文件
write_txt(zengjia(dtm_train,"train"), os.path.join(dataset_dir, 'dtm_train.txt'))
print("train_dtm.txt写入成功")
write_txt(zengjia(dtm_val,"val"), os.path.join(dataset_dir, 'dtm_val.txt'))
print("val_dtm.txt写入成功")
write_txt(zengjia(dtm_test,"test"), os.path.join(dataset_dir, 'dtm_test.txt'))
print("test_dtm.txt写入成功")

# 定义一个函数，用来移动图片和对应的标签到指定文件夹
def move_files(file_list, split):
    for file_path in file_list:
        filename = os.path.basename(file_path)
        # 移动图片文件
        shutil.move(file_path, os.path.join(images_dir, split, filename))

        # 移动标签文件
        if "images" in file_path:
            label_file = filename.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path): # 如果文件存在就移动。
                shutil.move(label_path, os.path.join(labels_dir, split, label_file))

# 移动文件到相应的文件夹
move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')
print("image 图片移动成功")

#移动dtm文件
move_files(dtm_train, 'train')
move_files(dtm_val, 'val')
move_files(dtm_test, 'test')
print("dtm 图片移动成功")

