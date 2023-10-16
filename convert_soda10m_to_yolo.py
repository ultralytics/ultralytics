
#SODA10M 格式的数据集转化为 YOLO 格式的数据集
#--json_path 输入的json文件路径
#--save_path 保存的文件夹名字，默认为当前目录下的labels。
 
import os
import json
from tqdm import tqdm
import argparse
 
parser = argparse.ArgumentParser()
#这里根据自己的json文件位置，换成自己的就行
parser.add_argument('--json_path', default='H:/MasterDegreeCourse/Yolo/yolov8/ultralytics/ultralytics/data/datasets/labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json',type=str, help="input: SODA10M format(json)")
#这里设置.txt文件保存位置
parser.add_argument('--save_path', default='H:/MasterDegreeCourse/Yolo/yolov8/ultralytics/ultralytics/data/datasets/soda10m/val/labels', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
#round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)
 
if __name__ == '__main__':
    json_file =   arg.json_path # SODA10M Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径
 
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
 
    id_map = {} # SODA10M 数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(id_map)
    #这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    # list_file = open(os.path.join(ana_txt_save_path, 'train_images_path.txt'), 'w')
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        #将图片的相对路径写入train_images_path或val_images_path的路径
    #     list_file.write('H:/MasterDegreeCourse/Yolo/yolov8/ultralytics/ultralytics/data/datasets/soda10m/val/images/%s.jpg\n' %(head))
    # list_file.close()
