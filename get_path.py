import os


dataset_dir = 'D:\\Desktop\\ultralytics\\datasets\\LROC'
a = ["train", "val", "test"]
for type in a:
    image_path = os.path.join(dataset_dir, "images", type)
    txt_path = os.path.join(dataset_dir, type+".txt")
    with open(txt_path, 'w') as f:
        for jpg in os.listdir(image_path):
            jpg_path = os.path.join(image_path, jpg)
            f.write(f'{jpg_path}\n')

