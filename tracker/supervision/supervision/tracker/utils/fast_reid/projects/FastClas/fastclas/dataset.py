# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from supervision.tracker.utils.fast_reid.fastreid.data.data_utils import read_image


class ClasDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, idx_to_class=None):
        self.img_items = img_items
        self.transform = transform

        if idx_to_class is not None:
            self.idx_to_class = idx_to_class
            self.class_to_idx = {clas_name: int(i) for i, clas_name in self.idx_to_class.items()}
            self.classes = sorted(list(self.idx_to_class.values()))
        else:
            classes = set()
            for i in img_items:
                classes.add(i[1])

            self.classes = sorted(list(classes))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.idx_to_class = {idx: clas for clas, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        label = self.class_to_idx[img_item[1]]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)

        return {
            "images": img,
            "targets": label,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.classes)
