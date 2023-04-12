import random

import numpy as np
import torch
import os
import warnings
import imagesize
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from skyNet.yolo_custom.utils.utils import resize_image
from skyNet.yolo_custom.utils.bboxes_utils import iou_width_height, coco_to_yolo_tensors, non_max_suppression
from skyNet.yolo_custom.utils.plot_utils import plot_image, cells_to_bboxes
import skyNet.config as cfg

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


class Training_Dataset(Dataset):
    def __init__(self,
                 root_directory=cfg.ROOT_DIR,
                 transform=None,
                 train=True,
                 rect_training=False,
                 default_size=640,
                 batch_size=64,
                 bboxes_format="yolo",
                 ultralytics_loss=False,
                 ):

        assert bboxes_format in ["coco", "yolo"], 'bboxes_format must be either "coco" or "yolo"'
        self.batch_size = batch_size
        self.batch_range = 64 if batch_size < 64 else 128

        self.bboxes_format = bboxes_format
        self.ultralytics_loss = ultralytics_loss
        self.root_directory = root_directory
        self.transform = transform
        self.rect_training = rect_training
        self.default_size = default_size
        self.train = train

        if train:
            fname = 'images/train'
            annot_file = "train.txt"
            # class instance because it's used in the __getitem__
            self.annot_folder = "train"
        else:
            fname = 'images/val'
            annot_file = "val.txt"
            # class instance because it's used in the __getitem__
            self.annot_folder = "val"
        self.fname = fname

        try:
            # self.annotations = pd.read_csv(os.path.join(root_directory, "labels", annot_file),
            #                                header=None, index_col=0).sort_values(by=[0])
            self.annotations = pd.read_csv(os.path.join(root_directory, "labels", annot_file),
                                           header=None).sort_values(by=[0])
            self.annotations = self.annotations.head((len(self.annotations)-1))  # just removes last line
        except FileNotFoundError:
            annotations = []
            for img_txt in os.listdir(os.path.join(self.root_directory, "labels", self.annot_folder)):
                img = img_txt.split(".txt")[0]
                try:
                    w, h = imagesize.get(os.path.join(self.root_directory, "images", self.annot_folder, f"{img}.png"))
                except FileNotFoundError:
                    continue
                annotations.append([str(img) + ".png", h, w])
            self.annotations = pd.DataFrame(annotations)
            self.annotations.to_csv(os.path.join(self.root_directory, "labels", annot_file))

        self.len_ann = len(self.annotations)
        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        target_height = self.annotations.iloc[idx, 2] if self.rect_training else cfg.IMAGE_SIZE
        target_width = self.annotations.iloc[idx, 2] if self.rect_training else cfg.IMAGE_SIZE
        # img_name[:-4] to remove the .jpg or .png which are coco img formats
        label_path = os.path.join(self.root_directory, "labels", self.annot_folder, img_name[:-4] + ".txt")

        # print(f'{label_path}')

        # to avoid an annoying "UserWarning: loadtxt: Empty input file"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = np.loadtxt(fname=label_path, dtype=float, delimiter=" ", ndmin=2)
            # removing annotations with negative values
            labels = labels[np.all(labels >= 0, axis=1), :]
            # to avoid negative values
            labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        img = Image.open(os.path.join(self.root_directory, self.fname, img_name)).convert("RGB")
        img = np.array(img)

        if self.bboxes_format == "coco":
            labels[:, -1] -= 1  # 0-indexing the classes of coco labels (1-80 --> 0-79)
            labels = np.roll(labels, axis=1, shift=1)
            # normalized coordinates are scale invariant, hence after resizing the img we don't resize labels
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], w0=img.shape[1], h0=img.shape[0])

        img = resize_image(img, (int(target_width), int(target_height)))

        if self.transform:
            # albumentations requires bboxes to be (x,y,w,h,class_idx)
            batch_n = idx // self.batch_size
            if batch_n % 2 == 0:
                self.transform[1].p = 1
            else:
                self.transform[1].p = 0

            augmentations = self.transform(
                image=img,
                bboxes=np.roll(labels, shift=-1, axis=1)
            )
            img = augmentations["image"]
            # loss fx requires bboxes to be (class_idx, x, y, w, h)
            labels = np.array(augmentations["bboxes"])
            if len(labels):
                labels = np.roll(labels, axis=1, shift=1)
            # print(f'labels: {labels}')

        if len(labels):
            plot_labels = xywhn2xyxy(labels[:, 1:], w=img.shape[1], h=img.shape[0])
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for box in plot_labels:
                rect = Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none"
                )
                # Add the patch to the Axes
                ax.add_patch(rect)

            plt.show()

        if self.ultralytics_loss:
            labels = torch.from_numpy(labels)
            out_bboxes = torch.zeros((labels.shape[0], 6))
            if len(labels):
                out_bboxes[..., 1:] = labels

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), out_bboxes if self.ultralytics_loss else labels

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "labels",
            "adaptive_ann_{}_{}_br_{}.csv".format(name, self.len_ann, int(self.batch_range))
        )

        if os.path.isfile(path):
            print(f"==> Loading cached annotations for rectangular training on {self.annot_folder}")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for 'rectangular training' on training set...")
            annotations["w_h_ratio"] = annotations.iloc[:, 2] / annotations.iloc[:, 1]
            annotations.sort_values(["w_h_ratio"], ascending=True, inplace=True)

            for i in range(0, len(annotations), self.batch_range):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                max_dim = max(size)
                max_idx = size.index(max_dim)
                size[~max_idx] += 32
                sz = random.randrange(int(self.default_size * 0.9), int(self.default_size * 1.1)) // 32 * 32
                size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                size[max_idx] = sz
                if i + self.batch_range <= len(annotations):
                    bs = self.batch_range
                else:
                    bs = len(annotations) - i

                annotations.iloc[i:bs, 2] = size[0]
                annotations.iloc[i:bs, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:i+bs, :] = annotations.iloc[i:i+bs, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations.iloc[:,:3])
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        return torch.stack(im, 0), label

    @staticmethod
    def collate_fn_ultra(batch):
        im, label = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0)


class Validation_Dataset(Dataset):
    """COCO 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 anchors,
                 root_directory=cfg.ROOT_DIR,
                 transform=None,
                 train=True,
                 S=(8, 16, 32),
                 rect_training=False,
                 default_size=640,
                 bs=64,
                 bboxes_format="coco",
                 ):
        """
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            root_directory (path): path to the COCO2017 dataset
            transform: set of Albumentations transformations to be performed with A.Compose
        """
        assert bboxes_format in ["coco", "yolo"], 'bboxes_format must be either "coco" or "yolo"'

        self.batch_range = 64 if bs < 64 else 128
        self.bs = bs
        self.bboxes_format = bboxes_format
        self.transform = transform
        self.S = S
        self.nl = len(anchors[0])
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.S).repeat(6, 1).T.reshape(3, 3, 2)
        self.num_anchors = self.anchors.reshape(9,2).shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.rect_training = rect_training
        self.default_size = default_size
        self.root_directory = root_directory
        self.train = train

        if train:
            fname = 'images/train'
            annot_file = "train.txt"
            # class instance because it's used in the __getitem__
            self.annot_folder = "train"
        else:
            fname = 'images/val'
            annot_file = "val.txt"
            # class instance because it's used in the __getitem__
            self.annot_folder = "val"

        self.fname = fname

        try:
            self.annotations = pd.read_csv(os.path.join(root_directory, "labels", annot_file),
                                           header=None).sort_values(by=[0])
            self.annotations = self.annotations.head((len(self.annotations)-1))  # just removes last line
        except FileNotFoundError:
            annotations = []
            for img_txt in os.listdir(os.path.join(self.root_directory, "labels", self.annot_folder)):
                img = img_txt.split(".txt")[0]
                try:
                    w, h = imagesize.get(os.path.join(self.root_directory, "images", self.annot_folder, f"{img}.png"))
                except FileNotFoundError:
                    continue
                annotations.append([str(img) + ".png", h, w])
            self.annotations = pd.DataFrame(annotations)
            self.annotations.to_csv(os.path.join(self.root_directory, "labels", annot_file))

        self.len_ann = len(self.annotations)
        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        tg_height = self.annotations.iloc[idx, 1] if self.rect_training else cfg.IMAGE_SIZE
        tg_width = self.annotations.iloc[idx, 2] if self.rect_training else cfg.IMAGE_SIZE
        # img_name[:-4] to remove the .jpg or .png which are coco img formats
        label_path = os.path.join(os.path.join(self.root_directory, "labels", self.annot_folder, img_name[:-4] + ".txt"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = np.loadtxt(fname=label_path, dtype=float, delimiter=" ", ndmin=2)
            # removing annotations with negative values
            labels = labels[np.all(labels >= 0, axis=1), :]
            # to avoid negative values
            labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        img = Image.open(os.path.join(self.root_directory, self.fname, img_name)).convert("RGB")
        img = np.array(img)

        if self.bboxes_format == "coco":
            labels[:, -1] -= 1  # 0-indexing the classes of coco labels (1-80 --> 0-79)
            labels = np.roll(labels, axis=1, shift=1)
            # normalized coordinates are scale invariant, hence after resizing the img we don't resize labels
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], w0=img.shape[1], h0=img.shape[0])

        img = resize_image(img, (int(tg_width), int(tg_height)))

        if self.transform:
            # albumentations requires bboxes to be (x,y,w,h,class_idx)
            batch_n = idx // self.bs
            if batch_n % 2 == 0:
                self.transform[2].p = 1
            else:
                self.transform[2].p = 0

            augmentations = self.transform(
                image=img,
                bboxes=np.roll(labels, shift=-1, axis=1)
            )

            img = augmentations["image"]
            # loss fx requires bboxes to be (class_idx, x, y, w, h)
            labels = np.array(augmentations["bboxes"])
            if len(labels):
                labels = np.roll(labels, axis=1, shift=1)

        classes = labels[:, 0].tolist() if len(labels) else []
        bboxes = labels[:, 1:] if len(labels) else []

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # 6 because (p_o, x, y, w, h, class)
        # targets is a list of len 3 and targets[0] has shape (3, 13, 13 ,6)
        # ?where is batch_size?
        targets = [torch.zeros((self.num_anchors // 3, int(img.shape[0]/S),
                                int(img.shape[1]/S), 6))
                   for S in self.S]

        for idx, box in enumerate(bboxes):
            # this iou() computer iou just by comparing widths and heights
            # torch.tensor(box[2:4] -> shape (2,) - self.anchors shape -> (9,2)
            # iou_anchors --> tensor of shape (9,)
            iou_anchors = iou_width_height(torch.from_numpy(box[2:4]), self.anchors)
            # sorting anchors from the one with best iou with gt_box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, = box

            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                # i.e. if the best anchor idx is 8, num_anchors_per_scale
                # we know that 8//3 = 2 --> the best scale_idx is 2 -->
                # best_anchor belongs to last scale (52,52)
                # scale_idx will be used to slice the variable "targets"
                # another pov: scale_idx searches the best scale of anchors
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode="floor")
                # print(scale_idx)
                # anchor_on_scale searches the idx of the best anchor in a given scale
                # found via index in the line below
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # slice anchors based on the idx of the best scales of anchors

                scale_y = targets[scale_idx].shape[1]
                scale_x = targets[scale_idx].shape[2]
                # S = self.S[scale_idx]
                # scale_y = int(img.shape[1]/S)
                # scale_x = int(img.shape[2]/S)

                # another problem: in the labels the coordinates of the objects are set
                # with respect to the whole image, while we need them wrt the corresponding (?) cell
                # next line idk how --> i tells which y cell, j which x cell
                # i.e x = 0.5, S = 13 --> int(S * x) = 6 --> 6th cell
                i, j = int(scale_y * y), int(scale_x * x)  # which cell
                # targets[scale_idx] --> shape (3, 13, 13, 6) best group of anchors
                # targets[scale_idx][anchor_on_scale] --> shape (13,13,6)
                # i and j are needed to slice to the right cell
                # 0 is the idx corresponding to p_o
                # I guess [anchor_on_scale, i, j, 0] equals to [anchor_on_scale][i][j][0]
                # check that the anchor hasn't been already taken by another object (rare)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]
                # if not anchor_taken == if anchor_taken is still == 0 cause in the following
                # lines will be set to one
                # if not has_anchor[scale_idx] --> if this scale has not been already taken
                # by another anchor which were ordered in descending order by iou, hence
                # the previous ones are better
                if not anchor_taken and not has_anchor[scale_idx]:
                    # here below we are going to populate all the
                    # 6 elements of targets[scale_idx][anchor_on_scale, i, j]
                    # setting p_o of the chosen cell = 1 since there is an object there
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                    # setting the values of the coordinates x, y
                    # i.e (6.5 - 6) = 0.5 --> x_coord is in the middle of this particular cell
                    # both are between [0,1]
                    x_cell, y_cell = scale_x * x - j, scale_y * y - i  # both between [0,1]
                    # width = 0.5 would be 0.5 of the entire image
                    # and as for x_cell we need the measure w.r.t the cell
                    # i.e S=13, width = 0.5 --> 6.5
                    width_cell, height_cell = (
                        width * scale_x,
                        height * scale_y,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 0:4] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(classes[idx])
                    has_anchor[scale_idx] = True
                # not understood

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = -1  # ignore prediction

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), tuple(targets)

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "labels",
            "adaptive_ann_{}_{}_br_{}.csv".format(name, self.len_ann, int(self.batch_range))
        )

        if os.path.isfile(path):
            print(f"==> Loading cached annotations for rectangular training on {self.annot_folder}")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for 'rectangular training' on training set...")
            annotations["w_h_ratio"] = annotations.iloc[:, 2] / annotations.iloc[:, 1]
            annotations.sort_values(["w_h_ratio"], ascending=True, inplace=True)

            for i in range(0, len(annotations), self.batch_range):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                size[0] = size[0] // 32 * 32
                size[1] = size[1] // 32 * 32
                if i + self.batch_range <= len(annotations):
                    bs = self.batch_range
                else:
                    bs = len(annotations) - i

                annotations.iloc[i:bs, 2] = size[0]
                annotations.iloc[i:bs, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:i+bs, :] = annotations.iloc[i:i+bs, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations.iloc[:, :3])
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == "__main__":
    S = [8, 16, 32]

    anchors = cfg.ANCHORS
    dataset = Training_Dataset(root_directory=cfg.ROOT_DIR,
                               transform=cfg.TRAIN_TRANSFORMS, train=True, rect_training=False,
                               batch_size=1, bboxes_format="yolo", ultralytics_loss=False)
    # dataset = Validation_Dataset(anchors=anchors,
    #                              root_directory=cfg.ROOT_DIR, transform=cfg.TRAIN_TRANSFORMS,
    #                              train=False, S=S, rect_training=False, default_size=640, bs=4,
    #                              bboxes_format="yolo")
    #
    # anchors = torch.tensor(anchors)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                        collate_fn=dataset.collate_fn
                        )


    for idx, (images, bboxes) in enumerate(loader):
        """boxes = cells_to_bboxes(y, anchors, S)[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")"""
        print(f'idx: {idx}', end=' ')
        # boxes = cells_to_bboxes(bboxes, torch.tensor(anchors), S, device=cfg.DEVICE, to_list=False)
        # boxes = non_max_suppression(boxes, iou_threshold=0.6, threshold=0.01, max_detections=300)
        # plot_image(images[0].permute(1, 2, 0).to("cpu"), boxes[0])
