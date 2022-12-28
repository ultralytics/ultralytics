import cv2
import numpy as np
from omegaconf import OmegaConf

from ultralytics.yolo.data import build_dataloader


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_one_box(x, img, keypoints=None, color=None, label=None, line_thickness=None):
    import random

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    if keypoints is not None:
        plot_keypoint(img, keypoints, color, tl)


def plot_keypoint(img, keypoints, color, tl):
    num_l = len(keypoints)
    # clors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(0, 255, 255)]
    # clors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_l)]
    for i in range(num_l):
        point_x = int(keypoints[i][0])
        point_y = int(keypoints[i][1])
        cv2.circle(img, (point_x, point_y), tl + 3, color, -1)


with open("ultralytics/tests/data/dataloader/hyp_test.yaml") as f:
    hyp = OmegaConf.load(f)


def test(augment, rect):
    dataloader, _ = build_dataloader(
        img_path="/d/dataset/COCO/images/val2017",
        imgsz=640,
        label_path=None,
        cache=False,
        hyp=hyp,
        augment=augment,
        prefix="",
        rect=rect,
        batch_size=4,
        stride=32,
        pad=0.5,
        use_segments=False,
        use_keypoints=True,
    )

    for d in dataloader:
        idx = 1  # show which image inside one batch
        img = d["img"][idx].numpy()
        img = np.ascontiguousarray(img.transpose(1, 2, 0))
        ih, iw = img.shape[:2]
        # print(img.shape)
        bidx = d["batch_idx"]
        cls = d["cls"][bidx == idx].numpy()
        bboxes = d["bboxes"][bidx == idx].numpy()
        bboxes[:, [0, 2]] *= iw
        bboxes[:, [1, 3]] *= ih
        keypoints = d["keypoints"][bidx == idx]
        keypoints[..., 0] *= iw
        keypoints[..., 1] *= ih
        # print(keypoints, keypoints.shape)
        # print(d["im_file"])

        for i, b in enumerate(bboxes):
            x, y, w, h = b
            x1 = x - w / 2
            x2 = x + w / 2
            y1 = y - h / 2
            y2 = y + h / 2
            c = int(cls[i][0])
            # print(x1, y1, x2, y2)
            plot_one_box([int(x1), int(y1), int(x2), int(y2)],
                         img,
                         keypoints=keypoints[i],
                         label=f"{c}",
                         color=colors(c))
        cv2.imshow("p", img)
        if cv2.waitKey(0) == ord("q"):
            break


if __name__ == "__main__":
    test(augment=True, rect=False)
    test(augment=False, rect=True)
    test(augment=False, rect=False)
