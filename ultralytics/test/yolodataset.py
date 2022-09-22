from ultralytics.yolo.data import YOLODetectionDataset, YOLOSegmentDataset, YOLOPoseDataset, MixAndRectDataset
import cv2
import numpy as np


dataset = MixAndRectDataset(
    # dataset=YOLODetectionDataset(
    dataset=YOLOSegmentDataset(
        img_path="/d/dataset/COCO/images/val2017",
        img_size=640,
        label_path=None,
        cache_images=False,
        augment=False,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
    )
)

color = (255, 255, 0)
for d in dataset:
    img = d["img"]
    # print(img.shape)
    cls = d["cls"]
    bboxes = d["bboxes"]
    # print(cls.shape, bboxes.shape)
    masks = d["masks"]
    # keypoints = d["keypoints"]

    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = b
        # print(x1, y1, x2, y2)
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # mask = masks[i]
        # img[mask] = img[mask] * 0.5 + np.array(color) * 0.5
    cv2.imshow("p", img)
    if cv2.waitKey(0) == ord("q"):
        break
