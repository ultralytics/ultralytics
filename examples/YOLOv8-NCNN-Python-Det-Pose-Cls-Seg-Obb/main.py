import argparse
import os

import cv2
import ncnn
import numpy as np
import yaml


# ultralytics.utils.plotting Colors
class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (np.ndarray): A numpy array of shape (N, 5) representing rotated bounding boxes, with xywhr format.
    Returns:
        (np.ndarray): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignored the center points (the first two columns) cause it's not needed here.
    gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)

    cov_matrix_1 = a * np.cos(c) ** 2 + b * np.sin(c) ** 2
    cov_matrix_2 = a * np.sin(c) ** 2 + b * np.cos(c) ** 2
    cov_matrix_3 = a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c)
    return (cov_matrix_1, cov_matrix_2, cov_matrix_3)


def xywhr2xyxyxyxy(rboxes):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4].

    Rotation values should
    be in degrees from 0 to 90.
    Args:
        rboxes (numpy.ndarray): Input data in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).
    Returns:
        numpy.ndarray: Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin = (np.cos, np.sin)

    ctr = rboxes[..., :2]
    w, h, angle = (rboxes[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    Args:
        obb1 (np.ndarray): An array of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (np.ndarray): An array of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, oy1ptional): A small value to avoid division by zero. Defaults to 1e-7.
    Returns:
        np.ndarray: An array of shape (N, M) representing obb similarities.
    """
    x1, y1 = obb1[:, 0].reshape(-1, 1), obb1[:, 1].reshape(-1, 1)
    x2, y2 = obb2[:, 0].reshape(1, -1), obb2[:, 1].reshape(1, -1)

    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    a2 = a2.reshape(1, -1)
    b2 = b2.reshape(1, -1)
    c2 = c2.reshape(1, -1)

    t1 = (
        ((a1 + a2) * (np.power(y1 - y2, 2)) + (b1 + b2) * (np.power(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)))
            / (4 * np.sqrt((a1 * b1 - np.power(c1, 2)).clip(0) * (a2 * b2 - np.power(c2, 2)).clip(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = np.clip(bd, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for obbs, powered by probiou and fast-nms.

    Args:
        boxes (np.ndarray): (N, 5), xywhr.
        scores (np.ndarray): (N, ).
        threshold (float): IoU threshold.
    Returns:
        np.ndarray: Indices of selected boxes after NMS.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)

    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)

    pick = np.where(ious.max(axis=0) < threshold)[0]
    return sorted_idx[pick]


class YoloV8:
    def __init__(
        self,
        modelPath,
        modelType,  # Detection: "", Pose: "pose", Classification: "cls", Segmentation: "seg", OBB: "obb"
        targetSize=640,
        probThreshold=0.25,
        probObbThreshold=0.15,
        nmsThreshold=0.45,
        topK=5,
        numThreads=1,
        useGpu=False,
    ):
        self.modelPath = modelPath
        assert modelType in ["det", "pose", "cls", "seg", "obb"]
        self.modelType = modelType
        self.targetSize = targetSize

        self.probThreshold = probThreshold
        self.probObbThreshold = probObbThreshold
        self.nmsThreshold = nmsThreshold
        self.topK = topK
        self.numThreads = numThreads
        self.useGpu = useGpu

        self.cocoPosePointNum = 17

        self.meanVals = []
        self.normVals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.useGpu
        self.net.opt.num_threads = self.numThreads

        self.colorPalette = Colors()

        self.KPS_COLORS = [
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (51, 153, 255),
            (51, 153, 255),
            (51, 153, 255),
            (51, 153, 255),
            (51, 153, 255),
            (51, 153, 255),
        ]

        self.SKELETON = [
            (16, 14),
            (14, 12),
            (17, 15),
            (15, 13),
            (12, 13),
            (6, 12),
            (7, 13),
            (6, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (9, 11),
            (2, 3),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
        ]

        self.LIMB_COLORS = [
            (51, 153, 255),
            (51, 153, 255),
            (51, 153, 255),
            (51, 153, 255),
            (255, 51, 255),
            (255, 51, 255),
            (255, 51, 255),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0),
        ]

        self.CocoClasses = None
        CocoYamlPath = "../ultralytics/cfg/datasets/coco.yaml"
        with open(CocoYamlPath, "r", encoding="utf-8") as f:
            CocoClasses = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.CocoClasses = list(CocoClasses["names"].values())

        self.CocoPoseClasses = None
        self.CocoPoseKptShape = None
        CocoPoseYamlPath = "../ultralytics/cfg/datasets/coco-pose.yaml"
        with open(CocoPoseYamlPath, "r", encoding="utf-8") as f:
            CocoPoseClasses = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.CocoPoseClasses = list(CocoPoseClasses["names"].values())
            self.CocoPoseKptShape = CocoPoseClasses["kpt_shape"]

        self.ImageNetClasses = None
        ImageNetYamlPath = "../ultralytics/cfg/datasets/ImageNet.yaml"
        with open(ImageNetYamlPath, "r", encoding="utf-8") as f:
            ImageNetYaml = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.ImageNetClasses = list(ImageNetYaml["names"].values())

        self.CocoSegClasses = None
        self.segNumMask = 32
        CocoSegYamlPath = "../ultralytics/cfg/datasets/coco128-seg.yaml"
        with open(CocoSegYamlPath, "r", encoding="utf-8") as f:
            CocoSegClasses = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.CocoSegClasses = list(CocoSegClasses["names"].values())

        self.DOTAv1Classes = None
        DOTAv1YamlPath = "../ultralytics/cfg/datasets/DOTAv1.yaml"
        with open(DOTAv1YamlPath, "r", encoding="utf-8") as f:
            DOTAv1Classes = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.DOTAv1Classes = list(DOTAv1Classes["names"].values())

        self.net.load_param(os.path.join(self.modelPath, "model.ncnn.param"))
        self.net.load_model(os.path.join(self.modelPath, "model.ncnn.bin"))

    def __del__(self):
        self.net = None

    def __call__(self, img):
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]

        w = imgWidth
        h = imgHeight
        scale = 1.0
        if w > h:
            scale = float(self.targetSize) / w
            w = self.targetSize
            h = int(h * scale)
        else:
            scale = float(self.targetSize) / h
            h = self.targetSize
            w = int(w * scale)

        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, imgWidth, imgHeight, w, h)

        # ultralytics/data/augment.py LetterBox(center=True, stride=32)
        wpad = (self.targetSize + 31) // 32 * 32 - w
        hpad = (self.targetSize + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )

        mat_in_pad.substract_mean_normalize(self.meanVals, self.normVals)

        ex = self.net.create_extractor()
        ex.input("in0", mat_in_pad)

        out = []
        if self.modelType == "seg":
            _, out0 = ex.extract("out0")
            out.append(np.expand_dims(np.array(out0), axis=0))
            _, out1 = ex.extract("out1")
            out.append(np.expand_dims(np.array(out1), axis=0))
        else:
            _, out0 = ex.extract("out0")
            out.append(np.expand_dims(np.array(out0), axis=0))

        # yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
        if self.modelType == "det":
            output = out[0].squeeze()
            output = np.array(cv2.transpose(output))
            scores = output[:, 4:]
            boxes = output[:, 0:4]
            maxScores = np.max(np.array(scores), axis=1)
            maxIndex = np.argmax(np.array(scores), axis=1)

            mask = maxScores > self.probThreshold

            maxScores = np.array(maxScores)[mask]
            maxIndex = np.array(maxIndex)[mask]
            boxes = np.array(boxes)[mask]

            boxesId = cv2.dnn.NMSBoxes(boxes.tolist(), maxScores.tolist(), self.probThreshold, self.nmsThreshold)

            maxScores = maxScores[boxesId]
            maxIndex = maxIndex[boxesId]

            retBoxes = np.array(boxes[boxesId]) / scale
            retBoxes[:, 0] -= int(wpad // 2 / scale)
            retBoxes[:, 1] -= int(hpad // 2 / scale)

            return (maxScores, retBoxes, maxIndex)

        # yolov8 has an output of shape (batchSize, 56,  8400) (17 x point[x,y,prop] + prop + box[x,y,w,h])
        if self.modelType == "pose":
            output = out[0].squeeze()
            output = np.array(cv2.transpose(output))
            scores = output[:, 4]
            boxes = output[:, 0:4]
            kpts = output[:, 5:]

            mask = np.array(scores) > self.probThreshold

            scores = np.array(scores)[mask]
            boxes = np.array(boxes)[mask]
            kpts = np.array(kpts)[mask]

            boxesId = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.probThreshold, 0.5)

            scores = scores[boxesId]
            ret_boxes = np.array(boxes[boxesId]) / scale
            ret_kpts = np.array(kpts[boxesId]) / scale

            ret_boxes[:, 0] -= int(wpad // 2 / scale)
            ret_boxes[:, 1] -= int(hpad // 2 / scale)
            ret_kpts[:, 0::3] -= int(wpad // 2 / scale)
            ret_kpts[:, 1::3] -= int(hpad // 2 / scale)

            return (scores, ret_boxes, ret_kpts)

        # yolov8 has an output of shape (batchSize, 1000,  1) (Num classes)
        if self.modelType == "cls":
            output = out[0].squeeze()
            output = np.array(cv2.transpose(output))
            scores = output[:]
            topkIdx = self._topkIndices(scores.squeeze(), self.topK)

            return topkIdx

        if self.modelType == "seg":
            output1 = out[0].squeeze()
            output1 = np.array(cv2.transpose(output1))
            output2 = out[1].squeeze()

            scores = output1[:, 4 : -1 * self.segNumMask]
            boxes = output1[:, :4]
            outputMask = output1[:, -1 * self.segNumMask :]

            maxScores = np.max(np.array(scores), axis=1)
            maxIndex = np.argmax(np.array(scores), axis=1)

            mask = maxScores > self.probThreshold

            boxes = boxes[mask]
            maxScores = maxScores[mask]
            maxIndex = maxIndex[mask]
            outputMask = outputMask[mask]

            boxesId = cv2.dnn.NMSBoxes(boxes, maxScores, self.probThreshold, self.nmsThreshold)

            ret_socres = maxScores[boxesId]
            max_index = maxIndex[boxesId]
            outputMask = outputMask[boxesId]

            ret_boxes = np.array(boxes[boxesId])
            ret_boxes[..., [0, 1]] -= ret_boxes[..., [2, 3]] / 2
            ret_boxes[..., [2, 3]] += ret_boxes[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            ret_boxes[..., :4] -= [wpad // 2, hpad // 2, wpad - wpad // 2, hpad - hpad // 2]
            ret_boxes[..., :4] /= scale

            im0 = img
            # Bounding boxes boundary clamp
            ret_boxes[..., [0, 2]] = ret_boxes[:, [0, 2]].clip(0, im0.shape[1])
            ret_boxes[..., [1, 3]] = ret_boxes[:, [1, 3]].clip(0, im0.shape[0])

            c, mh, mw = output2.shape
            masks = np.matmul(outputMask, output2.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
            masks = np.ascontiguousarray(masks)

            # masks rescale
            im1_shape = masks.shape[:2]
            im0_shape = img.shape[:2]

            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding

            # Calculate tlbr of mask
            top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
            bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))

            masks = masks[top:bottom, left:right]
            masks = cv2.resize(
                masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
            )  # INTER_CUBIC would be better

            masks = masks.transpose(2, 0, 1)

            # crop mask
            n, h, w = masks.shape
            boxes2 = ret_boxes
            x1, y1, x2, y2 = np.split(boxes2[:, :, None], 4, 1)
            r = np.arange(w, dtype=x1.dtype)[None, None, :]
            c = np.arange(h, dtype=x1.dtype)[None, :, None]
            masks = masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
            masks = np.greater(masks, 0.5)

            # segment
            segments = []
            for x in masks.astype("uint8"):
                c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
                if c:
                    c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
                else:
                    c = np.zeros((0, 2))  # no segments found
                segments.append(c.astype("float32"))

            ret_box = (ret_boxes, ret_socres, max_index)

            return (ret_box, segments)

        # yolov8 has an output of shape (batchSize, 21504,  20) (Num classes + box[x,y,w,h] + r)
        if self.modelType == "obb":
            output = out[0].squeeze()
            output = np.array(cv2.transpose(output))
            scores = output[:, 4:-1]
            xywhr = np.concatenate((output[:, :4], np.expand_dims(output[:, -1], axis=1)), axis=1)
            boxes = output[:, :4]

            maxScores = np.max(np.array(scores), axis=1)
            maxIndex = np.argmax(np.array(scores), axis=1)

            mask = maxScores > self.probObbThreshold  # too high

            maxScores = np.array(maxScores)[mask]
            maxIndex = np.array(maxIndex)[mask]
            boxes = np.array(boxes)[mask]
            xywhr = xywhr[mask]

            boxesId = nms_rotated(xywhr, maxScores)
            xywhr = xywhr[boxesId]
            maxScores = maxScores[boxesId]
            maxIndex = maxIndex[boxesId]
            dotaPoint = xywhr2xyxyxyxy(xywhr)

            return dotaPoint

    def _draw_det(self, img, confidence, box, max_index):
        """
        Draw bounding boxes and labels on the input image based on detection results.

        Args:
            img (np.ndarray): Input image.
            confidence (np.ndarray): Confidence scores for each detected object.
            box (np.ndarray): Bounding boxes in (x, y, width, height) format.
            max_index (np.ndarray): Index of the predicted class for each detection.
        """

        for idx in range(len(confidence)):
            score = f"{confidence[idx]:.2f}"
            # Clip coordinates to ensure they are within the image boundaries
            x = int(np.clip(box[idx][0] - box[idx][2] / 2, 0, img.shape[1]))
            y = int(np.clip(box[idx][1] - box[idx][3] / 2, 0, img.shape[0]))
            x_plus_w = int(np.clip(x + box[idx][2], 0, img.shape[1]))
            y_plus_h = int(np.clip(y + box[idx][3], 0, img.shape[0]))

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)

            # Draw label with class name and confidence score
            label = f"{self.CocoClasses[max_index[idx]]}:{score}"
            cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def _draw_pose(self, img, confidence, box, kpts):
        """
        Draw skeleton and keypoints on the input image based on pose detection results.
        Args:
            img (np.ndarray): Input image.
            confidence (np.ndarray): Confidence scores for each detected pose.
            box (np.ndarray): Bounding boxes in (x, y, width, height) format.
            kpts (np.ndarray): Detected keypoints for each pose.
        Note:
            `kpts` should be a NumPy array of shape (num_poses, num_keypoints * 3), where each row contains
            (x, y, score) for each keypoint in the order specified by your implementation.
        """

        for idx in range(len(confidence)):
            score = f"{confidence[idx]:.2f}"
            x = int(np.clip(box[idx][0] - box[idx][2] / 2, 0, img.shape[1]))
            y = int(np.clip(box[idx][1] - box[idx][3] / 2, 0, img.shape[0]))
            x_plus_w = int(np.clip(x + box[idx][2], 0, img.shape[1]))
            y_plus_h = int(np.clip(y + box[idx][3], 0, img.shape[0]))

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
            cv2.putText(img, score, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for k in range(self.cocoPosePointNum + 2):
                if k < self.cocoPosePointNum:
                    kps_x = int(kpts[idx][k * 3 + 0])
                    kps_y = int(kpts[idx][k * 3 + 1])
                    kps_s = kpts[idx][k * 3 + 2]

                    if kps_s > 0.5:
                        kps_color = self.KPS_COLORS[k]
                        cv2.circle(img, (kps_x, kps_y), 5, kps_color, -1)

                    ske = self.SKELETON[k]
                    pos1_x = int(kpts[idx][(ske[0] - 1) * 3])
                    pos1_y = int(kpts[idx][(ske[0] - 1) * 3 + 1])

                    pos2_x = int(kpts[idx][(ske[1] - 1) * 3])
                    pos2_y = int(kpts[idx][(ske[1] - 1) * 3 + 1])

                    pos1_s = int(kpts[idx][(ske[0] - 1) * 3 + 2])
                    pos2_s = int(kpts[idx][(ske[1] - 1) * 3 + 2])

                    if pos1_s > 0.5 and pos2_s > 0.5:
                        limb_color = self.LIMB_COLORS[k]
                        cv2.line(img, (pos1_x, pos1_y), (pos2_x, pos2_y), limb_color, 2)

    def _topkIndices(self, arr, k):
        """
        Returns the indices of the top-k elements in the input array.

        Args:
            arr (np.ndarray): Input one-dimensional array.
            k (int): Number of top elements to retrieve.
        Returns:
            np.ndarray: Indices of the top-k elements.
        """
        if k <= 0 or k > len(arr):
            raise ValueError("Invalid value for 'k'")
        indices = np.argsort(arr)[-k:]

        return indices

    def _draw_seg(self, im, bboxes, segments):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for idx in range(len(segments)):
            box = bboxes[0][idx]
            conf = bboxes[1][idx]
            cls_ = bboxes[2][idx]
            segment = segments[idx]
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.colorPalette(int(cls_), bgr=True))

            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.colorPalette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.CocoSegClasses[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colorPalette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

    def _draw_rbox(self, image, dotaPoint):
        """
        Draw rotated bounding box on the input image based on four points.

        Args:
            image (np.ndarray): Input image.
            dotaPoint (np.ndarray): Four points representing the rotated bounding box.
        Note:
            `dotaPoint` should be a NumPy array of shape (4, 2), where each row contains (x, y) coordinates of a point.
        """

        scaledCoordinates = dotaPoint.astype(np.int16)
        for data_points in scaledCoordinates:
            for x, y in data_points:
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            for i in range(3):
                cv2.line(image, tuple(data_points[i]), tuple(data_points[i + 1]), (0, 255, 0), 2)
            # Connect the fourth point and the first point to form a closed shape
            cv2.line(image, tuple(data_points[3]), tuple(data_points[0]), (0, 255, 0), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/yolov8s_ncnn_model", help="Input NCNN model path.")
    parser.add_argument(
        "--img", default="../ultralytics/assets/bus.jpg", help="Path to input image."
    )  # DOTAv1 P3185.png(imgs folder)
    args = parser.parse_args()

    img = cv2.imread(args.img)

    if args.model.split("/")[-1] == "yolov8s_ncnn_model":
        yolov8 = YoloV8(args.model, "det")
        maxScores, retBoxes, maxIndex = yolov8(img)
        yolov8._draw_det(img, maxScores, retBoxes, maxIndex)
        cv2.imwrite(os.path.join(args.model, "det.png"), img)

    elif args.model.split("/")[-1] == "yolov8s-pose_ncnn_model":
        yolov8pose = YoloV8(args.model, "pose")
        scores, ret_boxes, ret_kpts = yolov8pose(img)
        yolov8pose._draw_pose(img, scores, ret_boxes, ret_kpts)
        cv2.imwrite(os.path.join(args.model, "pose.png"), img)

    elif args.model.split("/")[-1] == "yolov8s-cls_ncnn_model":
        yolov8cls = YoloV8(args.model, "cls", 224)
        topkIdx = yolov8cls(img)
        for idx in range(len(topkIdx)):
            idx = topkIdx[idx]
            print(yolov8cls.ImageNetClasses[idx])

    elif args.model.split("/")[-1] == "yolov8s-seg_ncnn_model":
        yolov8seg = YoloV8(args.model, "seg")
        ret_box, segments = yolov8seg(img)
        yolov8seg._draw_seg(img, ret_box, segments)
        cv2.imwrite(os.path.join(args.model, "seg.png"), img)

    elif args.model.split("/")[-1] == "yolov8s-obb_ncnn_model":
        yolov8obb = YoloV8(args.model, "obb", 1024)  # obb has not use letterBox
        dotaPoint = yolov8obb(img)
        yolov8obb._draw_rbox(img, dotaPoint)
        cv2.imwrite(os.path.join(args.model, "obb.png"), img)
