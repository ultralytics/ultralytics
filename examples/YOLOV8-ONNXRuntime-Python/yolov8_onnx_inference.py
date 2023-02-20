import os
import cv2
import argparse
import onnxruntime
import numpy as np
from typing import List, Tuple, Dict

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    '''
    box and boxes are format as [x1, y1, x2, y2]
    '''
    # inter area
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, xmax-xmin) * np.maximum(0, ymax-ymin)

    # union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / union_area

def nms_process(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx+1]
    return keep_idx
    
class Yolov8Inference(object):
    ''' yolov8 onnxruntime inference
    '''

    def __init__(self,
                 onnx_path: str,
                 input_size: Tuple[int],
                 class_names: Tuple[str],
                 score_thr=0.25,
                 nms_thr=0.2) -> None:
        assert onnx_path.endswith('.onnx'), f"invalid onnx model: {onnx_path}"
        assert os.path.exists(onnx_path), f"model not found: {onnx_path}"
        self.sess = onnxruntime.InferenceSession(onnx_path)
        print("input info: ", self.sess.get_inputs()[0])
        print("output info: ", self.sess.get_outputs()[0])
        self.input_size = input_size
        self.class_names = class_names
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        np.random.seed(0)
        self.color_list = np.random.randint(0, 255, (len(class_names), 3)).tolist()

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        ''' preprocess image for model inference
        '''
        input_w, input_h = self.input_size
        if len(img.shape) == 3:
            padded_img = np.ones((input_w, input_h, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114
        r = min(input_w / img.shape[0], input_h / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        # (H, W, C) BGR -> (C, H, W) RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def _postprocess(self, output: List[np.ndarray], ratio) -> Dict:
        pred = np.squeeze(output[0]).T
        scores = np.max(pred[:,4:], axis=1)
        pred = pred[scores > self.score_thr, :]
        scores = scores[scores > self.score_thr]

        result = {'boxes': [], 'idxes': [], 'scores': []}
        if pred.shape[0] == 0:
            return result
        
        class_idx = np.argmax(pred[:, 4:], axis=1)
        boxes = pred[:, :4] / ratio
        boxes = xywh2xyxy(boxes)
        keep_idx = nms_process(boxes, scores, self.nms_thr)
        result['boxes'] = boxes[keep_idx, ].tolist()
        result['idxes'] = class_idx[keep_idx, ].tolist()
        result['scores'] = scores[keep_idx, ].tolist()
        return result
    
    def detect(self, img: np.ndarray) -> Dict:
        img, ratio = self._preprocess(img)
        ort_input = {self.sess.get_inputs()[0].name: img[None, :]/255}
        output = self.sess.run(None, ort_input)
        result = self._postprocess(output, ratio)
        return result
    
    def draw_result(self, img: np.ndarray, result: Dict) -> np.ndarray:
        boxes = result['boxes']
        idxes = result['idxes']
        scores = result['scores']

        for box, idx, score in zip(boxes, idxes, scores):
            x1, y1, x2, y2 = list(map(int, box))
            cls_name = self.class_names[idx]
            color = self.color_list[idx]
            label = "{}: {:.0f}%".format(cls_name, score*100)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1), (x1+label_size[0], y1+label_size[1]+baseline),
                color, -1)
            cv2.putText(img, label, (x1, y1+label_size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
        return img

def parse_args():
    parser = argparse.ArgumentParser("yolov8 onnx inference demo")
    parser.add_argument("demo", type=str,
        help="demo type, image or stream")
    parser.add_argument("model", type=str,
        help="onnx model path")
    parser.add_argument("input", type=str, default="0",
        help="camera id | video path | image path")
    parser.add_argument("--score_thr", type=float, default=0.25,
        help="score threshold")
    parser.add_argument("--nms_thr", type=float, default=0.5,
        help="nms threshold")
    parser.add_argument("--input_size", type=int, default=640,
        help="inpur size")
    parser.add_argument("--classes_txt", type=str, default='',
        help="class names txt file.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def demo_image(args):
    input_size = (args.input_size, args.input_size)
    class_names = COCO_CLASSES
    if os.path.exists(args.classes_txt):
        with open(args.classes_txt, 'r') as f:
            class_names = tuple([name.strip() for name in f.readlines()])

    yolo = Yolov8Inference(args.model, input_size, class_names,
                           args.score_thr, args.nms_thr)
    img = cv2.imread(args.input)
    if img is None:
        print(f"read image faild: {args.input}")
        return
    result = yolo.detect(img)
    img = yolo.draw_result(img, result)
    boxes = result['boxes']
    idxes = result['idxes']
    scores = result['scores']
    for box, idx, score in zip(boxes, idxes, scores):
        box = list(map(int, box))
        print(f"{class_names[idx]}: {score:.3}\t{box}")
        
    if (args.show):
        cv2.imshow("detect", img)
        cv2.waitKey(0)

def demo_stream(args):
    input_size = (args.input_size, args.input_size)
    class_names = COCO_CLASSES
    if os.path.exists(args.classes_txt):
        with open(args.classes_txt, 'r') as f:
            class_names = tuple([name.strip() for name in f.readlines()])

    yolo = Yolov8Inference(args.model, input_size, class_names,
                           args.score_thr, args.nms_thr)

    if (len(args.input)) == 1:
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = yolo.detect(frame)
        frame = yolo.draw_result(frame, result)
        if args.show:
            cv2.imshow("detect", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            num = len(result['boxes'])
            print(f"detect {num} objects")

def main():
    args = parse_args()
    if args.demo == 'image':
        return demo_image(args)
    elif args.demo == 'stream':
        return demo_stream(args)

if __name__ == "__main__":
    main()
