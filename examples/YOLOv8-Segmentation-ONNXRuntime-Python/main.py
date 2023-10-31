import argparse
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class Model:
    def __init__(self, *, p, **kwargs):
        self.session = ort.InferenceSession(
            p, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        )  # session
        self.ndtype = np.half if self.session.get_inputs()[0].type == 'tensor(float16)' else np.single  # dtype
        self.iShapes = [x.shape for x in self.session.get_inputs()] # input shapes
        self.im_meta = dict()
        self.conf_threshold = kwargs.get('conf_threshold', 0.4)
        self.iou_threshold = kwargs.get('iou_threshold', 0.45)
        self.nm = kwargs.get('nms', 32)
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']
        self.palette = np.random.uniform(0, 255, size=(len(self.classes), 3))


    def __call__(self, x, **kwargs):
        self._preprocess(x)
        preds = self.session.run(None, {self.session.get_inputs()[0].name: self.im_meta['im']})
        x, proto = np.einsum('bcn->bnc', preds[0]), preds[1][0]   # prediction, proto
        x = x[np.amax(x[..., 4: -self.nm], axis=-1) > self.conf_threshold]  # filter by conf
        x = np.c_[x[..., :4], np.amax(x[..., 4: -self.nm], axis=-1), np.argmax(x[..., 4: -self.nm], axis=-1), x[..., -self.nm: ]]  # box, score, cls
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], self.conf_threshold, self.iou_threshold)]  # nms
        if len(x) == 0:
            return [], [], []
        else: 
            # de-scale boxes
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]
            x[..., :4] -= [self.im_meta['dw'], self.im_meta['dh'], self.im_meta['dw'], self.im_meta['dh']]
            x[..., :4] /= min(self.im_meta['ratio'])
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, self.im_meta['im0'].shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, self.im_meta['im0'].shape[0])
            y_segments, y_masks, y_boxes = list(), list(), list()
            masks = self.process_mask_hq(proto, x[:, 6:], x[:, :4], self.im_meta['im0'].shape)
            segments = self.mask2segs(masks)  # mask2segments

            # visualize
            for segment in segments:
                cv2.polylines(self.im_meta['im0'], np.int32([segment]), True, (255, 255, 0), 2)
                # cv2.fillPoly(self.im_meta['im0'], np.int32([segment]), (255, 255, 0))
            for *box, conf, cls_ in x[..., :6]:
                cv2.rectangle(
                    self.im_meta['im0'],
                    (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                    self.palette[int(cls_)], 0, cv2.LINE_AA
                 )
                cv2.putText(
                    self.im_meta['im0'], 
                    f'{self.classes[cls_]}: {conf:.3f}', 
                    (int(box[0]), int(box[1] - 9)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    self.palette[int(cls_)], 
                    2, cv2.LINE_AA
                )
            cv2.imshow('demo', self.im_meta['im0'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('demo.jpg', self.im_meta['im0'])

            return x[..., :6], segments, masks


    def _preprocess(self, x) -> dict:
        self.im_meta['im0'] = x
        shape = x.shape[:2]  # current shape
        new_shape = self.iShapes[0][-2:]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        self.im_meta['ratio'] = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        self.im_meta['dw'], self.im_meta['dh'] = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            x = cv2.resize(x, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.im_meta['dh'] - 0.1)), int(round(self.im_meta['dh'] + 0.1))
        left, right = int(round(self.im_meta['dw'] - 0.1)), int(round(self.im_meta['dw'] + 0.1))
        x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
        x = np.ascontiguousarray(np.einsum('asd->das', x)[::-1], dtype=self.ndtype) / 255.0  # tranform
        x = x[None] if len(x.shape) == 3 else x
        self.im_meta['im'] = x


    @staticmethod
    def mask2segs(masks):
        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments


    @staticmethod
    def crop_mask(masks, boxes):
        """
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


    def process_mask_hq(self, protos, masks_in, bboxes, im0shape):
        c, mh, mw = protos.shape  
        mask_protos = np.reshape(protos, (c, -1))
        matmulres = np.matmul(masks_in, mask_protos) 
        masks = np.reshape(matmulres, (-1, mh, mw))  
        masks = np.ascontiguousarray(masks.transpose(1, 2, 0))
        masks = self.scale_mask(masks.shape[:2], masks, im0shape) 
        masks = np.einsum('zxc->czx', masks)
        masks = self.crop_mask(masks, bboxes.copy()) 
        masks_gt = np.greater(masks, 0.5)
        masks_gt = masks_gt.astype(float)
        return masks_gt


    @staticmethod
    def scale_mask(im1_shape, masks, im0_shape, ratio_pad=None):
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]   
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_CUBIC)  # 
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks



if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model ')
    parser.add_argument('--source', type=str, required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    args = parser.parse_args()

    # build model
    model = Model(p=args.model, conf_threshold=args.conf, iou_threshold=args.iou)
    
    # load image
    img = cv2.imread(args.source)

    # infer
    y_bboxes, y_segments, y_masks = model(img)
    print(y_bboxes)
