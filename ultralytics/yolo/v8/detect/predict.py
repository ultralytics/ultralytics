import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.utils import ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def postprocess(self, preds):
        return ops.non_max_suppression(preds,
                                       self.args.conf_thres,
                                       self.args.iou_thres,
                                       agnostic=self.args.agnostic_nms,
                                       max_det=self.args.max_det)

    def write_results(self, pred, img, orig_img):
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()

        # Print results
        for c in pred[:, 5].unique():
            n = (pred[:, 5] == c).sum()  # detections per class
            s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        gn = torch.tensor(orig_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(pred):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.save_img or self.args.save_crop or self.args.view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = orig_img.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG.parent, config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "n.pt"
    cfg.source = ROOT / "assets/"
    sz = cfg.img_size
    if type(sz) != int:  # recieved listConfig
        cfg.img_size = [sz[0], sz[0]] if len(cfg.img_size) == 1 else [sz[0], sz[1]]  # expand
    else:
        cfg.img_size = [sz, sz]
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/segment/train.py cfg=yolov5n-seg.yaml data=coco128-segments epochs=100 img_size=640

    TODO:
    Direct cli support, i.e, yolov8 classify_train args.epochs 10
    """
    predict()
