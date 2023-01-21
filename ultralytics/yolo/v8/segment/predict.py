# Ultralytics YOLO 🚀, GPL-3.0 license

import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, is_git_directory, ops
from ultralytics.yolo.utils.plotting import colors, save_one_box
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class SegmentationPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_img, classes=None):
        # TODO: filter by classes
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nm=32,
                                    classes=self.args.classes)
        results = []
        proto = preds[1][-1]
        for i, pred in enumerate(p):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append(Results(boxes=pred[:, :6], orig_shape=shape[:2]))  # save empty boxes
                continue
            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred[:, :6], masks=masks, orig_shape=shape[:2]))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.webcam or self.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        result = results[idx]
        if len(result) == 0:
            return log_string
        det, mask = result.boxes, result.masks  # getting tensors TODO: mask mask,box inherit for tensor

        # Print results
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Mask plotting
        self.annotator.masks(
            mask.masks,
            colors=[colors(x, True) for x in det.cls],
            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
            255 if self.args.retina_masks else im[idx])

        # Segments
        if self.args.save_txt:
            segments = mask.segments

        # Write results
        for j, d in enumerate(reversed(det)):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                seg = segments[j].copy()
                seg = seg.reshape(-1)  # (n,2) to (n*2)
                line = (cls, *seg, conf) if self.args.save_conf else (cls, *seg)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG):
    cfg.model = cfg.model or "yolov8n-seg.pt"
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets" if is_git_directory() \
        else "https://ultralytics.com/images/bus.jpg"
    predictor = SegmentationPredictor(cfg)
    predictor.predict_cli()


if __name__ == "__main__":
    predict()
