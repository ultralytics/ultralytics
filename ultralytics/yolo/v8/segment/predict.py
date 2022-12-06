from pathlib import Path

import hydra
import torch

from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.utils import ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from ..detect.predict import DetectionPredictor


class SegmentationPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_img):
        masks = []
        if len(preds) == 2:  # eval
            p, proto, = preds
        else:  # len(3) train
            p, proto, _ = preds
        # TODO: filter by classes
        p = ops.non_max_suppression(p,
                                    self.args.conf_thres,
                                    self.args.iou_thres,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nm=32)
        for i, pred in enumerate(p):
            if not len(pred):
                continue
            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
                masks.append(ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2]))  # HWC
            else:
                masks.append(ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True))  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()

        return (p, masks)

    def write_results(self, preds, batch, log_string):
        path, im, im0s, vid_cap, s = batch
        preds, masks = preds

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        for i, det in enumerate(preds):  # per image
            self.seen += 1
            if self.webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                log_string += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

            p = Path(p)  # to Path
            self.data_path = p
            save_path = str(self.save_dir / p.name)  # im.jpg
            self.txt_path = str(
                self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
            log_string += '%gx%g ' % im.shape[2:]  # print string
            self.annotator = self.get_annotator(im0)

            if len(det):
                # Segments
                if self.args.save_txt:
                    segments = [
                        ops.scale_segments(im0.shape if self.arg.retina_masks else im.shape[2:],
                                           x,
                                           im0.shape,
                                           normalize=True) for x in reversed(ops.masks2segments(masks[i]))]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                self.annotator.masks(masks[i],
                                     colors=[colors(x, True) for x in det[:, 5]],
                                     im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(
                                         2, 0, 1).flip(0).contiguous() / 255 if self.args.retina_masks else im[i])

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if self.args.save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg, conf) if self.args.save_conf else (cls, *seg)  # label format
                        with open(f'{self.txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.args.save_crop or self.args.view_img:
                        c = int(cls)  # integer class
                        label = None if self.args.hide_labels else (
                            self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                        self.annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if self.args.save_crop:
                        imc = im0s.copy()
                        save_one_box(xyxy,
                                     imc,
                                     file=self.save_dir / 'crops' / self.model.names[c] / f'{p.stem}.jpg',
                                     BGR=True)
            self._stream_results(p)
            self._save_preds(vid_cap, im0, i, save_path)


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG.parent, config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "n.pt"
    cfg.source = cfg.source or ROOT / "assets/"
    sz = cfg.img_size
    if type(sz) != int:  # recieved listConfig
        cfg.img_size = [sz[0], sz[0]] if len(cfg.img_size) == 1 else [sz[0], sz[1]]  # expand
    else:
        cfg.img_size = [sz, sz]
    predictor = SegmentationPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
