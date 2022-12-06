import platform
from pathlib import Path

import cv2
import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.utils import LOGGER, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf_thres,
                                        self.args.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for pred in preds:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()

        return preds

    def _stream_results(self, p):
        im0 = self.annotator.result()
        if self.args.view_img:
            if platform.system() == 'Linux' and p not in self.windows:
                self.windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

    def _save_preds(self, vid_cap, im0, idx, save_path):
        # save imgs
        if self.save_img:
            if self.dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if self.vid_path[idx] != save_path:  # new video
                    self.vid_path[idx] = save_path
                    if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                        self.vid_writer[idx].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.vid_writer[idx].write(im0)

    def write_results(self, preds, batch, log_string):
        path, im, im0s, vid_cap, s = batch
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        for i, pred in enumerate(preds):  # per image
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

            if len(pred):
                for c in pred[:, 5].unique():
                    n = (pred[:, 5] == c).sum()  # detections per class
                    print_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

                # write
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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
                        imc = im0s.copy()
                        save_one_box(xyxy,
                                     imc,
                                     file=self.save_dir / 'crops' / self.model.model.names[c] /
                                     f'{self.data_path.stem}.jpg',
                                     BGR=True)

            self._stream_results(p)
            self._save_preds(vid_cap, im0, i, save_path)


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
    predict()
