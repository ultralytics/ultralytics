# Ultralytics YOLO 🚀, GPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT
from ultralytics.yolo.utils.plotting import Annotator


class ClassificationPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, example=str(self.model.names), pil=True)

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, probs=pred))

        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        result = results[idx]
        if len(result) == 0:
            return log_string
        prob = result.probs
        # Print results
        n5 = min(len(self.model.names), 5)
        top5i = prob.argsort(0, descending=True)[:n5].tolist()  # top 5 indices
        log_string += f"{', '.join(f'{self.model.names[j]} {prob[j]:.2f}' for j in top5i)}, "

        # write
        text = '\n'.join(f'{prob[j]:.2f} {self.model.names[j]}' for j in top5i)
        if self.args.save or self.args.show:  # Add bbox to image
            self.annotator.text((32, 32), text, txt_color=(255, 255, 255))
        if self.args.save_txt:  # Write to file
            with open(f'{self.txt_path}.txt', 'a') as f:
                f.write(text + '\n')

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = ClassificationPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
