# predictor engine by Ultralytics
"""
Run prection on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ yolo task=... mode=predict  model=s.pt --source 0                         # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ yolo task=... mode=predict --weights yolov5s.pt          # PyTorch
                                    yolov5s.torchscript        # TorchScript
                                    yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov5s_openvino_model     # OpenVINO
                                    yolov5s.engine             # TensorRT
                                    yolov5s.mlmodel            # CoreML (macOS-only)
                                    yolov5s_saved_model        # TensorFlow SavedModel
                                    yolov5s.pb                 # TensorFlow GraphDef
                                    yolov5s.tflite             # TensorFlow Lite
                                    yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov5s_paddle_model       # PaddlePaddle
    """
import platform
from pathlib import Path

import cv2
import torch

from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadScreenshots, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS, check_dataset, check_dataset_yaml
from ultralytics.yolo.utils import LOGGER, ROOT, TQDM_BAR_FORMAT, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imshow
from ultralytics.yolo.utils.configs import get_config
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.modeling.autobackend import AutoBackend
from ultralytics.yolo.utils.plotting import Annotator
from ultralytics.yolo.utils.torch_utils import check_img_size, select_device, strip_optimizer

DEFAULT_CONFIG = ROOT / "yolo/utils/configs/default.yaml"


class BasePredictor:

    def __init__(self, config=DEFAULT_CONFIG, overrides={}):
        self.args = get_config(config, overrides)
        self.save_dir = increment_path(Path(self.args.project) / self.args.name, exist_ok=self.args.exist_ok)
        (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        self.done_setup = False

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.view_img = None
        self.annotator = None
        self.data_path = None

    def get_annotator(self, img):
        raise NotImplementedError("get_annotator function needs to be implemented")

    def write_results(self, pred, img, orig_img, print_string):
        raise NotImplementedError("print_results function needs to be implemented")

    def postprocess(self, preds):
        return preds

    def setup(self, source=None, model=None):
        # source
        source = str(source or self.args.source)
        self.save_img = not self.args.nosave and not source.endswith('.txt')
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # data
        if self.data:
            if self.data.endswith(".yaml"):
                self.data = check_dataset_yaml(self.data)
            else:
                self.data = check_dataset(self.data)

        # model
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)  # NOTE: not passing data
        stride, pt = model.stride, model.pt
        imgsz = check_img_size(self.args.img_size, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            self.view_img = check_imshow(warn=True)
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.args.vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.args.vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

        self.model = model
        self.webcam = webcam
        self.screenshot = screenshot
        self.imgsz = imgsz
        self.done_setup = True

        return model

    def __call__(self, source=None, model=None):
        if not self.done_setup:
            model = self.setup(source, model)
        else:
            model = self.model

        seen, windows, dt = 0, [], (ops.Profile(), ops.Profile(), ops.Profile())
        for path, im, im0s, vid_cap, s in self.dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(self.save_dir /
                                           Path(path).stem, mkdir=True) if self.args.visualize else False
                preds = model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with dt[2]:
                preds = self.postprocess(preds)

            for i, pred in enumerate(preds):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                self.data_path = p
                save_path = str(self.save_dir / p.name)  # im.jpg
                self.txt_path = str(
                    self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % im.shape[2:]  # print string

                self.annotator = self.annotator or self.get_annotator(im0)  # initialize only once
                self.write_results(pred=pred, img=im, orig_img=im0, print_string=s)

                # stream
                im0 = self.annotator.result()
                if self.args.view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # save imgs
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if self.vid_path[i] != save_path:  # new video
                            self.vid_path[i] = save_path
                            if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                self.vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                                 (w, h))
                        self.vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(pred) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms postprocess per image at shape {(1, 3, *self.imgsz)}'
            % t)
        if self.args.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
