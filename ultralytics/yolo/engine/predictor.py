# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
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
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS-only)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """
import platform
from collections import defaultdict
from pathlib import Path

import cv2

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadScreenshots, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (OmegaConf): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        view_img (bool): Whether to view image output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        """
        Initializes the BasePredictor class.

        Args:
            config (str, optional): Path to a configuration file. Defaults to DEFAULT_CONFIG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        if overrides is None:
            overrides = {}
        self.args = get_config(config, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.save:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_setup = False

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.view_img = None
        self.annotator = None
        self.data_path = None
        self.callbacks = defaultdict(list, {k: [v] for k, v in callbacks.default_callbacks.items()})  # add callbacks
        callbacks.add_integration_callbacks(self)

    def preprocess(self, img):
        pass

    def get_annotator(self, img):
        raise NotImplementedError("get_annotator function needs to be implemented")

    def write_results(self, pred, batch, print_string):
        raise NotImplementedError("print_results function needs to be implemented")

    def postprocess(self, preds, img, orig_img):
        return preds

    def setup(self, source=None, model=None):
        # source
        source = str(source or self.args.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # model
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
        stride, pt = model.stride, model.pt
        imgsz = check_imgsz(self.args.imgsz, stride=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            self.view_img = check_imshow(warn=True)
            self.dataset = LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=self.args.vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(source, imgsz=imgsz, stride=stride, auto=pt)
        else:
            self.dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=self.args.vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

        self.model = model
        self.webcam = webcam
        self.screenshot = screenshot
        self.imgsz = imgsz
        self.done_setup = True
        self.device = device

        return model

    @smart_inference_mode()
    def __call__(self, source=None, model=None):
        self.run_callbacks("on_predict_start")
        model = self.model if self.done_setup else self.setup(source, model)
        model.eval()
        self.seen, self.windows, self.dt = 0, [], (ops.Profile(), ops.Profile(), ops.Profile())
        self.all_outputs = []
        for batch in self.dataset:
            self.run_callbacks("on_predict_batch_start")
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                preds = model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with self.dt[2]:
                preds = self.postprocess(preds, im, im0s)

            for i in range(len(im)):
                if self.webcam:
                    path, im0s = path[i], im0s[i]
                p = Path(path)
                s += self.write_results(i, preds, (p, im, im0s))

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(preds) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

            self.run_callbacks("on_predict_batch_end")

        # Print results
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms postprocess per image at shape {(1, 3, *self.imgsz)}'
            % t)
        if self.args.save_txt or self.args.save:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")
        return self.all_outputs

    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        im0 = self.annotator.result()
        # save imgs
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

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
