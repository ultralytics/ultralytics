import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.data import build_manitou_dataset, ManitouAPI, build_dataloader, get_manitou_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.models.yolo_manitou.detect import ManitouValidator


class ManitouValidator_MultiCam(ManitouValidator):
    """
    A class extending the DetectionValidator class for validation based on a Manitou detection model.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm, optional): Progress bar for displaying progress.
            args (SimpleNamespace, optional): Configuration for validation.
            _callbacks (Callbacks, optional): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        self.nt_per_class = None
        self.nt_per_image = None
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        
        self.eval_tracking = False

    def preprocess(self, batch):
        key_frames = batch["key_frames"]
        ref_frames = batch["ref_frames"]
        
        key_frames["img"] = key_frames["img"].to(self.device, non_blocking=True)
        key_frames["img"] = (key_frames["img"].half() if self.args.half else key_frames["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            key_frames[k] = key_frames[k].to(self.device)
        
        if ref_frames is not None:
            ref_frames["img"] = ref_frames["img"].to(self.device, non_blocking=True)
            ref_frames["img"] = (ref_frames["img"].half() if self.args.half else ref_frames["img"].float()) / 255
            for k in ["batch_idx", "cls", "bboxes"]:
                ref_frames[k] = ref_frames[k].to(self.device)

        return batch

    def init_metrics(self, model):
        self._init_det_metrics(model)

    def _init_det_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        self.class_map = list(range(1, len(model.names) + 1))
        if self.names is not None:
            assert self.names == model.names, f"Model names: {model.names} do not match dataloader names: {self.names}"
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            stats (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            self.stride = model.stride
            pt, jit, engine = model.pt, model.jit, model.engine
            if engine:
                self.args.batch = model.batch_size
            elif not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch}")
            
            self.data = get_manitou_dataset(self.args.data)
            self.names = self.data["names"]
            dataset = build_manitou_dataset(
                                            cfg=self.args,
                                            ann_path=self.data["val"],
                                            batch=self.args.batch,
                                            data=self.data,
                                            mode="val",
                                            stride=self.stride,
                                            multi_cam=True,
                                        )
            
            self.dataloader = self.dataloader or build_dataloader(dataset, self.args.batch, self.args.workers, shuffle=False, rank=-1)
            
            imgsz = dataset.imgsz
            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz[0], imgsz[1]))  # warmup
            
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
                batch = batch["key_frames"]
            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)
                
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            return stats
            
            
        
        