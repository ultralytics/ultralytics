import torch
import math
import numpy as np
import cv2
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.engine.predictor import STREAM_WARNING
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER, ops, colorstr
from ultralytics.data import load_inference_source
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.data.augmentV1 import (
    ManitouResizeCrop,
    LetterBox,
)
from ultralytics.models.yolo_manitou.utils import invert_manitou_resize_crop_xyxy


class ManitouPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a Manitou detection model.

    """

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Stream real-time inference on camera feed and save results to file.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor | None):
                Source for inference.
            model (str | Path | torch.nn.Module | None): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, self.model.ch, *self.imgsz)
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im0s[i], s)

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")
    
    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = self.pre_transform(im)
            im = np.stack(im)
            
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List[np.ndarray]): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (List[np.ndarray]): A list of transformed images.
        """
        resize_crop = ManitouResizeCrop(self.pre_crop_cfg["scale"],
                                        self.pre_crop_cfg["target_size"],
                                        self.pre_crop_cfg["original_size"],
                                        1.0 if self.pre_crop_cfg["is_crop"] else 0.0)
        
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )
        
        for transform in [resize_crop]:
            im = [transform(image=x) for x in im]
            
        return im
    
    def setup_source(self, source):
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """
        # Check image size
        if isinstance(self.args.imgsz, int):
            self.args.imgsz = (self.args.imgsz, self.args.imgsz)
        
        h = self.args.imgsz[0] // self.model.stride * self.model.stride
        w = math.ceil(self.args.imgsz[1] / self.model.stride) * self.model.stride
        self.pre_crop_cfg = {"is_crop": False, 
                             "scale": 1, 
                             "target_size": (self.args.imgsz[0], self.args.imgsz[1]), 
                             "original_size": (self.args.imgsz[0], self.args.imgsz[1])}
        if (h, w) != (self.args.imgsz[0], self.args.imgsz[1]):
            self.pre_crop_cfg["is_crop"] = True
            self.pre_crop_cfg["scale"] = w / self.args.imgsz[1]
            self.pre_crop_cfg["target_size"] = (h, w)
            self.imgsz = (h, w)
        else:
            self.imgsz = (self.args.imgsz[0], self.args.imgsz[1])

        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "save_feats", False)
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)


        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results
    
    def construct_results(self, preds, img, orig_imgs, **kwargs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing detection information for each image.
        """
        res_list = []
        
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            assert orig_img.shape[:2] == self.pre_crop_cfg["original_size"], f"Original image size {orig_img.shape[:2]} does not match pre-crop cfg {self.pre_crop_cfg['original_size']}"
            pred[:, :4] = invert_manitou_resize_crop_xyxy(pred[:, :4], self.pre_crop_cfg)
            res_list.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
    
        return res_list
    


