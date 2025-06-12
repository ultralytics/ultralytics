import torch
import math
import numpy as np
import cv2
from pathlib import Path

from ultralytics.models.yolo_manitou.detect import ManitouPredictor
from ultralytics.engine.predictor import STREAM_WARNING
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER, DEFAULT_CFG, ops, colorstr
from ultralytics.data.manitou_loaders import LoadManitouImagesAndRadar
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.data.augmentV2 import ManitouResizeCrop_MultiImg
from ultralytics.models.yolo_manitou.utils import invert_manitou_resize_crop_xyxy


class ManitouPredictor_MultiCam(ManitouPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a Manitou detection model.

    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.use_radar = self.args.use_radar            

    def __call__(self, data_cfg=None, model=None, *args, **kwargs):
        return list(self.stream_inference(data_cfg, model, *args, **kwargs))  # merge list of Result into one

    @smart_inference_mode()
    def stream_inference(self, data_cfg, model=None, *args, **kwargs):
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
            self.setup_source(data_cfg)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(4 if self.model.pt or self.model.triton else self.dataset.bs * 4, self.model.ch, *self.imgsz)  # `4` means images from 4 cameras
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
                paths, _batch, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(_batch)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, _batch)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                self.results = self.results["cameras"]  # get results for each camera
                n = len(_batch)
                for i in range(n):
                    self.seen += 1
                    for idx in range(1, 5):   
                        self.results[i*4+idx-1].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, paths, _batch, s)

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield self.results

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
    
    def preprocess(self, batch):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        cam1 = [b["cam1"] for b in batch]
        cam2 = [b["cam2"] for b in batch]
        cam3 = [b["cam3"] for b in batch]
        cam4 = [b["cam4"] for b in batch]
        im = [*cam1, *cam2, *cam3, *cam4]
        im = np.stack(im)
        
        if im.shape[-1] == 3:
            im = im[..., ::-1]  # BGR to RGB
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n*4, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
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
        # resize_crop = ManitouResizeCrop(self.pre_crop_cfg["scale"],
        #                                 self.pre_crop_cfg["target_size"],
        #                                 self.pre_crop_cfg["original_size"],
        #                                 1.0 if self.pre_crop_cfg["is_crop"] else 0.0)
        
        # same_shapes = len({x.shape for x in im}) == 1
        # letterbox = LetterBox(
        #     self.imgsz,
        #     auto=same_shapes
        #     and self.args.rect
        #     and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
        #     stride=self.model.stride,
        # )
        
        # for transform in [resize_crop]:
        #     im = [transform(image=x) for x in im]
            
        return im
    
    def get_pre_transform(self):
        """
        Get the pre-transforms used for the input images.
        Returns:
            (List): A list of pre-transform objects.
        """
        resize_crop = ManitouResizeCrop_MultiImg(self.pre_crop_cfg["scale"],
                                                 self.pre_crop_cfg["target_size"],
                                                 self.pre_crop_cfg["original_size"],
                                                 1.0 if self.pre_crop_cfg["is_crop"] else 0.0)
        return [resize_crop]  
    
    def setup_source(self, data_cfg):
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """
        self.calib_params = data_cfg["calib_params"]
        
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
            
        if self.use_radar:
            if self.pre_crop_cfg["is_crop"]:  # if use the ManitouResizeCrop_MultiImg, we need to update the camera intrinsics
                LOGGER.info("Updating camera intrinsics for Manitou dataset with pre-crop configuration.")
                # update camera intrinsics
                h, w = self.pre_crop_cfg["original_size"]
                crop_h, crop_w = self.pre_crop_cfg["target_size"]
                new_h, new_w = int(h *self.pre_crop_cfg["scale"]), int(w * self.pre_crop_cfg["scale"])
                y_off = new_h - crop_h
                for cam_idx in range(1, 5):
                    mat_K = self.calib_params[f"camera{cam_idx}_K"]
                    cvt_mat = np.array([
                                    [self.pre_crop_cfg["scale"], 0,                           0],
                                    [0,                          self.pre_crop_cfg["scale"],  -y_off],
                                    [0,                          0,                           1]
                                ], dtype=mat_K.dtype)
                    self.calib_params[f"new_camera{cam_idx}_K"] = cvt_mat @ mat_K

        radar_accumulation = data_cfg.pop("radar_accumulation", 3)
        
        self.dataset = LoadManitouImagesAndRadar(data_cfg=data_cfg,
                                                 radar_accumulation=radar_accumulation,
                                                 batch=1,
                                                 pre_transform=self.get_pre_transform())
        

    def postprocess(self, preds, _batch):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
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
        orig_imgs = [_batch[i]["orig_images"][f"cam{j}"] for i in range(len(_batch)) for j in range(1, 5)]  # get original images from batch
        paths = [self.batch[0][i][f"cam{j}"] for i in range(len(self.batch[0])) for j in range(1, 5)]  # get paths from batch
        
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

        results = self.construct_results(preds, orig_imgs, paths)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results
                
        results = {"cameras": results, "radars": [_batch[i]["radar"] for i in range(len(_batch))]}  # add radar data to results

        return results
    
    def construct_results(self, preds, orig_imgs, paths):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing detection information for each image.
        """
        res_list = []
        
        for pred, orig_img, img_path in zip(preds, orig_imgs, paths):
            assert orig_img.shape[:2] == self.pre_crop_cfg["original_size"], f"Original image size {orig_img.shape[:2]} does not match pre-crop cfg {self.pre_crop_cfg['original_size']}"
            pred[:, :4] = invert_manitou_resize_crop_xyxy(pred[:, :4], self.pre_crop_cfg)
            res_list.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
    
        return res_list
    
    def write_results(self, i, p, batch, s):
        """
        Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p ([Paths]): list of paths to different cameras.
            batch (list(dict)): Batch of images with original images for each camera.
            s (List[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        cam1 = batch[i]["orig_images"]["cam1"]
        cam2 = batch[i]["orig_images"]["cam2"]
        cam3 = batch[i]["orig_images"]["cam3"]
        cam4 = batch[i]["orig_images"]["cam4"]
        
        radar = batch[i]["radar"]
        # draw radar on cameras
        cam1 = radar.get_overlay_image(1, cam1)
        cam2 = radar.get_overlay_image(2, cam2)
        cam3 = radar.get_overlay_image(3, cam3)
        cam4 = radar.get_overlay_image(4, cam4)
        
        cams = [cam1, cam2, cam3, cam4]
        string = ""  # print string

        string += f"{i}: "
        rosbag = Path(p[i]["cam1"]).parent.parent.name
        frame = Path(p[i]["cam1"]).stem

        self.txt_path_cam1 = self.save_dir / "labels" / rosbag / f"{frame}_cam1"
        self.txt_path_cam2 = self.save_dir / "labels" / rosbag / f"{frame}_cam2"
        self.txt_path_cam3 = self.save_dir / "labels" / rosbag / f"{frame}_cam3"
        self.txt_path_cam4 = self.save_dir / "labels" / rosbag / f"{frame}_cam4"
        
        string += "{:g}x{:g} ".format(*cam1.shape[:2])
        
        plotted_imgs = []
        for idx in range(1, 5):
            result = self.results[i*4 + idx - 1]  
            result.save_dir = self.save_dir.__str__()  # used in other locations
            string += f"{result.verbose()}, "

            # Add predictions to image
            if self.args.save or self.args.show:
                _plotted_img = result.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
                    img=cams[idx - 1],
                )
                if _plotted_img is not None:
                    plotted_imgs.append(_plotted_img)
        if plotted_imgs:
            # Concatenate images from all cameras
            # |cam1|cam3|
            # |cam2|cam4|
            top_row = np.concatenate((plotted_imgs[0], plotted_imgs[2]), axis=1)
            bottom_row = np.concatenate((plotted_imgs[1], plotted_imgs[3]), axis=1)
            self.plotted_img = np.concatenate((top_row, bottom_row), axis=0)
            # resize to 1/2
            self.plotted_img = cv2.resize(self.plotted_img, (0, 0), fx=0.5, fy=0.5)

        string += f"{result.speed['inference']:.1f}ms"
        
        # Save results
        if self.args.save_txt:
            for txt_path in [self.txt_path_cam1, self.txt_path_cam2, self.txt_path_cam3, self.txt_path_cam4]:
                result.save_txt(f"{txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            for txt_path in [self.txt_path_cam1, self.txt_path_cam2, self.txt_path_cam3, self.txt_path_cam4]:
                result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p[i]["cam1"]))
        if self.args.save:
            (self.save_dir / rosbag).mkdir(parents=True, exist_ok=True)
            self.save_predicted_images(str(self.save_dir / rosbag / Path(p[i]["cam1"]).name))

        return string
    
    def save_predicted_images(self, save_path=""):
        im = self.plotted_img
        cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im) 


