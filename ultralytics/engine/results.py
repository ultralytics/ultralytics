# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode


class BaseTensor(SimpleClass):
    """Base tensor class with additional methods for easy manipulation and device handling."""

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (tuple): Original shape of the image, typically in the format (height, width).

        Returns:
            (None)

        Example:
            ```python
            import torch
            from ultralytics.engine.results import BaseTensor

            data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            orig_shape = (720, 1280)
            base_tensor = BaseTensor(data, orig_shape)
            ```
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Returns the shape of the underlying data tensor for easier manipulation and device handling."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor stored in CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Returns a copy of the tensor as a numpy array for efficient numerical operations."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Moves the tensor to GPU memory, returning a new instance if necessary."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the underlying data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a new BaseTensor instance containing the specified indexed elements of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes, optional): Object containing detection bounding boxes.
        masks (Masks, optional): Object containing detection masks.
        probs (Probs, optional): Object containing class probabilities for classification tasks.
        keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.

    Methods:
        update(boxes=None, masks=None, probs=None, obb=None): Updates object attributes with new detection results.
        cpu(): Returns a copy of the Results object with all tensors on CPU memory.
        numpy(): Returns a copy of the Results object with all tensors as numpy arrays.
        cuda(): Returns a copy of the Results object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the Results object with tensors on a specified device and dtype.
        new(): Returns a new Results object with the same image, path, and names.
        plot(...): Plots detection results on an input image, returning an annotated image.
        show(): Show annotated results to screen.
        save(filename): Save annotated results to file.
        verbose(): Returns a log string for each task, detailing detections and classifications.
        save_txt(txt_file, save_conf=False): Saves detection results to a text file.
        save_crop(save_dir, file_name=Path("im.jpg")): Saves cropped detection images.
        tojson(normalize=False): Converts detection results to JSON format.
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.tensor, optional): A 2D tensor of keypoint coordinates for each detection. For default pose
                model, Keypoint indices for human body pose estimation are:
                0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
                5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
                9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
                13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
            obb (torch.tensor, optional): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (dict, optional): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Returns:
            None

        Example:
            ```python
            results = model("path/to/image.jpg")
            ```
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """Return a Results object for a specific index of inference results."""
        return self._apply("__getitem__", idx)

    def __len__(self):
        """Return the number of detections in the Results object from a non-empty attribute set (boxes, masks, etc.)."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """Updates detection results attributes including boxes, masks, probs, and obb with new data."""
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Example:
            ```python
            results = model("path/to/image.jpg")
            for result in results:
                result_cuda = result.cuda()
                result_cpu = result.cpu()
            ```
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Returns a copy of the Results object with all its tensors moved to CPU memory."""
        return self._apply("cpu")

    def numpy(self):
        """Returns a copy of the Results object with all tensors as numpy arrays."""
        return self._apply("numpy")

    def cuda(self):
        """Moves all tensors in the Results object to GPU memory."""
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Moves all tensors in the Results object to the specified device and dtype."""
        return self._apply("to", *args, **kwargs)

    def new(self):
        """Returns a new Results object with the same image, path, names, and speed attributes."""
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability.
            show (bool): Whether to display the annotated image directly.
            save (bool): Whether to save the annotated image to `filename`.
            filename (str): Filename to save image to if save is True.

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('bus.jpg')  # results list
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.show()  # show image
                im.save('results.jpg')  # save image
            ```
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        """Show the image with annotated inference results."""
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """Save annotated inference results image to file."""
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """Returns a log string for each task in the results, detailing detection and classification outcomes."""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.

        Args:
            txt_file (str): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Example:
            ```python
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model("path/to/image.jpg")
            for result in results:
                result.save_txt("output.txt")
            ```

        Notes:
            - The file will contain one line per detection or classification with the following structure:
                - For detections: `class confidence x_center y_center width height`
                - For classifications: `confidence class_name`
                - For masks and keypoints, the specific formats will vary accordingly.

            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.

            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Save cropped detection images to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Directory path where the cropped images should be saved.
            file_name (str | pathlib.Path): Filename for the saved cropped image.

        Notes:
            This function does not support Classify or Oriented Bounding Box (OBB) tasks. It will warn and exit if
            called for such tasks.

        Example:
            ```python
            from ultralytics import YOLO

            model = YOLO("yolov8n.pt")
            results = model("path/to/image.jpg")

            # Save cropped images to the specified directory
            for result in results:
                result.save_crop(save_dir="path/to/save/crops", file_name="crop")
            ```
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """Convert inference results to a summarized dictionary with optional normalization for box coordinates."""
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def tojson(self, normalize=False, decimals=5):
        """Converts detection results to JSON format."""
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)


class Boxes(BaseTensor):
    """
    Manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class
    identifiers, and optional tracking IDs. Supports multiple formats for box coordinates, including both absolute and
    normalized forms.

    Attributes:
        data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.

    Attributes:
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray, optional): Tracking IDs for each box, if available.
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, calculated on demand.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes, relative to `orig_shape`.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes, relative to `orig_shape`.

    Methods:
        cpu(): Moves the boxes to CPU memory.
        numpy(): Converts the boxes to a numpy array format.
        cuda(): Moves the boxes to CUDA (GPU) memory.
        to(device, dtype=None): Moves the boxes to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6)
                or (num_boxes, 7). Columns should contain [x1, y1, x2, y2, confidence, class, (optional) track_id].
                The track ID  column is included if present.
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.

        Returns:
            (None)
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Returns bounding boxes in [x1, y1, x2, y2] format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Returns the confidence scores for each detection box."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Class ID tensor representing category predictions for each bounding box."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the tracking IDs for each box if available."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Returns boxes in [x, y, width, height] format."""
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Normalize box coordinates to [x1, y1, x2, y2] relative to the original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Returns normalized bounding boxes in [x, y, width, height] format."""
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    Attributes:
        xy (list): A list of segments in pixel coordinates.
        xyn (list): A list of normalized segments.

    Methods:
        cpu(): Returns the masks tensor on CPU memory.
        numpy(): Returns the masks tensor as a numpy array.
        cuda(): Returns the masks tensor on GPU memory.
        to(device, dtype): Returns the masks tensor with the specified device and dtype.
    """

    def __init__(self, masks, orig_shape) -> None:
        """Initializes the Masks class with a masks tensor and original image shape."""
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Return normalized xy-coordinates of the segmentation masks."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Returns the [x, y] normalized mask coordinates for each segment in the mask tensor."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    Attributes
        xy (torch.Tensor): A collection of keypoints containing x, y coordinates for each detection.
        xyn (torch.Tensor): A normalized version of xy with coordinates in the range [0, 1].
        conf (torch.Tensor): Confidence values associated with keypoints if available, otherwise None.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(device, dtype): Returns a copy of the keypoints tensor with the specified device and dtype.
    """

    @smart_inference_mode()  # avoid keypoints < conf in-place error
    def __init__(self, keypoints, orig_shape) -> None:
        """Initializes the Keypoints object with detection keypoints and original image dimensions."""
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # points with conf < 0.5 (not visible)
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Returns x, y coordinates of keypoints."""
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Returns normalized coordinates (x, y) of keypoints relative to the original image size."""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """Returns confidence values for each keypoint."""
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification predictions.

    Attributes
        top1 (int): Index of the top 1 class.
        top5 (list[int]): Indices of the top 5 classes.
        top1conf (torch.Tensor): Confidence of the top 1 class.
        top5conf (torch.Tensor): Confidences of the top 5 classes.

    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(): Returns a copy of the probs tensor with the specified device and dtype.
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """Initialize Probs with classification probabilities and optional original image shape."""
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """Return the index of the class with the highest probability."""
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """Return the indices of the top 5 class probabilities."""
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """Retrieves the confidence score of the highest probability class."""
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """Returns confidence scores for the top 5 classification predictions."""
        return self.data[self.top5]


class OBB(BaseTensor):
    """
    A class for storing and manipulating Oriented Bounding Boxes (OBB).

    Args:
        boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values.
            If present, the third last column contains track IDs, and the fifth column from the left contains rotation.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes
        xywhr (torch.Tensor | numpy.ndarray): The boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray): The track IDs of the boxes (if available).
        xyxyxyxyn (torch.Tensor | numpy.ndarray): The rotated boxes in xyxyxyxy format normalized by orig image size.
        xyxyxyxy (torch.Tensor | numpy.ndarray): The rotated boxes in xyxyxyxy format.
        xyxy (torch.Tensor | numpy.ndarray): The horizontal boxes in xyxyxyxy format.
        data (torch.Tensor): The raw OBB tensor (alias for `boxes`).

    Methods:
        cpu(): Move the object to CPU memory.
        numpy(): Convert the object to a numpy array.
        cuda(): Move the object to CUDA memory.
        to(*args, **kwargs): Move the object to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """Initialize an OBB instance with oriented bounding box data and original image shape."""
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        """Return boxes in [x_center, y_center, width, height, rotation] format."""
        return self.data[:, :5]

    @property
    def conf(self):
        """Gets the confidence values of Oriented Bounding Boxes (OBBs)."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Returns the class values of the oriented bounding boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the tracking IDs of the oriented bounding boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        """Convert OBB format to 8-point (xyxyxyxy) coordinate format of shape (N, 4, 2) for rotated bounding boxes."""
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        """Converts rotated bounding boxes to normalized xyxyxyxy format of shape (N, 4, 2)."""
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        """
        Convert the oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format (x1, y1, x2, y2).

        Returns:
            (torch.Tensor | numpy.ndarray): Axis-aligned bounding boxes in xyxy format with shape (num_boxes, 4).

        Example:
            ```python
            import torch
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('path/to/image.jpg')
            for result in results:
                obb = result.obb
                if obb is not None:
                    xyxy_boxes = obb.xyxy
                    # Do something with xyxy_boxes
            ```

        Note:
            This method is useful to perform operations that require axis-aligned bounding boxes, such as IoU
            calculation with non-rotated boxes. The conversion approximates the OBB by the minimal enclosing rectangle.
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
