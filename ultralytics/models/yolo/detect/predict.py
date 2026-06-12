# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

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
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        return_probs = getattr(self.args, "return_probs", False) and not save_feats
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            kwargs.pop("iou", self.args.iou),  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
            return_probs=return_probs,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        probs_list = None
        if return_probs and isinstance(preds, tuple):
            preds, probs_list = preds

        extra = {"probs_list": probs_list} if probs_list is not None else {}
        results = self.construct_results(preds, img, orig_imgs, **extra, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs, probs_list=None):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            probs_list (list[torch.Tensor] | None): Optional per-image class probability tensors from NMS.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        if probs_list is None:
            probs_list = [None] * len(preds)
        return [
            self.construct_result(pred, img, orig_img, img_path, **({}  if cp is None else {"class_probs": cp}))
            for pred, orig_img, img_path, cp in zip(preds, orig_imgs, self.batch[0], probs_list)
        ]

    def construct_result(self, pred, img, orig_img, img_path, class_probs=None):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.
            class_probs (torch.Tensor | None): Optional (N, nc) tensor of per-class probabilities from NMS.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        from ultralytics.engine.results import Boxes

        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        boxes = Boxes(pred[:, :6], orig_img.shape[:2], class_probs=class_probs)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=boxes)
