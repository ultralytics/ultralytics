# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import DEFAULT_CFG, YAML, SEMSEG_CFG

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import (SegmentMetrics, box_iou, mask_iou, mask_mcr, SemSegMetrics,
                                       mask_precision, dice_score, mask_recall, mask_accuracy)
from ultralytics.utils.plotting import plot_images, plot_masks
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_semantic_dataset, build_dataloader

class SemSegValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml")
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "semseg"
        self.metrics = SemSegMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
            Build YOLO Dataset.

            Args:
                img_path (str): Path to the folder containing images.
                mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
                batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_semantic_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        # more accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(
            precision=[],
            recall=[],
            dice_score=[],
            mIoU=[],
            mcr=[]
        )

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Precision",
            "Recall",
            "mIoU",
            "Dice-Score",
            "MCR",
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        prepared_batch["batch_idx"] = si
        return prepared_batch

    def _prepare_pred(self, pred):
        """Prepares a batch for training or inference by processing images and targets."""
        return pred

    def mask_iou_for_each_catgoriy(self, pred_mask_for_category, gt_mask_for_catgory):
        nc, length = pred_mask_for_category.shape
        ious = torch.zeros([nc], dtype=pred_mask_for_category.dtype, device=self.device)
        for i in range(nc):
            pred_mask = pred_mask_for_category[i, :]
            gt_mask = gt_mask_for_catgory[i, :]
            ious[i] = mask_iou(pred_mask[None,:], gt_mask[None,:])
        return ious

    def mask_precision_for_each_catgoriy(self, pred_mask_for_category, gt_mask_for_catgory):
        nc, length = pred_mask_for_category.shape
        precisions = torch.zeros([nc], dtype=pred_mask_for_category.dtype, device=self.device)
        for i in range(nc):
            pred_mask = pred_mask_for_category[i, :]
            gt_mask = gt_mask_for_catgory[i, :]
            precisions[i] = mask_precision(pred_mask[None,:], gt_mask[None,:])
        return precisions

    def mask_recall_for_each_catgoriy(self, pred_mask_for_category, gt_mask_for_catgory):
        nc, length = pred_mask_for_category.shape
        recalls = torch.zeros([nc], dtype=pred_mask_for_category.dtype, device=self.device)
        for i in range(nc):
            pred_mask = pred_mask_for_category[i, :]
            gt_mask = gt_mask_for_catgory[i, :]
            recalls[i] = mask_recall(pred_mask[None,:], gt_mask[None,:])
        return recalls

    def mask_accuracy_for_each_catgoriy(self, pred_mask_for_category, gt_mask_for_catgory):
        nc, length = pred_mask_for_category.shape
        accuracys = torch.zeros([nc], dtype=pred_mask_for_category.dtype, device=self.device)
        for i in range(nc):
            pred_mask = pred_mask_for_category[i, :]
            gt_mask = gt_mask_for_catgory[i, :]
            accuracys[i] = mask_accuracy(pred_mask[None,:], gt_mask[None,:])
        return accuracys

    def mask_mcr_for_each_catgoriy(self, pred_mask_for_category, gt_mask_for_catgory):
        nc, length = pred_mask_for_category.shape
        mcrs = torch.zeros([nc], dtype=pred_mask_for_category.dtype, device=self.device)
        for i in range(nc):
            pred_mask = pred_mask_for_category[i, :]
            gt_mask = gt_mask_for_catgory[i, :]
            mcrs[i] = mask_mcr(pred_mask[None,:], gt_mask[None,:])
        return mcrs

    def mask_dice_score_each_catgoriy(self,pred_mask_for_category, gt_mask_for_catgory):
        nc, length = pred_mask_for_category.shape
        dice_scores = torch.zeros([nc], dtype=pred_mask_for_category.dtype, device=self.device)
        for i in range(nc):
            pred_mask = pred_mask_for_category[i, :]
            gt_mask = gt_mask_for_catgory[i, :]
            dice_scores[i] = dice_score(pred_mask[None, :], gt_mask[None, :])
        return dice_scores


    def update_metrics(self, preds, batch):
        """Metrics."""
        self.args.plots = False  # no plots
        for si, pred in enumerate(preds):
            self.seen += 1
            stat = dict(
                precision = torch.zeros(self.nc, dtype=torch.float, device=self.device),
                recall = torch.zeros(self.nc, dtype=torch.float, device=self.device),
                dice_score=torch.zeros(self.nc, dtype=torch.float, device=self.device),
                mIoU=torch.zeros(self.nc, dtype=torch.float, device=self.device),
                mcr=torch.zeros(self.nc, dtype=torch.float, device=self.device)
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            # Masks
            gt_mask = batch["masks"][si] / 255

            #for k in range(self.nc):
            #    p = (pred[k,:,:] * 255).cpu().numpy().astype(np.uint8)
            #    m = (gt_mask[k,:,:] * 255).cpu().numpy().astype(np.uint8)
            #    cv2.imwrite("/media/yanggang/847C02507C023D84/view/" + self.names[k] + "_pred.jpg", p)
            #    cv2.imwrite("/media/yanggang/847C02507C023D84/view/" + self.names[k] + "_mask.jpg", m)
            #    cv2.imwrite("/media/yanggang/847C02507C023D84/view/" + batch['im_file'][si].split(os.sep)[-1], cv2.imread(batch['im_file'][si]))
            #    cv2.imwrite("/media/yanggang/847C02507C023D84/view/" + batch['msk_file'][si].split(os.sep)[-1],
            #                cv2.imread(batch['msk_file'][si]))
            # Evaluate
            stat['precision'] = self.mask_precision_for_each_catgoriy(
                (pred > 0.5).float().view(self.nc, -1),
                (gt_mask > 0.5).float().view(self.nc, -1)
            )

            stat['recall'] = self.mask_recall_for_each_catgoriy(
                (pred > 0.5).float().view(self.nc, -1),
                (gt_mask > 0.5).float().view(self.nc, -1)
            )

            stat['dice_score'] = self.mask_dice_score_each_catgoriy(
                (pred > 0.5).float().view(self.nc, -1),
                (gt_mask > 0.5).float().view(self.nc, -1)
            )


            stat['mIoU'] = self.mask_iou_for_each_catgoriy(
                (pred > 0.5).float().view(self.nc,-1),
                (gt_mask > 0.5).float().view(self.nc,-1)
            )


            stat['mcr'] = self.mask_mcr_for_each_catgoriy(
                (pred > 0.5).float().view(self.nc,-1),
                (gt_mask > 0.5).float().view(self.nc,-1)
            )

            if self.args.plots:
                self.confusion_matrix.process_batch(pred, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k].unsqueeze(0))

            pred_masks = torch.as_tensor((pred > 0.5).float(), dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks.cpu())



    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor | None): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor | None): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.
            masks (bool): Flag indicating if the batch contains mask data.

        Returns:
            (torch.Tensor): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Example:
            ```python
            detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            gt_cls = torch.tensor([1, 0])
            correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
            ```
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        plot_masks(
            images=batch["img"],
            masks=batch["masks"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            nc=self.data["nc"],
            names=self.data["names"],
            colors=self.data["colors"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            mname=self.save_dir / f"mask_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
        # pred
        self.plot_masks.clear()

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {}
        for k,v in self.stats.items():
            stats[k] = torch.cat(v, 0).cpu().numpy()

        #self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        #self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        #stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all",  *self.metrics.mean_results))

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i in range(self.nc):
                LOGGER.info(
                    pf % (self.names[i],  *self.metrics.class_result(i))
                )

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        bs, cls, h, w = preds.shape
        processed_preds = []

        for i in range(bs):
            processed_preds_batch = torch.zeros_like(preds[i,:,:,:])
            pred_batch_cls_indexes = preds[i,:,:,:].argmax(dim=0)

            for j in range(cls):
                mask = (pred_batch_cls_indexes == j)
                processed_preds_batch[j] = mask
            processed_preds.append(processed_preds_batch)

        processed_preds = torch.stack(processed_preds, dim=0)
        return preds.sigmoid()

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats

def validate(cfg=DEFAULT_CFG):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or 'yolov11n-seg.pt'
    data = cfg.data or 'coco128-seg.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''
    cfg.name = os.path.join(cfg.name, 'val')
    data = YAML.load(data)
    img_path = os.path.join(data['path'], data['val'])
    bs = cfg.batch
    #args = dict(model=model, data=data, device=device, name=cfg.name, task='skyseg')
    val_dataset = build_semantic_dataset(
        cfg,
        img_path,
        bs,
        data,
        mode='val'
    )
    val_dataloader = build_dataloader(
        val_dataset,
        bs,
        cfg.workers,
        shuffle=False,
    )

    validator = SemSegValidator(val_dataloader, Path(cfg.name))
    validator(model=model)

if __name__ == '__main__':
    validate(cfg=SEMSEG_CFG)