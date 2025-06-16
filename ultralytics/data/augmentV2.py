import cv2
import numpy as np
import functools
from copy import deepcopy
import random
from ultralytics.utils.radar_data import ManitouRadarPC
from .augmentV1 import *
    

class ManitouResizeCrop_MultiImg(ManitouResizeCrop):
    """
        Applies resize and crop to multiple images.
    """
    def __init__(self, scale, target_size, original_size, p):
        super().__init__(scale, target_size, original_size, p)

    def __call__(self, labels=None, images=None):
        """
        Args:
            labels (Dict, None): Dictionary containing key "labels" with a list of label dictionaries and "ref_labels" for reference labels.
            image (list, ndarray): List of images to be resized and cropped. 
        Returns:
            (Dict | np.ndarray): If `labels` is provided, returns an updated dictionary with the resized and cropped image,
                updated labels, and additional metadata. If `labels` is empty, returns a list containing the resized 
                and cropped image.
        """
        if labels is not None:
            # If labels are provided, apply resize and crop to the image in the labels dictionary
            for i, (k_l, r_ls) in enumerate(zip(labels["labels"], labels["ref_labels"])):
                labels["labels"][i] = self.apply_one_label(k_l, None)
                if r_ls is not None:
                    labels["ref_labels"][i] = [self.apply_one_label(r_l, None) for r_l in r_ls]
            return labels
        
        return [self.apply_one_label(None, img) for img in images]
    
    def apply_one_label(self, label=None, image=None):
        assert label is not None or image is not None, "Either label or image must be provided."
        if label is not None:
            return super().__call__(label, None)
        
        return super().__call__(None, image)

        
class RandomHSV_MultiImg(RandomHSV):
    """
        Applies random HSV augmentation to multiple images.
    """
    def __call__(self, labels):
        
        if self.hgain or self.sgain or self.vgain:
            dtype = labels["labels"][0]["img"].dtype  # uint8
            lut_hue, lut_sat, lut_val = self.get_params(dtype=dtype)
            for i, (k_l, r_ls) in enumerate(zip(labels["labels"], labels["ref_labels"])):
                labels["labels"][i] = self.apply_one_label(k_l, lut_hue, lut_sat, lut_val)
                if r_ls is not None:
                    labels["ref_labels"][i] = [self.apply_one_label(r_l, lut_hue, lut_sat, lut_val) for r_l in r_ls]
            
        return labels
    
    def get_params(self, dtype=np.uint8):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
        x = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79
        return lut_hue, lut_sat, lut_val
    
    def apply_one_label(self, label, lut_hue, lut_sat, lut_val):
        img = label["img"]
        if img.shape[-1] != 3:  # only apply to RGB images
            return label
        
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return label
    

class RandomFlip_MultiImg(RandomFlip):
    """
        Applies random flip augmentation to multiple images.
    """
    def __call__(self, labels):
        for i, (k_l, r_ls) in enumerate(zip(labels["labels"], labels["ref_labels"])):
            run_flip = random.random() < self.p
            labels["labels"][i] = self.apply_one_label(k_l, run_flip)
            if r_ls is not None:
                labels["ref_labels"][i] = [self.apply_one_label(r_l, run_flip) for r_l in r_ls]
        return labels
    
    def apply_one_label(self, label, run_flip):        
        img = label["img"]
        instances = label.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and run_flip:
            img = np.flipud(img)
            instances.flipud(h)
            if label.get("intrinsic_K", None) is not None:
                _h = img.shape[0]  # don't use the normalized height
                cvt_mat = np.array([[1, 0, 0], [0, -1, _h-1], [0, 0, 1]], dtype=label["intrinsic_K"].dtype)
                label["intrinsic_K"] = cvt_mat @ label["intrinsic_K"]
        if self.direction == "horizontal" and run_flip:
            img = np.fliplr(img)
            instances.fliplr(w)
            if label.get("intrinsic_K", None) is not None:
                _w = img.shape[1]  # don't use the normalized width
                cvt_mat = np.array([[-1, 0, _w-1], [0, 1, 0], [0, 0, 1]], dtype=label["intrinsic_K"].dtype)
                label["intrinsic_K"] = cvt_mat @ label["intrinsic_K"]
                
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        label["img"] = np.ascontiguousarray(img)
        label["instances"] = instances
        return label
  
    
class FormatManitou_MultiImg(Format):
    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        super().__init__(
            bbox_format=bbox_format,
            normalize=normalize,
            return_mask=return_mask,
            return_keypoint=False,
            return_obb=False,
            mask_ratio=mask_ratio,
            mask_overlap=mask_overlap,
            batch_idx=batch_idx,
            bgr=bgr,
        )
        
    def __call__(self, labels):
        for i, (k_l, r_ls) in enumerate(zip(labels["labels"], labels["ref_labels"])):
            labels["labels"][i] = self.apply_one_label(k_l)
            if r_ls is not None:
                labels["ref_labels"][i] = [self.apply_one_label(r_l) for r_l in r_ls]
                
        # collect labels from 4 cameras to a mini-batch
        labels["labels"] = self.collect_4camera_labels(labels["labels"])
        labels["ref_labels"] = self.collect_4camera_labels(labels["ref_labels"]) if labels["ref_labels"][0] is not None else None
            
        return labels              
        
    def apply_one_label(self, labels):
        labels.pop("prev", None)  # in case recursive call when using pin_memory=True
        labels.pop("next", None)  # in case recursive call when using pin_memory=True

        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        ins_ids = labels.pop("ins_ids")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["ins_ids"] = torch.from_numpy(ins_ids) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels
    
    def collect_4camera_labels(self, labels):
        """
        collect labels from 4 cameras to a mini-batch.
        """
        _labels = []
        for label in labels:
            if isinstance(label, dict):
                _labels.append(label)
            elif isinstance(label, list):
                _labels.extend(label)
            else:
                raise TypeError(f"Unsupported label type: {type(label)}")

        mb = [dict(sorted(cam_dict.items())) for cam_dict in _labels]
        keys = list(mb[0].keys())
        
        values = list(zip(*[[b[k] for k in keys] for b in mb]))
        
        nb = {}
        for idx_key, k in enumerate(keys):
            v = values[idx_key]
            
            if k in {"img",}:
                v = torch.stack(v, dim=0)  # stack images
            elif k == "visuals":
                v = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                v = torch.cat(v, 0)
        
            nb[k] = v
        
        nb["batch_idx"] = list(nb["batch_idx"])
        for i in range(len(nb["batch_idx"])):
            nb["batch_idx"][i] += i
        nb["batch_idx"] = torch.cat(nb["batch_idx"], 0)
        
        # TODO: process radar data if needed
        return nb

class Debug_Radar:
    """Debug radar data by projecting it onto camera images and saving the overlay images."""
    
    def __init__(self, calib_params, filter_cfg={}):
        self.calib_params = calib_params
        self.filter_cfg = filter_cfg
    
    def __call__(self, labels):
        radar1 = labels[0]["radar"]
        radar2 = labels[1]["radar"]
        radar3 = labels[2]["radar"]
        radar4 = labels[3]["radar"]
        
        camera1_K = labels[0]["intrinsic_K"]
        camera2_K = labels[1]["intrinsic_K"]
        camera3_K = labels[2]["intrinsic_K"]
        camera4_K = labels[3]["intrinsic_K"]
        
        new_params = {
            "camera1_K": camera1_K,
            "camera2_K": camera2_K,
            "camera3_K": camera3_K,
            "camera4_K": camera4_K,
        }
        calib_params = self.calib_params
        calib_params.update(new_params)
        
        radar_pc = ManitouRadarPC(
            radar1=radar1,
            radar2=radar2,
            radar3=radar3,
            radar4=radar4,
            calib_params=calib_params,
            filter_cfg=self.filter_cfg
        )
        
        # save projected image
        img1 = labels[0]["img"].copy()
        img1 = radar_pc.get_overlay_image(cam_idx=1, img=img1)
        img2 = labels[1]["img"].copy()
        img2 = radar_pc.get_overlay_image(cam_idx=2, img=img2)
        img3 = labels[2]["img"].copy()
        img3 = radar_pc.get_overlay_image(cam_idx=3, img=img3)
        img4 = labels[3]["img"].copy()
        img4 = radar_pc.get_overlay_image(cam_idx=4, img=img4)
        
        import cv2
        rosbag = labels[0]["im_file"].split("/")[6]
        cv2.imwrite(f"/home/shu/Documents/{rosbag}_camera1_overlay.jpg", img1)
        cv2.imwrite(f"/home/shu/Documents/{rosbag}_camera2_overlay.jpg", img2)
        cv2.imwrite(f"/home/shu/Documents/{rosbag}_camera3_overlay.jpg", img3)
        cv2.imwrite(f"/home/shu/Documents/{rosbag}_camera4_overlay.jpg", img4)
        
        return labels
        