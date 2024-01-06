# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ultralytics.utils import TQDM


class FastSAMPrompt:
    """
    Fast Segment Anything Model class for image annotation and visualization.

    Attributes:
        device (str): Computing device ('cuda' or 'cpu').
        results: Object detection or segmentation results.
        source: Source image or image path.
        clip: CLIP model for linear assignment.
    """

    def __init__(self, source, results, device='cuda') -> None:
        """Initializes FastSAMPrompt with given source, results and device, and assigns clip for linear assignment."""
        self.device = device
        self.results = results
        self.source = source

        # Import and assign clip
        try:
            import clip  # for linear_assignment
        except ImportError:
            from ultralytics.utils.checks import check_requirements
            check_requirements('git+https://github.com/openai/CLIP.git')
            import clip
        self.clip = clip

    @staticmethod
    def _segment_image(image, bbox):
        """Segments the given image according to the provided bounding box coordinates."""
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new('RGB', image.size, (255, 255, 255))
        # transparency_mask = np.zeros_like((), dtype=np.uint8)
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image

    @staticmethod
    def _format_results(result, filter=0):
        """Formats detection results into list of annotations each containing ID, segmentation, bounding box, score and
        area.
        """
        annotations = []
        n = len(result.masks.data) if result.masks is not None else 0
        for i in range(n):
            mask = result.masks.data[i] == 1.0
            if torch.sum(mask) >= filter:
                annotation = {
                    'id': i,
                    'segmentation': mask.cpu().numpy(),
                    'bbox': result.boxes.data[i],
                    'score': result.boxes.conf[i]}
                annotation['area'] = annotation['segmentation'].sum()
                annotations.append(annotation)
        return annotations

    @staticmethod
    def _get_bbox_from_mask(mask):
        """Applies morphological transformations to the mask, displays it, and if with_contours is True, draws
        contours.
        """
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
        return [x1, y1, x2, y2]

    def plot(self,
             annotations,
             output,
             bbox=None,
             points=None,
             point_label=None,
             mask_random_color=True,
             better_quality=True,
             retina=False,
             with_contours=True):
        """
        Plots annotations, bounding boxes, and points on images and saves the output.

        Args:
            annotations (list): Annotations to be plotted.
            output (str or Path): Output directory for saving the plots.
            bbox (list, optional): Bounding box coordinates [x1, y1, x2, y2]. Defaults to None.
            points (list, optional): Points to be plotted. Defaults to None.
            point_label (list, optional): Labels for the points. Defaults to None.
            mask_random_color (bool, optional): Whether to use random color for masks. Defaults to True.
            better_quality (bool, optional): Whether to apply morphological transformations for better mask quality. Defaults to True.
            retina (bool, optional): Whether to use retina mask. Defaults to False.
            with_contours (bool, optional): Whether to plot contours. Defaults to True.
        """
        pbar = TQDM(annotations, total=len(annotations))
        for ann in pbar:
            result_name = os.path.basename(ann.path)
            image = ann.orig_img[..., ::-1]  # BGR to RGB
            original_h, original_w = ann.orig_shape
            # For macOS only
            # plt.switch_backend('TkAgg')
            plt.figure(figsize=(original_w / 100, original_h / 100))
            # Add subplot with no margin.
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(image)

            if ann.masks is not None:
                masks = ann.masks.data
                if better_quality:
                    if isinstance(masks[0], torch.Tensor):
                        masks = np.array(masks.cpu())
                    for i, mask in enumerate(masks):
                        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                        masks[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

                self.fast_show_mask(masks,
                                    plt.gca(),
                                    random_color=mask_random_color,
                                    bbox=bbox,
                                    points=points,
                                    pointlabel=point_label,
                                    retinamask=retina,
                                    target_height=original_h,
                                    target_width=original_w)

                if with_contours:
                    contour_all = []
                    temp = np.zeros((original_h, original_w, 1))
                    for i, mask in enumerate(masks):
                        mask = mask.astype(np.uint8)
                        if not retina:
                            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contour_all.extend(iter(contours))
                    cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
                    color = np.array([0 / 255, 0 / 255, 1.0, 0.8])
                    contour_mask = temp / 255 * color.reshape(1, 1, -1)
                    plt.imshow(contour_mask)

            # Save the figure
            save_path = Path(output) / result_name
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            pbar.set_description(f'Saving {result_name} to {save_path}')

    @staticmethod
    def fast_show_mask(
        annotation,
        ax,
        random_color=False,
        bbox=None,
        points=None,
        pointlabel=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        """
        Quickly shows the mask annotations on the given matplotlib axis.

        Args:
            annotation (array-like): Mask annotation.
            ax (matplotlib.axes.Axes): Matplotlib axis.
            random_color (bool, optional): Whether to use random color for masks. Defaults to False.
            bbox (list, optional): Bounding box coordinates [x1, y1, x2, y2]. Defaults to None.
            points (list, optional): Points to be plotted. Defaults to None.
            pointlabel (list, optional): Labels for the points. Defaults to None.
            retinamask (bool, optional): Whether to use retina mask. Defaults to True.
            target_height (int, optional): Target height for resizing. Defaults to 960.
            target_width (int, optional): Target width for resizing. Defaults to 960.
        """
        n, h, w = annotation.shape  # batch, height, width

        areas = np.sum(annotation, axis=(1, 2))
        annotation = annotation[np.argsort(areas)]

        index = (annotation != 0).argmax(axis=0)
        if random_color:
            color = np.random.random((n, 1, 1, 3))
        else:
            color = np.ones((n, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 1.0])
        transparency = np.ones((n, 1, 1, 1)) * 0.6
        visual = np.concatenate([color, transparency], axis=-1)
        mask_image = np.expand_dims(annotation, -1) * visual

        show = np.zeros((h, w, 4))
        h_indices, w_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

        show[h_indices, w_indices, :] = mask_image[indices]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
        # Draw point
        if points is not None:
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
                s=20,
                c='y',
            )
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
                s=20,
                c='m',
            )

        if not retinamask:
            show = cv2.resize(show, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        ax.imshow(show)

    @torch.no_grad()
    def retrieve(self, model, preprocess, elements, search_text: str, device) -> int:
        """Processes images and text with a model, calculates similarity, and returns softmax score."""
        preprocessed_images = [preprocess(image).to(device) for image in elements]
        tokenized_text = self.clip.tokenize([search_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100.0 * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)

    def _crop_image(self, format_results):
        """Crops an image based on provided annotation format and returns cropped images and related data."""
        if os.path.isdir(self.source):
            raise ValueError(f"'{self.source}' is a directory, not a valid source for this function.")
        image = Image.fromarray(cv2.cvtColor(self.results[0].orig_img, cv2.COLOR_BGR2RGB))
        ori_w, ori_h = image.size
        annotations = format_results
        mask_h, mask_w = annotations[0]['segmentation'].shape
        if ori_w != mask_w or ori_h != mask_h:
            image = image.resize((mask_w, mask_h))
        cropped_boxes = []
        cropped_images = []
        not_crop = []
        filter_id = []
        for _, mask in enumerate(annotations):
            if np.sum(mask['segmentation']) <= 100:
                filter_id.append(_)
                continue
            bbox = self._get_bbox_from_mask(mask['segmentation'])  # mask 的 bbox
            cropped_boxes.append(self._segment_image(image, bbox))  # 保存裁剪的图片
            cropped_images.append(bbox)  # 保存裁剪的图片的bbox

        return cropped_boxes, cropped_images, not_crop, filter_id, annotations

    def box_prompt(self, bbox):
        """Modifies the bounding box properties and calculates IoU between masks and bounding box."""
        if self.results[0].masks is not None:
            assert (bbox[2] != 0 and bbox[3] != 0)
            if os.path.isdir(self.source):
                raise ValueError(f"'{self.source}' is a directory, not a valid source for this function.")
            masks = self.results[0].masks.data
            target_height, target_width = self.results[0].orig_shape
            h = masks.shape[1]
            w = masks.shape[2]
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height), ]
            bbox[0] = max(round(bbox[0]), 0)
            bbox[1] = max(round(bbox[1]), 0)
            bbox[2] = min(round(bbox[2]), w)
            bbox[3] = min(round(bbox[3]), h)

            # IoUs = torch.zeros(len(masks), dtype=torch.float32)
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

            masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            iou = masks_area / union
            max_iou_index = torch.argmax(iou)

            self.results[0].masks.data = torch.tensor(np.array([masks[max_iou_index].cpu().numpy()]))
        return self.results

    def point_prompt(self, points, pointlabel):  # numpy
        """Adjusts points on detected masks based on user input and returns the modified results."""
        if self.results[0].masks is not None:
            if os.path.isdir(self.source):
                raise ValueError(f"'{self.source}' is a directory, not a valid source for this function.")
            masks = self._format_results(self.results[0], 0)
            target_height, target_width = self.results[0].orig_shape
            h = masks[0]['segmentation'].shape[0]
            w = masks[0]['segmentation'].shape[1]
            if h != target_height or w != target_width:
                points = [[int(point[0] * w / target_width), int(point[1] * h / target_height)] for point in points]
            onemask = np.zeros((h, w))
            for annotation in masks:
                mask = annotation['segmentation'] if isinstance(annotation, dict) else annotation
                for i, point in enumerate(points):
                    if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                        onemask += mask
                    if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                        onemask -= mask
            onemask = onemask >= 1
            self.results[0].masks.data = torch.tensor(np.array([onemask]))
        return self.results

    def text_prompt(self, text):
        """Processes a text prompt, applies it to existing results and returns the updated results."""
        if self.results[0].masks is not None:
            format_results = self._format_results(self.results[0], 0)
            cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)
            clip_model, preprocess = self.clip.load('ViT-B/32', device=self.device)
            scores = self.retrieve(clip_model, preprocess, cropped_boxes, text, device=self.device)
            max_idx = scores.argsort()
            max_idx = max_idx[-1]
            max_idx += sum(np.array(filter_id) <= int(max_idx))
            self.results[0].masks.data = torch.tensor(np.array([annotations[max_idx]['segmentation']]))
        return self.results

    def everything_prompt(self):
        """Returns the processed results from the previous methods in the class."""
        return self.results
