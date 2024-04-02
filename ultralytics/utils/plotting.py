# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings, threaded
from .checks import check_font, check_version, is_ascii
from .files import increment_path


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # use cv2
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                # NOTE: PIL-version polygon needs tuple type.
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                # NOTE: cv2-version polylines needs np.asarray type.
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """
        Plot masks on image.

        Args:
            masks (tensor): Predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
            retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True):
        """
        Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note:
            `kpt_line=True` currently only supports human pose plotting.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                outside = xy[1] - h >= 3
                p2 = xy[0] + w, xy[1] - h - 3 if outside else xy[1] + h + 3
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title=None):
        """Show the annotated image."""
        Image.fromarray(np.asarray(self.im)[..., ::-1]).show(title)

    def save(self, filename="image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0), fontsize=0.7):
        """Displays queue counts on an image centered at the points with customizable font size and colors."""
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontsize, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw text
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontsize,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_counts(
        self, counts=None, tf=2, fontScale=0.6, line_color=(0, 0, 0), txt_color=(255, 255, 255), classwise_txtgap=55
    ):
        """
        Display counts on im0.

        Args:
            counts (str): objects count data
            tf (int): text thickness for display
            fontScale (float): text fontsize for display
            line_color (RGB Color): counts highlighter color
            txt_color (RGB Color): counts display color
            classwise_txtgap (int): Gap between each class count data
        """

        tl = tf or round(0.002 * (self.im.shape[0] + self.im.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)

        t_sizes = [cv2.getTextSize(str(count), 0, fontScale=0.8, thickness=tf)[0] for count in counts]

        max_text_width = max([size[0] for size in t_sizes])
        max_text_height = max([size[1] for size in t_sizes])

        text_x = self.im.shape[1] - max_text_width - 20
        text_y = classwise_txtgap

        for i, count in enumerate(counts):
            text_x_pos = text_x
            text_y_pos = text_y + i * classwise_txtgap

            cv2.putText(
                self.im,
                str(count),
                (text_x_pos, text_y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontScale,
                color=txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

            line_y_pos = text_y_pos + max_text_height + 5
            cv2.line(self.im, (text_x_pos, line_y_pos), (text_x_pos + max_text_width, line_y_pos), line_color, tf)

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle for object.

        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c

        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=[2, 5, 7], shape=(640, 640), radius=2):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): list of keypoints data to be plotted
            indices (list): keypoints ids list to be plotted
            shape (tuple): imgsz for model inference
            radius (int): Keypoint radius value
        """
        for i, k in enumerate(keypoints):
            if i in indices:
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return self.im

    def plot_angle_and_count_and_stage(self, angle_text, count_text, stage_text, center_kpt, line_thickness=2):
        """
        Plot the pose angle, count value and step stage.

        Args:
            angle_text (str): angle value for workout monitoring
            count_text (str): counts value for workout monitoring
            stage_text (str): stage decision for workout monitoring
            center_kpt (int): centroid pose index for workout monitoring
            line_thickness (int): thickness for text display
        """
        angle_text, count_text, stage_text = (f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}")
        font_scale = 0.6 + (line_thickness / 10.0)

        # Draw angle
        (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, font_scale, line_thickness)
        angle_text_position = (int(center_kpt[0]), int(center_kpt[1]))
        angle_background_position = (angle_text_position[0], angle_text_position[1] - angle_text_height - 5)
        angle_background_size = (angle_text_width + 2 * 5, angle_text_height + 2 * 5 + (line_thickness * 2))
        cv2.rectangle(
            self.im,
            angle_background_position,
            (
                angle_background_position[0] + angle_background_size[0],
                angle_background_position[1] + angle_background_size[1],
            ),
            (255, 255, 255),
            -1,
        )
        cv2.putText(self.im, angle_text, angle_text_position, 0, font_scale, (0, 0, 0), line_thickness)

        # Draw Counts
        (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 0, font_scale, line_thickness)
        count_text_position = (angle_text_position[0], angle_text_position[1] + angle_text_height + 20)
        count_background_position = (
            angle_background_position[0],
            angle_background_position[1] + angle_background_size[1] + 5,
        )
        count_background_size = (count_text_width + 10, count_text_height + 10 + (line_thickness * 2))

        cv2.rectangle(
            self.im,
            count_background_position,
            (
                count_background_position[0] + count_background_size[0],
                count_background_position[1] + count_background_size[1],
            ),
            (255, 255, 255),
            -1,
        )
        cv2.putText(self.im, count_text, count_text_position, 0, font_scale, (0, 0, 0), line_thickness)

        # Draw Stage
        (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 0, font_scale, line_thickness)
        stage_text_position = (int(center_kpt[0]), int(center_kpt[1]) + angle_text_height + count_text_height + 40)
        stage_background_position = (stage_text_position[0], stage_text_position[1] - stage_text_height - 5)
        stage_background_size = (stage_text_width + 10, stage_text_height + 10)

        cv2.rectangle(
            self.im,
            stage_background_position,
            (
                stage_background_position[0] + stage_background_size[0],
                stage_background_position[1] + stage_background_size[1],
            ),
            (255, 255, 255),
            -1,
        )
        cv2.putText(self.im, stage_text, stage_text_position, 0, font_scale, (0, 0, 0), line_thickness)

    def seg_bbox(self, mask, mask_color=(255, 0, 255), det_label=None, track_label=None):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (list): masks data list for instance segmentation area plotting
            mask_color (tuple): mask foreground color
            det_label (str): Detection label text
            track_label (str): Tracking label text
        """
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)

        label = f"Track ID: {track_label}" if track_label else det_label
        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)

        cv2.rectangle(
            self.im,
            (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
            (int(mask[0][0]) + text_size[0] // 2 + 5, int(mask[0][1] + 5)),
            mask_color,
            -1,
        )

        cv2.putText(
            self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (255, 255, 255), 2
        )

    def plot_distance_and_line(self, distance_m, distance_mm, centroids, line_color, centroid_color):
        """
        Plot the distance and line on frame.

        Args:
            distance_m (float): Distance between two bbox centroids in meters.
            distance_mm (float): Distance between two bbox centroids in millimeters.
            centroids (list): Bounding box centroids data.
            line_color (RGB): Distance line color.
            centroid_color (RGB): Bounding box centroid color.
        """
        (text_width_m, text_height_m), _ = cv2.getTextSize(
            f"Distance M: {distance_m:.2f}m", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 10, 25 + text_height_m + 20), (255, 255, 255), -1)
        cv2.putText(
            self.im,
            f"Distance M: {distance_m:.2f}m",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        (text_width_mm, text_height_mm), _ = cv2.getTextSize(
            f"Distance MM: {distance_mm:.2f}mm", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(self.im, (15, 75), (15 + text_width_mm + 10, 75 + text_height_mm + 20), (255, 255, 255), -1)
        cv2.putText(
            self.im,
            f"Distance MM: {distance_mm:.2f}mm",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255), thickness=2, pins_radius=10):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
            thickness (int): int value for line thickness
            pins_radius (int): visioneye point radius value
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, pins_radius, pin_color, -1)
        cv2.circle(self.im, center_bbox, pins_radius, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, thickness)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics."""
    import pandas as pd
    import seaborn as sn

    # Filter matplotlib>=3.7.2 warning and Seaborn use_inf and is_categorical FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = pd.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib labels
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    sn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # Rectangles
    boxes[:, 0:2] = 0.5  # center
    boxes = ops.xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
        im (numpy.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.jpg'.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.

    Returns:
        (numpy.ndarray): The cropped image.

    Example:
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread('image.jpg')
        cropped_im = save_one_box(xyxy, im, file='cropped.jpg', square=True)
        ```
    """

    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop


@threaded
def plot_images(
    images,
    batch_idx,
    cls,
    bboxes=np.zeros(0, dtype=np.float32),
    confs=None,
    masks=np.zeros(0, dtype=np.uint8),
    kpts=np.zeros((0, 51), dtype=np.float32),
    paths=None,
    fname="images.jpg",
    names=None,
    on_plot=None,
    max_subplots=16,
    save=True,
    conf_thres=0.25,
):
    """Plot image grid with labels."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920  # max image size
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., 0::2] *= w  # scale to pixels
                        boxes[..., 1::2] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0::2] += x
                boxes[..., 1::2] += y
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j])

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results('path/to/results.csv', segment=True)
        ```
    """
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 4, 2, 3]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def plt_color_scatter(v, f, bins=20, cmap="viridis", alpha=0.8, edgecolors="none"):
    """
    Plots a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        cmap (str, optional): Colormap for the scatter plot. Defaults to 'viridis'.
        alpha (float, optional): Alpha for the scatter plot. Defaults to 0.8.
        edgecolors (str, optional): Edge colors for the scatter plot. Defaults to 'none'.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """

    # Calculate 2D histogram and corresponding colors
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # Scatter plot
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


def plot_tune_results(csv_file="tune_results.csv"):
    """
    Plot the evolution results stored in an 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.

    Examples:
        >>> plot_tune_results('path/to/tune_results.csv')
    """

    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    # Scatter plots for each hyperparameter
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.values
    fitness = x[:, 0]  # fitness
    j = np.argmax(fitness)  # max fitness index
    n = math.ceil(len(keys) ** 0.5)  # columns and rows in plot
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]  # best single result
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        plt.tick_params(axis="both", labelsize=8)  # Set axis label size to 8
        if i % n != 0:
            plt.yticks([])

    file = csv_file.with_name("tune_scatter_plots.png")  # filename
    plt.savefig(file, dpi=200)
    plt.close()
    LOGGER.info(f"Saved {file}")

    # Fitness vs iteration
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)  # smoothing line
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()

    file = csv_file.with_name("tune_fitness.png")  # filename
    plt.savefig(file, dpi=200)
    plt.close()
    LOGGER.info(f"Saved {file}")


def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def output_to_rotated_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, angle = o[:max_det].cpu().split((4, 1, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box, angle, conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    for m in ["Detect", "Pose", "Segment"]:
        if m in module_type:
            return
    _, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis("off")

        LOGGER.info(f"Saving {f}... ({n}/{channels})")
        plt.savefig(f, dpi=300, bbox_inches="tight")
        plt.close()
        np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save
