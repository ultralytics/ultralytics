# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

if TYPE_CHECKING:
    from ultralytics.data.stereo.box3d import Box3D

try:
    from ultralytics.data.stereo.calib import CalibrationParameters
except ImportError:
    CalibrationParameters = None  # type: ignore
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.checks import check_font, check_version, is_ascii
from ultralytics.utils.files import increment_path


class Colors:
    """Ultralytics color palette for visualization and plotting.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to RGB
    values and accessing predefined color schemes for object detection and pose estimation.

    Attributes:
        palette (list[tuple]): List of RGB color tuples for general use.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array for pose estimation with dtype np.uint8.

    Examples:
        >>> from ultralytics.utils.plotting import Colors
        >>> colors = Colors()
        >>> colors(5, True)  # Returns BGR format: (221, 111, 255)
        >>> colors(5, False)  # Returns RGB format: (255, 111, 221)

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand).
        Please use the official Ultralytics colors for all marketing materials.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
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

    def __call__(self, i: int | torch.Tensor, bgr: bool = False) -> tuple:
        """Convert hex color codes to RGB values.

        Args:
            i (int | torch.Tensor): Color index.
            bgr (bool, optional): Whether to return BGR format instead of RGB.

        Returns:
            (tuple): RGB or BGR color tuple.
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """Convert hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


@dataclass
class VisualizationConfig:
    """Configuration container for stereo 3D visualization helpers."""

    line_width: int = 2
    font_size: float = 0.5
    show_labels: bool = True
    show_conf: bool = True
    camera_view: str = "both"
    pred_color_scheme: dict[int, tuple[int, int, int]] = field(
        default_factory=lambda: {
            0: (0, 128, 255),  # Car - orange tone in BGR
            1: (64, 64, 255),  # Pedestrian - reddish in BGR
            2: (221, 111, 255),  # Cyclist - magenta
        }
    )
    gt_color_scheme: dict[int, tuple[int, int, int]] = field(
        default_factory=lambda: {
            0: (0, 255, 0),  # Car - green
            1: (255, 180, 0),  # Pedestrian - cyan/blue tone
            2: (0, 223, 183),  # Cyclist - teal
        }
    )

    def __post_init__(self) -> None:
        if self.camera_view not in {"left", "right", "both"}:
            raise ValueError(f"camera_view must be 'left', 'right', or 'both', got '{self.camera_view}'")
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")


EDGE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


class Annotator:
    """Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image | np.ndarray): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype | ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (list[list[int]]): Skeleton structure for keypoints.
        limb_color (list[int]): Color palette for limbs.
        kpt_color (list[int]): Color palette for keypoints.
        dark_colors (set): Set of colors considered dark for text contrast.
        light_colors (set): Set of colors considered light for text contrast.

    Examples:
        >>> from ultralytics.utils.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if not input_is_pil:
            if im.shape[2] == 1:  # handle grayscale
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] > 3:  # multispectral
                im = np.ascontiguousarray(im[..., :3])
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)
            if self.im.mode not in {"RGB", "RGBA"}:  # multispectral
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
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
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """Assign text color based on background color.

        Args:
            color (tuple, optional): The background color of the rectangle for text (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Returns:
            (tuple): Text color for label.

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # return (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(self, box, label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """Draw a bounding box on an image with a given label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str, optional): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        multi_points = isinstance(box[0], list)  # multiple points with shape (n, 2)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))
        if self.pil:
            self.draw.polygon(
                [tuple(b) for b in box], width=self.lw, outline=color
            ) if multi_points else self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text([box[0], box[1]], label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            cv2.polylines(
                self.im, [np.asarray(box, dtype=int)], True, color, self.lw
            ) if multi_points else cv2.rectangle(
                self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu: torch.Tensor = None, alpha: float = 0.5, retina_masks: bool = False):
        """Plot masks on image.

        Args:
            masks (torch.Tensor | np.ndarray): Predicted masks with shape: [n, h, w]
            colors (list[list[int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (torch.Tensor | None): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float, optional): Mask transparency: 0.0 fully transparent, 1.0 opaque.
            retina_masks (bool, optional): Whether to use high resolution masks or not.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            assert isinstance(masks, np.ndarray), "`masks` must be a np.ndarray if `im_gpu` is not provided."
            overlay = self.im.copy()
            for i, mask in enumerate(masks):
                overlay[mask.astype(bool)] = colors[i]
            self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        else:
            assert isinstance(masks, torch.Tensor), "'masks' must be a torch.Tensor if 'im_gpu' is provided."
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
                return
            if im_gpu.device != masks.device:
                im_gpu = im_gpu.to(masks.device)

            ih, iw = self.im.shape[:2]
            if not retina_masks:
                # Use scale_masks to properly remove padding and upsample, convert bool to float first
                masks = ops.scale_masks(masks[None].float(), (ih, iw))[0] > 0.5
                # Convert original BGR image to RGB tensor
                im_gpu = (
                    torch.from_numpy(self.im).to(masks.device).permute(2, 0, 1).flip(0).contiguous().float() / 255.0
                )

            colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)
            inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

            im_gpu = im_gpu.flip(dims=[0]).permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
            self.im[:] = (im_gpu * 255).byte().cpu().numpy()
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(
        self,
        kpts,
        shape: tuple = (640, 640),
        radius: int | None = None,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
        kpt_color: tuple | None = None,
    ):
        """Plot keypoints on the image.

        Args:
            kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
            shape (tuple, optional): Image shape (h, w).
            radius (int, optional): Keypoint radius.
            kpt_line (bool, optional): Draw lines between keypoints.
            conf_thres (float, optional): Confidence threshold.
            kpt_color (tuple, optional): Keypoint color (B, G, R).

        Notes:
            - `kpt_line=True` currently only supports human pose plotting.
            - Modifies self.im in-place.
            - If self.pil is True, converts image to numpy array and back to PIL.
        """
        radius = radius if radius is not None else self.lw
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
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
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text: str, txt_color: tuple = (255, 255, 255), anchor: str = "top", box_color: tuple = ()):
        """Add text to an image using PIL or cv2.

        Args:
            xy (list[int]): Top-left coordinates for text placement.
            text (str): Text to be drawn.
            txt_color (tuple, optional): Text color (R, G, B).
            anchor (str, optional): Text anchor position ('top' or 'bottom').
            box_color (tuple, optional): Box color (R, G, B, A) with optional alpha.
        """
        if self.pil:
            w, h = self.font.getsize(text)
            if anchor == "bottom":  # start y from font bottom
                xy[1] += 1 - h
            for line in text.split("\n"):
                if box_color:
                    # Draw rectangle for each line
                    w, h = self.font.getsize(line)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=box_color)
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            if box_color:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, box_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title: str | None = None):
        """Show the annotated image."""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # Convert numpy array to PIL Image with RGB to BGR
        if IS_COLAB or IS_KAGGLE:  # can not use IS_JUPYTER as will run for all ipython environments
            try:
                display(im)  # noqa - display() function only available in ipython environments
            except ImportError as e:
                LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        else:
            im.show(title=title)

    def save(self, filename: str = "image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox: tuple | None = None):
        """Calculate the dimensions and area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            area (float): Area enclosed by the bounding box.

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_bbox_dimension(bbox=[10, 20, 30, 40])
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height


@TryExcept()
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics.

    Args:
        boxes (np.ndarray): Bounding box coordinates in format [x, y, width, height].
        cls (np.ndarray): Class indices.
        names (dict, optional): Dictionary mapping class indices to class names.
        save_dir (Path, optional): Directory to save the plot.
        on_plot (Callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars
    from matplotlib.colors import LinearSegmentedColormap

    # Filter matplotlib>=3.7.2 warning
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])

    # Matplotlib labels
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        ax[0].bar_label(y[2])
    else:
        ax[0].set_xlabel("classes")
    boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax[3].set_xlabel("width")
    ax[3].set_ylabel("height")
    for a in {0, 1, 2, 3}:
        for s in {"top", "right", "left", "bottom"}:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(
    xyxy,
    im,
    file: Path = Path("im.jpg"),
    gain: float = 1.02,
    pad: int = 10,
    square: bool = False,
    BGR: bool = False,
    save: bool = True,
):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according to the
    bounding box. Optionally, the crop can be squared, and the function allows for gain and padding adjustments to the
    bounding box.

    Args:
        xyxy (torch.Tensor | list): A tensor or list representing the bounding box in xyxy format.
        im (np.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box.
        square (bool, optional): If True, the bounding box will be transformed into a square.
        BGR (bool, optional): If True, the image will be returned in BGR format, otherwise in RGB.
        save (bool, optional): If True, the cropped image will be saved to disk.

    Returns:
        (np.ndarray): The cropped image.

    Examples:
        >>> from ultralytics.utils.plotting import save_one_box
        >>> xyxy = [50, 50, 150, 150]
        >>> im = cv2.imread("image.jpg")
        >>> cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
    """
    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    grayscale = im.shape[2] == 1  # grayscale image
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR or grayscale else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        crop = crop.squeeze(-1) if grayscale else crop[..., ::-1] if BGR else crop
        Image.fromarray(crop).save(f, quality=95, subsampling=0)  # save RGB
    return crop


@threaded
def plot_images(
    labels: dict[str, Any],
    images: torch.Tensor | np.ndarray = np.zeros((0, 3, 640, 640), dtype=np.float32),
    paths: list[str] | None = None,
    fname: str = "images.jpg",
    names: dict[int, str] | None = None,
    on_plot: Callable | None = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> np.ndarray | None:
    """Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        labels (dict[str, Any]): Dictionary containing detection data with keys like 'cls', 'bboxes', 'conf', 'masks',
            'keypoints', 'batch_idx', 'img'.
        images (torch.Tensor | np.ndarray]): Batch of images to plot. Shape: (batch_size, channels, height, width).
        paths (Optional[list[str]]): List of file paths for each image in the batch.
        fname (str): Output filename for the plotted image grid.
        names (Optional[dict[int, str]]): Dictionary mapping class indices to class names.
        on_plot (Optional[Callable]): Optional callback function to be called after saving the plot.
        max_size (int): Maximum size of the output image grid.
        max_subplots (int): Maximum number of subplots in the image grid.
        save (bool): Whether to save the plotted image grid to a file.
        conf_thres (float): Confidence threshold for displaying detections.

    Returns:
        (np.ndarray): Plotted image grid as a numpy array if save is False, None otherwise.

    Notes:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.

        Channel Support:
        - 1 channel: Grayscale
        - 2 channels: Third channel added as zeros
        - 3 channels: Used as-is (standard RGB)
        - 4+ channels: Cropped to first 3 channels
    """
    for k in {"cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images"}:
        if k not in labels:
            continue
        if k == "cls" and labels[k].ndim == 2:
            labels[k] = labels[k].squeeze(1)  # squeeze if shape is (n, 1)
        if isinstance(labels[k], torch.Tensor):
            labels[k] = labels[k].cpu().numpy()

    cls = labels.get("cls", np.zeros(0, dtype=np.int64))
    batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
    bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
    confs = labels.get("conf", None)
    masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
    kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
    images = labels.get("img", images)  # default to input images

    if len(images) and isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # Handle 2-ch and n-ch images
    c = images.shape[1]
    if c == 2:
        zero = np.zeros_like(images[:, :1])
        images = np.concatenate((images, zero), axis=1)  # pad 2-ch with a black channel
    elif c > 3:
        images = images[:, :3]  # crop multispectral images to first 3 channels

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
    fs = max(fs, 18)  # ensure that the font size is large enough to be easily readable.
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None
            conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)

            if len(bboxes):
                boxes = bboxes[idx]
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                # TODO: this transformation might be unnecessary
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    label = f"{c}" if labels else f"{c} {conf[0]:.1f}"
                    annotator.text([x, y], label, txt_color=color, box_color=(64, 64, 64, 128))

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
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0] and masks.max() <= 1:  # overlap_mask=False
                    image_masks = masks[idx]
                else:  # overlap_mask=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(1, nl + 1).reshape((nl, 1, 1))
                    image_masks = (image_masks == index).astype(np.float32)

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
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None):
    """Plot training results from a results CSV file. The function supports various types of data including
    segmentation, pose estimation, and classification. Plots are saved as 'results.png' in the directory where the
    CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.

    Examples:
        >>> from ultralytics.utils.plotting import plot_results
        >>> plot_results("path/to/results.csv", segment=True)
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = (
                    loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                )
                fig, ax = plt.subplots(2, len(columns) // 2, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel()
            x = data.select(data.columns[0]).to_numpy().flatten()
            for i, j in enumerate(columns):
                y = data.select(j).to_numpy().flatten().astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(j, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def plt_color_scatter(v, f, bins: int = 20, cmap: str = "viridis", alpha: float = 0.8, edgecolors: str = "none"):
    """Plot a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram.
        cmap (str, optional): Colormap for the scatter plot.
        alpha (float, optional): Alpha for the scatter plot.
        edgecolors (str, optional): Edge colors for the scatter plot.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

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


@plt_settings()
def plot_tune_results(csv_file: str = "tune_results.csv", exclude_zero_fitness_points: bool = True):
    """Plot the evolution results stored in a 'tune_results.csv' file. The function generates a scatter plot for each
    key in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on
    the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results.
        exclude_zero_fitness_points (bool, optional): Don't include points with zero fitness in tuning plots.

    Examples:
        >>> plot_tune_results("path/to/tune_results.csv")
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"Saved {file}")

    # Scatter plots for each hyperparameter
    csv_file = Path(csv_file)
    data = pl.read_csv(csv_file, infer_schema_length=None)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.to_numpy()
    fitness = x[:, 0]  # fitness
    if exclude_zero_fitness_points:
        mask = fitness > 0  # exclude zero-fitness points
        x, fitness = x[mask], fitness[mask]
    # Iterative sigma rejection on lower bound only
    for _ in range(3):  # max 3 iterations
        mean, std = fitness.mean(), fitness.std()
        lower_bound = mean - 3 * std
        mask = fitness >= lower_bound
        if mask.all():  # no more outliers
            break
        x, fitness = x[mask], fitness[mask]
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
    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))

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
    _save_one_file(csv_file.with_name("tune_fitness.png"))


@plt_settings()
def feature_visualization(x, module_type: str, stage: int, n: int = 32, save_dir: Path = Path("runs/detect/exp")):
    """Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot.
        save_dir (Path, optional): Directory to save results.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:
            return
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.rsplit('.', 1)[-1]}_features.png"  # filename

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


def project_3d_to_2d(
    box3d: "Box3D",
    calib: CalibrationParameters,
) -> tuple[float, float, float, float]:
    """Project 3D bounding box to 2D bounding box using camera calibration.

    Projects the 8 corners of a 3D box to the 2D image plane and computes
    the axis-aligned bounding box that contains all projected corners.

    Args:
        box3d: 3D bounding box (Box3D object).
        calib: Camera calibration parameters dict with keys:
            fx, fy, cx, cy (focal lengths and principal point).

    Returns:
        tuple: 2D bounding box (x_min, y_min, x_max, y_max) in pixels.
    """
    from ultralytics.data.stereo.box3d import Box3D

    x, y, z = box3d.center_3d
    length, width, height = box3d.dimensions
    orientation = box3d.orientation

    # Extract calibration parameters (support both dict and CalibrationParameters)
    if isinstance(calib, CalibrationParameters):
        fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
        image_width = calib.image_width
        image_height = calib.image_height
    else:
        fx = calib.get("fx", 721.5377)
        fy = calib.get("fy", 721.5377)
        cx = calib.get("cx", 609.5593)
        cy = calib.get("cy", 172.8540)
        image_width = calib.get("image_width", 1242)
        image_height = calib.get("image_height", 375)       

    # Generate 8 corners in object coordinate system
    # KITTI convention: rotation_y=0 means object faces camera X direction
    # So object's length (forward direction) should be along X axis
    # Coordinate system: x: right (length), y: down (height), z: forward (width)
    # EDGE_CONNECTIONS expects: bottom face (0,1,2,3), top face (4,5,6,7)
    # In KITTI camera coords: Y points down, so bottom has y=+height/2, top has y=-height/2
    corners_obj = np.array(
        [
            # Bottom face corners (y = +height/2): 0, 1, 2, 3
            # Top face corners (y = -height/2): 4, 5, 6, 7
            [-length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2],  # x (length)
            [height / 2, height / 2, height / 2, height / 2, -height / 2, -height / 2, -height / 2, -height / 2],  # y (height) - bottom first, then top
            [width / 2, width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2],  # z (width)
        ]
    )

    # Rotation matrix around y-axis
    cos_rot = np.cos(orientation)
    sin_rot = np.sin(orientation)
    R = np.array([[cos_rot, 0, sin_rot], [0, 1, 0], [-sin_rot, 0, cos_rot]])

    # Rotate and translate to world coordinates
    corners_world = R @ corners_obj
    corners_world[0, :] += x
    corners_world[1, :] += y
    corners_world[2, :] += z

    # Project to 2D image plane
    # u = fx × X/Z + cx, v = fy × Y/Z + cy
    X, Y, Z = corners_world
    # Avoid division by zero
    Z = np.maximum(Z, 1e-6)
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # Compute axis-aligned bounding box
    x_min = float(np.min(u))
    y_min = float(np.min(v))
    x_max = float(np.max(u))
    y_max = float(np.max(v))

    return (x_min, y_min, x_max, y_max)


def project_box3d_corners(
    box3d: "Box3D",
    calib: CalibrationParameters | dict[str, float],
    letterbox_scale: float | None = None,
    letterbox_pad_left: float | None = None,
    letterbox_pad_top: float | None = None,
) -> np.ndarray:
    """Project the eight corners of a 3D bounding box to 2D pixel coordinates.
    
    Args:
        box3d: 3D bounding box to project
        calib: Camera calibration parameters (for original image size)
        letterbox_scale: Scale factor from letterboxing (if images were letterboxed)
        letterbox_pad_left: Left padding from letterboxing (if images were letterboxed)
        letterbox_pad_top: Top padding from letterboxing (if images were letterboxed)
    
    Returns:
        Array of 2D pixel coordinates [8, 2] with shape (u, v) for each corner
    """
    from ultralytics.data.stereo.box3d import Box3D  # Import here to avoid circular import

    def _get_calib_params(cal: CalibrationParameters | dict[str, float]) -> tuple[float, float, float, float]:
        if isinstance(cal, dict):
            return cal["fx"], cal["fy"], cal["cx"], cal["cy"]
        # Assume it's a CalibrationParameters object
        return cal.fx, cal.fy, cal.cx, cal.cy

    fx, fy, cx, cy = _get_calib_params(calib)

    x, y, z = box3d.center_3d
    length, width, height = box3d.dimensions
    orientation = box3d.orientation

    # Skip boxes with invalid Z depth (too shallow or negative)
    # Z < MIN_VALID_Z can cause extreme pixel coordinates when projecting (u = fx * X / Z)
    # Example: Z=1.36 produced 3.4 billion pixels causing cv2.clipLine TypeError
    # Use conservative 2.0m threshold for KITTI dataset stability
    # Objects closer than 2m are rare and can cause projection artifacts
    # Lower thresholds (1.0, 1.5) can be used but may cause edge cases
    MIN_VALID_Z = 2.0  # Minimum valid Z depth in meters (conservative threshold)
    if z < MIN_VALID_Z or not np.isfinite(z):
        return np.zeros((8, 2), dtype=np.float32)  # Return dummy corners (will be clipped out)

    # KITTI convention: rotation_y=0 means object faces camera X direction
    # So object's length (forward direction) should be along X axis
    # EDGE_CONNECTIONS expects: bottom face (0,1,2,3), top face (4,5,6,7)
    # In KITTI camera coords: Y points down, so bottom has y=+height/2, top has y=-height/2
    corners_obj = np.array(
        [
            # Bottom face corners (y = +height/2): 0, 1, 2, 3
            # Top face corners (y = -height/2): 4, 5, 6, 7
            [-length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2],  # x (length)
            [height / 2, height / 2, height / 2, height / 2, -height / 2, -height / 2, -height / 2, -height / 2],  # y (height) - bottom first, then top
            [width / 2, width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2],  # z (width)
        ]
    )

    cos_rot = np.cos(orientation)
    sin_rot = np.sin(orientation)
    rotation = np.array([[cos_rot, 0, sin_rot], [0, 1, 0], [-sin_rot, 0, cos_rot]])

    corners_world = rotation @ corners_obj
    corners_world[0, :] += x
    corners_world[1, :] += y
    corners_world[2, :] += z

    X, Y, Z = corners_world
    Z = np.maximum(Z, 1e-6)
    
    # Project to original image coordinates
    u_orig = fx * X / Z + cx
    v_orig = fy * Y / Z + cy
    
    # Adjust for letterboxing if provided
    if letterbox_scale is not None and letterbox_pad_left is not None and letterbox_pad_top is not None:
        u = u_orig * letterbox_scale + letterbox_pad_left
        v = v_orig * letterbox_scale + letterbox_pad_top
    else:
        u = u_orig
        v = v_orig

    corners = np.stack((u, v), axis=-1).astype(np.float32)
    return corners


def _select_color(
    class_id: int,
    scheme: dict[int, tuple[int, int, int]],
) -> tuple[int, int, int]:
    color = scheme.get(class_id)
    if color is None:
        color = colors(class_id, bgr=True)
    return tuple(int(c) for c in color)


def plot_boxes3d(
    img: np.ndarray,
    boxes3d: list["Box3D"] | None,
    calib: CalibrationParameters | dict[str, float],
    config: VisualizationConfig | None = None,
    is_ground_truth: bool = False,
    letterbox_scale: float | None = None,
    letterbox_pad_left: float | None = None,
    letterbox_pad_top: float | None = None,
) -> np.ndarray:
    """Draw wireframe representations of Box3D objects onto an image.
    
    Args:
        img: Image to draw on (may be letterboxed)
        boxes3d: List of 3D bounding boxes to draw
        calib: Camera calibration parameters (for original image size)
        config: Visualization configuration
        is_ground_truth: Whether boxes are ground truth (affects color scheme)
        letterbox_scale: Scale factor from letterboxing (if images were letterboxed)
        letterbox_pad_left: Left padding from letterboxing (if images were letterboxed)
        letterbox_pad_top: Top padding from letterboxing (if images were letterboxed)
    """
    from ultralytics.data.stereo.box3d import Box3D  # Import here to avoid circular import

    config = config or VisualizationConfig()
    
    # Ensure input image is uint8 and properly initialized
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Create a properly initialized copy of the image
    canvas = img.copy().astype(np.uint8)
    
    if not boxes3d:
        return canvas

    height, width = canvas.shape[:2]
    rect = (0, 0, width - 1, height - 1)
    line_width = max(1, config.line_width)
    scheme = config.gt_color_scheme if is_ground_truth else config.pred_color_scheme

    for box in boxes3d:
        try:
            corners = project_box3d_corners(
                box, 
                calib,
                letterbox_scale=letterbox_scale,
                letterbox_pad_left=letterbox_pad_left,
                letterbox_pad_top=letterbox_pad_top,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping invalid Box3D during visualization: %s", exc)
            continue
        
        # Skip if corners are invalid (all zeros from Z < MIN_VALID_Z)
        if np.allclose(corners, 0.0, atol=1e-6):
            LOGGER.debug(f"Skipping Box3D with invalid Z depth (corners all zero)")
            continue

        color = _select_color(getattr(box, "class_id", 0), scheme)

        for start, end in EDGE_CONNECTIONS:
            pt1 = (int(round(corners[start][0])), int(round(corners[start][1])))
            pt2 = (int(round(corners[end][0])), int(round(corners[end][1])))
            clipped, clip_pt1, clip_pt2 = cv2.clipLine(rect, pt1, pt2)
            if clipped:
                cv2.line(canvas, clip_pt1, clip_pt2, color, line_width, lineType=cv2.LINE_AA)

        if config.show_labels:
            label = getattr(box, "class_label", "object")
            if config.show_conf and hasattr(box, "confidence"):
                label = f"{label} {box.confidence:.2f}"
            anchor = (
                int(np.clip(corners[0][0], 0, width - 1)),
                int(np.clip(corners[0][1], 0, height - 1)),
            )
            cv2.putText(
                canvas,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_size,
                color,
                max(1, line_width // 2),
                cv2.LINE_AA,
            )

    return canvas


def plot_boxes2d(
    img: np.ndarray,
    boxes3d: list["Box3D"] | None,
    config: VisualizationConfig | None = None,
    calib: dict[str, float] | None = None,
) -> np.ndarray:
    """Draw 2D bounding boxes (projected from 3D) onto an image.

    Args:
        img: Image to draw on.
        boxes3d: List of Box3D objects to project and draw.
        config: Visualization configuration (default: VisualizationConfig()).
        calib: Calibration dict for 3D-to-2D projection.
    """
    config = config or VisualizationConfig()

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img.copy().astype(np.uint8)

    canvas = img.astype(np.uint8)

    if not boxes3d or calib is None:
        return canvas

    height, width = canvas.shape[:2]
    line_width = max(1, config.line_width)

    for box in boxes3d:
        bbox_2d = box.project_to_2d(calib, image_size=(width, height))
        x_min, y_min, x_max, y_max = int(bbox_2d[0]), int(bbox_2d[1]), int(bbox_2d[2]), int(bbox_2d[3])
        if x_max <= x_min or y_max <= y_min:
            continue
        color = _select_color(0, config.pred_color_scheme)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, line_width, cv2.LINE_AA)
    return canvas


def plot_stereo3d_boxes(
    left_img: np.ndarray,
    right_img: np.ndarray,
    pred_boxes3d: list["Box3D"] | None = None,
    gt_boxes3d: list["Box3D"] | None = None,
    left_calib: CalibrationParameters | dict[str, float] | None = None,
    right_calib: CalibrationParameters | dict[str, float] | None = None,
    config: VisualizationConfig | None = None,
    letterbox_scale: float | None = None,
    letterbox_pad_left: float | None = None,
    letterbox_pad_top: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw predictions and ground truth on stereo image pairs.
    
    Args:
        left_img: Left camera image (may be letterboxed)
        right_img: Right camera image (may be letterboxed)
        pred_boxes3d: List of predicted 3D bounding boxes
        gt_boxes3d: List of ground truth 3D bounding boxes
        left_calib: Left camera calibration parameters (for original image size)
        right_calib: Right camera calibration parameters (defaults to left_calib)
        config: Visualization configuration
        letterbox_scale: Scale factor from letterboxing (if images were letterboxed)
        letterbox_pad_left: Left padding from letterboxing (if images were letterboxed)
        letterbox_pad_top: Top padding from letterboxing (if images were letterboxed)
    """
    from ultralytics.data.stereo.box3d import Box3D  # Import here to avoid circular import

    if left_calib is None:
        raise ValueError("left_calib is required for stereo visualization")
    config = config or VisualizationConfig()
    right_calib = right_calib or left_calib

    left_canvas = plot_boxes3d(
        left_img, pred_boxes3d, left_calib, config, is_ground_truth=False,
        letterbox_scale=letterbox_scale, letterbox_pad_left=letterbox_pad_left, letterbox_pad_top=letterbox_pad_top
    )
    left_canvas = plot_boxes3d(
        left_canvas, gt_boxes3d, left_calib, config, is_ground_truth=True,
        letterbox_scale=letterbox_scale, letterbox_pad_left=letterbox_pad_left, letterbox_pad_top=letterbox_pad_top
    )

    # Convert calib to dict if needed for project_to_2d
    calib_dict = left_calib.to_dict() if hasattr(left_calib, "to_dict") else left_calib
    right_canvas = plot_boxes2d(
        left_img, pred_boxes3d, config, calib=calib_dict
    )
    right_canvas = plot_boxes2d(
        right_canvas, gt_boxes3d, config, calib=calib_dict
    )

    combined = combine_stereo_views(left_canvas, right_canvas)
    return left_canvas, right_canvas, combined


def combine_stereo_views(
    left_img: np.ndarray,
    right_img: np.ndarray,
    pad_value: int = 0,
) -> np.ndarray:
    """Horizontally stack stereo images, padding the shorter view if necessary."""

    if left_img.ndim != 3 or right_img.ndim != 3:
        raise ValueError("Stereo images must be rank-3 tensors shaped [H, W, C].")

    # Ensure images are uint8 and properly initialized
    if left_img.dtype != np.uint8:
        left_img = np.clip(left_img, 0, 255).astype(np.uint8)
    if right_img.dtype != np.uint8:
        right_img = np.clip(right_img, 0, 255).astype(np.uint8)

    max_height = max(left_img.shape[0], right_img.shape[0])

    def _pad_to_height(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == max_height:
            return img.copy()  # Make a copy to avoid modifying original
        pad_rows = max_height - img.shape[0]
        # Create a new array with proper initialization
        padded = np.full((max_height, img.shape[1], img.shape[2]), pad_value, dtype=np.uint8)
        padded[:img.shape[0], :, :] = img
        return padded

    left_padded = _pad_to_height(left_img)
    right_padded = _pad_to_height(right_img)
    return np.hstack((left_padded, right_padded))
