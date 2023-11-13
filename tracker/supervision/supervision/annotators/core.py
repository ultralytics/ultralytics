from math import sqrt
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from supervision.annotators.base import BaseAnnotator
from supervision.annotators.utils import ColorLookup, Trace, resolve_color
from supervision.detection.core import Detections
from supervision.detection.utils import clip_boxes
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


class BoundingBoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with bounding boxes based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> bounding_box_annotator = sv.BoundingBoxAnnotator()
            >>> annotated_frame = bounding_box_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![bounding-box-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/bounding-box-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
        return scene


class MaskAnnotator(BaseAnnotator):
    """
    A class for drawing masks on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with masks based on the provided detections.

        Args:
            scene (np.ndarray): The image where masks will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> mask_annotator = sv.MaskAnnotator()
            >>> annotated_frame = mask_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/mask-annotator-example-purple.png)
        """
        if detections.mask is None:
            return scene

        for detection_idx in np.flip(np.argsort(detections.area)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            mask = detections.mask[detection_idx]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()
            scene[mask] = cv2.addWeighted(
                colored_mask, self.opacity, scene, 1 - self.opacity, 0
            )[mask]

        return scene


class BoxMaskAnnotator(BaseAnnotator):
    """
    A class for drawing box masks on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.5,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.color_lookup: ColorLookup = color_lookup
        self.opacity = opacity

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with box masks based on the provided detections.

        Args:
            scene (np.ndarray): The image where bounding boxes will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> box_mask_annotator = sv.BoxMaskAnnotator()
            >>> annotated_frame = box_mask_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![box-mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-mask-annotator-example-purple.png)
        """
        mask_image = scene.copy()
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=-1,
            )
        scene = cv2.addWeighted(
            scene, self.opacity, mask_image, 1 - self.opacity, gamma=0
        )
        return scene


class HaloAnnotator(BaseAnnotator):
    """
    A class for drawing Halos on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        opacity: float = 0.8,
        kernel_size: int = 40,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            kernel_size (int): The size of the average pooling kernel used for creating
                the halo.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.opacity = opacity
        self.color_lookup: ColorLookup = color_lookup
        self.kernel_size: int = kernel_size

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with halos based on the provided detections.

        Args:
            scene (np.ndarray): The image where masks will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> halo_annotator = sv.HaloAnnotator()
            >>> annotated_frame = halo_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![halo-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/halo-annotator-example-purple.png)
        """
        if detections.mask is None:
            return scene
        colored_mask = np.zeros_like(scene, dtype=np.uint8)
        fmask = np.array([False] * scene.shape[0] * scene.shape[1]).reshape(
            scene.shape[0], scene.shape[1]
        )

        for detection_idx in np.flip(np.argsort(detections.area)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            mask = detections.mask[detection_idx]
            fmask = np.logical_or(fmask, mask)
            color_bgr = color.as_bgr()
            colored_mask[mask] = color_bgr

        colored_mask = cv2.blur(colored_mask, (self.kernel_size, self.kernel_size))
        colored_mask[fmask] = [0, 0, 0]
        gray = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
        alpha = self.opacity * gray / gray.max()
        alpha_mask = alpha[:, :, np.newaxis]
        scene = np.uint8(scene * (1 - alpha_mask) + colored_mask * self.opacity)
        return scene


class EllipseAnnotator(BaseAnnotator):
    """
    A class for drawing ellipses on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        start_angle: int = -45,
        end_angle: int = 235,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the ellipse lines.
            start_angle (int): Starting angle of the ellipse.
            end_angle (int): Ending angle of the ellipse.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.start_angle: int = start_angle
        self.end_angle: int = end_angle
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with ellipses based on the provided detections.

        Args:
            scene (np.ndarray): The image where ellipses will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> ellipse_annotator = sv.EllipseAnnotator()
            >>> annotated_frame = ellipse_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![ellipse-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/ellipse-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            center = (int((x1 + x2) / 2), y2)
            width = x2 - x1
            cv2.ellipse(
                scene,
                center=center,
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=self.start_angle,
                endAngle=self.end_angle,
                color=color.as_bgr(),
                thickness=self.thickness,
                lineType=cv2.LINE_4,
            )
        return scene


class BoxCornerAnnotator(BaseAnnotator):
    """
    A class for drawing box corners on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 4,
        corner_length: int = 15,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the corner lines.
            corner_length (int): Length of each corner line.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.corner_length: int = corner_length
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with box corners based on the provided detections.

        Args:
            scene (np.ndarray): The image where box corners will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> corner_annotator = sv.BoxCornerAnnotator()
            >>> annotated_frame = corner_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![box-corner-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-corner-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

            for x, y in corners:
                x_end = x + self.corner_length if x == x1 else x - self.corner_length
                cv2.line(
                    scene, (x, y), (x_end, y), color.as_bgr(), thickness=self.thickness
                )

                y_end = y + self.corner_length if y == y1 else y - self.corner_length
                cv2.line(
                    scene, (x, y), (x, y_end), color.as_bgr(), thickness=self.thickness
                )
        return scene


class CircleAnnotator(BaseAnnotator):
    """
    A class for drawing circle on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the circle line.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with circles based on the provided detections.

        Args:
            scene (np.ndarray): The image where box corners will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> circle_annotator = sv.CircleAnnotator()
            >>> annotated_frame = circle_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```


        ![circle-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/circle-annotator-example-purple.png)
        """
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.circle(
                img=scene,
                center=center,
                radius=int(distance),
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        return scene


class DotAnnotator(BaseAnnotator):
    """
    A class for drawing dots on an image at specific coordinates based on provided
    detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        radius: int = 4,
        position: Position = Position.CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            radius (int): Radius of the drawn dots.
            position (Position): The anchor position for placing the dot.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.radius: int = radius
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with dots based on the provided detections.

        Args:
            scene (np.ndarray): The image where dots will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> dot_annotator = sv.DotAnnotator()
            >>> annotated_frame = dot_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![dot-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/dot-annotator-example-purple.png)
        """
        xy = detections.get_anchor_coordinates(anchor=self.position)
        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            center = (int(xy[detection_idx, 0]), int(xy[detection_idx, 1]))
            cv2.circle(scene, center, self.radius, color.as_bgr(), -1)
        return scene


class LabelAnnotator:
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_position: Position = text_position
        self.color_lookup: ColorLookup = color_lookup

    @staticmethod
    def resolve_text_background_xyxy(
        detection_xyxy: Tuple[int, int, int, int],
        text_wh: Tuple[int, int],
        text_padding: int,
        position: Position,
    ) -> Tuple[int, int, int, int]:
        padded_text_wh = (text_wh[0] + 2 * text_padding, text_wh[1] + 2 * text_padding)
        x1, y1, x2, y2 = detection_xyxy
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if position == Position.TOP_LEFT:
            return x1, y1 - padded_text_wh[1], x1 + padded_text_wh[0], y1
        elif position == Position.TOP_RIGHT:
            return x2 - padded_text_wh[0], y1 - padded_text_wh[1], x2, y1
        elif position == Position.TOP_CENTER:
            return (
                center_x - padded_text_wh[0] // 2,
                y1 - padded_text_wh[1],
                center_x + padded_text_wh[0] // 2,
                y1,
            )
        elif position == Position.CENTER:
            return (
                center_x - padded_text_wh[0] // 2,
                center_y - padded_text_wh[1] // 2,
                center_x + padded_text_wh[0] // 2,
                center_y + padded_text_wh[1] // 2,
            )
        elif position == Position.BOTTOM_LEFT:
            return x1, y2, x1 + padded_text_wh[0], y2 + padded_text_wh[1]
        elif position == Position.BOTTOM_RIGHT:
            return x2 - padded_text_wh[0], y2, x2, y2 + padded_text_wh[1]
        elif position == Position.BOTTOM_CENTER:
            return (
                center_x - padded_text_wh[0] // 2,
                y2,
                center_x + padded_text_wh[0] // 2,
                y2 + padded_text_wh[1],
            )

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: List[str] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided detections.

        Args:
            scene (np.ndarray): The image where labels will be drawn.
            detections (Detections): Object detections to annotate.
            labels (List[str]): Optional. Custom labels for each detection.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            >>> annotated_frame = label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![label-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/label-annotator-example-purple.png)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection_idx in range(len(detections)):
            detection_xyxy = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            text = (
                f"{detections.class_id[detection_idx]}"
                if (labels is None or len(detections) != len(labels))
                else labels[detection_idx]
            )
            text_wh = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_background_xyxy = self.resolve_text_background_xyxy(
                detection_xyxy=detection_xyxy,
                text_wh=text_wh,
                text_padding=self.text_padding,
                position=self.text_position,
            )

            text_x = text_background_xyxy[0] + self.text_padding
            text_y = text_background_xyxy[1] + self.text_padding + text_wh[1]

            cv2.rectangle(
                img=scene,
                pt1=(text_background_xyxy[0], text_background_xyxy[1]),
                pt2=(text_background_xyxy[2], text_background_xyxy[3]),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


class BlurAnnotator(BaseAnnotator):
    """
    A class for blurring regions in an image using provided detections.
    """

    def __init__(self, kernel_size: int = 15):
        """
        Args:
            kernel_size (int): The size of the average pooling kernel used for blurring.
        """
        self.kernel_size: int = kernel_size

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        """
        Annotates the given scene by blurring regions based on the provided detections.

        Args:
            scene (np.ndarray): The image where blurring will be applied.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> blur_annotator = sv.BlurAnnotator()
            >>> annotated_frame = circle_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![blur-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/blur-annotator-example-purple.png)
        """
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)

        for x1, y1, x2, y2 in clipped_xyxy:
            roi = scene[y1:y2, x1:x2]
            roi = cv2.blur(roi, (self.kernel_size, self.kernel_size))
            scene[y1:y2, x1:x2] = roi

        return scene


class TraceAnnotator:
    """
    A class for drawing trace paths on an image based on detection coordinates.

    !!! warning

        This annotator utilizes the `tracker_id`. Read
        [here](https://supervision.roboflow.com/trackers/) to learn how to plug
        tracking into your inference pipeline.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        position: Position = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color to draw the trace, can be
                a single color or a color palette.
            position (Position): The position of the trace.
                Defaults to `CENTER`.
            trace_length (int): The maximum length of the trace in terms of historical
                points. Defaults to `30`.
            thickness (int): The thickness of the trace lines. Defaults to `2`.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACE`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.position = position
        self.trace = Trace(max_size=trace_length)
        self.thickness = thickness
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draws trace paths on the frame based on the detection coordinates provided.

        Args:
            scene (np.ndarray): The image on which the traces will be drawn.
            detections (Detections): The detections which include coordinates for
                which the traces will be drawn.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            np.ndarray: The image with the trace paths drawn on it.

        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> model = YOLO('yolov8x.pt')

            >>> trace_annotator = sv.TraceAnnotator()

            >>> video_info = sv.VideoInfo.from_video_path(video_path='...')
            >>> frames_generator = sv.get_video_frames_generator(source_path='...')
            >>> tracker = sv.ByteTrack()

            >>> with sv.VideoSink(target_path='...', video_info=video_info) as sink:
            ...    for frame in frames_generator:
            ...        result = model(frame)[0]
            ...        detections = sv.Detections.from_ultralytics(result)
            ...        detections = tracker.update_with_detections(detections)
            ...        annotated_frame = trace_annotator.annotate(
            ...            scene=frame.copy(),
            ...            detections=detections)
            ...        sink.write_frame(frame=annotated_frame)
            ```

        ![trace-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/trace-annotator-example-purple.png)
        """
        self.trace.put(detections)

        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            xy = self.trace.get(tracker_id=tracker_id)
            if len(xy) > 1:
                scene = cv2.polylines(
                    scene,
                    [xy.astype(np.int32)],
                    False,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return scene


class HeatMapAnnotator:
    """
    A class for drawing heatmaps on an image based on provided detections.
    Heat accumulates over time and is drawn as a semi-transparent overlay
    of blurred circles.
    """

    def __init__(
        self,
        position: Position = Position.BOTTOM_CENTER,
        opacity: float = 0.2,
        radius: int = 40,
        kernel_size: int = 25,
        top_hue: int = 0,
        low_hue: int = 125,
    ):
        """
        Args:
            position (Position): The position of the heatmap. Defaults to
                `BOTTOM_CENTER`.
            opacity (float): Opacity of the overlay mask, between 0 and 1.
            radius (int): Radius of the heat circle.
            kernel_size (int): Kernel size for blurring the heatmap.
            top_hue (int): Hue at the top of the heatmap. Defaults to 0 (red).
            low_hue (int): Hue at the bottom of the heatmap. Defaults to 125 (blue).
        """
        self.position = position
        self.opacity = opacity
        self.radius = radius
        self.kernel_size = kernel_size
        self.heat_mask = None
        self.top_hue = top_hue
        self.low_hue = low_hue

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the scene with a heatmap based on the provided detections.

        Args:
            scene (np.ndarray): The image where the heatmap will be drawn.
            detections (Detections): Object detections to annotate.

        Returns:
            np.ndarray: Annotated image.

        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> model = YOLO('yolov8x.pt')

            >>> heat_map_annotator = sv.HeatMapAnnotator()

            >>> video_info = sv.VideoInfo.from_video_path(video_path='...')
            >>> frames_generator = get_video_frames_generator(source_path='...')

            >>> with sv.VideoSink(target_path='...', video_info=video_info) as sink:
            ...    for frame in frames_generator:
            ...        result = model(frame)[0]
            ...        detections = sv.Detections.from_ultralytics(result)
            ...        annotated_frame = heat_map_annotator.annotate(
            ...            scene=frame.copy(),
            ...            detections=detections)
            ...        sink.write_frame(frame=annotated_frame)
            ```

        ![heatmap-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/heat-map-annotator-example-purple.png)
        """

        if self.heat_mask is None:
            self.heat_mask = np.zeros(scene.shape[:2])
        mask = np.zeros(scene.shape[:2])
        for xy in detections.get_anchor_coordinates(self.position):
            cv2.circle(mask, (int(xy[0]), int(xy[1])), self.radius, 1, -1)
        self.heat_mask = mask + self.heat_mask
        temp = self.heat_mask.copy()
        temp = self.low_hue - temp / temp.max() * (self.low_hue - self.top_hue)
        temp = temp.astype(np.uint8)
        if self.kernel_size is not None:
            temp = cv2.blur(temp, (self.kernel_size, self.kernel_size))
        hsv = np.zeros(scene.shape)
        hsv[..., 0] = temp
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(self.heat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) > 0
        scene[mask] = cv2.addWeighted(temp, self.opacity, scene, 1 - self.opacity, 0)[
            mask
        ]
        return scene
