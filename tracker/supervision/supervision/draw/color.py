from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_COLOR_PALETTE = [
    "#a351fb",
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")


@dataclass
class Color:
    """
    Represents a color in RGB format.

    Attributes:
        r (int): Red channel.
        g (int): Green channel.
        b (int): Blue channel.
    """

    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str) -> Color:
        """
        Create a Color instance from a hex string.

        Args:
            color_hex (str): Hex string of the color.

        Returns:
            Color: Instance representing the color.

        Example:
            ```
            >>> Color.from_hex('#ff00ff')
            Color(r=255, g=0, b=255)
            ```
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    def as_rgb(self) -> Tuple[int, int, int]:
        """
        Returns the color as an RGB tuple.

        Returns:
            Tuple[int, int, int]: RGB tuple.

        Example:
            ```
            >>> color.as_rgb()
            (255, 0, 255)
            ```
        """
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """
        Returns the color as a BGR tuple.

        Returns:
            Tuple[int, int, int]: BGR tuple.

        Example:
            ```
            >>> color.as_bgr()
            (255, 0, 255)
            ```
        """
        return self.b, self.g, self.r

    @classmethod
    def white(cls) -> Color:
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    def black(cls) -> Color:
        return Color.from_hex(color_hex="#000000")

    @classmethod
    def red(cls) -> Color:
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    def green(cls) -> Color:
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    def blue(cls) -> Color:
        return Color.from_hex(color_hex="#0000ff")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classmethod
    def default(cls) -> ColorPalette:
        """
        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```
            >>> ColorPalette.default()
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]) -> ColorPalette:
        """
        Create a ColorPalette instance from a list of hex strings.

        Args:
            color_hex_list (List[str]): List of color hex strings.

        Returns:
            ColorPalette: A ColorPalette instance.

        Example:
            ```
            >>> ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        """
        Return the color at a given index in the palette.

        Args:
            idx (int): Index of the color in the palette.

        Returns:
            Color: Color at the given index.

        Example:
            ```
            >>> color_palette.by_idx(1)
            Color(r=0, g=255, b=0)
            ```
        """
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]
