# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    """A class for real-time counting of objects within user-defined regions in a video stream.

    This class inherits from `BaseSolution` and provides functionality to define polygonal regions in a video frame,
    track objects, and count those objects that pass through each defined region. Useful for applications requiring
    counting in specified areas, such as monitoring zones or segmented sections.

    Attributes:
        region_template (dict): Template for creating new counting regions with default attributes including name,
            polygon coordinates, and display colors.
        counting_regions (list): List storing all defined regions, where each entry is based on `region_template` and
            includes specific region settings like name, coordinates, and color.
        region_counts (dict): Dictionary storing the count of objects for each named region.

    Methods:
        add_region: Add a new counting region with specified attributes.
        process: Process video frames to count objects in each region.
        initialize_regions: Initialize zones to count the objects in each one. Zones could be multiple as well.

    Examples:
        Initialize a RegionCounter and add a counting region
        >>> counter = RegionCounter()
        >>> counter.add_region("Zone1", [(100, 100), (200, 100), (200, 200), (100, 200)], (255, 0, 0), (255, 255, 255))
        >>> results = counter.process(frame)
        >>> print(f"Total tracks: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the RegionCounter for real-time object counting in user-defined regions."""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()

    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        """Add a new region to the counting list based on the provided template with specific attributes.

        Args:
            name (str): Name assigned to the new region.
            polygon_points (list[tuple]): List of (x, y) coordinates defining the region's polygon.
            region_color (tuple[int, int, int]): BGR color for region visualization.
            text_color (tuple[int, int, int]): BGR color for the text within the region.

        Returns:
            (dict[str, Any]): Region information including name, polygon, and display colors.
        """
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region

    def initialize_regions(self):
        """Initialize regions from `self.region` only once."""
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):  # Ensure self.region is initialized and structured as a dictionary
            self.region = {"Region#01": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            region = self.add_region(name, pts, colors(i, True), (255, 255, 255))
            region["prepared_polygon"] = self.prep(region["polygon"])

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Process the input frame to detect and count objects within each defined region.

        Args:
            im0 (np.ndarray): Input image frame where objects and regions are annotated.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, 'total_tracks' (int, total number of tracked
                objects), and 'region_counts' (dict, counts of objects per region).
        """
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            center = self.Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        # Display region counts
        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            annotator.adaptive_label(
                [x1, y1, x2, y2],
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
                shape="rect",
            )
            region["counts"] = 0  # Reset for next frame
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)
