# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    """
    A class designed for real-time counting of objects within user-defined regions in a video stream.

    This class inherits from `BaseSolution` and offers functionalities to define polygonal regions in a video
    frame, track objects, and count those objects that pass through each defined region. This makes it useful
    for applications that require counting in specified areas, such as monitoring zones or segmented sections.

    Attributes:
        region_template (dict): A template for creating new counting regions with default attributes including
                                the name, polygon coordinates, and display colors.
        counting_regions (list): A list storing all defined regions, where each entry is based on `region_template`
                                 and includes specific region settings like name, coordinates, and color.

    Methods:
        add_region: Adds a new counting region with specified attributes, such as the region's name, polygon points,
                    region color, and text color.
        count: Processes video frames to count objects in each region, drawing regions and displaying counts
               on the frame. Handles object detection, region definition, and containment checks.
    """

    def __init__(self, **kwargs):
        """Initializes the RegionCounter class for real-time counting in different regions of the video streams."""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "dragging": False,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []

    def add_region(self, name, polygon_points, region_color, text_color):
        """
        Adds a new region to the counting list based on the provided template with specific attributes.

        Args:
            name (str): Name assigned to the new region.
            polygon_points (list[tuple]): List of (x, y) coordinates defining the region's polygon.
            region_color (tuple): BGR color for region visualization.
            text_color (tuple): BGR color for the text within the region.
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

    def process(self, im0):
        """
        Processes the input frame to detect and count objects within each defined region.

        Args:
            im0 (numpy.ndarray): Input image frame where objects and regions are annotated.

        Returns:
            results (SolutionResults): Contains processed image `im0`, 'total_tracks' (int, total number of tracked objects).
        """
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        # Ensure self.region is initialized and structured as a dictionary
        if not isinstance(self.region, dict):
            self.region = {"Region#01": self.region or self.initialize_region()}

        # Draw only valid regions
        for idx, (region_name, reg_pts) in enumerate(self.region.items(), start=1):
            color = colors(idx, True)
            annotator.draw_region(reg_pts, color, self.line_width * 2)
            self.add_region(region_name, reg_pts, color, annotator.get_txt_color())

        # Prepare regions for containment check (only process valid ones)
        for region in self.counting_regions:
            if "prepared_polygon" not in region:
                region["prepared_polygon"] = self.prep(region["polygon"])

        # Convert bounding boxes to NumPy array
        boxes_np = np.array([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in self.boxes], dtype=np.float32)
        points = [self.Point(pt) for pt in boxes_np]  # Convert centers to Point objects

        # Vectorized processing for bounding boxes & containment checks
        if points:
            for (point, cls), box in zip(zip(points, self.clss), self.boxes):
                annotator.box_label(box, label=self.names[cls], color=colors(cls))

                for region in self.counting_regions:  # Efficient containment check using precomputed polygons
                    if region["prepared_polygon"].contains(point):
                        region["counts"] += 1
                        self.region_counts[region["name"]] = region["counts"]

        # Display region counts efficiently
        for region in self.counting_regions:
            annotator.text_label(
                region["polygon"].bounds,
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
            )
            region["counts"] = 0  # Reset for next frame
        plot_im = annotator.result()
        self.display_output(plot_im)

        # Return a SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)
