# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors


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

    def count(self, im0):
        """
        Processes the input frame to detect and count objects within each defined region.

        Args:
            im0 (numpy.ndarray): Input image frame where objects and regions are annotated.

        Returns:
           im0 (numpy.ndarray): Processed image frame with annotated counting information.
        """
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        # Region initialization and conversion
        if self.region is None:
            self.initialize_region()
            regions = {"Region#01": self.region}
        else:
            regions = self.region if isinstance(self.region, dict) else {"Region#01": self.region}

        # Draw regions and process counts for each defined area
        for idx, (region_name, reg_pts) in enumerate(regions.items(), start=1):
            if not isinstance(reg_pts, list) or not all(isinstance(pt, tuple) for pt in reg_pts):
                LOGGER.warning(f"Invalid region points for {region_name}: {reg_pts}")
                continue  # Skip invalid entries
            color = colors(idx, True)
            self.annotator.draw_region(reg_pts=reg_pts, color=color, thickness=self.line_width * 2)
            self.add_region(region_name, reg_pts, color, self.annotator.get_txt_color())

        # Prepare regions for containment check
        for region in self.counting_regions:
            region["prepared_polygon"] = self.prep(region["polygon"])

        # Process bounding boxes and count objects within each region
        for box, cls in zip(self.boxes, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            for region in self.counting_regions:
                if region["prepared_polygon"].contains(self.Point(bbox_center)):
                    region["counts"] += 1

        # Display counts in each region
        for region in self.counting_regions:
            self.annotator.text_label(
                region["polygon"].bounds,
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
            )
            region["counts"] = 0  # Reset count for next frame

        self.display_output(im0)
        return im0
