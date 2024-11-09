# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution, cv2
from ultralytics.utils.plotting import Annotator, colors


class RegionCounter(BaseSolution):
    def __init__(self, **kwargs):
        """Initializes the RegionCounter class for real-time counting in different regions of the video streams."""
        super().__init__(**kwargs)
        # Template for a region with default values
        self.region_template = {
            "name": "Default Region",
            "polygon": None,  # Will be updated with specific coordinates
            "counts": 0,
            "dragging": False,
            "region_color": (255, 255, 255),  # Default color (can be customized per region)
            "text_color": (0, 0, 0),  # Default text color
        }
        self.counting_regions = []

    def add_region(self, name, polygon_points, region_color, text_color):
        """
        Adds a new region based on the template with specific attributes.

        Args:
           name (str): Name of the region.
           polygon_points (list of tuples): List of (x, y) coordinates defining the polygon.
           region_color (tuple): BGR color tuple for the region.
           text_color (tuple): BGR color tuple for the text.
       """
        region = self.region_template.copy()     # Create a deep copy of the template and update it
        region.update({
            "name": name,
            "polygon": self.Polygon(polygon_points),
            "region_color": region_color,
            "text_color": text_color
        })
        self.counting_regions.append(region)    # Append new region

    def count(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Convert single region list to dictionary format if needed
        regions = self.region if isinstance(self.region, dict) else {"Region#01": self.region}

        # Iterate through all regions (single or multiple)
        for idx, (region_name, reg_pts) in enumerate(regions.items(), start=1):
            color = colors(idx, True)
            self.annotator.draw_region(reg_pts=reg_pts, color=color, thickness=self.line_width * 2)
            self.add_region(region_name, reg_pts, color, self.annotator.get_txt_color())

        # Preprocess regions for faster containment checks (if many regions)
        for region in self.counting_regions:
            region["prepared_polygon"] = self.prep(region["polygon"])

        # Process bounding boxes and track information
        for box, cls in zip(self.boxes, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))   # Draw bounding box
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)    # Calculate the bounding box center once

            for region in self.counting_regions:    # Check if the bbox center is inside any region
                if region["prepared_polygon"].contains(self.Point(bbox_center)):
                    region["counts"] += 1  # Increment count if inside region

        # Batch draw region counts and reset
        for region in self.counting_regions:
            self.annotator.text_label(region["polygon"].bounds, label=str(region["counts"]),
                                      color=region["region_color"], txt_color=region["text_color"])
            region["counts"] = 0  # Reset count for next frame

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
