# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
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
        self.current_region = None

    def add_region(self, name, polygon_points, region_color, text_color):
        """
               Adds a new region based on the template with specific attributes.

               Args:
                   name (str): Name of the region.
                   polygon_points (list of tuples): List of (x, y) coordinates defining the polygon.
                   region_color (tuple): BGR color tuple for the region.
                   text_color (tuple): BGR color tuple for the text.
               """
        # Create a deep copy of the template and update it
        region = self.region_template.copy()
        region.update({
            "name": name,
            "polygon": self.Polygon(polygon_points),
            "region_color": region_color,
            "text_color": text_color
        })
        self.counting_regions.append(region)

    def count(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Check if `self.region` is a list (single region) or dictionary (multiple regions)
        if isinstance(self.region, list):
            # Single region passed as a list
            self.annotator.draw_region(
                reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
            )
        elif isinstance(self.region, dict):
            # Multiple regions passed as a dictionary
            idx = 0
            for region_name, reg_pts in self.region.items():
                idx+=1
                color=colors(int(idx), True)
                txt_color=self.annotator.get_txt_color()
                self.annotator.draw_region(
                    reg_pts=reg_pts, color=color, thickness=self.line_width * 2
                )
                self.add_region(region_name, reg_pts, color, txt_color)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # Store track history
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

            # Check if the bbox center is inside the polygon region
            for region in self.counting_regions:
                if region["polygon"].contains(self.Point(bbox_center)):
                    region["counts"] += 1  # Update the count for the region if an object is inside

        # Optionally, reset counts for each frame if needed
        for region in self.counting_regions:
            box = list(region["polygon"].exterior.coords)
            # Get bounding box as (min_x, min_y, max_x, max_y)
            bbox = (min(x for x, y in box), min(y for x, y in box),
                    max(x for x, y in box), max(y for x, y in box))

            print(region["polygon"].centroid)
            self.annotator.box_label(bbox)
            self.annotator.text_label(bbox, label=str(region["counts"]))

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
