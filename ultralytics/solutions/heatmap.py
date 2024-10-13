# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np

from ultralytics.solutions.object_counter import ObjectCounter  # Import object counter class
from ultralytics.utils.plotting import Annotator


class Heatmap(ObjectCounter):
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initializes function for heatmap class with default values."""
        super().__init__(**kwargs)

        self.initialized = False  # bool variable for heatmap initialization
        if self.region is not None:  # check if user provided the region coordinates
            self.initialize_region()

        # store colormap
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]

    def heatmap_effect(self, box):
        """
        Efficient calculation of heatmap area and effect location for applying colormap.

        Args:
            box (list): Bounding Box coordinates data [x0, y0, x1, y1]
        """
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # Calculate squared distances from the center
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # Create a mask of points within the radius
        within_radius = dist_squared <= radius_squared

        # Update only the values within the bounding box in a single vectorized operation
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def generate_heatmap(self, im0):
        """
        Generate heatmap for each frame using Ultralytics.

        Args:
            im0 (ndarray): Input image array for processing
        Returns:
            im0 (ndarray): Processed image for further usage
        """
        self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99 if not self.initialized else self.heatmap
        self.initialized = True  # Initialize heatmap only once

        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)  # Store track history
                self.store_classwise_counts(cls)  # store classwise counts in dict

                # Store tracking previous position and perform object counting
                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                self.count_objects(self.track_line, box, track_id, prev_position, cls)  # Perform object counting

        self.display_counts(im0) if self.region is not None else None  # Display the counts on the frame

        # Normalize, apply colormap to heatmap and combine with original image
        im0 = (
            im0
            if self.track_data.id is None
            else cv2.addWeighted(
                im0,
                0.5,
                cv2.applyColorMap(
                    cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.colormap
                ),
                0.5,
                0,
            )
        )

        self.display_output(im0)  # display output with base class function
        return im0  # return output image for more usage
