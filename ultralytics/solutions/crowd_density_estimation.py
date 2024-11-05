# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics.solutions.solutions import BaseSolution


class CrowdDensityEstimation(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.density_threshold = kwargs.get("density_threshold", 10)  # Density threshold for crowd detection

    def process_data(self, im0):
        """Process the input frame and return the annotated frame."""
        self.extract_tracks(im0)
        density = self.calculate_density()
        im0 = self.display_output(im0, density)
        return im0

    def calculate_density(self):
        """Calculate the crowd density based on the number of detected objects."""
        return len(self.boxes) / (self.region_area if self.region_area else 1)

    def display_output(self, im0, density):
        """Annotate the frame with crowd density information."""
        cv2.putText(im0, f"Density: {density:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return im0
