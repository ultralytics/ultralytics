# Ultralytics YOLO 🚀, AGPL-3.0 license

import os

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectBlurrer(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.blur_ratio = int(self.CFG["blur_ratio"] * 100)
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"] if self.CFG["conf"] is not None else 0.25

    def blur(self, im0):
        annotator = SolutionAnnotator(im0, self.line_width)
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        for box, cls in zip(self.boxes, self.clss):
            # Crop the detected object
            blur_obj = cv2.blur(im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])],
                                (self.blur_ratio, self.blur_ratio))
            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))  # Bounding box plot

        self.display_output(im0)  # display output with base class function

        # Return output dictionary with summary for usage
        return SolutionResults(im0=im0, total_tracks=len(self.track_ids)).summary()
