# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectBlur(BaseSolution):
    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

    def blur(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes and classes index
        for box, cls in zip(self.boxes, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        self.display_counts(im0)  # Display the counts on the frame
        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
