# Ultralytics YOLO 🚀, AGPL-3.0 license



from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class InstanceSegmentation(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def segment(self, im0):
        annotator = SolutionAnnotator(im0, self.line_width)

        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes and classes
        for cls, t_id, mask in zip(self.clss, self.track_ids, self.masks):
            annotator.seg_bbox(mask=mask, mask_color=colors(t_id, True), label=self.names[cls])  # Draw segmentation box

        self.display_output(im0)  # Display the output using the base class function

        # Return a summary dictionary for usage
        return SolutionResults(im0=im0, total_tracks=len(self.track_ids)).summary()
