# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the heatmap class with default values for Visual, Image, track and heatmap parameters."""

        # Visual Information
        self.annotator = None
        self.view_img = False

        # Image Information
        self.imw = None
        self.imh = None
        self.im0 = None

        # Heatmap Colormap and heatmap np array
        self.colormap = None
        self.heatmap = None
        self.heatmap_alpha = 0.5

        # Predict/track information
        self.boxes = None
        self.track_ids = None
        self.clss = None

    def set_args(self, imw, imh, colormap=cv2.COLORMAP_JET, heatmap_alpha=0.5, view_img=False):
        """
        Configures the heatmap colormap, width, height and display parameters.

        Args:
            colormap (cv2.COLORMAP): The colormap to be set.
            imw (int): The width of the frame.
            imh (int): The height of the frame.
            heatmap_alpha (float): alpha value for heatmap display
            view_img (bool): Flag indicating frame display
        """
        self.imw = imw
        self.imh = imh
        self.colormap = colormap
        self.heatmap_alpha = heatmap_alpha
        self.view_img = view_img

        # Heatmap new frame
        self.heatmap = np.zeros((int(self.imw), int(self.imh)), dtype=np.float32)

    def extract_results(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        if tracks[0].boxes.id is None:
            return
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.track_ids = tracks[0].boxes.id.int().cpu().tolist()

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.extract_results(tracks)
        self.im0 = im0

        for box, cls in zip(self.boxes, self.clss):
            self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 1

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)
        im0_with_heatmap = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha, heatmap_colored, self.heatmap_alpha, 0)

        if self.view_img:
            self.display_frames(im0_with_heatmap)

        return im0_with_heatmap

    def display_frames(self, im0_with_heatmap):
        """
        Display heatmap.

        Args:
            im0_with_heatmap (nd array): Original Image with heatmap
        """
        cv2.imshow('Ultralytics Heatmap', im0_with_heatmap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


if __name__ == '__main__':
    Heatmap()
