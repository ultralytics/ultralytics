# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import Point, Polygon


class QueueManager:
    """A class to manage the queue management in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the queue manager with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 60), (20, 680), (1120, 680), (1120, 60)]
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_queue_counts = True
        self.fontsize = 0.6

        self.names = None  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Queue Manager"

        # Object counting Information
        self.counts = 0
        self.count_txt_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = None

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        region_color=(255, 0, 255),
        view_queue_counts=True,
        draw_tracks=False,
        count_txt_color=(255, 255, 255),
        track_color=None,
        region_thickness=5,
        fontsize=0.7,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_queue_counts (bool): Flag to control whether to display the counts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            region_color (RGB color): Color of queue region
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_color (RGB color): count text color value
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            fontsize (float): Text display font size
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_queue_counts = view_queue_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.region_color = region_color

        if len(reg_pts) >= 3:
            print("Queue region initiated...")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid region points provided...")
            print("Using default region now....")
            self.counting_region = Polygon(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_color = count_txt_color
        self.region_thickness = region_thickness
        self.fontsize = fontsize

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for queue management in a video stream."""

        # Annotator Init and queue region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        label = "Queue Counts : " + str(self.counts)

        if label is not None:
            self.annotator.queue_counts_display(
                label,
                points=self.reg_pts,
                region_color=self.region_color,
                txt_color=self.count_txt_color,
                fontsize=self.fontsize,
            )

        self.counts = 0
        self.display_frames()

    def display_frames(self):
        """Display frame."""
        if self.env_check:
            self.annotator.draw_region(reg_pts=self.reg_pts, thickness=self.region_thickness, color=self.region_color)
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def process_queue(self, im0, tracks):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        if self.view_img:
            self.display_frames()
        return self.im0


if __name__ == "__main__":
    QueueManager()
