# Ultralytics YOLO 🚀, AGPL-3.0 license

from time import time
import cv2
from . import (tf, cls_names, bg_color_rgb, display_tracks, rg_pts, track_history,
               txt_color_rgb, extract_tracks, env_check, Annotator, colors,
               d_thresh, np, display_frames)


class SpeedEstimator:
    """A class to estimation speed of objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the speed-estimator class with default values for Visual, Image, track and speed parameters."""
        self.im0 = None
        self.annotator = None
        self.trk_previous_times = {}
        self.trk_previous_points = {}
        self.trk_idslist = []
        self.dist_data = {}
        self.window_name = "Ultralytics YOLOv8 Speed Estimation"
        print("Speed estimation app initialized...")


    def calculate_speed(self, trk_id, track):
        """
        Calculation of object speed.

        Args:
            trk_id (int): object track id.
            track (list): tracking history for tracks path drawing
        """

        if not rg_pts[0][0] < track[-1][0] < rg_pts[1][0]:
            return
        if rg_pts[1][1] - d_thresh < track[-1][1] < rg_pts[1][1] + d_thresh:
            direction = "known"

        elif rg_pts[0][1] - d_thresh < track[-1][1] < rg_pts[0][1] + d_thresh:
            direction = "known"

        else:
            direction = "unknown"

        if self.trk_previous_times[trk_id] != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            self.trk_idslist.append(trk_id)

            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                speed = dist_difference / time_difference
                self.dist_data[trk_id] = speed

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]

    def estimate_speed(self, im0, tracks):
        """
        Calculate object based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0

        boxes, clss, track_ids = extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=tf)
        self.annotator.draw_region(reg_pts=rg_pts, color=bg_color_rgb, thickness=tf)

        if track_ids is not None:
            for box, trk_id, cls in zip(boxes, track_ids, clss):
                color = colors(int(trk_id), True)
                track_line = track_history[trk_id]

                x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                track_line.append((x_center, y_center))
                if len(track_line) > 30:
                    track_line.pop(0)

                if display_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=color,
                        track_thickness=tf,
                    )

                if trk_id not in self.trk_previous_times:
                    self.trk_previous_times[trk_id] = 0

                speed_label = f"{int(self.dist_data[trk_id])}km/ph" if trk_id in self.dist_data else cls_names[int(cls)]
                self.annotator.draw_label_in_center(speed_label, txt_color_rgb, color, x_center, y_center, 10)
                self.calculate_speed(trk_id, track_line)

        display_frames(self.im0, self.window_name)
        return self.im0
