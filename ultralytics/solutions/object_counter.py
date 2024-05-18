# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics.solutions import Solutions
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(Solutions):
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def extract_and_process_tracks(self, tracks):
        """
        Extracts and processes tracks for object counting in a video stream.

        Args:
            tracks (list): List of tracks obtained from the object tracking process
        """

        self.annotator = Annotator(self.im0, line_width=self.tf)
        self.annotator.draw_region(
            self.reg_pts, self.count_bg_color, self.rt
        ) if self.count_type == "classwise" else None

        # Extract tracks
        self.extract_tracks(tracks)

        if self.track_ids is not None:
            for box, trk_id, cls in zip(self.boxes, self.track_ids, self.clss):
                color = colors(int(trk_id), True)

                track_line = self.track_history[trk_id]
                x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

                track_line.append((x_center, y_center))
                if len(track_line) > 30:
                    track_line.pop(0)

                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(track_line, color=self.track_color, track_thickness=self.tt)

                self.annotator.draw_label_in_center(
                    (str(self.classes_names[int(cls)]) + ":" + str(trk_id)),
                    self.count_txt_color,
                    color,
                    x_center,
                    y_center,
                    5,
                )

                prev_position = self.track_history[trk_id][-2] if len(self.track_history[trk_id]) > 1 else None
                self.object_counts(prev_position, box, cls, trk_id, track_line)

            if self.count_type == "classwise":
                labels_dict = {}

                for key, value in self.clswise_dict.items():
                    if value["IN"] != 0 or value["OUT"] != 0:
                        if not self.view_in_counts and not self.view_out_counts:
                            continue
                        elif not self.view_in_counts:
                            labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                        elif not self.view_out_counts:
                            labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                        else:
                            labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

                if labels_dict is not None:
                    self.annotator.display_analytics(
                        self.im0, labels_dict, txt_color=self.count_txt_color, bg_color=self.count_bg_color, margin=5
                    )
            else:
                if not self.view_in_counts and not self.view_out_counts:
                    label = 0
                elif not self.view_in_counts:
                    label = f"Out : {self.out_count}"
                elif not self.view_out_counts:
                    label = f"In : {self.in_count}"
                else:
                    label = f"In : {self.in_count}, Out : {self.out_count}"

                self.annotator.line_counter(
                    points=self.reg_pts,
                    bg_color=self.count_bg_color,
                    txt_color=self.count_txt_color,
                    text=label,
                    margin=5,
                    gap=10,
                )

        self.display_frames(self.im0, self.window_name)

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """

        self.im0 = im0
        self.extract_and_process_tracks(tracks)

        return self.im0
