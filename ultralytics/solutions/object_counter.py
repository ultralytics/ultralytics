# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
from . import (tf, cls_names, bg_color_rgb, display_tracks, rg_pts, count_type,
               track_history, display_img, display_in, display_out, clswise_dict,
               txt_color_rgb, draw_region, extract_tracks, object_counts, env_check,
               Annotator, colors)


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self, ):
        """Initializes object counting module."""
        self.im0 = None
        self.annotator = None
        self.window_name = "Ultralytics YOLOv8 Object Counter"

    def extract_and_process_tracks(self, tracks):
        """
        Extracts and processes tracks for object counting in a video stream.

        Args:
            tracks: object tracking data for complete frame
        """

        global in_count, out_count

        self.annotator = Annotator(self.im0, line_width=tf)
        if count_type == "classwise":
            draw_region(self.annotator)

        # Extract tracks
        boxes, clss, track_ids = extract_tracks(tracks)

        if track_ids is not None:
            for box, track_id, cls in zip(boxes, track_ids, clss):
                self.annotator.box_label(box, label=f"{cls_names[cls]}#{track_id}",
                                         color=colors(int(track_id), True))

                track_line = track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if display_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=colors(int(track_id), True),
                        track_thickness=tf,
                    )

                prev_position = track_history[track_id][-2] if len(track_history[track_id]) > 1 else None
                in_count, out_count, clswise_dict = object_counts(prev_position, box, cls, track_id, track_line)

        if count_type == "classwise":
            labels_dict = {}

            for key, value in clswise_dict.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    if not display_in and not display_out:
                        continue
                    elif not display_in:
                        labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                    elif not display_out:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                    else:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

            if labels_dict is not None:
                self.annotator.display_analytics(self.im0, labels_dict, txt_color=txt_color_rgb, bg_color=bg_color_rgb, margin=10)
        else:
            if not display_in and not display_out:
                label = 0
            elif not display_in:
                label = f"Out : {out_count}"
            elif not display_out:
                label = f"In : {in_count}"
            else:
                label = f"In : {in_count}, Out : {out_count}"

            self.annotator.line_counter(points=rg_pts, bg_color=bg_color_rgb, txt_color=txt_color_rgb, text=label, margin=10, gap=10)

        display_frames(self.im0, self.window_name)

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
