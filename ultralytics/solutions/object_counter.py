# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from . import (
    Annotator,
    bg_color_rgb,
    cls_names,
    clswise_dict,
    colors,
    count_type,
    display_frames,
    display_in,
    display_out,
    display_tracks,
    enable_count,
    env_check,
    extract_tracks,
    object_counts,
    rg_pts,
    tf,
    track_history,
    txt_color_rgb,
)


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
    ):
        """Initializes object counting module."""
        self.im0 = None
        self.annotator = None
        self.window_name = "Ultralytics YOLOv8 Object Counter"
        print("Object counting app initialized...")

    def extract_and_process_tracks(self, tracks):
        """
        Extracts and processes tracks for object counting in a video stream.

        Args:
            tracks (list): List of tracks obtained from the object tracking process
        """

        global in_count, out_count, clswise_dict

        self.annotator = Annotator(self.im0, line_width=tf)
        self.annotator.draw_region(
            reg_pts=rg_pts, color=bg_color_rgb, thickness=tf
        ) if count_type == "classwise" and enable_count else None

        # Extract tracks
        boxes, clss, track_ids = extract_tracks(tracks)

        if track_ids is not None:
            for box, trk_id, cls in zip(boxes, track_ids, clss):
                color = colors(int(trk_id), True)
                track_line = track_history[trk_id]
                x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                track_line.append((x_center, y_center))
                if len(track_line) > 30:
                    track_line.pop(0)

                self.annotator.draw_label_in_center(
                    (str(cls_names[int(cls)]) + ":" + str(trk_id)), txt_color_rgb, color, x_center, y_center, 5
                )

                if display_tracks:
                    self.annotator.draw_centroid_and_tracks(track_line, color=color, track_thickness=tf)

                if enable_count:
                    prev_position = track_history[trk_id][-2] if len(track_history[trk_id]) > 1 else None
                    in_count, out_count, clswise_dict = object_counts(prev_position, box, cls, trk_id, track_line)

            if enable_count:
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
                        self.annotator.display_analytics(
                            self.im0, labels_dict, txt_color=txt_color_rgb, bg_color=bg_color_rgb, margin=5
                        )
                else:
                    if not display_in and not display_out:
                        label = 0
                    elif not display_in:
                        label = f"Out : {out_count}"
                    elif not display_out:
                        label = f"In : {in_count}"
                    else:
                        label = f"In : {in_count}, Out : {out_count}"

                    self.annotator.line_counter(
                        points=rg_pts, bg_color=bg_color_rgb, txt_color=txt_color_rgb, text=label, margin=5, gap=10
                    )

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
