# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np
from ultralytics.solutions import Solutions
from ultralytics.utils.plotting import Annotator, colors


class Heatmap(Solutions):
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap_im0 based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """

        self.im0 = im0
        self.heatmap_im0 = (
            np.zeros((int(self.im0.shape[0]), int(self.im0.shape[1])), dtype=np.float32)
            if self.heatmap_im0 is None
            else self.heatmap_im0
        )

        self.annotator = Annotator(self.im0, line_width=self.tf)
        self.heatmap_im0 *= self.decay_factor

        self.extract_tracks(tracks)

        self.annotator.draw_region(
            reg_pts=self.reg_pts, color=self.count_bg_color, thickness=self.tt
        ) if self.count_type == "classwise" else None

        heatmap_normalized = cv2.normalize(self.heatmap_im0, None, 0, 255, cv2.NORM_MINMAX)

        if self.track_ids is not None:
            for box, cls, trk_id in zip(self.boxes, self.clss, self.track_ids):

                color = colors(int(trk_id), True)

                if self.shape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0: self.heatmap_im0.shape[0], 0: self.heatmap_im0.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

                    self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += (
                        2 * mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                    )
                else:
                    self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += 2

                track_line = self.track_history[trk_id]
                x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                track_line.append((x_center, y_center))
                if len(track_line) > 30:
                    track_line.pop(0)

                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(track_line, color=self.track_color, track_thickness=tf)
                    self.annotator.draw_label_in_center(
                        (str(cls_names[int(cls)]) + ":" + str(trk_id)), self.count_txt_color, color,
                        x_center, y_center, 5
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
                        points=self.reg_pts, bg_color=self.count_bg_color, txt_color=self.count_txt_color,
                        text=label, margin=5, gap=10
                    )

        else:
            for box in boxes:
                if self.shape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0: self.heatmap_im0.shape[0], 0: self.heatmap_im0.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

                    self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += (
                        2 * mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                    )
                else:
                    self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += 2

        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)

        self.im0 = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha, heatmap_colored,
                                   self.heatmap_alpha, 0)
        self.display_frames(self.im0, self.window_name)

        return self.im0
