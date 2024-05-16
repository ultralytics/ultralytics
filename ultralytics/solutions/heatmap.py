# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
from ..utils.plotting import Annotator, colors
from . import (tf, cls_names, bg_color_rgb, display_tracks, rg_pts, count_type,
               track_history, display_img, display_in, display_out, clswise_dict,
               txt_color_rgb, draw_region, extract_tracks, object_counts, env_check,
               h_alpha, h_decay, heatshape, display_frames, np, colormap, enable_count)


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the heatmap_im0 class with default values for Visual, Image, track, count and heatmap_im0 parameters."""
        self.im0 = None
        self.annotator = None
        self.heatmap_im0 = None
        self.window_name = "Ultralytics Heatmaps"

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap_im0 based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        global in_count, out_count

        self.heatmap_im0 = np.zeros((int(im0.shape[0]), int(im0.shape[1])), dtype=np.float32) if self.heatmap_im0 is None else self.heatmap_im0

        self.im0 = im0
        self.annotator = Annotator(self.im0, line_width=tf)
        self.heatmap_im0 *= h_decay

        boxes, clss, track_ids = extract_tracks(tracks)

        if enable_count:
            if rg_pts is not None and count_type == "classwise":
                draw_region(self.annotator)

            if track_ids is not None:
                for box, cls, track_id in zip(boxes, clss, track_ids):
                    if heatshape == "circle":
                        center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                        radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                        y, x = np.ogrid[0: self.heatmap_im0.shape[0], 0: self.heatmap_im0.shape[1]]
                        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

                        self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += (
                                2 * mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                        )
                    else:
                        self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += 2

                    track_line = track_history[track_id]
                    track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                    if len(track_line) > 30:
                        track_line.pop(0)

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

        else:
            for box in boxes:
                if heatshape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0: self.heatmap_im0.shape[0], 0: self.heatmap_im0.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

                    self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += (
                            2 * mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                    )
                else:
                    self.heatmap_im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += 2

        # Normalize, apply colormap to heatmap_im0 and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap_im0, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), colormap)

        self.im0 = cv2.addWeighted(self.im0, 1 - h_alpha, heatmap_colored, h_alpha, 0)
        display_frames(self.im0, self.window_name)

        return self.im0


if __name__ == "__main__":
    Heatmap()
