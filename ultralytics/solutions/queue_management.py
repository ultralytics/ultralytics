# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
from . import (tf, cls_names, bg_color_rgb, display_tracks, counting_region,
               track_history, txt_color_rgb, draw_region, extract_tracks,
               env_check, Annotator, colors, Point, rg_pts, display_frames)


class QueueManager:
    """A class to manage the queue management in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the queue manager with default values for various tracking and counting parameters."""
        self.im0 = None
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Queue Manager"
        print("Queue management app initialized...")

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for queue management in a video stream."""

        # Annotator Init and queue region drawing
        global counts
        counts = 0
        self.annotator = Annotator(self.im0, line_width=tf)
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

                if len(rg_pts) >= 3:
                    is_inside = counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        counts += 1

        label = "Queue Counts: " + str(counts)
        self.annotator.queue_counts_display(label,
                                            points=rg_pts,
                                            region_color=bg_color_rgb,
                                            txt_color=txt_color_rgb)

        counts = 0
        display_frames(self.im0, self.window_name)


    def process_queue(self, im0, tracks):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        return self.im0


if __name__ == "__main__":
    QueueManager()
