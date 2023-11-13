import argparse
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from tracker.utils.parse_config import ConfigParser
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.default()


class VideoProcessor:
    def __init__(self, config) -> None:
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]

        self.source_video_path = config["source_video_path"]

        self.output_dir = config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")

        self.device = config["device"]
        self.video_stride = config["video_stride"]

        self.model = YOLO(config["source_weights_path"])
        self.tracker = sv.ByteTrack(
            track_thresh=config["track_threshold"],
            track_buffer=config["track_buffer"],
            match_thresh=config["match_threshold"],
            frame_rate=config["frame_rate"],
        )

        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )

        self.display = config["display"]

        self.class_names = {
            0: "person",
            1: "car",
            2: "truck",
            3: "uav",
            4: "airplane",
            5: "boat",
        }

    def process_video(self):
        print(f"Processing video: {os.path.basename(self.source_video_path)} ...")
        print(f"Original video size: {self.video_info.resolution_wh}")
        print(f"Original video fps: {self.video_info.fps}")
        print(f"Original video number of frames: {self.video_info.total_frames}\n")

        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        if not self.display:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for i, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                    if i % self.video_stride == 0:
                        annotated_frame = self.process_frame(frame)
                        sink.write_frame(annotated_frame)
        else:
            for i, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                if i % self.video_stride == 0:
                    annotated_frame = self.process_frame(frame)
                    cv2.imshow("Processed Video", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        return self.annotate_frame(frame, detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()

        labels = [f"#{tracker_id} {self.class_names[class_id]} {confidence:.2f}"
                  for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and ByteTrack")
    parser.add_argument(
        "-c",
        "--config",
        default="./track_config.json",
        type=str,
        help="config file path (default: None)",
    )
    config = ConfigParser.from_args(parser)
    processor = VideoProcessor(config)
    processor.process_video()
