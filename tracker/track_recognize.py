import sys
from PIL.ImageQt import QImage
import argparse
import csv

import cv2
import numpy as np
import supervision as sv
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from tqdm import tqdm


from tracker import ByteTrack
from tracker.action_recognition import ActionRecognizer
from tracker.gui.GUI import VideoDisplay
from tracker.gui.frameCapture import FrameCapture
from tracker.gui.frameProcessing import VideoWriter
from tracker.utils.cfg.parse_config import ConfigParser
from tracker.utils.timer.utils import FrameRateCounter, Timer
from ultralytics import YOLO

COLORS = sv.ColorPalette.default()
import os

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"


class VideoProcessor(QObject):
    frame_ready = pyqtSignal(QImage, float)

    def __init__(self, config) -> None:
        super(VideoProcessor, self).__init__()
        # Initialize the YOLO parameters
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]

        self.source_video_path = config["source_stream_path"]

        self.output_dir = config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")

        self.device = config["device"]
        self.video_stride = config["video_stride"]

        self.model = YOLO(config["source_weights_path"])
        self.model.fuse()
        self.model.to(self.device)
        self.paused = False

        # TODO: CHECK IF MAINTAIN THIS
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        # TODO : CHECK TO PUT IN A THREAD
        self.tracker = ByteTrack(config, frame_rate=self.video_info.fps)

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100,
                                                 thickness=2)

        self.frame_capture = FrameCapture(self.source_video_path, stabilize=config["stabilize"],
                                          stream_mode=config["stream_mode"], logging=config["logging"])

        self.display = config["display"]
        self.save_video = config["save_video"]
        self.save_results = config["save_results"]
        self.csv_path = str(self.output_dir) + "/track_data.csv"
        if self.save_video and self.display:
            raise ValueError("Cannot display and save video at the same time")

        self.class_names = {
            0: "person",
            1: "car",
            2: "truck",
            3: "uav",
            4: "airplane",
            5: "boat",
        }

        self.action_recognizer = ActionRecognizer(config["action_recognition"], self.video_info)


    def process_video(self):
        print(f"Processing video: {self.source_video_path} ...")
        print(f"Original video size: {self.video_info.resolution_wh}")
        print(f"Original video FPS: {self.video_info.fps}")
        print(f"Original video number of frames: {self.video_info.total_frames}\n")

        if self.save_video:
            self.video_writer = VideoWriter(self.target_video_path, frame_size=self.frame_capture.get_frame_size(),
                                            compression_mode=config["compression_mode"],
                                            logging=config["logging"],
                                            fps=self.frame_capture.get_fps())
            self.video_writer.start()

        fps_counter = FrameRateCounter()
        timer = Timer()
        self.frame_capture.start()

        pbar = tqdm(total=self.video_info.total_frames, desc="Processing Frames", unit="frame")

        # Data dictionary to accumulate CSV data
        if self.save_results:
            data_dict = {
                "frame_id": [],
                "tracker_id": [],
                "class_id": [],
                "x1": [],
                "y1": [],
                "x2": [],
                "y2": []
            }
        while True:
            if not self.paused:
                frame = self.frame_capture.read()
                if frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame = self.process_frame(frame_rgb, self.frame_capture.get_frame_count(), fps_counter.value())
                fps_counter.step()
                if self.save_video and not self.display:
                    self.video_writer.write_frame(annotated_frame)
                if self.display:
                    height, width, channel = annotated_frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(q_image, fps_counter.value())
                if self.save_results:
                    for track in self.tracker.tracked_stracks:
                        data_dict["frame_id"].append(self.frame_capture.get_frame_count())
                        data_dict["tracker_id"].append(track.track_id)
                        data_dict["class_id"].append(track.class_id)
                        data_dict["x1"].append(track.tlbr[0])
                        data_dict["y1"].append(track.tlbr[1])
                        data_dict["x2"].append(track.tlbr[2])
                        data_dict["y2"].append(track.tlbr[3])
                pbar.update(1)
            if self.save_video:
                self.video_writer.stop()

            if self.save_results:
                # Write the collected data to a CSV file
                with open(self.csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data_dict.keys())
                    writer.writerows(zip(*data_dict.values()))

        pbar.close()
        print(f"\nTracking complete over {self.video_info.total_frames} frames.")
        print(f"Total time: {timer.elapsed():.2f} seconds")
        avg_fps = self.video_info.total_frames / timer.elapsed()
        print(f"Average FPS: {avg_fps:.2f}")

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections, tracks = self.tracker.update(detections, frame)

        ar_results = self.action_recognizer.recognize_frame(tracks)
        return self.annotate_frame(frame, detections, ar_results, frame_number, fps)

    def annotate_frame(self, annotated_frame: np.ndarray, detections: sv.Detections, ar_results: None,
                       frame_number: int,
                       fps: float) -> np.ndarray:

        labels = [f"#{tracker_id} {self.class_names[class_id]} {confidence:.2f}"
                  for tracker_id, class_id, confidence in
                  zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = self.action_recognizer.annotate(annotated_frame, ar_results)
        # cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated_frame

    def toggle_pause(self):
        self.paused = not self.paused

    def cleanup(self):
        self.frame_capture.stop()
        if self.save_video:
            self.video_writer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and ByteTrack")
    parser.add_argument(
        "-c",
        "--config",
        default="./ByteTrack.json",
        type=str,
        help="config file path (default: None)",
    )
    config = ConfigParser.from_args(parser)
    if config["display"]:
        app = QApplication(sys.argv)
        video_display = VideoDisplay(processor=VideoProcessor(config))
        video_display.show()
        sys.exit(app.exec())
    else:
        video_processor = VideoProcessor(config)
        video_processor.process_video()
