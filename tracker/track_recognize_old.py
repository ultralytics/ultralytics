import argparse
import csv
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

from tracker.byte_track import ByteTrack
from tracker.action_recognition import ActionRecognizer
from tracker.utils.cfg.parse_config import ConfigParser
from tracker.utils.timer.utils import FrameRateCounter, Timer
from ultralytics import YOLO
import supervision as sv

COLORS = sv.ColorPalette.default()


class VideoProcessor:
    def __init__(self, config) -> None:

        # Initialize the YOLO parameters
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]

        self.source_video_path = config["source_video_path"]

        self.output_dir = config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")

        self.device = config["device"]
        self.video_stride = config["video_stride"]
        self.wait_time = 1
        self.slow_factor = 1

        self.model = YOLO(config["source_weights_path"])
        self.model.fuse()

        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        self.tracker = ByteTrack(config, frame_rate=self.video_info.fps)

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)

        self.display = config["display"]
        self.save_results = config["save_results"]
        self.csv_path = str(self.output_dir) + "/track_data.csv"


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
        print(f"Processing video: {os.path.basename(self.source_video_path)} ...")
        print(f"Original video size: {self.video_info.resolution_wh}")
        print(f"Original video FPS: {self.video_info.fps}")
        print(f"Original video number of frames: {self.video_info.total_frames}\n")

        #TODO: use get_stream_frames_generator instead of get_video_frames_generator for real-time processing
        # https://www.youtube.com/watch?v=hAWpsIuem10
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        data_dict = {
            "frame_id": [],
            "tracker_id": [],
            "class_id": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
        }

        video_sink = sv.VideoSink(self.target_video_path, self.video_info) if not self.display else nullcontext()
        with video_sink as sink:
            fps_counter = FrameRateCounter()
            timer = Timer()

            for i, frame in enumerate(pbar := tqdm(frame_generator, total=self.video_info.total_frames)):
                pbar.set_description(f"[FPS: {fps_counter.value():.2f}] ")
                if i % self.video_stride == 0:
                    annotated_frame = self.process_frame(frame, i, fps_counter.value())
                    fps_counter.step() # here

                    if not self.display:
                        sink.write_frame(annotated_frame)
                    else:
                        cv2.imshow("Processed Video", annotated_frame)

                        k = cv2.waitKey(int(self.wait_time * self.slow_factor))  # dd& 0xFF

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        elif k == ord('p'):  # pause the video
                            cv2.waitKey(-1)  # wait until any key is pressed
                        elif k == ord('r'):  # resume the video
                            continue
                        elif k == ord('d'):
                            slow_factor = self.slow_factor - 1
                            print(slow_factor)
                        elif k == ord('i'):
                            slow_factor = self.slow_factor + 1
                            print(slow_factor)
                            break

                    # Store results
                    if self.save_results:
                        for track in self.tracker.tracked_stracks:
                            data_dict["frame_id"].append(track.frame_id)
                            data_dict["tracker_id"].append(track.track_id)
                            data_dict["class_id"].append(track.class_ids)
                            data_dict["x1"].append(track.tlbr[0])
                            data_dict["y1"].append(track.tlbr[1])
                            data_dict["x2"].append(track.tlbr[2])
                            data_dict["y2"].append(track.tlbr[3])

            if self.display:
                cv2.destroyAllWindows()

        # Print time and fps
        time_taken = f"{int(timer.elapsed() / 60)} min {int(timer.elapsed() % 60)} sec"
        avg_fps = self.video_info.total_frames / timer.elapsed()
        print(f"\nTracking complete over {self.video_info.total_frames} frames.")
        print(f"Total time: {time_taken}")
        print(f"Average FPS: {avg_fps:.2f}")

        # Save datadict in csv
        if self.save_results:
            with open(self.csv_path, "w") as f:
                w = csv.writer(f)
                w.writerow(data_dict.keys())
                w.writerows(zip(*data_dict.values()))

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
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
        detections, tracks = self.tracker.update(detections, frame)

        ar_results = self.action_recognizer.recognize_frame(tracks)

        return self.annotate_frame(frame, detections, ar_results,  frame_number, fps)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, ar_results: None, frame_number: int,
                       fps: float) -> np.ndarray:
        annotated_frame = frame.copy()

        labels = [f"#{tracker_id} {self.class_names[class_id]} {confidence:.2f}"
                  for tracker_id, class_id, confidence in
                  zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = self.action_recognizer.annotate(annotated_frame, ar_results)
        cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame


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
    processor = VideoProcessor(config)
    processor.process_video()
