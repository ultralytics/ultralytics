import argparse
import os
import queue
import threading
import time
import urllib.request

import gi
import numpy as np

from ultralytics import YOLO

gi.require_version("Gst", "1.0")
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from gi.repository import GLib, Gst
from PIL import Image, ImageTk

# ----------------------------
#   COLOR MAPPINGS
# ----------------------------
COLOR_MAP = {
    2: (0, 0, 255),  # car: red
    3: (255, 0, 0),  # motorcycle: blue
    5: (0, 255, 0),  # bus: green
    7: (0, 165, 255),  # truck: orange
}
DEFAULT_COLOR = (255, 255, 0)  # cyan fallback


# ----------------------------
#   GSTREAMER OVERLAY PROCESSOR
# ----------------------------
class GStreamerOverlayProcessor:
    def __init__(self, source, imgsz=640, fps=30):
        self.source = source
        self.imgsz = imgsz
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.pipeline = None
        self.loop = None
        self.error = None

        Gst.init(None)

    def on_new_sample(self, sink_pad, data):
        sample = sink_pad.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)
            width = s.get_value("width")
            height = s.get_value("height")

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame_np = np.ndarray((height, width, 4), buffer=map_info.data, dtype=np.uint8)
                buffer.unmap(map_info)
                frame_bgr = frame_np[:, :, :3].copy()

                try:
                    self.frame_queue.put_nowait(frame_bgr)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_bgr)
                    except queue.Empty:
                        pass

        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.error = f"GStreamer error: {err.message}"
            print(f"[ERROR] {self.error}")
            print(f"[DEBUG] {debug}")
        elif msg_type == Gst.MessageType.EOS:
            print("[INFO] End of stream")
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            print(f"[INFO] Pipeline state changed: {old_state.value_name} -> {new_state.value_name}")
        return True

    def create_pipeline(self):
        """Create GStreamer pipeline with appsink only (no display window)."""
        if self.source.startswith("rtsp://"):
            # RTSP stream pipeline
            pipeline_str = (
                f"rtspsrc location={self.source} latency=0 ! "
                f"rtph264depay ! h264parse ! "
                f"nvv4l2decoder enable-max-performance=true ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx, width={self.imgsz}, height={self.imgsz} ! "
                f"appsink name=mysink drop=true max-buffers=1"
            )
        else:
            # Local file pipeline
            pipeline_str = (
                f"filesrc location={self.source} ! "
                f"qtdemux ! h264parse ! "
                f"nvv4l2decoder enable-max-performance=true ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx, width={self.imgsz}, height={self.imgsz} ! "
                f"appsink name=mysink drop=true max-buffers=1"
            )

        print(f"[INFO] Creating pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Add bus message handler
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        appsink = self.pipeline.get_by_name("mysink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)  # Disable sync for maximum FPS
        # appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self.on_new_sample, None)

    def start(self):
        try:
            self.create_pipeline()
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to set pipeline to PLAYING state")

            self.running = True

            self.loop = GLib.MainLoop()
            self.thread = threading.Thread(target=self.loop.run, daemon=True)
            self.thread.start()

            print("[INFO] GStreamer pipeline started successfully")
            return True
        except Exception as e:
            self.error = str(e)
            print(f"[ERROR] Failed to start GStreamer pipeline: {e}")
            return False

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)

    def read(self):
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def isOpened(self):
        return self.running and not self.error


# ----------------------------
#   TKINTER GUI WITH VIDEO DISPLAY
# ----------------------------
class SimpleDroneAnalysisGUI:
    def __init__(self, root, processor, model, args):
        self.root = root
        self.processor = processor
        self.model = model
        self.args = args

        self.setup_gui()

        # Processing variables
        self.fi = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.running = False

    def setup_gui(self):
        self.root.title("Simple Drone Analysis - GStreamer Overlay")
        self.root.geometry("1000x700")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls and stats
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Control panel
        control_frame = ttk.LabelFrame(left_panel, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_stop_btn = ttk.Button(control_frame, text="Start Processing", command=self.toggle_processing)
        self.start_stop_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)

        # Performance frame
        perf_frame = ttk.LabelFrame(left_panel, text="Performance")
        perf_frame.pack(fill=tk.X, pady=(0, 10))

        self.fps_label = ttk.Label(perf_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.frame_label = ttk.Label(perf_frame, text="Frame: 0")
        self.frame_label.pack(side=tk.LEFT, padx=20, pady=5)

        self.detection_label = ttk.Label(perf_frame, text="Detections: 0")
        self.detection_label.pack(side=tk.LEFT, padx=20, pady=5)

        self.tracking_label = ttk.Label(perf_frame, text="Tracks: 0")
        self.tracking_label.pack(side=tk.LEFT, padx=20, pady=5)

        # Model info frame
        info_frame = ttk.LabelFrame(left_panel, text="Model Info")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.model_label = ttk.Label(info_frame, text=f"Model: {os.path.basename(args.model)}")
        self.model_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.size_label = ttk.Label(info_frame, text=f"Size: {args.imgsz}x{args.imgsz}")
        self.size_label.pack(side=tk.LEFT, padx=20, pady=5)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Video display frame
        video_frame = ttk.LabelFrame(right_panel, text="Video Display")
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def toggle_processing(self):
        if not self.running:
            if not self.processor.isOpened():
                messagebox.showerror("Error", f"GStreamer pipeline not ready: {self.processor.error}")
                return

            self.running = True
            self.start_stop_btn.config(text="Stop Processing")
            self.status_label.config(text="Processing...")
            self.process_frames()
        else:
            self.running = False
            self.start_stop_btn.config(text="Start Processing")
            self.status_label.config(text="Stopped")

    def process_frames(self):
        if not self.running:
            return

        ret, frame = self.processor.read()
        if not ret:
            self.root.after(100, self.process_frames)
            return

        # Run YOLO inference
        results = self.model.track(
            frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            tracker=self.args.tracker,
            classes=[2, 3, 5, 7],
            device="cuda:0",
            verbose=False,
        )

        res = results[0] if results else None

        # Process detections
        if res and res.boxes:
            db = res.boxes.xyxy.cpu().numpy()
            dc = res.boxes.cls.cpu().numpy().astype(int)
            tid_t = res.boxes.id
            tr = tid_t.cpu().numpy().astype(int) if (tid_t is not None) else np.empty(0, int)
        else:
            db = np.empty((0, 4))
            dc = np.empty((0,), int)
            tr = np.empty((0,), int)

        # Draw detections on frame
        display_frame = frame.copy()

        # Draw detections
        for i, (x1, y1, x2, y2) in enumerate(db):
            if i < len(dc):
                cls_id = dc[i]
                color = COLOR_MAP.get(cls_id, DEFAULT_COLOR)
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw vehicle ID if tracking
                if i < len(tr):
                    vid = tr[i]
                    cv2.putText(
                        display_frame, f"ID:{vid}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

        # Update video display
        self.update_video_display(display_frame)

        # Update GUI
        self.update_gui(len(db), len(tr))

        # Continue processing
        self.fi += 1
        self.fps_counter += 1
        self.root.after(1, self.process_frames)

    def update_video_display(self, frame):
        """Update the video display in the canvas."""
        try:
            # Resize frame to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Resize frame to fit canvas while maintaining aspect ratio
                h, w = frame.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)

                resized_frame = cv2.resize(frame, (new_w, new_h))

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)

                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(pil_image)

                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    canvas_width // 2, canvas_height // 2, image=self.photo, anchor=tk.CENTER
                )
        except Exception as e:
            print(f"[WARNING] Failed to update video display: {e}")

    def update_gui(self, detections, tracks):
        # Calculate FPS using rolling window (last 1 second)
        current_time = time.time()
        time_diff = current_time - self.last_fps_time

        if time_diff >= 1.0:  # Update FPS every second
            fps = self.fps_counter / time_diff
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.last_fps_time = current_time
            self.fps_counter = 0

        self.frame_label.config(text=f"Frame: {self.fi}")
        self.detection_label.config(text=f"Detections: {detections}")
        self.tracking_label.config(text=f"Tracks: {tracks}")

    def on_closing(self):
        self.running = False
        self.processor.stop()

        elapsed = time.time() - self.start_time
        fps_final = self.fi / elapsed if elapsed > 0 else 0
        print(f"[DONE] Processed {self.fi} frames in {elapsed:.1f}s ({fps_final:.1f} FPS)")

        self.root.destroy()


# ----------------------------
#   MAIN
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple Drone Detection with GStreamer Overlay & Tkinter GUI")
    p.add_argument("video", help="Video file or RTSP URL")
    p.add_argument("--model", default="yolo11m.pt", help="YOLO11m PyTorch weights file")
    p.add_argument("--engine", default="yolo11m.engine", help="YOLO11m TensorRT engine file")
    p.add_argument(
        "--model-url",
        default="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        help="URL to download YOLO11m PyTorch model if missing",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference size (px)")
    p.add_argument("--conf", type=float, default=0.03, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.03, help="NMS IoU threshold")
    p.add_argument("--tracker", default="bytetrack.yaml", help="ByteTrack config")
    p.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    args = p.parse_args()

    # Load model
    if os.path.exists(args.engine):
        print(f"[INFO] Loading TensorRT engine: {args.engine}")
        model = YOLO(args.engine)
    elif os.path.exists(args.model):
        print(f"[INFO] Loading PyTorch model: {args.model}")
        model = YOLO(args.model)
        print(f"[INFO] Exporting to TensorRT engine: {args.engine}")
        model.export(
            format="engine",
            name=os.path.basename(args.engine).split(".")[0],
            save_dir=os.path.dirname(args.engine),
            imgsz=args.imgsz,
        )
        model = YOLO(args.engine)
    else:
        print(f"[INFO] Downloading PyTorch model: {args.model_url}")
        urllib.request.urlretrieve(args.model_url, args.model)
        print(f"[INFO] Loading PyTorch model: {args.model}")
        model = YOLO(args.model)
        print(f"[INFO] Exporting to TensorRT engine: {args.engine}")
        model.export(
            format="engine",
            name=os.path.basename(args.engine).split(".")[0],
            save_dir=os.path.dirname(args.engine),
            imgsz=args.imgsz,
        )
        model = YOLO(args.engine)

    print("[INFO] Classes:", model.names)

    # Setup GStreamer processor
    print(f"[INFO] Using GStreamer for: {args.video}")
    processor = GStreamerOverlayProcessor(args.video, args.imgsz, args.fps)

    if not processor.start():
        print("[ERROR] Failed to start GStreamer pipeline")
        exit(1)

    time.sleep(2)
    print(f"[INFO] GStreamer video {args.imgsz}×{args.imgsz} @ {args.fps:.2f} FPS")

    # Create Tkinter GUI
    root = tk.Tk()
    app = SimpleDroneAnalysisGUI(root, processor, model, args)

    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    print("[INFO] Starting GUI application...")
    root.mainloop()
