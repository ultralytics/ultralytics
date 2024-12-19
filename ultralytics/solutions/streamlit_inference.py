# Ultralytics YOLO 🚀, AGPL-3.0 license

import io
import time

import cv2

from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from ultralytics.solutions.solutions import check_requirements, YOLO, LOGGER


class Inference:
    def __init__(self, **kwargs):
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
        import streamlit as st
        self.st = st

        self.temp_dict = {"model": None}    # Temporary dict to store the model path
        self.temp_dict.update(kwargs)

        self.model_path = None      # Store model file name with path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        with self.st.sidebar:        # Add Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox("Video", ("webcam", "video"),)  # Add source selection dropdown
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))   # Enable object tracking
        self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))  # Slider for confidence
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))     # Slider for NMS threshold

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()
        self.fps_display = self.st.sidebar.empty()  # Placeholder for FPS display

    def source_upload(self):
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO Object
                with open("ultralytics.mp4", "wb") as out:  # Open temporary file as bytes
                    out.write(g.read())  # Read bytes into file
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0

    def configure(self):
        # Add dropdown menu for model selection
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:     # If user provided the custom model, insert model without suffix as *.pt is added later
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")  # Load the YOLO model
            class_names = list(self.model.names.values())  # Convert dictionary to list of class names
        self.st.success("Model loaded successfully!")

        # Multiselect box with class names and get indices of selected classes
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # Ensure selected_options is a list
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        self.web_ui()       # Initialize the web interface
        self.sidebar()      # Create the sidebar
        self.source_upload()    # Upload the video source
        self.configure()    # Configure the app

        if self.st.sidebar.button("Start"):
            stop_button = self.st.button("Stop")  # Button to stop the inference
            cap = cv2.VideoCapture(self.vid_file_name)  # Capture the video
            if not cap.isOpened():
                self.st.error("Could not open webcam.")
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                    break

                prev_time = time.time()  # Store initial time for FPS calculation

                # Store model predictions
                if self.enable_trk == "Yes":
                    results = self.model.track(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind,
                                               persist=True)
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_frame = results[0].plot()  # Add annotations on frame

                fps = 1 / (time.time() - prev_time)     # Calculate model FPS

                if stop_button:
                    cap.release()  # Release the capture
                    self.st.stop()  # Stop streamlit app

                self.fps_display.metric("FPS", f"{fps:.2f}")    # Display FPS in sidebar
                self.org_frame.image(frame, channels="BGR")     # Display original frame
                self.ann_frame.image(annotated_frame, channels="BGR")   # Display processed frame

            cap.release()   # Release the capture
        cv2.destroyAllWindows()     # Destroy window
