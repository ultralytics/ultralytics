# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import torch
from ultralytics import YOLO
import streamlit as st

# Hide main menu style
menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

# Main title of streamlit application
main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                         font-family: 'Archivo', sans-serif; margin-top:-50px;">
                Ultralytics YOLOv8 Streamlit Application
                </h1></div>"""

# Subtitle of streamlit application
sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                font-family: 'Archivo', sans-serif; margin-top:-15px;">
                Experience real-time object detection on your webcam with the power of Ultralytics YOLOv8! 🚀</h4>
                </div>"""


def inference():
    # Set html page configuration
    st.set_page_config(page_title='Ultralytics Streamlit App', layout='wide',
                       initial_sidebar_state='auto')

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # Add elements to vertical setting menu
    st.sidebar.title("USER Configuration")
    yolov8_model = st.sidebar.radio("Model", ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'))
    conf_thres = st.sidebar.text_input("Confidence Threshold", "0.25")
    nms_thres = st.sidebar.text_input("NMS Threshold", "0.45")

    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    ann_frame = col2.empty()

    if st.sidebar.button("Start"):
        model = YOLO(yolov8_model + ".pt")     # Load the yolov8 model
        videocapture = cv2.VideoCapture(0)       # Capture the webcam

        if not videocapture.isOpened():
            st.error("Could not open webcam.")

        stop_button = st.button("Stop")     # Button to stop the inference
        # execute code until webcam closed
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                break

            # Store model predictions
            results = model(frame, conf=float(conf_thres), iou=float(nms_thres))
            annotated_frame = results[0].plot()     # Add annotations on frame

            # display frame
            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()       # Release the capture
                torch.cuda.empty_cache()    # Clear CUDA memory

        # Release the capture
        videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()


# Main function call
if __name__ == "__main__":
    try:
        inference()
    except SystemExit:
        pass
