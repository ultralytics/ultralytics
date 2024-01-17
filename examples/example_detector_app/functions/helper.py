import cv2
import streamlit as st

from ultralytics import YOLO


# load model function (#used)
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


# display tracker option (#used)
def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))
    is_display_tracker = True if display_tracker == "Yes" else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


# detected frames display (#used)
def display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Display object tracking, if specified
    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption="Detected Video", channels="BGR", use_column_width=True)
