# Ultralytics YOLO 🚀, AGPL-3.0 license
# Documentation: https://docs.ultralytics.com/analytics/index.md
# Example usage: yolo dashboard "path/to/custom_data.yaml"

import os
import sys
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from ultralytics.utils import ROOT
from ultralytics.utils.analytics_utils import Analytics
from ultralytics.utils.plotting import Annotator

# Application configuration
app = Flask(__name__, static_url_path="/static")
app.config["DEBUG"] = True

# Initialize analytics class object
analytics = Analytics()

success = True  # bool variable for success
fps = 0  # store fps
mode = None  # store mode name
in_counts = 0  # store in counts
out_counts = 0  # store out counts
start_time = time.time()  # store starting time
total_detections = 0  # store total detection counts
yaml_filepath = None  # Path to yaml file


def detect_objects():
    """Function for detection, tracking and objects counting."""
    global total_detections, fps, in_counts, out_counts, success, start_time, mode

    # Load and process YAML file
    model, mode, video_file = analytics.load_and_proces_yaml(yaml_filepath)

    # Capture the video file using OpenCV
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Error reading video file!"

    while cap.isOpened():
        success, im0 = cap.read()

        if success:
            annotator = Annotator(im0, example=model.names)

            # Processing results
            if mode == "track" or mode == "count":
                results = model.track(im0, persist=True, show=False)

                if mode == "track":
                    total_detections = analytics.process_results(results, annotator)

                elif mode == "count":
                    total_detections, in_counts, out_counts = analytics.process_results(results, annotator)

            else:
                results = model.predict(im0, show=False)
                total_detections = analytics.process_results(results, annotator)

            # Calculate FPS
            elapsed_time = time.time()
            fps = int(1 / (elapsed_time - start_time))
            start_time = elapsed_time

            # Yield frame
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", im0)[1].tobytes() + b"\r\n"
            )

        else:
            break

    # Release and destroy window
    cap.release()
    cv2.destroyAllWindows()


@app.route("/detected_frame")
def detected_frame():
    return Response(detect_objects(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/total_detections")
def total_detections_endpoint():
    """
    Send total detections to webpage.

    Returns:
        response_data (json, optional): Return a JSON data
    """
    if success:
        response_data = {
            "status": "Detection complete",
            "total_detections": total_detections,
            "in_counts": in_counts,
            "out_counts": out_counts,
            "fps": fps,
            "mode": mode,
            "class_wise_dict": analytics.classwise_detections,
        }
        return jsonify(response_data), 200
    else:
        sys.exit()


@app.route("/", methods=["GET"])
def index():
    return render_template("dashboard.html")


if __name__ == "__main__":
    """Main function for app start and processing data."""
    yaml_filepath = sys.argv[1]

    if not os.path.exists(yaml_filepath):
        print(f"Error: YAML file '{yaml_filepath}' not found.")
        print("Using default dashboard.yaml!")
        yaml_filepath = ROOT / "cfg/analytics/dashboard.yaml"

    app.run(debug=False)
