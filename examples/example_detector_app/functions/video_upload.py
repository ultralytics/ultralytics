import os

import cv2
import streamlit as st


# upload video function
def upload_video():
    video_file = st.file_uploader("Upload a video", type=["mp4", "mpeg"])

    if video_file is not None:
        # st.video(video_file)
        st.success("Video uploaded successfully!")
        return video_file


# save and upload a video for detection
def save_uploaded_file(uploaded_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def save_file():
    # Upload a video file
    uploaded_file = upload_video()

    if uploaded_file is not None:
        # Save the uploaded video file to a directory
        save_dir = "uploaded_videos"
        file_path = save_uploaded_file(uploaded_file, save_dir)

        # Display the saved file path
        # st.write(f"Uploaded video file saved to: {file_path}")
        return file_path


def detect_video(confidence, model):
    # call the save_file function
    file_path = save_file()
    if file_path is not None:
        with open(str(f"{file_path}"), "rb") as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            # st.video(video_bytes)
            pass
        if st.button("Detect Objects"):
            vid_cap = cv2.VideoCapture(f"{file_path}")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720 * (9 / 16))))
                    res = model.predict(image, conf=confidence)
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted, caption="Detected Video", channels="BGR", use_column_width=True)
                else:
                    vid_cap.release()
                    break
