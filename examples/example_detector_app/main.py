from pathlib import Path

import streamlit as st
from functions import helper, html, settings
from functions.upload_image import image_detection, image_upload
from functions.video_upload import detect_video
from functions.webcam import play_webcam
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")
st.title("Detector App")

style = html.STYLE
about_dev = html.ABOUT_DEV
about_app = html.ABOUT_APP
link = html.LINK

# Upper sidebar
with st.sidebar:
    choose = option_menu(
        "Upload Menu",
        ["Webcam", "Video", "Image", "Info"],
        icons=["camera-video", "film", "camera", "clipboard"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "white"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "orange"},
        },
    )

if choose == "Webcam":
    st.markdown(
        """ <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: black;}
        </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">WebCam</p>', unsafe_allow_html=True)
    # confidence chooser
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100
    # path
    model_path = Path(settings.DETECTION_MODEL)
    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    play_webcam(confidence, model)

elif choose == "Video":
    st.markdown(
        """ <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: black;}
        </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">Video</p>', unsafe_allow_html=True)

    # path
    model_path = Path(settings.DETECTION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

    detect_video(confidence=confidence, model=model)

elif choose == "Image":
    st.markdown(
        """ <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: black;}
        </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">Upload an Image</p>', unsafe_allow_html=True)  # For image
    uploaded_file = image_upload()

    # Run object detection when an image or video is uploaded
    if uploaded_file is not None:
        # Perform object detection using YOLOv5
        detected_output = image_detection(uploaded_file)

        # # Display the original and detected images/videos in the app
        # if detected_output is not None:
        #         st.image(detected_output, caption='Detected Objects', use_column_width=True)

# elif choose == "Multiple Images":
#     st.markdown(""" <style> .font {
#         font-size:30px ; font-family: 'Cooper Black'; color: black;}
#         </style> """, unsafe_allow_html=True)
#     st.markdown('<p class="font">Multiple Images</p>', unsafe_allow_html=True)
#     upload_multiple()

elif choose == "Info":
    with st.sidebar:
        info = option_menu(
            "Info Menu",
            ["About the app", "Developer"],
            icons=["book", "person lines fill"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "white"},
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "orange"},
                "nav-link-selected": {"background-color": "orange"},
            },
        )

    if info == "About the app":
        st.markdown(
            """ <style> .font {
            font-size:30px ; font-family: 'Cooper Black'; color: black;}
            </style> """,
            unsafe_allow_html=True,
        )
        st.markdown('<h2 class="font"> About the App </h2>', unsafe_allow_html=True)
        st.markdown(style, unsafe_allow_html=True)

        st.markdown(about_app, unsafe_allow_html=True)

    elif info == "Developer":
        st.markdown(
            """ <style> .font {
                font-size:30px ; font-family: 'Cooper Black'; color: black;}
                </style> """,
            unsafe_allow_html=True,
        )
        st.markdown('<h2 class="font"> About the Developer </h2>', unsafe_allow_html=True)
        st.markdown(style, unsafe_allow_html=True)
        st.markdown(about_dev, unsafe_allow_html=True)
        st.markdown(link, unsafe_allow_html=True)
    else:
        pass
