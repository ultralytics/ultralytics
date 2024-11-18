# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import cv2

# # Load the YOLO model
# MODEL_PATH = "/Users/solomon1.odum/Documents/Anemia_prediction/runs/classify/train8/weights/best.pt"
# model = YOLO(MODEL_PATH)

# # Streamlit App
# st.set_page_config(
#     page_title="Anemia Detection",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Header and Instructions
# st.title("🩺 Anemia Detection from Conjunctiva Images")
# st.markdown(
#     """
#     Welcome to the **Anemia Detection App**! This tool uses a deep learning model to predict whether an uploaded image of the conjunctiva indicates **anemia** or not.
#     \n**Instructions**:
#     - Upload a **clear image** of the conjunctiva (in JPG, PNG, or JPEG format).
#     - For **batch predictions**, upload multiple images at once.
#     - View predictions for each image along with confidence scores.
#     """
# )

# # File uploader for single or multiple images
# uploaded_files = st.file_uploader(
#     "Upload Image(s) of the Conjunctiva",
#     type=["jpg", "png", "jpeg"],
#     accept_multiple_files=True,
# )

# # Handle uploaded images
# if uploaded_files:
#     results_container = st.container()
#     for uploaded_file in uploaded_files:
#         try:
#             # Read the image
#             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#             image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#             # Check if the uploaded file is a valid image
#             if image is None:
#                 st.error(f"Error: {uploaded_file.name} is not a valid image.")
#                 continue

#             # Convert BGR to RGB for display
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Display the uploaded image
#             with results_container:
#                 st.image(image_rgb, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

#             # Perform inference
#             results = model.predict(image, verbose=False)

#             # Access classification probabilities
#             probs = results[0].probs  # Probabilities object
#             if probs is None:
#                 st.error(f"No predictions could be made for {uploaded_file.name}.")
#                 continue

#             # Get top prediction and confidence
#             predicted_class_idx = probs.top1  # Top-1 predicted class index
#             predicted_confidence = float(probs.top1conf)  # Convert tensor to float

#             # Map class index to class name
#             class_names = ["anemic", "non_anemic"]
#             predicted_class = class_names[predicted_class_idx]

#             # Display results
#             with results_container:
#                 st.success(f"Prediction for **{uploaded_file.name}**: **{predicted_class}**")
#                 st.info(f"Confidence: {predicted_confidence * 100:.2f}%")

#         except Exception as e:
#             st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

# else:
#     st.info("Upload image(s) to see predictions.")

# # Footer
# st.markdown(
#     """
#     ---
#     Built with ❤️ using [Streamlit](https://streamlit.io/) and YOLOv8.
#     """
# )
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLO model
MODEL_PATH = "/Users/solomon1.odum/Documents/Anemia_prediction/runs/classify/train8/weights/best.pt"
model = YOLO(MODEL_PATH)

# Streamlit App
st.set_page_config(
    page_title="Anemia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header and Instructions
st.title("🩺 Anemia Detection from Conjunctiva Images")
st.markdown(
    """
    Welcome to the **Anemia Detection App**! This tool uses a deep learning model to predict whether an uploaded image of the conjunctiva indicates **anemia** or not.
    \n**Instructions**:
    - Upload a **clear image** of the conjunctiva (in JPG, PNG, or JPEG format).
    - For **batch predictions**, upload multiple images at once.
    - View predictions for each image along with confidence scores.
    """
)

# File uploader for single or multiple images
uploaded_files = st.file_uploader(
    "Upload Image(s) of the Conjunctiva",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True,
)

# Handle uploaded images
if uploaded_files:
    results_container = st.container()
    for uploaded_file in uploaded_files:
        try:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Check if the uploaded file is a valid image
            if image is None:
                st.error(f"Error: {uploaded_file.name} is not a valid image.")
                continue

            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the uploaded image
            with results_container:
                st.image(image_rgb, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

            # Perform inference
            results = model.predict(image, verbose=False)

            # Access classification probabilities
            probs = results[0].probs  # Probabilities object
            if probs is None:
                st.error(f"No predictions could be made for {uploaded_file.name}.")
                continue

            # Get top prediction and confidence
            predicted_class_idx = probs.top1  # Top-1 predicted class index
            predicted_confidence = float(probs.top1conf)  # Convert tensor to float

            # Map class index to class name
            class_names = ["anemic", "non_anemic"]
            predicted_class = class_names[predicted_class_idx]

            # Display results
            with results_container:
                st.success(f"Prediction for **{uploaded_file.name}**: **{predicted_class}**")
                st.info(f"Confidence: {predicted_confidence * 100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

else:
    st.info("Upload image(s) to see predictions.")

# Footer
st.markdown(
    """
    ---
    Built with ❤️ using [Streamlit](https://streamlit.io/) and YOLOv8.
    """
)
