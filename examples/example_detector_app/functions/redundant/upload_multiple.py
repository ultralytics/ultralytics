import io

import streamlit as st
from PIL import Image


def upload_multiple():
    uploaded_files = st.file_uploader(
        "Please select multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        st.write("filename:", uploaded_file.name)
        return st.image(image, use_column_width=True)
