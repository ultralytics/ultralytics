### Detector Application

#### About:

```
    This application is built with streamlit.
    It allows for three major ways of detection:
        * Detection by uploading image(s)
        * Detection by uploading video(s)
        * Detection using webcam
    The streamlit library is used for building the interface of the app.
    There are two major models(pre-trained) present in this project:
        - YOLOv3
        - YOLOv8
```

#### Files and Folders:

```
    There are multiple files and folders in this project:
    **FOLDERS**
        **functions**
            This folder contains all the necessary individual functions of the app:
                - webcam
                    This function contains all the necessary codes for opening and displaying the detected images using webcam.
                - video_upload
                    This function contains the codes for
                        - uploading
                        - saving
                        - and detecting videos only.
                - image_upload
                    This function contains the codes to upload and detect images.
                - settings
                    This file contains the necessary settings needed for each functions written in the functions folder to avoid repetition.
                - helper
                    This functions contains various functions that aids the main app file outside the functions folder.

        **images**
            This folder contains local images used for testing the app.

        **weights**
            This folder contains the different YOLO weights used in the project.

        **uploaded_videos**
            This is a `code` generated folder that will store the uploaded videos to be re-read for detection in the upload_video function.

    **FILES**
        **main**
            This is the main file that contains the streamlit interface code and the calling of the various functions in the functions folder.
```

#### How to run

```
    * Fork the repository/ download the zip file.
    * Download the requirements from the requirements.txt file.
    * cd into the directory.
    * run the following command in your terminal/command prompt 'python -m streamlit run main.py'.
    * click on the link (localhost).
```

### NB: Test the app automatically here:
