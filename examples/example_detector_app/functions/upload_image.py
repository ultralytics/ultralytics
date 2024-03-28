import urllib

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# Image upload function
def image_upload():
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.success("Image uploaded successfully")
        image = Image.open(uploaded_file)
        return uploaded_file


# Image detection
def image_detection(file):
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        # st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)
        confThreshold = st.slider("Confidence", 0, 100, 50)
        nmsThreshold = st.slider("Threshold", 0, 100, 20)
        # classNames = []
        whT = 320
        url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
        f = urllib.request.urlopen(url)
        classNames = [line.decode("utf-8").strip() for line in f]
        # f = open(r'C:\Users\Olazaah\Downloads\stream\labels\coconames.txt','r')
        # lines = f.readlines()
        # classNames = [line.strip() for line in lines]
        config_path = r"C:\\Users\\BAB AL SAFA\\Desktop\\MINE\\Detector-App\\weights\\yolov3-spp.cfg"
        weights_path = r"C:\\Users\\BAB AL SAFA\\Desktop\\MINE\\Detector-App\\weights\\yolov3-spp.weights"
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs, img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold / 100):
                        w, h = int(det[2] * wT), int(det[3] * hT)
                        x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                        bbox.append([x, y, w, h])
                        classIds.append(classId)
                        confs.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold / 100, nmsThreshold / 100)
            obj_list = []
            confi_list = []
            # drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (240, 54, 230), 2)
                # print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())

                confi_list.append(int(confs[i] * 100))
                cv2.putText(
                    img2,
                    f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (240, 0, 240),
                    2,
                )
            df = pd.DataFrame(list(zip(obj_list, confi_list)), columns=["Object Name", "Confidence"])
            if st.checkbox("Show Object's list"):
                st.write(df)

            if st.checkbox("Show Confidence chart"):
                st.subheader("Bar chart for Confidence levels")

                st.bar_chart(df["Confidence"])

        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img2)

        st.image(img2, caption="Processed Image.")

        cv2.waitKey(0)

        cv2.destroyAllWindows()
        my_bar.progress(100)
