#!/bin/bash

#######      chmod +x script_device1.sh
#######      ./script_device1.sh
#######      device1 is 2080ti22g

yolo detect train data=VisDrone.yaml model=yolo11n-CAAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-CAAM-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11n-SSAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-SSAM-VD200-d1'
