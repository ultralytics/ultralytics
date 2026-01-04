#!/bin/bash

#######      chmod +x script_device0.sh
#######      ./script_device0.sh
#######      device0 is 3080 20g

# yolo detect train data=VisDrone.yaml model=yolo11n-CAAM.yaml epochs=200 batch=16 imgsz=640 device=0 name='yolo11n-CAAM-VD200'
# yolo detect train data=VisDrone.yaml model=yolo11n-SSAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-SSAM-VD200'

yolo detect train data=VisDrone.yaml model=yolo11x.yaml epochs=200 batch=4 imgsz=640 device=0 name='v11x-VD200-d0'
# yolo detect train data=VisDrone.yaml model=yolo11l.yaml epochs=200 batch=4 imgsz=640 device=0 name='v11l-VD200-d0'
yolo detect train data=VisDrone.yaml model=yolo11m.yaml epochs=200 batch=8 imgsz=640 device=0 name='v11m-VD200-d0'
yolo detect train data=VisDrone.yaml model=yolo11s.yaml epochs=200 batch=16 imgsz=640 device=0 name='v11s-VD200-d0'
# yolo detect train data=VisDrone.yaml model=yolo11n.yaml epochs=200 batch=16 imgsz=640 device=0 name='v11n-VD200-d0'