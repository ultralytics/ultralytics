#!/bin/bash

python yolov8_pruning.py --max-map-drop=1.0 
python yolov8_pruning.py --max-map-drop=1.0 --epochs=25