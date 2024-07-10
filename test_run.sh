#!/bin/bash

python yolov8_pruning.py --max-map-drop=0.5 
python yolov8_pruning.py --max-map-drop=0.5 --epochs=25