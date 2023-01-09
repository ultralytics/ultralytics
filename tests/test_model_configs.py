import glob

from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT

def test_all_configs():
    for cfg in glob.glob(str(ROOT/ "models/hub/*.yaml")):
        YOLO(cfg).info()
