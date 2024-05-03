# simple file to run the benchmarks without
# having to type multiple python commands at the
# command line

from ultralytics.utils.benchmarks import ProfileModels, benchmark

ProfileModels(['yolov8n.yaml']).profile()

df = benchmark(model='yolov8n.pt', imgsz=160)
