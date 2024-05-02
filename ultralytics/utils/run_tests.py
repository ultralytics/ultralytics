from ultralytics.utils.benchmarks import ProfileModels, benchmark
ProfileModels(['yolov8n.yaml', 'yolov8s.yaml']).profile()
benchmark(model='yolov8n.pt', imgsz=160)
