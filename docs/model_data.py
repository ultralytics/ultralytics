# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

data = {
    "YOLO11": {
        "n": {"size": 640, "map": 39.5, "cpu": 56.1, "t4": 1.5, "params": 2.6, "flops": 6.5},
        "s": {"size": 640, "map": 47.0, "cpu": 90.0, "t4": 2.5, "params": 9.4, "flops": 21.5},
        "m": {"size": 640, "map": 51.5, "cpu": 183.2, "t4": 4.7, "params": 20.1, "flops": 68.0},
        "l": {"size": 640, "map": 53.4, "cpu": 238.6, "t4": 6.2, "params": 25.3, "flops": 86.9},
        "x": {"size": 640, "map": 54.7, "cpu": 462.8, "t4": 11.3, "params": 56.9, "flops": 194.9},
    },
    "YOLOv10": {
        "n": {"size": 640, "map": 39.5, "cpu": "", "t4": 1.56, "params": 2.3, "flops": 6.7},
        "s": {"size": 640, "map": 46.7, "cpu": "", "t4": 2.66, "params": 7.2, "flops": 21.6},
        "m": {"size": 640, "map": 51.3, "cpu": "", "t4": 5.48, "params": 15.4, "flops": 59.1},
        "b": {"size": 640, "map": 52.7, "cpu": "", "t4": 6.54, "params": 24.4, "flops": 92.0},
        "l": {"size": 640, "map": 53.3, "cpu": "", "t4": 8.33, "params": 29.5, "flops": 120.3},
        "x": {"size": 640, "map": 54.4, "cpu": "", "t4": 12.2, "params": 56.9, "flops": 160.4},
    },
    "YOLOv9": {
        "t": {"size": 640, "map": 38.3, "cpu": "", "t4": 2.3, "params": 2.0, "flops": 7.7},
        "s": {"size": 640, "map": 46.8, "cpu": "", "t4": 3.54, "params": 7.1, "flops": 26.4},
        "m": {"size": 640, "map": 51.4, "cpu": "", "t4": 6.43, "params": 20.0, "flops": 76.3},
        "c": {"size": 640, "map": 53.0, "cpu": "", "t4": 7.16, "params": 25.3, "flops": 102.1},
        "e": {"size": 640, "map": 55.6, "cpu": "", "t4": 16.77, "params": 57.3, "flops": 189.0},
    },
    "YOLOv8": {
        "n": {"size": 640, "map": 37.3, "cpu": 80.4, "t4": 1.47, "params": 3.2, "flops": 8.7},
        "s": {"size": 640, "map": 44.9, "cpu": 128.4, "t4": 2.66, "params": 11.2, "flops": 28.6},
        "m": {"size": 640, "map": 50.2, "cpu": 234.7, "t4": 5.86, "params": 25.9, "flops": 78.9},
        "l": {"size": 640, "map": 52.9, "cpu": 375.2, "t4": 9.06, "params": 43.7, "flops": 165.2},
        "x": {"size": 640, "map": 53.9, "cpu": 479.1, "t4": 14.37, "params": 68.2, "flops": 257.8},
    },
    "YOLOv7": {
        "l": {"size": 640, "map": 51.4, "cpu": "", "t4": 6.84, "params": 36.9, "flops": 104.7},
        "x": {"size": 640, "map": 53.1, "cpu": "", "t4": 11.57, "params": 71.3, "flops": 189.9},
    },
    "YOLOv6-3.0": {
        "n": {"size": 640, "map": 37.5, "cpu": "", "t4": 1.17, "params": 4.7, "flops": 11.4},
        "s": {"size": 640, "map": 45.0, "cpu": "", "t4": 2.66, "params": 18.5, "flops": 45.3},
        "m": {"size": 640, "map": 50.0, "cpu": "", "t4": 5.28, "params": 34.9, "flops": 85.8},
        "l": {"size": 640, "map": 52.8, "cpu": "", "t4": 8.95, "params": 59.6, "flops": 150.7},
    },
    "YOLOv5": {
        "n": {"size": 640, "map": 28.0, "cpu": 73.6, "t4": 1.12, "params": 2.6, "flops": 7.7},
        "s": {"size": 640, "map": 37.4, "cpu": 120.7, "t4": 1.92, "params": 9.1, "flops": 24.0},
        "m": {"size": 640, "map": 45.4, "cpu": 233.9, "t4": 4.03, "params": 25.1, "flops": 64.2},
        "l": {"size": 640, "map": 49.0, "cpu": 408.4, "t4": 6.61, "params": 53.2, "flops": 135.0},
        "x": {"size": 640, "map": 50.7, "cpu": 763.2, "t4": 11.89, "params": 97.2, "flops": 246.4},
    },
    "PP-YOLOE+": {
        "t": {"size": 640, "map": 39.9, "cpu": "", "t4": 2.84, "params": 4.85, "flops": 19.15},
        "s": {"size": 640, "map": 43.7, "cpu": "", "t4": 2.62, "params": 7.93, "flops": 17.36},
        "m": {"size": 640, "map": 49.8, "cpu": "", "t4": 5.56, "params": 23.43, "flops": 49.91},
        "l": {"size": 640, "map": 52.9, "cpu": "", "t4": 8.36, "params": 52.20, "flops": 110.07},
        "x": {"size": 640, "map": 54.7, "cpu": "", "t4": 14.3, "params": 98.42, "flops": 206.59},
    },
    "DAMO-YOLO": {
        "t": {"size": 640, "map": 42.0, "cpu": "", "t4": 2.32, "params": 8.5, "flops": 18.1},
        "s": {"size": 640, "map": 46.0, "cpu": "", "t4": 3.45, "params": 16.3, "flops": 37.8},
        "m": {"size": 640, "map": 49.2, "cpu": "", "t4": 5.09, "params": 28.2, "flops": 61.8},
        "l": {"size": 640, "map": 50.8, "cpu": "", "t4": 7.18, "params": 42.1, "flops": 97.3},
    },
    "YOLOX": {
        "nano": {"size": 416, "map": 25.8, "cpu": "", "t4": "", "params": 0.91, "flops": 1.08},
        "tiny": {"size": 416, "map": 32.8, "cpu": "", "t4": "", "params": 5.06, "flops": 6.45},
        "s": {"size": 640, "map": 40.5, "cpu": "", "t4": 2.56, "params": 9.0, "flops": 26.8},
        "m": {"size": 640, "map": 46.9, "cpu": "", "t4": 5.43, "params": 25.3, "flops": 73.8},
        "l": {"size": 640, "map": 49.7, "cpu": "", "t4": 9.04, "params": 54.2, "flops": 155.6},
        "x": {"size": 640, "map": 51.1, "cpu": "", "t4": 16.1, "params": 99.1, "flops": 281.9},
    },
    "RTDETRv2": {
        "s": {"size": 640, "map": 48.1, "cpu": "", "t4": 5.03, "params": 20, "flops": 60},
        "m": {"size": 640, "map": 51.9, "cpu": "", "t4": 7.51, "params": 36, "flops": 100},
        "l": {"size": 640, "map": 53.4, "cpu": "", "t4": 9.76, "params": 42, "flops": 136},
        "x": {"size": 640, "map": 54.3, "cpu": "", "t4": 15.03, "params": 76, "flops": 259},
    },
    "EfficientDet": {
        "d0": {"size": 640, "map": 34.6, "cpu": 10.2, "t4": 3.92, "params": 3.9, "flops": 2.54},
        "d1": {"size": 640, "map": 40.5, "cpu": 13.5, "t4": 7.31, "params": 6.6, "flops": 6.10},
        "d2": {"size": 640, "map": 43.0, "cpu": 17.7, "t4": 10.92, "params": 8.1, "flops": 11.0},
        "d3": {"size": 640, "map": 47.5, "cpu": 28.0, "t4": 19.59, "params": 12.0, "flops": 24.9},
        "d4": {"size": 640, "map": 49.7, "cpu": 42.8, "t4": 33.55, "params": 20.7, "flops": 55.2},
        "d5": {"size": 640, "map": 51.5, "cpu": 72.5, "t4": 67.86, "params": 33.7, "flops": 130.0},
        "d6": {"size": 640, "map": 52.6, "cpu": 92.8, "t4": 89.29, "params": 51.9, "flops": 226.0},
        "d7": {"size": 640, "map": 53.7, "cpu": 122.0, "t4": 128.07, "params": 51.9, "flops": 325.0},
    },
}

if __name__ == "__main__":
    import json

    with open("model_data.json", "w") as f:
        json.dump(data, f)
