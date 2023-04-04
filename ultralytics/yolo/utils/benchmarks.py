# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Benchmark a YOLO model formats for speed and accuracy

Usage:
    from ultralytics.yolo.utils.benchmarks import run_benchmarks
    run_benchmarks(model='yolov8n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlmodel
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/
"""

import platform
import time
from pathlib import Path

from ultralytics import YOLO
from ultralytics.yolo.engine.exporter import export_formats
from ultralytics.yolo.utils import LINUX, LOGGER, MACOS, ROOT, SETTINGS
from ultralytics.yolo.utils.checks import check_yolo
from ultralytics.yolo.utils.downloads import download
from ultralytics.yolo.utils.files import file_size
from ultralytics.yolo.utils.torch_utils import select_device


def benchmark(model=Path(SETTINGS['weights_dir']) / 'yolov8n.pt', imgsz=160, half=False, device='cpu', hard_fail=False):
    import pandas as pd
    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():  # index, (name, format, suffix, CPU, GPU)
        emoji, filename = '❌', None  # export defaults
        try:
            assert i != 9 or LINUX, 'Edge TPU export only supported on Linux'
            if i == 10:
                assert MACOS or LINUX, 'TF.js export only supported on macOS and Linux'
            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                assert gpu, 'inference not supported on GPU'

            # Export
            if format == '-':
                filename = model.ckpt_path or model.cfg
                export = model  # PyTorch format
            else:
                filename = model.export(imgsz=imgsz, format=format, half=half, device=device)  # all others
                export = YOLO(filename, task=model.task)
                assert suffix in str(filename), 'export failed'
            emoji = '❎'  # indicates export succeeded

            # Predict
            assert i not in (9, 10), 'inference not supported'  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == 'Darwin', 'inference only supported on macOS>=10.13'  # CoreML
            if not (ROOT / 'assets/bus.jpg').exists():
                download(url='https://ultralytics.com/images/bus.jpg', dir=ROOT / 'assets')
            export.predict(ROOT / 'assets/bus.jpg', imgsz=imgsz, device=device, half=half)

            # Validate
            if model.task == 'detect':
                data, key = 'coco128.yaml', 'metrics/mAP50-95(B)'
            elif model.task == 'segment':
                data, key = 'coco128-seg.yaml', 'metrics/mAP50-95(M)'
            elif model.task == 'classify':
                data, key = 'imagenet100', 'metrics/accuracy_top5'

            results = export.val(data=data, batch=1, imgsz=imgsz, plots=False, device=device, half=half, verbose=False)
            metric, speed = results.results_dict[key], results.speed['inference']
            y.append([name, '✅', round(file_size(filename), 1), round(metric, 4), round(speed, 2)])
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f'Benchmark hard_fail for {name}: {e}'
            LOGGER.warning(f'ERROR ❌️ Benchmark failure for {name}: {e}')
            y.append([name, emoji, round(file_size(filename), 1), None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(y, columns=['Format', 'Status❔', 'Size (MB)', key, 'Inference time (ms/im)'])

    name = Path(model.ckpt_path).name
    s = f'\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n'
    LOGGER.info(s)
    with open('benchmarks.log', 'a', errors='ignore', encoding='utf-8') as f:
        f.write(s)

    if hard_fail and isinstance(hard_fail, float):
        metrics = df[key].array  # values to compare to floor
        floor = hard_fail  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f'HARD FAIL: one or more metric(s) < floor {floor}'

    return df


if __name__ == '__main__':
    benchmark()
