"""Time one engine with Esat's timer: Ultralytics ProfileModels at stock defaults.

Called once per process from a shell loop with a cooldown between, mirroring how he runs
working_dir/profile_onnx.py. Only profile_tensorrt_model is invoked, since the ONNX Runtime stage that
follows it in run() is CPU work that cannot affect the TensorRT number.
"""

import sys

sys.path.insert(0, "/root/autodl-tmp/code/ultravit-lane-b")

import tensorrt as trt

from ultralytics.utils.benchmarks import ProfileModels

engine = sys.argv[1]
pm = ProfileModels(paths=[engine], imgsz=640)
mean, std = pm.profile_tensorrt_model(engine)
print(f"RESULT\t{engine.split('/')[-1].split('_op17')[0]}\t{mean:.4f}\t{std:.4f}\ttrt{trt.__version__}", flush=True)
