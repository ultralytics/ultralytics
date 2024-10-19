import itertools

import numpy as np

step_size = 2

augment_params = {
    "hsv_h": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "hsv_s": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "hsv_v": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "degrees": {"from-limit": -180, "to-limit": 180, "function": np.linspace},
    "translate": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "scale": {"from-limit": 0.0, "to-limit": 0.5, "function": np.linspace},
    "shear": {"from-limit": -180, "to-limit": 180, "function": np.linspace},
    "perspective": {"from-limit": 0.0, "to-limit": 0.001, "function": np.linspace},
    "flipud": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "fliplr": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "bgr": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "mosaic": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "mixup": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "copy_paste": {"from-limit": 0.0, "to-limit": 1.0, "function": np.linspace},
    "auto_augment": ["randaugment", "autoaugment", "augmix"],
    "erasing": {"from-limit": 0.0, "to-limit": 0.9, "function": np.linspace},
    "crop_fraction": {"from-limit": 0.1, "to-limit": 1.0, "function": np.linspace},
}

cases = dict()


for key, values in augment_params.items():
    if isinstance(values, list):
        cases[key] = values
    else:
        cases[key] = values["function"](values["from-limit"], values["to-limit"], step_size).tolist()


keys = cases.keys()
values = cases.values()


combinations = itertools.product(*values)

permutations = []
for combination in combinations:
    result = dict(zip(keys, combination))
    permutations.append(result)


def test_permutation(permutation):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(data="coco8.yaml", epochs=2)


for permutation in permutations:
    test_permutation(permutation)
