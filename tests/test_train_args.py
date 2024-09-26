import numpy as np

step_size = 10

augmenentation = {
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
  "crop_fraction": {"from-limit": 0.1, "to-limit": 1.0, "function": np.linspace}
}

