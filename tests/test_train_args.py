import numpy as np

step_size = 10

def generate():
  lr0 =  np.logspace(np.log10(1e-5), np.log10(1e-1), step_size)
  lrf = np.logspace(np.log10(1e-5), np.log10(1e-1), step_size)
  momentum = np.linspace(0.7, 0.99, step_size)
  weight_decay = weight_decay_values = np.logspace(np.log10(1e-5), np.log10(1e-2), step_size)
  warmup_epochs= np.linspace(0, 10, step_size)
  warmup_momentum = np.linspace(0.5, 0.99, step_size)
  box=np.linspace(0, 10, step_size)
  cls = np.linspace(0, 0.99, step_size)
  dfl =  np.linspace(0.5, 3, step_size)
  pose =  np.linspace(5.0, 18.0, step_size)
  kobj = np.linspace(0.2, 3.0, step_size)
  label_smoothing = np.linspace(0.0, 1.0, step_size) 

  nbs = np.linspace(32, 512, step_size, dtype=int)


  hsv_h =  np.linspace(0.0, 0.015, step_size)
  hsv_s =  np.linspace(0.0, 0.7, step_size)
  hsv_v =  np.linspace(0.0, 0.4, step_size)
  degrees =  np.linspace(0.0, 180.0, step_size) 
  translate =  np.linspace(0.0, 0.1, step_size)
  scale =  np.linspace(0.0, 0.5, step_size)
  shear =  np.linspace(0.0, 45.0, step_size)
  perspective =  np.linspace(0.0, 0.001, step_size)
  flipud =  np.linspace(0.0, 1.0, step_size)
  fliplr =  np.linspace(0.0, 1.0, step_size)
  bgr =  np.linspace(0.0, 1.0, step_size)
  mosaic =  np.linspace(0.0, 1.0, step_size)
  mixup =  np.linspace(0.0, 1.0, step_size)
  copy_paste =  np.linspace(0.0, 1.0, step_size)


  num_values = np.linspace(0, 20, step_size, dtype=int)


  auto_augment = ["randaugment", "autoaugment", "augmix"]

  erasing = np.linspace(0.0, 0.9, step_size)
  crop_fraction = np.linspace(0.1, 1.0, step_size)
  return locals()

kwargs = generate()