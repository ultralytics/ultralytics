import yaml
from ultralytics.models.yolo.detect import DetectionTrainer

if __name__ == '__main__':
  with open(r"params.yaml") as f:
      params = yaml.safe_load(f)

  args = dict(
      model='yolov8n.pt',
      data='/content/content/drone_dataset/data.yaml',
      imgsz=params['imgsz'],
      batch=params['batch'],
      epochs=params['epochs'],
      patience=params['patience'],
      optimizer=params['optimizer'],
      lr0=params['lr0'],
      seed=params['seed'],
      pretrained=params['pretrained'],
      name=params['name']
  )
  trainer = DetectionTrainer(overrides=args)
  trainer.train()