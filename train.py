from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-RGBIR.yaml')
    model.train(data='ultralytics/cfg/datasets/LLVIP.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=64,
                device='0',
                optimizer='SGD',
                patience=0,
                amp=True, # Bypass the AMP check during RGB+IR training or set to False.
                project='runs/train',
                name='RGB-IR',
                )