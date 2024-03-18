# from ultralytics import YOLO
# import cv2
# import os


# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('ultralytics\\runs\\detect\\train5\\weights\\best.pt')

# results = model.predict(source="C:\\Users\\leejm\\Desktop\\project\\2024_0118_selaer2yoloset\\train_valid_dataset\\pe_module_24_1_26\\images\\test", show=False)

# for idx, frame in enumerate(results):
#     img_path = os.path.join("C:\\Users\\leejm\\Desktop\\project\\2024_0118_selaer2yoloset\\ultralytics\\result", 'img_'+str(idx)+'.png')
#     cv2.imwrite(img_path, frame.plot(font_size=0.1))
