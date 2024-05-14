from ultralytics import YOLO
from PIL import Image

if __name__=="__main__":
    model = YOLO('ultralytics/cfg/models/v8/yolov8x-obbwithkpt.yaml',task = 'obbwithkpt')
    model.train(data = 'datasets/bj_obbwithkpt.yaml',epochs=500,batch=16)
    #model.val(data = 'datasets\\bj_obbwithkpt.yaml')
    '''results = model('/aicc/userData/sdzl/AI_meter_src/AI_meter_src/sub_meter_api/sub_meter_cls/ultralytics-main/datasets/yolo_labels/images/train')
    for idx,r in enumerate(results):
        im_array = r.plot(line_width=2)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        im.save(f'plotted/results{idx}.jpg')  # save image'''
