from ultralytics import YOLO
from PIL import Image

if __name__=="__main__":
    model = YOLO('runs/obbwithkpt/train29/weights/best.pt',task = 'obbwithkpt')
    #model.train(data = '/aicc/userData/sdzl/AI_meter_src/AI_meter_src/sub_meter_api/sub_meter_cls/yolov8-3-25/ultralytics-main/datasets/bj_obbwithkpt.yaml',epochs=200,batch=16)
    #model.val(data = 'datasets\\bj_obbwithkpt.yaml')
    '''results = model('datasets/yolo_labels/images/train')
    for idx,r in enumerate(results):
        im_array = r.plot(line_width=2)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        im.save(f'plotted/results{idx}.jpg')  # save image'''
    
    model.export(format = 'onnx')