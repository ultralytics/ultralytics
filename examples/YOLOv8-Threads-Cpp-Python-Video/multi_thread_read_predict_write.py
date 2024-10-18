#############
###ZouJiu
###20240421
###1069679911@qq.com
# https://zoujiu.blog.csdn.net/
# https://zhihu.com/people/zoujiu1
# https://github.com/ZouJiu1
#############

from queue import Queue
from threading import Thread

import cv2

from ultralytics import YOLO


def predict_image(model, batch):
    """Get a frame and predict it, then push result to another queue."""
    images = []
    while True:
        image = 0
        while len(images) < batch:
            image = image_que.get()
            if isinstance(image, int):
                image_que.task_done()
                break
            images.append(image)
        if len(images) == 0:
            break
        # result = model.predict(images, batch = batch)
        result = model.track(images, batch=batch, persist=True)
        result_que.put(result)
        for _ in range(len(images)):
            image_que.task_done()
        images = []
        if isinstance(image, int):
            break
    result_que.put(0)


def get_image(pth):
    """Read video frame and put it to a queue."""
    cap = cv2.VideoCapture(pth)
    if not cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        image_que.put(0)
        informa_que.put((-1, -1, -1))
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    informa_que.put((fps, int(frame_width), int(frame_height)))
    success, frame = cap.read()
    while success:
        image_que.put(frame)
        success, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    image_que.put(0)


def write_video(write_path):
    """Get a result frame and write it to a video."""
    fps, frame_width, frame_height = informa_que.get()
    if fps < 0 and frame_width < 0 and frame_height < 0:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_writer = cv2.VideoWriter(write_path, fourcc, fps, (frame_width, frame_height))
    informa_que.task_done()
    informa_que.join()

    while True:
        result = result_que.get()
        if isinstance(result, int):
            result_que.task_done()
            video_writer.release()
            cv2.destroyAllWindows()
            break
        for ind, _ in enumerate(result):
            frame = result[ind].plot()
            # frame = result[ind].orig_img
            video_writer.write(frame)
        result_que.task_done()


if __name__ == "__main__":
    # model = YOLO('yolov8n.pt')
    # model = YOLO('yolov8n-seg.pt')
    model = YOLO("yolov8n-pose.pt")
    batch_size = 3
    Image_in_queue_maxsize = 900
    Result_in_queue_maxsize = 900
    video_path = r"C:\Users\10696\Desktop\CV\MOT16-06-raw.mp4"
    outpath = video_path.replace(".mp4", "_output.mp4")
    # outpath = video_path.replace('.mp4', "_output.avi")
    image_que = Queue(maxsize=Image_in_queue_maxsize)
    result_que = Queue(maxsize=Result_in_queue_maxsize)
    informa_que = Queue(1)
    t0 = Thread(target=get_image, args=(video_path,))
    t1 = Thread(target=predict_image, args=(model, batch_size))
    t2 = Thread(target=write_video, args=(outpath,))
    t0.setDaemon(True)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t0.start()
    t1.start()
    t2.start()
    t0.join()
    t1.join()
    t2.join()
    image_que.join()
    result_que.join()
