import base64
import io

import litserve as ls
import numpy as np
from PIL import Image

from ultralytics import YOLO


class YOLOServe(ls.LitAPI):
    def setup(self, device):
        self.model = YOLO("yolo11n.pt")

    def decode_request(self, request):
        base64_str = request["input"]
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        image_array = np.array(image)
        return image_array

    def predict(self, x):
        prediction = self.model(x)
        return prediction[0].to_json()


def run():
    server = ls.LitServer(YOLOServe(), accelerator="auto", max_batch_size=1)
    server.run(port=8000)
