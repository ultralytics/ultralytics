import base64
import io
from typing import Dict, List

import litserve as ls
import numpy as np
from PIL import Image
from pydantic import BaseModel

from ultralytics import YOLO
from ultralytics.engine.results import Results


class UltralyticsRequest(BaseModel):
    image: str


class UltralyticsResponse(BaseModel):
    status: str = "success"
    results: List[List[Dict]]


class YOLOServe(ls.LitAPI):
    def __init__(self, model) -> None:
        """
        Litserve API for YOLO model:
        Call order is:
            setup -> batch -> decode_request -> predict -> unbatch -> encode_response.

        Args:
            model (str): Model name to use
        """
        self.model_name = model
        super().__init__()

    def setup(self, device):
        self.model = YOLO(model=self.model_name)

    def batch(self, inputs):
        return list(inputs)

    def decode_request(self, request: UltralyticsRequest):
        base64_str = request.image
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        image_array = np.array(image)
        return image_array

    def predict(self, x):
        result = self.model(x)
        return result

    def unbatch(self, output):
        return list(output)

    def encode_response(self, result: List[Results]) -> UltralyticsResponse:
        return UltralyticsResponse(results=[r.to_dict() for r in result])


def run(args):
    if "model" in args:
        model = args["model"]
    else:
        model = "yolo11n.pt"
    server = ls.LitServer(YOLOServe(model), accelerator="auto", max_batch_size=15, batch_timeout=0.05)
    server.run(port=8000)
