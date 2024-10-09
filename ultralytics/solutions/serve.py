import base64
import io
import litserve as ls
import numpy as np
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics import YOLO
from pydantic import BaseModel  
from typing import List, Dict

class UltralyticsRequest(BaseModel):  
    image: str

class UltralyticsResponse(BaseModel):  
    status: str = "success"  
    results: List[List[Dict]]

class YOLOServe(ls.LitAPI):
    def __init__(self, model) -> None:
        self.model = model
        super().__init__()
        
    def setup(self, device):
        self.model = YOLO(model=self.model)

    def decode_request(self, request: UltralyticsRequest):
        base64_str = request.image
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        image_array = np.array(image)
        return image_array
    
    def encode_response(self, results: Results) -> UltralyticsResponse:  
        image_results = []  
        for result in results:
            image_results.append(result.to_json())  
        return UltralyticsResponse(results=[r.to_dict() for r in results])

    def predict(self, x):
        result = self.model(x)
        return result


def run(args):
    if "model" in args:
        model = args["model"]
    else:
        model = "yolo11n.pt"
    server = ls.LitServer(YOLOServe(model), accelerator="auto", max_batch_size=1, batch_timeout=0.05)
    server.run(port=8000)
