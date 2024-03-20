from ultralytics.utils.tlc.detect.model import TLCYOLO
from ultralytics.utils.tlc.detect.settings import Settings
from ultralytics.utils.tlc.detect.utils import reduce_all_embeddings

splits = ("train", "val")
data="coco128.yaml"

model = TLCYOLO("yolov8l.pt")

settings = Settings(
    image_embeddings_dim=2,
    conf_thres=0.2,
)

for split in splits:
    results = model.val(data=data, split=split, batch=32, imgsz=320, device=0, workers=0, settings=settings)

# Reduce the embeddings
reduce_all_embeddings(
    data_file=data,
    by="val",
    method=settings.image_embeddings_reducer,
    n_components=settings.image_embeddings_dim
)