from ultralytics.utils.tlc.detect.model import TLCYOLO  # noqa: E402
from ultralytics.utils.tlc.detect.settings import Settings

model = TLCYOLO("yolov8n.pt")  # initialize

# Set 3LC specific settings
settings = Settings(
    image_embeddings_dim=2,
    image_embeddings_reducer='umap',
    collection_epoch_start=0,
    collection_epoch_interval=2,
    conf_thres=0.2,
)

# Run training
results = model.train(
    data="coco128.yaml",
    device=0,
    epochs=5,
    batch=32,
    imgsz=320,
    workers=0,
    settings=settings,
)
