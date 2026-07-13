# SKU Recognition with YOLO Detection + ReID Retrieval

This example implements open-vocabulary retail SKU recognition by pairing an [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26) single-class product detector with a [YOLO ReID](https://docs.ultralytics.com/tasks/reid) embedding model. Instead of a fixed classifier head, each detected package is matched against a **folder-per-SKU gallery** of reference images by nearest-neighbor cosine similarity. Adding a new SKU means dropping a folder of reference images into the gallery, with no detector or embedding retraining.

This is the practical alternative to a closed-set classifier when the catalog changes often (new products, seasonal items, near-identical package variants), which is common for retail shelf and cigarette-pack recognition.

## 🧭 How It Works

```
shelf.jpg ─► [YOLO detector] ─► product boxes ─► crops ─┐
                                                        ▼
              gallery/<sku>/*.jpg ─► [YOLO ReID] ─► embeddings ─► L2-normalized (N, 512)
                                                        ▼
                       each crop embedding ─► top-k cosine vote over the gallery ─► SKU name + confidence
```

1. **Detect** package instances with a single-class SKU detector (`0: object`).
2. **Crop** each detection from the source image.
3. **Embed** every crop and every gallery image with the ReID model (L2-normalized embeddings).
4. **Retrieve** the top-k nearest gallery neighbors per crop with an in-memory NumPy dot product and assign the SKU by a similarity-weighted vote. Crops whose best confidence falls below `--sim-thresh` are labeled `unknown`.

Retrieval is a plain in-RAM matrix multiply, so a few hundred SKUs are trivial: 400 SKUs x 20 reference crops x 512-d float32 is about 16 MB. No vector database is needed at this scale.

## ⚙️ Setup

```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics/examples/YOLO-SKU-Recognition
pip install ultralytics
```

The example needs two model weights. Both default to published models on the [Ultralytics Platform](https://platform.ultralytics.com) that auto-download on first run, so you can try it with no weights of your own:

| Model        | What                                        | Default (auto-downloads)                                                                               |
| ------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `--detector` | single-class product detector (`0: object`) | `ul://fatih-enterprise/yolo26-sku-detection/yolo26l-sku-detector-sku-110k`, trained on SKU-110K        |
| `--reid`     | YOLO ReID embedding model                   | `ul://fatih-enterprise/yolo26-reid-sku-feature-extraction/yolo26l-reid-rp2k-pretrain`, RP2K-pretrained |

Platform downloads need an API key. Set it once, or pass your own local `.pt` to `--detector`/`--reid` to skip the platform entirely:

```bash
yolo settings api_key=YOUR_API_KEY # or export ULTRALYTICS_API_KEY=YOUR_API_KEY, key from https://platform.ultralytics.com/settings
```

### Train the SKU detector

[SKU-110K](https://docs.ultralytics.com/datasets/detect/sku-110k) is the standard dense retail-product detection set: one class (`object`), 8219 train / 588 val / 2936 test shelf photos with about 1.7M product boxes. It is class-agnostic, which is exactly what the detect-then-embed split wants, the detector only has to localize packages, and the ReID model decides identity.

```bash
yolo detect train data=SKU-110K.yaml model=yolo26l.pt imgsz=640 epochs=100
```

`model=yolo26l.pt` starts from COCO weights. If you have a domain-pretrained YOLO26l backbone (for example a product/retail-pretrained checkpoint), pass it as `model=<backbone>.pt` instead for a stronger starting point on dense shelf imagery.

### Fine-tune the ReID model on your SKUs

The published `yolo26{l,x}-reid.pt` weights are pretrained on RP2K product crops. To specialize on your own catalog, arrange your labeled SKU crops into ReID `train/query/gallery` splits (one folder per SKU identity) and fine-tune:

```bash
yolo reid train data=your-skus.yaml model=yolo26l-reid.pt imgsz=256
```

## 🚀 Run

Arrange a gallery where each immediate subfolder is one SKU holding a handful of reference crops:

```
gallery/
├── marlboro_red/      img0.jpg img1.jpg ...   (10-20 reference crops)
├── marlboro_gold/     img0.jpg img1.jpg ...
└── camel_blue/        img0.jpg img1.jpg ...
```

With no `--detector`/`--reid`, the platform defaults are used, so a gallery and a source image are enough:

```bash
python sku_recognition.py --gallery gallery/ --source shelf.jpg
```

To use your own weights, pass a local `.pt` or another platform id/url:

```bash
python sku_recognition.py --detector yolo26l-sku.pt --reid yolo26l-reid.pt --gallery gallery/ --source shelf.jpg
```

The script writes `shelf_sku.jpg` with each detected package boxed and labeled `SKU-name confidence`, and logs the same per box.

### Useful options

| Flag           | Default | Description                                                          |
| -------------- | ------- | -------------------------------------------------------------------- |
| `--imgsz`      | `256`   | ReID embedding image size (match how the ReID model was trained)     |
| `--det-imgsz`  | `640`   | detector image size                                                  |
| `--conf`       | `0.25`  | detector confidence threshold                                        |
| `--topk`       | `5`     | gallery neighbors per crop for the vote                              |
| `--sim-thresh` | `0.5`   | minimum confidence to accept a SKU, else `unknown`                   |
| `--cache`      | `None`  | `.pt` gallery-embedding cache, reused when gallery/model/imgsz match |
| `--device`     | `None`  | inference device, e.g. `0` or `cpu`                                  |

## 🖼️ Building a gallery from RP2K

[RP2K](https://www.pinlandata.com/rp2k_dataset) is a public retail-product dataset of about 384K **already-cropped** single-product images across roughly 2388 SKUs (folder-per-SKU). It has no bounding boxes, so it cannot train the detector, but it is ideal for the gallery and is what the published ReID weights were trained on. After downloading via the `rp2k-full-closedset.yaml` recipe, any subset of `datasets/rp2k_full_closedset/train/<sku>/` already has the folder-per-SKU layout this example expects, so you can point `--gallery` straight at it to prototype before collecting your own reference crops.

## 📦 Deployment

Both ReID backbones export to CoreML and TFLite/LiteRT (verified end-to-end, each emits an L2-normalized `(1, 512)` embedding), so on-device retrieval on iOS and Android is feasible: run the exported embedding model, then do the same NumPy top-k against a locally stored gallery matrix.

## 💡 Labeling Tip

For both detector training and gallery crops, each box should contain a **single** package instance. A box that mixes two packages produces a blended embedding that hurts both detector recall and retrieval accuracy, which matters most for near-identical variants where small text or color differences carry the identity.
