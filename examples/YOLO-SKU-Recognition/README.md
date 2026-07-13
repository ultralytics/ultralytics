# SKU Recognition with YOLO Detection + ReID Retrieval

This example implements open-vocabulary retail SKU recognition by pairing an [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26) single-class product detector with a [YOLO ReID](https://docs.ultralytics.com/tasks/reid) embedding model. Instead of a fixed classifier head, each detected package is matched against a **folder-per-SKU gallery** of reference images by nearest-neighbor cosine similarity. Adding a new SKU means dropping a folder of reference images into the gallery, with no detector or embedding retraining.

This is the practical alternative to a closed-set classifier when the catalog changes often (new products, seasonal items, near-identical package variants), which is common for retail shelf and cigarette-pack recognition.

## How It Works

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

## Setup

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

The default ReID model is pretrained on RP2K product crops, so it already embeds retail packaging well and is a strong seed for your own catalog. Arrange your labeled crops into folder-per-SKU `train/query/gallery` splits (the same layout the gallery uses):

```yaml
# your-skus.yaml
path: your_skus # dataset root, relative to your Ultralytics datasets dir
train: train # folder-per-SKU training crops
val: query # held-out query crops, one or more per SKU
gallery: gallery # reference crops each query is matched against
nc: 1200 # number of training SKU identities
```

```bash
yolo reid train data=your-skus.yaml imgsz=256 \
  model=ul://fatih-enterprise/yolo26-reid-sku-feature-extraction/yolo26l-reid-rp2k-pretrain
```

See the [custom ReID dataset guide](https://docs.ultralytics.com/guides/reid-custom-dataset/) for the folder-per-SKU dataset schema, and the [ReID fine-tuning guide](https://docs.ultralytics.com/guides/reid-finetuning/) for image-size and model-size guidance and mAP / Rank-1 evaluation. This example trains at `imgsz=256` to match the deployed RP2K seed and the on-device export, while the guides default to 448 for best standalone accuracy.

## Run

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

## Building a gallery from RP2K

[RP2K](https://www.pinlandata.com/rp2k_dataset) is a public retail-product dataset of about 384K **already-cropped** single-product images across roughly 2388 SKUs (folder-per-SKU). It has no bounding boxes, so it cannot train the detector, but it is ideal for the gallery and is what the published ReID weights were trained on. After downloading via the `rp2k-full-closedset.yaml` recipe, any subset of `datasets/rp2k_full_closedset/train/<sku>/` already has the folder-per-SKU layout this example expects, so you can point `--gallery` straight at it to prototype before collecting your own reference crops.

## On-device deployment (CoreML / TFLite)

Both models export to CoreML and TFLite/LiteRT, so the whole detect, embed, and retrieve pipeline can run on a phone. The ReID model emits an L2-normalized `(1, 512)` embedding. Pass a local `.pt` or a `ul://` platform id:

```bash
# CoreML for iOS. Use format=tflite for TFLite/LiteRT on Android.
yolo export model=yolo26l-sku.pt format=coreml imgsz=640  # detector -> yolo26l-sku.mlpackage
yolo export model=yolo26l-reid.pt format=coreml imgsz=256 # reid     -> yolo26l-reid.mlpackage
```

CoreML export is lightweight, TFLite/LiteRT pulls a larger TensorFlow and ONNX toolchain on first export.

On device only the models are heavy. The gallery, retrieval, and vote are a few lines of plain code. This Swift mirrors the Python `assign()`:

```swift
import Foundation

/// In-memory reference gallery: one SKU label per L2-normalized (512) embedding from the ReID CoreML model.
struct SKUGallery {
    let labels: [String]
    let embeddings: [[Float]]  // built once by running the ReID model over each reference crop

    /// Assign an SKU to a query embedding by a top-k similarity-weighted vote.
    func assign(_ query: [Float], topK: Int = 5, simThresh: Float = 0.5) -> (sku: String, confidence: Float) {
        // cosine similarity == dot product, because every vector is already unit length
        let scores = embeddings.map { dot($0, query) }
        let neighbors = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(topK)

        var total = [String: Float](), count = [String: Int]()
        for i in neighbors {
            total[labels[i], default: 0] += scores[i]
            count[labels[i], default: 0] += 1
        }
        guard let best = total.max(by: { $0.value < $1.value })?.key else { return ("unknown", 0) }
        let confidence = total[best]! / Float(count[best]!)  // mean similarity of the winning SKU's neighbors
        return (confidence >= simThresh ? best : "unknown", confidence)
    }
}

/// Dot product. For large galleries use Accelerate's vDSP_dotpr instead of this loop.
func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
```

Wire it up like the Python script: run the CoreML detector on the frame, crop each box, run the ReID model on each crop to get its `(512)` embedding, then call `gallery.assign(embedding)`. Adding an SKU is still just appending its reference embeddings to `labels` and `embeddings`, with no retraining.

## Labeling Tip

For both detector training and gallery crops, each box should contain a **single** package instance. A box that mixes two packages produces a blended embedding that hurts both detector recall and retrieval accuracy, which matters most for near-identical variants where small text or color differences carry the identity.
