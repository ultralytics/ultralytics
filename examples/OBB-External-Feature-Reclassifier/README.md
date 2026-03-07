# OBB External Feature Reclassifier

Bolt-on classification improvement for YOLO OBB models using a frozen, domain-agnostic spatial feature encoder. The encoder has never been trained on aerial imagery or any of the DOTA object categories. It captures spatial structure that detection models systematically discard, providing an orthogonal information source that improves classification without modifying the detector.

Detection is completely untouched. The same model, same weights, and same detections are used. The only thing that changes is how those detections are classified.

## How It Works

1. Run YOLO OBB detection normally
2. Crop each detection using the axis-aligned enclosure (`result.obb.xyxy`)
3. Resize to 128x128 grayscale and send to the encoder API
4. Concatenate the 920-dim feature vector with YOLO's class output
5. Train a lightweight LightGBM classifier on the concatenated features
6. Evaluate under 5-fold scene-level cross-validation

## Results on DOTA v1.0

**YOLOv8l-OBB** (50,348 matched detections, 458 original scenes):

|             | YOLO Direct | Bolt-On |
| ----------- | ----------- | ------- |
| Weighted F1 | 0.9925      | 0.9929  |
| Macro F1    | 0.9826      | 0.9827  |
| helicopter  | 0.502       | 0.916   |
| plane       | 0.976       | 0.998   |

**YOLO26l-OBB** (49,414 matched detections, 458 original scenes):

|             | YOLO Direct | Bolt-On |
| ----------- | ----------- | ------- |
| Weighted F1 | 0.9943      | 0.9947  |
| Macro F1    | 0.9891      | 0.9899  |

No class degraded on either model across all 15 categories. When given access to YOLOv8l-OBB's full 15-dim pre-NMS class distribution (not just the argmax label), the encoder still reduced classification error by 18.4%.

## Same Encoder on Other Benchmarks

The identical frozen encoder improves classification across domains and modalities it has never seen:

| Benchmark  | Domain              | Modality      | Baseline Model                    | Error Reduction |
| ---------- | ------------------- | ------------- | --------------------------------- | --------------- |
| xView3     | Maritime vessels    | C-band SAR    | 1st-place CircleNet               | 4.6%            |
| DOTA       | Aerial objects      | HR aerial     | YOLOv8l-OBB                       | 8.9%            |
| EuroSAT    | Land cover          | Multispectral | Fine-tuned ResNet-50              | 10.6%           |
| SpaceNet 6 | Building extraction | X-band SAR    | 1st-place competition winner      | 14.1%           |
| RarePlanes | Aircraft roles      | VHR satellite | Faster R-CNN (CosmiQ Works / IQT) | 39.5%           |
| xView2     | Building damage     | RGB optical   | 3rd-place BloodAxe ensemble       | 40.7%           |

## Try It on Your Own Task

The encoder is domain-agnostic. If you have a classification task where a model is producing errors, the encoder may capture spatial structure that your model misses. We encourage testing on diverse and unconventional problems beyond aerial imagery:

- **Medical imaging:** histopathology patches, radiology crops, cell classification
- **Industrial inspection:** defect classification on PCBs, welds, textiles, castings
- **Agriculture:** crop disease classification, pest identification from drone imagery
- **Autonomous driving:** traffic sign subtype classification, vehicle make/model
- **Retail / logistics:** product classification, package damage assessment
- **Scientific imaging:** mineral thin sections, satellite spectral classification, microscopy
- **Any detection pipeline where classification accuracy matters more than recall**

The pattern works with a wide variety of detectors, not just YOLO. Crop your detections, send them to the encoder, concatenate, and classify.

## Setup

```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics/examples/OBB-External-Feature-Reclassifier
pip install ultralytics==8.4.21 lightgbm scikit-learn numpy opencv-python requests pillow tqdm
export AUTHORIZE_EARTH_API_KEY=sk_ae_1289e06722678304582fcde59ba5573aab73f32d91601d22022f2890ed4a7833
```

The public evaluation key above works with no signup. Rate-limited to 3,000 requests/hour
per IP and 15,000/hour globally. Without the key set the encoding step returns zero vectors
and the script still runs but classification will not improve.

## Data

Download the DOTA v1.0 validation set from https://captain-whu.github.io/DOTA/. You need both the images and labelTxt directories.

## Usage

### Step 1: Tile DOTA images into 1024x1024 patches

DOTA images are very large (up to 20k x 20k pixels). This splits them into tiles that YOLO can process.

```bash
python obb_feature_reclassifier.py tile \
  --images path/to/val/images \
  --labels path/to/val/labelTxt \
  --output ./tiled
```

### Step 2: Run the benchmark

```bash
# YOLOv8 (default)
python obb_feature_reclassifier.py bench \
  --images ./tiled/images \
  --labels ./tiled/labels

# YOLO26
python obb_feature_reclassifier.py bench \
  --images ./tiled/images \
  --labels ./tiled/labels \
  --model yolo26l-obb.pt

# Full pre-NMS class scores (YOLOv8 only — see note below)
python obb_feature_reclassifier.py bench \
  --images ./tiled/images \
  --labels ./tiled/labels \
  --full-scores

# Skip detection on subsequent runs (uses cached results)
python obb_feature_reclassifier.py bench \
  --images ./tiled/images \
  --labels ./tiled/labels \
  --skip-detection
```

### Note on `--full-scores`

This extracts the raw 15-dim sigmoid class scores from YOLOv8's pre-NMS output and matches them back to post-NMS detections, giving the baseline a stronger representation than the default one-hot encoding. YOLO26's end-to-end NMS-free architecture uses a one-to-one prediction head, so intermediate activations are not interpretable class distributions. Use the default one-hot mode for YOLO26.

## API Key

The script uses a public evaluation key that works immediately with no signup. Rate limits are 3,000 requests/hour per IP address and 15,000 requests/hour globally. For higher limits or dedicated access, contact jackk@authorize.earth.

## Expected Output

```
5-fold scene-level cross-validation (50348 detections)
=================================================================
  Fold 1: bolt-on weighted F1 = 0.9930
  Fold 2: bolt-on weighted F1 = 0.9927
  Fold 3: bolt-on weighted F1 = 0.9931
  Fold 4: bolt-on weighted F1 = 0.9928
  Fold 5: bolt-on weighted F1 = 0.9929

  yolov8l-obb Direct   Bolt-On
  Weighted F1:  0.9925              0.9929
  Macro F1:     0.9826              0.9827
  Error reduction vs yolov8l-obb direct: 8.9%

  Class                  Direct     Bolt-On    Delta      n
  ──────────────────────────────────────────────────────────
  helicopter             0.502      0.916      +0.414     73
  plane                  0.976      0.998      +0.022     4858
  basketball-court       0.931      0.947      +0.015     222
  ...
```

## Links

- Encoder API docs and live demo: https://authorize.earth/r&d/spatial
- Contact: jackk@authorize.earth
- Discussion: [ultralytics/ultralytics#23821](https://github.com/ultralytics/ultralytics/discussions/23821)

## Contributor

[Jack Kowalik](https://github.com/jackkowalik) — [Authorize Earth](https://authorize.earth), Troy, MI
