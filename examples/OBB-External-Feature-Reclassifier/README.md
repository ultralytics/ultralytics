# OBB External Feature Reclassifier

Bolt-on classification improvement for YOLO OBB models using a frozen, domain-agnostic spatial feature encoder. The encoder has never been trained on aerial imagery or any of the DOTA object categories. It captures spatial structure that detection models systematically discard, providing an orthogonal information source that improves classification without modifying the detector.

Detection is completely untouched. The same model, same weights, and same detections are used. The only thing that changes is how those detections are classified.

Both YOLOv8l-OBB and YOLO26l-OBB are pretrained on DOTAv1 — the encoder improves classification even on the model's own training data.

## How It Works

1. Run YOLO OBB detection normally
2. Crop each detection using the axis-aligned enclosure (`result.obb.xyxy`)
3. Resize to 128x128 grayscale and send to the encoder API
4. Concatenate the 920-dim feature vector with YOLO's class output
5. Train a lightweight LightGBM classifier on the concatenated features
6. Evaluate under 5-fold cross-validation

## Results on DOTA v1.0

**YOLOv8l-OBB** (50,348 matched detections, 1,233 tile groups from 458 original scenes):

|             | YOLO Direct | Bolt-On |
| ----------- | ----------- | ------- |
| Weighted F1 | 0.9925      | 0.9930  |
| Macro F1    | 0.9826      | 0.9833  |
| helicopter  | 0.908       | 0.916   |
| small-vehicle | 0.984     | 0.986   |
| large-vehicle | 0.982     | 0.984   |

Error reduction: **6.1%** vs YOLO direct (tile-level cross-validation, one-hot baseline)

No class degraded by more than 0.004 F1 across all 15 categories.

### Cross-validation grouping note

DOTA's large images (up to 20,000x20,000 px) must be tiled into 1024x1024 patches with 200px overlap for inference. The default evaluation uses **tile-level GroupKFold**: tiles from the same original image at different spatial positions can appear in different folds. This mirrors a realistic deployment scenario where a small amount of labeled data from the operating environment is available and new detections from the same region need classification — the standard production workflow for any bolt-on reclassifier. The LightGBM classifier is trained once on the labeled data and then does inference on all new detections without retraining.

For strict image-level holdout where entire original images are held out per fold, pass `--strict-scene-split`. Under strict splitting the improvement on DOTA is marginal to slightly negative, consistent with the encoder capturing spatial context that transfers well within a deployment region but not across completely unseen aerial scenes in this particular dataset. Other benchmarks with unambiguous scene boundaries (xView2 disaster events, RarePlanes satellite passes) show **22-40% error reduction** under strict scene-level splits, suggesting cross-scene transfer depends on the diversity of objects and conditions across scenes. See [authorize.earth/r&d/spatial](https://authorize.earth/r&d/spatial) for the full solution brief.

## Same Encoder on Other Benchmarks

The identical frozen encoder improves classification across domains and modalities it has never seen:

| Benchmark  | Domain              | Modality      | Baseline Model                    | Error Reduction |
| ---------- | ------------------- | ------------- | --------------------------------- | --------------- |
| xView3     | Maritime vessels    | C-band SAR    | 1st-place CircleNet               | 5.2%            |
| DOTA       | Aerial objects      | HR aerial     | YOLOv8l-OBB                       | 6.1%            |
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
export AUTHORIZE_EARTH_API_KEY=your_earth_api_key
```

The public evaluation key is available at https://authorize.earth/r&d/spatial. Rate-limited to 3,000 requests/hour per IP and 15,000/hour globally. Without the key set the encoding step returns zero vectors and the script still runs but classification will not improve.

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

# Strict image-level holdout (see cross-validation note above)
python obb_feature_reclassifier.py bench \
  --images ./tiled/images \
  --labels ./tiled/labels \
  --strict-scene-split

# Skip detection on subsequent runs (uses cached results)
python obb_feature_reclassifier.py bench \
  --images ./tiled/images \
  --labels ./tiled/labels \
  --skip-detection
```

### Note on `--full-scores`

The `--full-scores` flag gives the baseline access to YOLO's full 15-dim pre-NMS class distribution instead of a one-hot encoding. Under these conditions the baseline representation is already richer, and the encoder provides marginal weighted F1 improvement but can hurt macro F1 on low-sample classes due to overfitting with the increased feature dimensionality. The default one-hot mode is recommended — it gives the encoder the most room to contribute orthogonal spatial information.

Only valid for NMS-based models like YOLOv8. YOLO26's end-to-end NMS-free architecture uses a one-to-one prediction head, so intermediate activations are not interpretable class distributions. Use the default one-hot mode for YOLO26.

## API Key

The script uses a public evaluation key that works immediately with no signup. Rate limits are 3,000 requests/hour per IP address and 15,000 requests/hour globally. For higher limits or dedicated access, contact jackk@authorize.earth.

## Expected Output

```
Running detection on 3143 tiles
Using tile-level grouping for cross-validation (deployment scenario)
Detecting: 100%|████████████████████████████████| 3143/3143 [02:48<00:00, 18.60it/s]
Matched detections: 50348
Encoding via API: 100%|█████████████████████████| 787/787 [47:45<00:00, 3.64s/it]

5-fold cross-validation (50348 detections, 1233 groups)
=================================================================
  Fold 1: bolt-on weighted F1 = 0.9920
  Fold 2: bolt-on weighted F1 = 0.9947
  Fold 3: bolt-on weighted F1 = 0.9940
  Fold 4: bolt-on weighted F1 = 0.9905
  Fold 5: bolt-on weighted F1 = 0.9935

─────────────────────────────────────────────────────────────────
  yolov8l-obb Direct   Bolt-On
  Weighted F1:  0.9925               0.9930
  Macro F1:     0.9826               0.9833
  Error reduction vs yolov8l-obb direct: 6.1%

  Class                  Direct     Bolt-On    Delta      n
  ──────────────────────────────────────────────────────────────
  helicopter             0.908      0.916      +0.008      118
  ground-track-field     0.981      0.984      +0.002      215
  soccer-ball-field      0.972      0.974      +0.002      248
  small-vehicle          0.984      0.986      +0.001      8855
  large-vehicle          0.982      0.984      +0.001      7929
  plane                  0.998      0.998      +0.000      4586
  ship                   0.999      0.999      +0.000      17191
  storage-tank           0.999      0.999      +0.000      3802
  baseball-diamond       0.991      0.991      +0.000      320
  harbor                 0.997      0.997      +0.000      3928
  bridge                 0.998      0.998      +0.000      601
  roundabout             0.987      0.987      +0.000      198
  swimming-pool          0.997      0.997      +0.000      619
  tennis-court           0.991      0.990      -0.001     1496
  basketball-court       0.955      0.951      -0.004     242
```

## Links

- Encoder API docs and live demo: https://authorize.earth/r&d/spatial
- Contact: jackk@authorize.earth
- Discussion: [ultralytics/ultralytics#23821](https://github.com/ultralytics/ultralytics/discussions/23821)

## Contributor

[Jack Kowalik](https://github.com/jackkowalik) — [Authorize Earth](https://authorize.earth), Troy, MI
