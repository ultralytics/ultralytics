---
title: Ultralytics Benchmark
comments: true
description: Ultralytics Benchmark measures object detection generalization across 37 curated datasets, from medical imaging and remote sensing to retail and sports.
keywords: Ultralytics Benchmark, cross-domain object detection, real-world datasets, model generalization, macro mAP, YOLO26, computer vision
---

# Ultralytics Benchmark

[COCO](coco.md) is not representative of many real-world [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv)
tasks. Ultralytics Benchmark is a [benchmark dataset](https://www.ultralytics.com/glossary/benchmark-dataset) for
measuring [model generalizability](<https://en.wikipedia.org/wiki/Generalization_(learning)>) across 37 curated
[object detection](https://www.ultralytics.com/glossary/object-detection) datasets with high-quality
[bounding box](https://www.ultralytics.com/glossary/bounding-box) labels. Its domains include
[medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis), remote sensing, traffic,
documents, user interfaces, retail, [machine vision](https://www.ultralytics.com/glossary/machine-vision), agriculture,
sports, wildlife, underwater imaging, and waste detection. Its sensor types include RGB, fisheye, thermal infrared,
X-ray, MRI, ultrasound, microscopy, and synthetic aperture radar (SAR).

The curated datasets are hosted on [Ultralytics Platform](../../platform/data/datasets.md). The
[VLM leaderboard](https://www.ultralytics.com/benchmarks/vlm) ranks vision language models and open-vocabulary
detectors on images from 36 of these datasets by accuracy, cost, and speed.

Candidate datasets were checked for exact and near-duplicate images across the training and validation splits. Flagged
pairs were reviewed, and confirmed split leaks were removed or the affected dataset was replaced.

## Datasets

| Domain          | Sensor / imagery               | Dataset                                                                                                            | Classes                                                                                                                                                                                                                                                      |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Aerial/RS       | RGB satellite imagery          | [Airbus Oil Storage Detection](https://platform.ultralytics.com/ultralytics/datasets/airbus-oil-storage-detection) | 1: oil-storage-tank                                                                                                                                                                                                                                          |
| Document/UI     | RGB and scanned pages          | [AnnoPage document elements](https://platform.ultralytics.com/ultralytics/datasets/annopage-doc-elements)          | 25: chemical formula/equation, symbol/logo, photograph, drawing, graph, map, mathematical expression, musical notation, image, decor, inscription, floor plan, barcode/QR code, stamp, advertisement, handwritten note, diagram, table, vignette, and frieze |
| Sports          | RGB camera                     | [Basketball Hoop Detection](https://platform.ultralytics.com/ultralytics/datasets/basketball-hoop-detection)       | 2: basketball, basketball net                                                                                                                                                                                                                                |
| Driving/Traffic | RGB road camera                | [CEYMO](https://platform.ultralytics.com/ultralytics/datasets/ceymo)                                               | 11: bus lane, cycle lane, diamond, junction box, left arrow, pedestrian crossing, right arrow, straight arrow, slow, straight-left arrow, straight-right arrow                                                                                               |
| Medical         | MRI                            | [Cheng Brain Tumor MRI](https://platform.ultralytics.com/ultralytics/datasets/cheng-brain-tumor-mri)               | 3: meningioma, glioma, pituitary                                                                                                                                                                                                                             |
| Medical         | Color fundus camera            | [DDR Lesion](https://platform.ultralytics.com/ultralytics/datasets/ddr-lesion)                                     | 4: hard exudates, hemorrhages, microaneurysms, soft exudates                                                                                                                                                                                                 |
| Industrial      | Linear-scan industrial camera  | [DeepPCB](https://platform.ultralytics.com/ultralytics/datasets/deeppcb)                                           | 6: open, short, mousebite, spur, copper, pin-hole                                                                                                                                                                                                            |
| Medical         | Dental X-ray                   | [DENTEX](https://platform.ultralytics.com/ultralytics/datasets/dentex)                                             | 4: Impacted, Caries, Periapical Lesion, Deep Caries                                                                                                                                                                                                          |
| Industrial      | RGB industrial camera          | [DSPCBSD](https://platform.ultralytics.com/ultralytics/datasets/dspcbsd-pcb-surface-defect)                        | 9: short, spur, spurious copper, open, mouse bite, hole breakout, conductor scratch, conductor foreign object, base material foreign object                                                                                                                  |
| Other           | RGB egocentric camera          | [EgoHands](https://platform.ultralytics.com/ultralytics/datasets/egohands-egocentric)                              | 4: own_left, own_right, other_left, other_right                                                                                                                                                                                                              |
| Other           | RGB and infrared trail camera  | [ENA24](https://platform.ultralytics.com/ultralytics/datasets/ena24)                                               | 22 wildlife, person, domestic-animal, and vehicle classes                                                                                                                                                                                                    |
| Driving/Traffic | RGB fisheye camera             | [Fisheye8K](https://platform.ultralytics.com/ultralytics/datasets/fisheye8k-traffic)                               | 5: Bus, Bike, Car, Pedestrian, Truck                                                                                                                                                                                                                         |
| Document/UI     | Digital and scanned plans      | [FloorPlanCAD](https://platform.ultralytics.com/ultralytics/datasets/floorplancad-2)                               | 30 architectural opening, furniture, appliance, bathroom fixture, stair, elevator, and escalator classes                                                                                                                                                     |
| Medical         | X-ray                          | [GRAZPEDWRI-DX](https://platform.ultralytics.com/ultralytics/datasets/grazpedwri-dx)                               | 9: boneanomaly, bonelesion, foreignbody, fracture, metal, periostealreaction, pronatorsign, softtissue, text                                                                                                                                                 |
| Document/UI     | Digital screenshots            | [GroundCUA UI](https://platform.ultralytics.com/ultralytics/datasets/groundcua-ui-yolo)                            | 9: Button, Information Display, Input Elements, Menu, Navigation, Others, Sidebar, Visual Elements, Unknown                                                                                                                                                  |
| Agriculture     | RGB field camera               | [GWHD 2021](https://platform.ultralytics.com/ultralytics/datasets/global-wheat-head-detection-2021)                | 1: wheat_head                                                                                                                                                                                                                                                |
| Aerial/RS       | Thermal infrared UAV           | [HIT-UAV](https://platform.ultralytics.com/ultralytics/datasets/hit-uav)                                           | 5: person, car, bicycle, other vehicle, ignored region (not scored)                                                                                                                                                                                          |
| Medical         | Security X-ray                 | [IED X-ray](https://platform.ultralytics.com/ultralytics/datasets/iedxray)                                         | 11 explosive, battery, laptop, pager, mobile phone, walkie-talkie, and modified-device classes                                                                                                                                                               |
| Medical         | RGB endoscope                  | [Kvasir-SEG Polyp](https://platform.ultralytics.com/ultralytics/datasets/kvasir-seg-polyp)                         | 1: polyp                                                                                                                                                                                                                                                     |
| Driving/Traffic | Thermal infrared               | [LLVIP](https://platform.ultralytics.com/ultralytics/datasets/llvip)                                               | 1: person                                                                                                                                                                                                                                                    |
| Aerial/RS       | 8-band multispectral imagery   | [MODA Multispectral](https://platform.ultralytics.com/ultralytics/datasets/moda-multispectral-obb)                 | 8: car, van, truck, bus, tricycle, bike, awning-bike, pedestrian                                                                                                                                                                                             |
| Agriculture     | RGB camera                     | [PlantDoc](https://platform.ultralytics.com/ultralytics/datasets/plantdoc)                                         | 29 crop leaf and plant-disease classes                                                                                                                                                                                                                       |
| Sports          | RGB video frames               | [RacketVision Tennis Ball](https://platform.ultralytics.com/ultralytics/datasets/racketvision-tennis-ball-yolo)    | 1: tennis ball                                                                                                                                                                                                                                               |
| Driving/Traffic | RGB road camera                | [RDD2022](https://platform.ultralytics.com/ultralytics/datasets/rdd2022)                                           | 5: longitudinal crack, transverse crack, alligator crack, other corruption, pothole                                                                                                                                                                          |
| Retail          | RGB checkout camera            | [RPC Retail Checkout](https://platform.ultralytics.com/ultralytics/datasets/rpc-retail-checkout)                   | 200 checkout product classes covering packaged food, drinks, milk, canned food, chocolate, gum, candy, seasonings, hygiene products, tissues, and stationery                                                                                                 |
| Other           | RGB underwater camera          | [RUOD Underwater](https://platform.ultralytics.com/ultralytics/datasets/ruod-underwater)                           | 10: holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish                                                                                                                                                              |
| Industrial      | RGB camera                     | [Safe Human 17](https://platform.ultralytics.com/ultralytics/datasets/safe-human-17)                               | 17 person, body-part, tool, and personal protective equipment classes                                                                                                                                                                                        |
| Aerial/RS       | RGB UAV camera                 | [SeaDronesSee](https://platform.ultralytics.com/ultralytics/datasets/seadronessee-2)                               | 6: ignored, swimmer, boat, jetski, life_saving_appliances, buoy                                                                                                                                                                                              |
| Retail          | RGB retail camera              | [SKU-110K](https://platform.ultralytics.com/ultralytics/datasets/sku110k-2)                                        | 1: shelf product                                                                                                                                                                                                                                             |
| Sports          | RGB broadcast camera           | [SoccerDB](https://platform.ultralytics.com/ultralytics/datasets/soccerdb)                                         | 3: player, ball, goal                                                                                                                                                                                                                                        |
| Aerial/RS       | Synthetic aperture radar (SAR) | [SSDD SAR Ship](https://platform.ultralytics.com/ultralytics/datasets/ssdd-sar-ship)                               | 1: ship                                                                                                                                                                                                                                                      |
| Other           | RGB camera                     | [TACO Litter](https://platform.ultralytics.com/ultralytics/datasets/taco-detect)                                   | 28 litter material and object classes                                                                                                                                                                                                                        |
| Medical         | Chest X-ray                    | [TBX11K](https://platform.ultralytics.com/ultralytics/datasets/tbx11k-tb-only)                                     | 2: ActiveTuberculosis, ObsoletePulmonaryTuberculosis                                                                                                                                                                                                         |
| Medical         | Ultrasound                     | [TN5000 Thyroid Ultrasound](https://platform.ultralytics.com/ultralytics/datasets/tn5000-thyroid-us)               | 2: benign, malignant                                                                                                                                                                                                                                         |
| Medical         | Light microscopy               | [TXL-PBC Blood Cells](https://platform.ultralytics.com/ultralytics/datasets/txl-pbc-blood-cells)                   | 3: white blood cells, red blood cells, platelets                                                                                                                                                                                                             |
| Other           | RGB camera                     | [UEC School Lunch Food](https://platform.ultralytics.com/ultralytics/datasets/uec-school-lunch-food-2)             | 21 school-lunch food and drink classes                                                                                                                                                                                                                       |
| Aerial/RS       | RGB UAV camera                 | [VisDrone](https://platform.ultralytics.com/ultralytics/datasets/visdrone)                                         | 10: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor                                                                                                                                                                      |

## Evaluation

- Sample up to 1,000 images from each training split, keep the full validation split, and exclude the test split
  with `fraction=[1000, 1.0, 0.0]`.
- Fine-tune the same pretrained checkpoint independently on each dataset for 100 epochs with patience 100,
  seed 0, batch 16, and four workers. Images use 640 pixels. Seed 0 is not fully deterministic, so YOLO27m and
  YOLO27l report the mean of 3 such runs.
- Enable class remapping, channels-last, and prediction JSON output, with plots disabled.
- Evaluate each dataset's best checkpoint and reuse its saved `max_det`, which accounts for its largest object count.
- Apply ignore regions as in the VisDrone DET toolkit: drop predictions and labels that an ignored region covers by at
  least half. These are VisDrone's 1,378 official ignored regions and HIT-UAV's 7 `ignored region` boxes, and the
  HIT-UAV class is then removed.
- For size-specific mAP<sub>50-95</sub>, multiply ground-truth and prediction box areas by
  `(640 / max(image_width, image_height)) ** 2` before applying the COCO Small, Medium, and Large bins.
- Give every dataset equal weight. Each size metric includes only datasets with at least 10 objects in that bin.

Macro [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) is
`sum(dataset validation mAP50-95) / 37`. Metrics are calculated from each dataset's saved predictions.

## Benchmark results

Metrics are macro averages across the 37 validation splits, reported as percentages. YOLO27 is an unreleased
[preview](../../models/yolo27.md); its weights are not yet available.

| Size | Model                                                                   | mAP<sub>50-95</sub> | Precision |    Recall | mAP<sub>50-95</sub>-S | mAP<sub>50-95</sub>-M | mAP<sub>50-95</sub>-L |
| :--: | ----------------------------------------------------------------------- | ------------------: | --------: | --------: | --------------------: | --------------------: | --------------------: |
| L/X  | YOLO27l                                                                 |           **55.97** | **79.11** | **75.14** |             **38.52** |             **55.90** |             **64.64** |
| L/X  | [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x)  |               50.19 |     76.93 |     69.08 |                 34.89 |                 50.78 |                 52.15 |
| L/X  | [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l)  |               50.05 |     75.28 |     69.53 |                 34.96 |                 50.37 |                 52.25 |
| L/X  | [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l)  |               48.71 |     74.26 |     68.50 |                 32.29 |                 50.84 |                 51.14 |
| L/X  | [YOLOv5l](https://platform.ultralytics.com/ultralytics/yolov5/yolov5lu) |               48.25 |     74.03 |     68.16 |                 32.15 |                 49.89 |                 50.27 |
| L/X  | [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l)  |               48.22 |     74.15 |     69.14 |                 32.44 |                 49.14 |                 49.98 |
| L/X  | [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x)  |               47.94 |     76.00 |     66.82 |                 31.60 |                 49.85 |                 50.16 |
| L/X  | [YOLOv5x](https://platform.ultralytics.com/ultralytics/yolov5/yolov5xu) |               47.58 |     75.05 |     68.15 |                 31.66 |                 49.60 |                 50.18 |
| L/X  | [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x)  |               47.49 |     73.75 |     67.50 |                 31.64 |                 48.82 |                 49.66 |
|  M   | YOLO27m                                                                 |           **53.68** | **76.66** | **73.20** |             **36.47** |             **53.65** |             **61.30** |
|  M   | [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m)  |               49.74 |     76.23 |     68.74 |                 33.40 |                 50.29 |                 51.78 |
|  M   | [YOLOv5m](https://platform.ultralytics.com/ultralytics/yolov5/yolov5mu) |               48.41 |     74.34 |     68.39 |                 32.12 |                 49.78 |                 52.39 |
|  M   | [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m)  |               48.39 |     73.87 |     68.85 |                 32.62 |                 49.87 |                 51.23 |
|  M   | [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m)  |               48.25 |     73.89 |     68.69 |                 31.21 |                 49.73 |                 52.41 |
|  S   | YOLO27s                                                                 |           **51.76** | **76.68** | **71.46** |             **35.19** |             **51.68** |             **54.53** |
|  S   | [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s)  |               48.00 |     75.00 |     67.85 |                 29.98 |                 49.81 |                 51.13 |
|  S   | [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s)  |               47.77 |     74.24 |     67.23 |                 32.00 |                 47.91 |                 51.00 |
|  S   | [YOLOv5s](https://platform.ultralytics.com/ultralytics/yolov5/yolov5su) |               47.55 |     74.18 |     67.37 |                 30.95 |                 49.13 |                 50.57 |
|  S   | [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s)  |               47.53 |     74.27 |     67.68 |                 31.38 |                 48.52 |                 50.57 |
|  N   | YOLO27n                                                                 |           **49.32** | **75.46** | **68.82** |             **32.41** |             **49.27** |             **54.05** |
|  N   | [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n)  |               46.37 |     72.86 |     65.94 |                 28.76 |                 47.53 |                 51.05 |
|  N   | [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n)  |               46.09 |     72.53 |     65.89 |                 29.02 |                 47.61 |                 49.62 |
|  N   | [YOLOv5n](https://platform.ultralytics.com/ultralytics/yolov5/yolov5nu) |               45.29 |     71.77 |     65.21 |                 29.03 |                 46.83 |                 48.59 |
|  N   | [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n)  |               44.37 |     70.25 |     63.40 |                 29.12 |                 44.77 |                 48.39 |

## Benchmarking

Use the `main` branch of Ultralytics with `ULTRALYTICS_API_KEY` set for dataset access:

```python
from ultralytics import YOLO

datasets = [
    "ul://ultralytics/datasets/airbus-oil-storage-detection",
    "ul://ultralytics/datasets/annopage-doc-elements",
    "ul://ultralytics/datasets/basketball-hoop-detection",
    "ul://ultralytics/datasets/ceymo",
    "ul://ultralytics/datasets/cheng-brain-tumor-mri",
    "ul://ultralytics/datasets/ddr-lesion",
    "ul://ultralytics/datasets/deeppcb",
    "ul://ultralytics/datasets/dentex",
    "ul://ultralytics/datasets/dspcbsd-pcb-surface-defect",
    "ul://ultralytics/datasets/egohands-egocentric",
    "ul://ultralytics/datasets/ena24",
    "ul://ultralytics/datasets/fisheye8k-traffic",
    "ul://ultralytics/datasets/floorplancad-2",
    "ul://ultralytics/datasets/grazpedwri-dx",
    "ul://ultralytics/datasets/groundcua-ui-yolo",
    "ul://ultralytics/datasets/global-wheat-head-detection-2021",
    "ul://ultralytics/datasets/hit-uav",
    "ul://ultralytics/datasets/iedxray",
    "ul://ultralytics/datasets/kvasir-seg-polyp",
    "ul://ultralytics/datasets/llvip",
    "ul://ultralytics/datasets/moda-multispectral-obb",
    "ul://ultralytics/datasets/plantdoc",
    "ul://ultralytics/datasets/racketvision-tennis-ball-yolo",
    "ul://ultralytics/datasets/rdd2022",
    "ul://ultralytics/datasets/rpc-retail-checkout",
    "ul://ultralytics/datasets/ruod-underwater",
    "ul://ultralytics/datasets/safe-human-17",
    "ul://ultralytics/datasets/seadronessee-2",
    "ul://ultralytics/datasets/sku110k-2",
    "ul://ultralytics/datasets/soccerdb",
    "ul://ultralytics/datasets/ssdd-sar-ship",
    "ul://ultralytics/datasets/taco-detect",
    "ul://ultralytics/datasets/tbx11k-tb-only",
    "ul://ultralytics/datasets/tn5000-thyroid-us",
    "ul://ultralytics/datasets/txl-pbc-blood-cells",
    "ul://ultralytics/datasets/uec-school-lunch-food-2",
    "ul://ultralytics/datasets/visdrone",
]

results = YOLO("yolo26n.pt").train(
    data=datasets,
    epochs=100,
    patience=100,
    imgsz=640,
    batch=16,
    fraction=[1000, 1.0, 0.0],
    seed=0,
    workers=4,
    cls_remap=True,
    channels_last=True,
    save_json=True,
    plots=False,
)
# The main branch reports native-area size bins. This benchmark separately rescales GT and prediction areas
# by (640 / max(image_width, image_height)) ** 2 before scoring size-specific AP, and applies the VisDrone and
# HIT-UAV ignore regions. This snippet does neither.
```

## Citation

If you use Ultralytics Benchmark in your work, please cite the
[Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748):

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @misc{jocher2026ultralyticsyolo26unifiedrealtime,
          title = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
          author = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
          year = {2026},
          eprint = {2606.03748},
          archivePrefix = {arXiv},
          primaryClass = {cs.CV},
          doi = {10.48550/arXiv.2606.03748},
          url = {https://arxiv.org/abs/2606.03748},
        }
        ```
