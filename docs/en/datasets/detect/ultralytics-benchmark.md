---
title: Ultralytics Benchmark
comments: true
description: Explore Ultralytics Benchmark across medical imaging, remote sensing, documents, user interfaces, retail, industry, agriculture, and sports.
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

The curated datasets are hosted on [Ultralytics Platform](../../platform/data/datasets.md).

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
| Aerial/RS       | Thermal infrared UAV           | [HIT-UAV](https://platform.ultralytics.com/ultralytics/datasets/hit-uav)                                           | 5: person, car, bicycle, other vehicle, ignored region                                                                                                                                                                                                       |
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
- For size-specific mAP50-95, multiply ground-truth and prediction box areas by
  `(640 / max(image_width, image_height)) ** 2` before applying the COCO Small, Medium, and Large bins.
- Give every dataset equal weight. Each size metric includes only datasets with at least 10 objects in that bin.

Macro [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) is
`sum(dataset validation mAP50-95) / 37`. Metrics are calculated from each dataset's saved predictions.

## Benchmark results

Metrics are macro averages across the 37 validation splits, reported as percentages.

| Size | Model                                                                   |  mAP50-95 | Precision |    Recall | mAP50-95-S | mAP50-95-M | mAP50-95-L |
| :--: | ----------------------------------------------------------------------- | --------: | --------: | --------: | ---------: | ---------: | ---------: |
| L/X  | YOLO27l                                                                 | **55.50** | **78.83** | **74.90** |  **38.17** |  **55.75** |  **63.36** |
| L/X  | [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x)  |     49.85 |     76.74 |     68.61 |      34.44 |      50.54 |      51.69 |
| L/X  | [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l)  |     49.72 |     75.05 |     69.31 |      34.49 |      50.16 |      51.72 |
| L/X  | [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l)  |     48.53 |     74.22 |     68.33 |      31.91 |      50.71 |      50.77 |
| L/X  | [YOLOv5l](https://platform.ultralytics.com/ultralytics/yolov5/yolov5lu) |     48.03 |     74.11 |     67.78 |      31.73 |      49.79 |      49.87 |
| L/X  | [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l)  |     48.02 |     73.95 |     68.80 |      32.05 |      49.04 |      49.56 |
| L/X  | [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x)  |     47.67 |     75.66 |     66.46 |      31.12 |      49.60 |      49.69 |
| L/X  | [YOLOv5x](https://platform.ultralytics.com/ultralytics/yolov5/yolov5xu) |     47.35 |     74.69 |     67.96 |      31.27 |      49.39 |      49.76 |
| L/X  | [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x)  |     47.24 |     73.69 |     67.08 |      31.22 |      48.57 |      49.24 |
|  M   | YOLO27m                                                                 | **53.38** |     75.85 | **72.95** |  **35.96** |  **53.40** |  **61.40** |
|  M   | [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m)  |     49.42 | **76.23** |     68.31 |      32.91 |      50.02 |      51.32 |
|  M   | [YOLOv5m](https://platform.ultralytics.com/ultralytics/yolov5/yolov5mu) |     48.19 |     74.21 |     68.07 |      31.68 |      49.67 |      51.93 |
|  M   | [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m)  |     48.15 |     73.72 |     68.56 |      32.14 |      49.76 |      50.75 |
|  M   | [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m)  |     47.99 |     73.86 |     68.15 |      30.73 |      49.61 |      51.97 |
|  S   | YOLO27s                                                                 | **51.23** | **76.96** | **70.74** |  **34.20** |  **51.47** |  **54.74** |
|  S   | [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s)  |     47.78 |     74.72 |     67.66 |      29.53 |      49.69 |      50.69 |
|  S   | [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s)  |     47.47 |     74.35 |     66.73 |      31.56 |      47.72 |      50.58 |
|  S   | [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s)  |     47.35 |     73.97 |     67.54 |      30.90 |      48.47 |      50.15 |
|  S   | [YOLOv5s](https://platform.ultralytics.com/ultralytics/yolov5/yolov5su) |     47.30 |     73.97 |     67.02 |      30.47 |      49.07 |      50.10 |
|  N   | YOLO27n                                                                 | **49.25** | **74.86** | **68.49** |  **31.93** |  **49.30** |  **53.20** |
|  N   | [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n)  |     46.19 |     72.83 |     65.69 |      28.35 |      47.51 |      50.66 |
|  N   | [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n)  |     45.89 |     72.29 |     65.56 |      28.56 |      47.55 |      49.22 |
|  N   | [YOLOv5n](https://platform.ultralytics.com/ultralytics/yolov5/yolov5nu) |     45.09 |     71.69 |     64.99 |      28.61 |      46.79 |      48.14 |
|  N   | [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n)  |     44.08 |     70.09 |     63.20 |      28.69 |      44.52 |      47.96 |

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

results = YOLO("yolo27n.pt").train(
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
# by (640 / max(image_width, image_height)) ** 2 before scoring size-specific AP.
```

## Citation

If you use Ultralytics Benchmark in your work, please cite the
[Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748):

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
